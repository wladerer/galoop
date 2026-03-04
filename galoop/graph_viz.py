"""
galoop/graph_viz.py

Interactive Plotly-based viewer for adsorbate chemical-environment graphs.
Generates a self-contained HTML file — no server required.
Uses scatter3d so the graph can be freely rotated with mouse drag.
"""

from __future__ import annotations

import json
from pathlib import Path

_CPK_COLORS: dict[str, str] = {
    "H":  "#ffffff",
    "C":  "#404040",
    "N":  "#3050f8",
    "O":  "#ff0d0d",
    "F":  "#90e050",
    "S":  "#ffff30",
    "Cl": "#1ff01f",
    "Fe": "#e06633",
    "Co": "#f090a0",
    "Ni": "#50d050",
    "Cu": "#c88033",
    "Zn": "#7d80b0",
    "Pd": "#006985",
    "Ag": "#c0c0c0",
    "Pt": "#d0d0e0",
    "Au": "#ffd123",
    "Ir": "#175487",
    "Rh": "#0a7d8c",
    "Ru": "#248f8f",
    "Os": "#266696",
}

_EDGE_COLORS = {
    0: "#ff7f0e",   # ads-ads bonds
    2: "#aaaaaa",   # slab / surface-ads bonds
}


def _element_color(symbol: str) -> str:
    """Return a CPK hex color for *symbol*, gray fallback."""
    return _CPK_COLORS.get(symbol, "#aaaaaa")


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex color and alpha to an 'rgba(r,g,b,a)' string."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _nx_to_traces(G, pos: dict) -> list[dict]:
    """
    Convert a networkx Graph *G* to a list of serializable Plotly scatter3d
    trace dicts.

    Parameters
    ----------
    G   : networkx.Graph with node attrs (index, ads, central_ads) and
          edge attrs (bond, ads_only)
    pos : {node_name: [x, y, z]} 3-D layout dict

    Returns
    -------
    list of trace dicts (edge traces + per-element node traces)
    """
    # --- Edge traces grouped by ads_only value ---
    edge_groups: dict[int, tuple[list, list, list]] = {}
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        ads_only = data.get("ads_only", 2)
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        xs, ys, zs = edge_groups.setdefault(ads_only, ([], [], []))
        xs += [x0, x1, None]
        ys += [y0, y1, None]
        zs += [z0, z1, None]

    traces: list[dict] = []
    for ads_only, (xs, ys, zs) in edge_groups.items():
        color = _EDGE_COLORS.get(ads_only, "#aaaaaa")
        label = "ads-ads" if ads_only == 0 else "slab/surface"
        traces.append({
            "type": "scatter3d",
            "x": xs, "y": ys, "z": zs,
            "mode": "lines",
            "line": {"color": color, "width": 3},
            "hoverinfo": "none",
            "name": label,
            "showlegend": True,
        })

    # --- Node traces grouped by element symbol ---
    element_nodes: dict[str, list] = {}
    for node, data in G.nodes(data=True):
        if node not in pos:
            continue
        symbol = node.split(":")[0] if ":" in node else node
        element_nodes.setdefault(symbol, []).append((node, data, pos[node]))

    for symbol, node_list in sorted(element_nodes.items()):
        xs, ys, zs, texts, sizes, colors = [], [], [], [], [], []
        for node, data, (x, y, z) in node_list:
            is_ads = data.get("ads", False)
            is_central = data.get("central_ads", False)
            idx = data.get("index", "?")
            xs.append(x); ys.append(y); zs.append(z)
            texts.append(f"{symbol} idx={idx}  ads={is_ads}  central={is_central}")
            sizes.append(10 if is_central else (7 if is_ads else 5))
            alpha = 1.0 if is_ads else 0.3
            colors.append(_hex_to_rgba(_element_color(symbol), alpha))

        traces.append({
            "type": "scatter3d",
            "x": xs, "y": ys, "z": zs,
            "mode": "markers",
            "marker": {
                "color": colors,
                "size": sizes,
                "line": {"color": "#333333", "width": 0.5},
            },
            "text": texts,
            "hoverinfo": "text",
            "name": symbol,
            "showlegend": True,
        })

    return traces


def _layout(G) -> dict:
    """
    Compute a 3-D spring layout for *G*.

    Returns ``{node_name: [x, y, z]}`` as a plain dict.
    """
    import networkx as nx
    raw = nx.spring_layout(G, seed=42, dim=3)
    return {k: [float(v[0]), float(v[1]), float(v[2])] for k, v in raw.items()}


def build_page(title: str, chem_envs: list) -> dict:
    """
    Build a single page dict from a list of chemical-environment graphs.

    Multiple environments are laid out side-by-side with x-axis offsets in
    3-D space.

    Parameters
    ----------
    title     : page title string (shown in the nav bar)
    chem_envs : list of networkx.Graph objects

    Returns
    -------
    {"title": str, "traces": list[dict]}
    """
    all_traces: list[dict] = []
    n = len(chem_envs)
    x_offset = 2.5

    for i, G in enumerate(chem_envs):
        pos = _layout(G)
        offset = (i - n / 2.0 + 0.5) * x_offset
        pos = {k: [v[0] + offset, v[1], v[2]] for k, v in pos.items()}
        traces = _nx_to_traces(G, pos)
        if i > 0:
            for t in traces:
                t["showlegend"] = False
        all_traces.extend(traces)

    return {"title": title, "traces": all_traces}


def generate_html(pages: list[dict], output: Path) -> None:
    """
    Write a self-contained interactive HTML file.

    Navigation: Prev / Next buttons, dropdown jump, keyboard arrow keys.
    The 3-D camera angle is preserved as you navigate between pages.

    Parameters
    ----------
    pages  : list of page dicts from :func:`build_page`
    output : destination Path
    """
    pages_json = json.dumps(pages)
    n = len(pages)

    scene_config = json.dumps({
        "bgcolor": "#1a1a2e",
        "xaxis": {"visible": False, "showgrid": False, "zeroline": False,
                  "showline": False, "showticklabels": False, "showspikes": False},
        "yaxis": {"visible": False, "showgrid": False, "zeroline": False,
                  "showline": False, "showticklabels": False, "showspikes": False},
        "zaxis": {"visible": False, "showgrid": False, "zeroline": False,
                  "showline": False, "showticklabels": False, "showspikes": False},
        "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 0.8}},
    })

    base_layout = json.dumps({
        "paper_bgcolor": "#1a1a2e",
        "font": {"color": "#e0e0e0", "size": 11},
        "legend": {
            "bgcolor": "rgba(30,30,60,0.85)",
            "bordercolor": "#444",
            "borderwidth": 1,
        },
        "margin": {"l": 0, "r": 0, "t": 40, "b": 0},
        "hovermode": "closest",
    })

    options_html = "\n".join(
        f'<option value="{i}">{p["title"]}</option>'
        for i, p in enumerate(pages)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>galoop — Chemical Environment Graphs</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #0f0f23;
      color: #e0e0e0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      display: flex;
      flex-direction: column;
      height: 100vh;
    }}
    #header {{
      background: #1a1a3e;
      padding: 8px 20px;
      border-bottom: 1px solid #333;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.4px;
    }}
    #graph-div {{ flex: 1; min-height: 0; }}
    #nav {{
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 8px 20px;
      background: #1a1a3e;
      border-top: 1px solid #333;
      flex-wrap: wrap;
    }}
    #nav button {{
      background: #2d2d5e;
      color: #e0e0e0;
      border: 1px solid #555;
      padding: 5px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 13px;
    }}
    #nav button:hover {{ background: #3d3d7e; }}
    #counter {{ font-size: 13px; min-width: 60px; text-align: center; }}
    #title-display {{
      font-size: 12px;
      color: #aaa;
      flex: 1;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    #jump {{
      background: #2d2d5e;
      color: #e0e0e0;
      border: 1px solid #555;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      max-width: 320px;
    }}
    #jump option {{ background: #1a1a3e; }}
    #hint {{ font-size: 11px; color: #666; }}
  </style>
</head>
<body>
  <div id="header">galoop &mdash; Chemical Environment Graphs ({n} page{'s' if n != 1 else ''})</div>
  <div id="graph-div"></div>
  <div id="nav">
    <button onclick="prev()">&#8592; Prev</button>
    <span id="counter">1 / {n}</span>
    <button onclick="next()">Next &#8594;</button>
    <span id="title-display"></span>
    <select id="jump" onchange="show(parseInt(this.value))">{options_html}</select>
    <span id="hint">drag to rotate &middot; scroll to zoom &middot; &larr;&rarr; keys</span>
  </div>
  <script>
    const pages = {pages_json};
    const baseLayout = {base_layout};
    const defaultScene = {scene_config};
    let current = 0;

    function show(i) {{
      current = ((i % pages.length) + pages.length) % pages.length;
      const page = pages[current];
      const gd = document.getElementById('graph-div');

      // Preserve the camera angle the user has rotated to
      let scene = defaultScene;
      if (gd.layout && gd.layout.scene && gd.layout.scene.camera) {{
        scene = Object.assign({{}}, defaultScene, {{camera: gd.layout.scene.camera}});
      }}

      Plotly.react(gd, page.traces, Object.assign({{}}, baseLayout, {{
        title: {{ text: page.title, font: {{ color: '#cccccc', size: 12 }} }},
        scene: scene,
      }}));

      document.getElementById('counter').textContent = (current + 1) + ' / ' + pages.length;
      document.getElementById('title-display').textContent = page.title;
      document.getElementById('jump').value = current;
    }}

    function prev() {{ show(current - 1); }}
    function next() {{ show(current + 1); }}

    document.addEventListener('keydown', function(e) {{
      if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')   {{ prev(); e.preventDefault(); }}
      if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {{ next(); e.preventDefault(); }}
    }});

    show(0);
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")
