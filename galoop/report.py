"""
galoop/report.py

Generate a self-contained HTML status report for a galoop run.

Core sections (pure SVG, no external deps, works offline):
  - Summary cards
  - Status breakdown
  - GCE evolution scatter
  - Operator performance stacked bar
  - Coverage landscape strip chart
  - Adsorbate sampling heatmap
  - Duplicate clustering table
  - Top-N structures table

3D structure viewer (requires internet for 3Dmol.js CDN):
  - Interactive viewer for top structures using 3Dmol.js
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from galoop.config import GaloopConfig

log = logging.getLogger(__name__)

# Viridis colormap key points (R, G, B)
_VIRIDIS = [
    (68,  1,  84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (110, 206, 88),
    (181, 222, 43), (253, 231, 37),
]


def _viridis(t: float) -> str:
    t = max(0.0, min(1.0, t))
    n = len(_VIRIDIS) - 1
    lo = int(t * n)
    hi = min(lo + 1, n)
    f = t * n - lo
    r = int(_VIRIDIS[lo][0] * (1 - f) + _VIRIDIS[hi][0] * f)
    g = int(_VIRIDIS[lo][1] * (1 - f) + _VIRIDIS[hi][1] * f)
    b = int(_VIRIDIS[lo][2] * (1 - f) + _VIRIDIS[hi][2] * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def _collect_adsorbate_xy(df, n_slab_atoms: int) -> list[tuple[float, float]]:
    """Return fractional (x, y) positions of all adsorbate atoms across all structures."""
    try:
        import numpy as np
        from ase.io import read as ase_read
    except ImportError:
        return []

    xys: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        gpath = row.get("geometry_path")
        if not gpath:
            continue
        gpath = Path(gpath)
        target = gpath.parent / "CONTCAR"
        if not target.exists():
            target = gpath.parent / "POSCAR"
        if not target.exists():
            continue
        try:
            atoms = ase_read(str(target), format="vasp")
            if len(atoms) <= n_slab_atoms:
                continue
            cell_inv = np.linalg.inv(atoms.cell[:].T)
            for pos in atoms[n_slab_atoms:].get_positions():
                frac = cell_inv @ pos
                fx, fy = float(frac[0] % 1.0), float(frac[1] % 1.0)
                if np.isfinite(fx) and np.isfinite(fy):
                    xys.append((fx, fy))
        except Exception:
            pass
    return xys


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

def _svg_heatmap(
    xys: list[tuple[float, float]],
    a_len: float,
    b_len: float,
    n_bins: int = 30,
    width: int = 560,
    height: int = 320,
) -> str:
    import numpy as np

    pad_l, pad_r, pad_t, pad_b = 50, 80, 24, 36
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    if not xys:
        return (
            f'<svg width="{width}" height="{height}" '
            f'style="background:#1e2a3a;border-radius:6px">'
            f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
            f'fill="#888" font-size="14">No adsorbate positions found</text></svg>'
        )

    hist = np.zeros((n_bins, n_bins), dtype=int)
    for fx, fy in xys:
        ix = min(int(fx * n_bins), n_bins - 1)
        iy = min(int(fy * n_bins), n_bins - 1)
        hist[iy, ix] += 1

    max_count = int(hist.max()) or 1
    bw = plot_w / n_bins
    bh = plot_h / n_bins

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#1e2a3a;border-radius:6px">',
    ]

    for iy in range(n_bins):
        for ix in range(n_bins):
            t = hist[iy, ix] / max_count
            x = pad_l + ix * bw
            y = pad_t + iy * bh
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" '
                f'width="{bw:.2f}" height="{bh:.2f}" fill="{_viridis(t)}"/>'
            )

    lines.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#aaa" stroke-width="1"/>'
    )
    lines += [
        f'<text x="{pad_l + plot_w / 2}" y="{height - 4}" '
        f'text-anchor="middle" fill="#aaa" font-size="11">a ({a_len:.2f} Å)</text>',
        f'<text x="12" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#aaa" font-size="11" '
        f'transform="rotate(-90,12,{pad_t + plot_h / 2})">b ({b_len:.2f} Å)</text>',
        f'<text x="{pad_l + plot_w // 2}" y="{pad_t - 6}" '
        f'text-anchor="middle" fill="#7a90a8" font-size="10">'
        f'{len(xys)} adsorbate atom positions</text>',
    ]

    cb_x = pad_l + plot_w + 10
    cb_w, n_stops = 14, 20
    for i in range(n_stops):
        t = i / n_stops
        seg_h = plot_h / n_stops + 1
        y = pad_t + (1.0 - t) * plot_h
        lines.append(
            f'<rect x="{cb_x}" y="{y:.2f}" width="{cb_w}" height="{seg_h:.2f}" '
            f'fill="{_viridis(t)}"/>'
        )
    lines += [
        f'<text x="{cb_x + cb_w + 4}" y="{pad_t + 4}" '
        f'fill="#888" font-size="9" dominant-baseline="middle">{max_count}</text>',
        f'<text x="{cb_x + cb_w + 4}" y="{pad_t + plot_h // 2}" '
        f'fill="#888" font-size="9" dominant-baseline="middle">{max_count // 2}</text>',
        f'<text x="{cb_x + cb_w + 4}" y="{pad_t + plot_h}" '
        f'fill="#888" font-size="9" dominant-baseline="middle">0</text>',
    ]
    lines.append("</svg>")
    return "\n".join(lines)


_STATUS_COLOR = {
    "converged": "#2ecc71",
    "pending":   "#3498db",
    "submitted": "#f1c40f",
    "failed":    "#e74c3c",
    "desorbed":  "#e67e22",
    "duplicate": "#95a5a6",
}
_STATUS_ORDER = ["converged", "pending", "submitted", "failed", "desorbed", "duplicate"]


def _svg_scatter(
    xs: list[float],
    ys: list[float],
    run_min: list[float],
    width: int = 700,
    height: int = 260,
) -> str:
    """Return an SVG scatter plot with a running-minimum line."""
    if not xs:
        return (
            f'<svg width="{width}" height="{height}" '
            f'style="background:#1e2a3a;border-radius:6px">'
            f'<text x="{width//2}" y="{height//2}" text-anchor="middle" '
            f'fill="#888" font-size="14">No converged structures yet</text></svg>'
        )

    pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    y_margin = max((max_y - min_y) * 0.08, 0.05)
    min_y -= y_margin
    max_y += y_margin

    def px(v: float) -> float:
        if max_x == min_x:
            return pad_l + plot_w / 2
        return pad_l + (v - min_x) / (max_x - min_x) * plot_w

    def py(v: float) -> float:
        if max_y == min_y:
            return pad_t + plot_h / 2
        return pad_t + (1.0 - (v - min_y) / (max_y - min_y)) * plot_h

    n_y_ticks = 5
    y_step = (max_y - min_y) / n_y_ticks
    y_ticks = [min_y + i * y_step for i in range(n_y_ticks + 1)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#1e2a3a;border-radius:6px">',
        *(
            f'<line x1="{pad_l}" y1="{py(t):.1f}" x2="{width - pad_r}" y2="{py(t):.1f}" '
            f'stroke="#2c3e50" stroke-width="1"/>'
            for t in y_ticks
        ),
        *(
            f'<text x="{pad_l - 6}" y="{py(t):.1f}" text-anchor="end" '
            f'fill="#888" font-size="10" dominant-baseline="middle">{t:.2f}</text>'
            for t in y_ticks
        ),
    ]

    n_x_ticks = min(6, len(xs))
    if n_x_ticks > 1:
        x_step = (max_x - min_x) / (n_x_ticks - 1)
        for i in range(n_x_ticks):
            xv = min_x + i * x_step
            lines.append(
                f'<text x="{px(xv):.1f}" y="{height - 8}" text-anchor="middle" '
                f'fill="#888" font-size="10">{int(round(xv))}</text>'
            )

    lines += [
        f'<text x="{pad_l + plot_w // 2}" y="{height - 0}" '
        f'text-anchor="middle" fill="#aaa" font-size="11">Evaluation</text>',
        f'<text x="12" y="{pad_t + plot_h // 2}" '
        f'text-anchor="middle" fill="#aaa" font-size="11" '
        f'transform="rotate(-90,12,{pad_t + plot_h // 2})">G (eV)</text>',
    ]

    if len(run_min) > 1:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, run_min))
        lines.append(
            f'<polyline points="{pts}" fill="none" stroke="#2ecc71" stroke-width="2"/>'
        )

    for x, y in zip(xs, ys):
        lines.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" '
            f'fill="#3498db" fill-opacity="0.75"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _svg_operator_stats(df, width: int = 700) -> str:
    """Horizontal stacked bar chart showing converged/duplicate/failed per operator."""
    from collections import defaultdict

    op_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        op = str(row.get("operator") or "init")
        st = str(row.get("status") or "unknown")
        op_status[op][st] += 1

    if not op_status:
        return "<p style='color:#7a90a8'>No operator data.</p>"

    # Order: most important statuses first (left to right in bar)
    bar_statuses = ["converged", "duplicate", "desorbed", "failed", "pending", "submitted"]
    bar_colors   = {
        "converged": "#2ecc71", "duplicate": "#95a5a6", "desorbed": "#e67e22",
        "failed": "#e74c3c", "pending": "#3498db", "submitted": "#f1c40f",
    }

    ops = sorted(op_status.keys())
    max_total = max(sum(op_status[op].values()) for op in ops) or 1

    row_h = 30
    pad_l, pad_r, pad_t, pad_b = 145, 50, 20, 44
    plot_w = width - pad_l - pad_r
    h = len(ops) * row_h + pad_t + pad_b

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" '
        f'style="background:#1e2a3a;border-radius:6px">',
    ]

    for i, op in enumerate(ops):
        y = pad_t + i * row_h
        cy = y + row_h * 0.5
        lines.append(
            f'<text x="{pad_l - 8}" y="{cy:.1f}" text-anchor="end" '
            f'fill="#aaa" font-size="11" dominant-baseline="middle">{op}</text>'
        )
        x_cur = float(pad_l)
        total = sum(op_status[op].values())
        for st in bar_statuses:
            cnt = op_status[op].get(st, 0)
            if cnt == 0:
                continue
            bw = cnt / max_total * plot_w
            color = bar_colors.get(st, "#888")
            lines.append(
                f'<rect x="{x_cur:.1f}" y="{y + 5}" width="{bw:.1f}" '
                f'height="{row_h - 10}" fill="{color}" rx="2"/>'
            )
            if bw > 18:
                lines.append(
                    f'<text x="{x_cur + bw / 2:.1f}" y="{cy:.1f}" text-anchor="middle" '
                    f'fill="#000" font-size="9" dominant-baseline="middle" font-weight="600">{cnt}</text>'
                )
            x_cur += bw
        lines.append(
            f'<text x="{pad_l + plot_w + 6}" y="{cy:.1f}" fill="#7a90a8" '
            f'font-size="10" dominant-baseline="middle">{total}</text>'
        )

    # Legend
    lx = pad_l
    ly = h - pad_b + 14
    for st in bar_statuses:
        color = bar_colors.get(st, "#888")
        lines += [
            f'<rect x="{lx}" y="{ly}" width="9" height="9" fill="{color}" rx="1"/>',
            f'<text x="{lx + 12}" y="{ly + 4.5}" fill="#888" font-size="9" '
            f'dominant-baseline="middle">{st}</text>',
        ]
        lx += 72

    lines.append("</svg>")
    return "\n".join(lines)


def _svg_coverage_strip(individuals, width: int = 700, height: int = 340) -> str:
    """Strip chart: GCE distribution per adsorbate coverage type."""
    import numpy as np
    from collections import defaultdict

    coverage_gce: dict[str, list[float]] = defaultdict(list)
    for ind in individuals:
        gce = getattr(ind, "grand_canonical_energy", None)
        if gce is None:
            continue
        counts = (ind.extra_data or {}).get("adsorbate_counts", {})
        label = " ".join(f"{k}×{v}" for k, v in sorted(counts.items()) if v > 0) or "bare"
        coverage_gce[label].append(float(gce))

    if not coverage_gce:
        return (
            f'<svg width="{width}" height="80" style="background:#1e2a3a;border-radius:6px">'
            f'<text x="{width//2}" y="40" text-anchor="middle" fill="#888" font-size="14">'
            f'No coverage data yet</text></svg>'
        )

    # Sort groups by median GCE (most stable leftmost)
    groups = sorted(coverage_gce.items(), key=lambda kv: float(np.median(kv[1])))

    pad_l, pad_r, pad_t, pad_b = 65, 20, 24, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    all_gces = [g for _, gs in groups for g in gs]
    lo, hi = min(all_gces), max(all_gces)
    margin = max((hi - lo) * 0.10, 0.05)
    lo -= margin
    hi += margin

    n_groups = len(groups)
    x_step = plot_w / max(n_groups, 1)

    def px(i: int) -> float:
        return pad_l + (i + 0.5) * x_step

    def py(v: float) -> float:
        if hi == lo:
            return pad_t + plot_h / 2
        return pad_t + (1.0 - (v - lo) / (hi - lo)) * plot_h

    palette = [
        "#e06c75", "#98c379", "#61afef", "#e5c07b",
        "#c678dd", "#56b6c2", "#d19a66", "#abb2bf",
    ]

    n_y = 5
    y_step_v = (hi - lo) / n_y
    y_ticks = [lo + i * y_step_v for i in range(n_y + 1)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#1e2a3a;border-radius:6px">',
        *(
            f'<line x1="{pad_l}" y1="{py(t):.1f}" x2="{width - pad_r}" y2="{py(t):.1f}" '
            f'stroke="#2c3e50" stroke-width="1"/>'
            for t in y_ticks
        ),
        *(
            f'<text x="{pad_l - 6}" y="{py(t):.1f}" text-anchor="end" '
            f'fill="#888" font-size="10" dominant-baseline="middle">{t:.2f}</text>'
            for t in y_ticks
        ),
        f'<text x="12" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" fill="#aaa" '
        f'font-size="11" transform="rotate(-90,12,{pad_t + plot_h / 2:.1f})">G (eV)</text>',
    ]

    rng = np.random.default_rng(0)
    for i, (label, gces) in enumerate(groups):
        x = px(i)
        color = palette[i % len(palette)]
        n = len(gces)

        # Thin vertical guide
        lines.append(
            f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="1" stroke-opacity="0.15"/>'
        )

        # Min–max range bar
        if n > 1:
            y_top = py(max(gces))
            y_bot = py(min(gces))
            bar_h = max(abs(y_bot - y_top), 2)
            lines.append(
                f'<rect x="{x - 3:.1f}" y="{y_top:.1f}" width="6" height="{bar_h:.1f}" '
                f'fill="{color}" fill-opacity="0.25" rx="3"/>'
            )

        # Median line
        med = float(np.median(gces))
        lines.append(
            f'<line x1="{x - 12:.1f}" y1="{py(med):.1f}" x2="{x + 12:.1f}" y2="{py(med):.1f}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )

        # Individual dots with jitter
        for gce in gces:
            jit = float(rng.uniform(-min(x_step * 0.3, 8), min(x_step * 0.3, 8))) if n > 1 else 0.0
            lines.append(
                f'<circle cx="{x + jit:.1f}" cy="{py(gce):.1f}" r="4" '
                f'fill="{color}" fill-opacity="0.85"/>'
            )

        # Rotated x-axis label
        lx, ly = x, pad_t + plot_h + 10
        short = label if len(label) <= 14 else label[:13] + "…"
        lines.append(
            f'<text x="{lx:.1f}" y="{ly}" text-anchor="end" fill="#aaa" font-size="10" '
            f'transform="rotate(-40,{lx:.1f},{ly})">{short} (n={n})</text>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


def _atoms_to_xyz(atoms) -> str:
    lines = [str(len(atoms)), "galoop"]
    for atom in atoms:
        x, y, z = atom.position
        lines.append(f"{atom.symbol} {x:.6f} {y:.6f} {z:.6f}")
    return "\n".join(lines)


def _structure_viewer_html(best, run_dir: Path, n_slab_atoms: int, n: int = 12) -> str:
    """Interactive 3Dmol.js viewer for top N structures.

    Requires an internet connection to load 3Dmol.js from CDN.
    Slab atoms are shown as small translucent spheres; adsorbates as
    full-size spheres + sticks.
    """
    try:
        from ase.io import read as ase_read
    except ImportError:
        return "<p style='color:#7a90a8'>ASE not available for structure viewer.</p>"

    structures = []
    for ind in best[:n]:
        geo = ind.geometry_path
        if not geo:
            continue
        contcar = Path(geo).parent / "CONTCAR"
        poscar  = Path(geo).parent / "POSCAR"
        src = contcar if contcar.exists() else (poscar if poscar.exists() else None)
        if src is None:
            continue
        try:
            atoms = ase_read(str(src), format="vasp")
            gce = f"{ind.grand_canonical_energy:.4f}" if ind.grand_canonical_energy is not None else "N/A"
            ads = ind.extra_data.get("adsorbate_counts", {})
            ads_str = " ".join(f"{k}×{v}" for k, v in sorted(ads.items()) if v > 0)
            structures.append({
                "id": ind.id,
                "gce": gce,
                "ads": ads_str or "—",
                "op": ind.operator,
                "n_atoms": len(atoms),
                "n_slab": n_slab_atoms,
                "xyz": _atoms_to_xyz(atoms),
            })
        except Exception as exc:
            log.debug("Could not read structure %s: %s", ind.id, exc)

    if not structures:
        return "<p style='color:#7a90a8'>No structure files found for viewer.</p>"

    structs_json = json.dumps(structures)
    n_structs = len(structures)

    return f"""<div>
  <div id="viewer3d" style="width:100%;height:440px;background:#050510;border-radius:6px;overflow:hidden;position:relative"></div>
  <div style="display:flex;align-items:center;gap:10px;padding:10px 0 4px;flex-wrap:wrap">
    <button onclick="v3d_show(v3d_cur-1)"
      style="background:#2d2d5e;color:#e0e0e0;border:1px solid #555;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:13px">&#8592; Prev</button>
    <span id="v3d-counter" style="font-size:13px;min-width:70px;text-align:center;color:#aaa">1 / {n_structs}</span>
    <button onclick="v3d_show(v3d_cur+1)"
      style="background:#2d2d5e;color:#e0e0e0;border:1px solid #555;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:13px">Next &#8594;</button>
    <span id="v3d-title" style="font-size:12px;color:#7a90a8;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
  </div>
  <p style="font-size:0.75rem;color:#3a5070;margin:0">
    Slab atoms: small translucent spheres &nbsp;·&nbsp; Adsorbates: full spheres + sticks
    &nbsp;·&nbsp; Drag to rotate &nbsp;·&nbsp; Scroll to zoom
    &nbsp;·&nbsp; <em>Requires internet (3Dmol.js CDN)</em>
  </p>
</div>
<script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
<script>
(function() {{
  const structs = {structs_json};
  let viewer = null;
  let v3d_cur = 0;
  window.v3d_cur = 0;

  function initViewer() {{
    const el = document.getElementById('viewer3d');
    if (typeof $3Dmol === 'undefined') {{
      el.innerHTML = '<p style="color:#e74c3c;padding:20px">Could not load 3Dmol.js — check your internet connection.</p>';
      return;
    }}
    viewer = $3Dmol.createViewer(el, {{backgroundColor: '0x050510', antialias: true}});
    v3d_show(0);
  }}

  window.v3d_show = function(i) {{
    if (!viewer) return;
    v3d_cur = ((i % structs.length) + structs.length) % structs.length;
    window.v3d_cur = v3d_cur;
    const s = structs[v3d_cur];
    viewer.removeAllModels();
    viewer.addModel(s.xyz, 'xyz');

    // Slab: small, translucent spheres
    const slabIdx = Array.from({{length: s.n_slab}}, (_, k) => k);
    if (slabIdx.length > 0) {{
      viewer.setStyle({{index: slabIdx}}, {{
        sphere: {{radius: 0.32, opacity: 0.30, colorscheme: 'Jmol'}}
      }});
    }}

    // Adsorbates: full spheres + sticks
    const adsIdx = Array.from({{length: s.n_atoms - s.n_slab}}, (_, k) => k + s.n_slab);
    if (adsIdx.length > 0) {{
      viewer.setStyle({{index: adsIdx}}, {{
        sphere: {{radius: 0.55, opacity: 0.95, colorscheme: 'Jmol'}},
        stick:  {{radius: 0.16, colorscheme: 'Jmol'}}
      }});
    }}

    viewer.zoomTo({{index: adsIdx.length > 0 ? adsIdx : slabIdx}});
    viewer.render();

    document.getElementById('v3d-counter').textContent =
      (v3d_cur + 1) + ' / ' + structs.length;
    document.getElementById('v3d-title').textContent =
      s.id + '  G=' + s.gce + ' eV  ' + s.ads + '  op=' + s.op;
  }};

  if (document.readyState === 'loading') {{
    document.addEventListener('DOMContentLoaded', initViewer);
  }} else {{
    initViewer();
  }}
}})();
</script>"""


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

def _css() -> str:
    return """
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1923;color:#dde3ec;
         font-size:14px;line-height:1.5;padding:24px}
    h1{font-size:1.6rem;font-weight:600;margin-bottom:4px;color:#e8edf5}
    h2{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#c5cfe0;
       border-bottom:1px solid #1e3048;padding-bottom:6px}
    .subtitle{color:#7a90a8;font-size:0.85rem;margin-bottom:24px}
    .grid{display:grid;gap:20px}
    .grid-2{grid-template-columns:repeat(2,1fr)}
    .grid-3{grid-template-columns:repeat(3,1fr)}
    .grid-4{grid-template-columns:repeat(4,1fr)}
    @media(max-width:700px){.grid-2,.grid-3,.grid-4{grid-template-columns:1fr}}
    .card{background:#162030;border:1px solid #1e3048;border-radius:8px;padding:18px}
    .stat-val{font-size:2rem;font-weight:700;color:#58a6ff;line-height:1.1}
    .stat-lbl{font-size:0.75rem;color:#7a90a8;text-transform:uppercase;letter-spacing:.06em;
              margin-top:4px}
    table{width:100%;border-collapse:collapse;font-size:0.88rem}
    th{text-align:left;padding:7px 10px;background:#1a2d42;color:#9ab;font-weight:600;
       font-size:0.78rem;text-transform:uppercase;letter-spacing:.05em}
    td{padding:6px 10px;border-bottom:1px solid #1a2638;color:#cdd6e0}
    tr:hover td{background:#1a2d42}
    .badge{display:inline-block;padding:2px 8px;border-radius:10px;
           font-size:0.75rem;font-weight:600}
    .bar-wrap{height:14px;background:#1a2638;border-radius:4px;overflow:hidden;
              margin-top:4px}
    .bar-fill{height:100%;border-radius:4px;transition:width .4s}
    .status-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
    .status-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
    .status-name{width:90px;font-size:0.85rem;color:#9ab}
    .status-count{margin-left:auto;font-weight:600;color:#dde3ec;min-width:30px;
                  text-align:right;font-size:0.85rem}
    code{font-family:'Consolas','Fira Mono',monospace;font-size:0.82rem;
         color:#58a6ff;word-break:break-all}
    .path{font-family:'Consolas','Fira Mono',monospace;font-size:0.78rem;
          color:#7a90a8;word-break:break-all}
    .gce{font-weight:600;color:#58a6ff}
    .section{margin-bottom:28px}
    """


def _status_panel(counts: dict[str, int]) -> str:
    total = sum(counts.values())
    if total == 0:
        return "<p style='color:#7a90a8'>No structures yet.</p>"
    rows = []
    for s in _STATUS_ORDER:
        n = counts.get(s, 0)
        if n == 0:
            continue
        pct = n / total * 100
        color = _STATUS_COLOR.get(s, "#888")
        rows.append(
            f'<div class="status-row">'
            f'  <div class="status-dot" style="background:{color}"></div>'
            f'  <span class="status-name">{s}</span>'
            f'  <div style="flex:1"><div class="bar-wrap">'
            f'    <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>'
            f'  </div></div>'
            f'  <span class="status-count">{n}</span>'
            f'</div>'
        )
    return "\n".join(rows)


def _chemical_potentials_table(cfg) -> str:
    rows = []
    for ads in cfg.adsorbates:
        rows.append(
            f"<tr>"
            f"<td>{html.escape(ads.symbol)}</td>"
            f"<td class='gce'>{ads.chemical_potential:.4f}</td>"
            f"<td>{ads.min_count} – {ads.max_count}</td>"
            f"<td>{ads.binding_index}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>Species</th><th>μ (eV)</th><th>Count range</th><th>Binding atom</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table>"
    )


def _conditions_table(cfg) -> str:
    cond = cfg.conditions
    rows = [
        f"<tr><td>Temperature</td><td class='gce'>{cond.temperature:.2f} K</td></tr>",
        f"<tr><td>Pressure</td><td class='gce'>{cond.pressure:.4g} bar</td></tr>",
        f"<tr><td>Potential</td><td class='gce'>{cond.potential:.4f} V</td></tr>",
        f"<tr><td>pH</td><td class='gce'>{cond.pH:.2f}</td></tr>",
        f"<tr><td>Slab energy</td><td class='gce'>{cfg.slab.energy:.6f} eV</td></tr>",
    ]
    return (
        "<table><thead><tr><th>Parameter</th><th>Value</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _top_structures_table(individuals: list, n: int = 20) -> str:
    if not individuals:
        return "<p style='color:#7a90a8'>No converged structures.</p>"
    rows = []
    for rank, ind in enumerate(individuals[:n], 1):
        gce = f"{ind.grand_canonical_energy:.4f}" if ind.grand_canonical_energy is not None else "—"
        raw = f"{ind.raw_energy:.4f}" if ind.raw_energy is not None else "—"
        ads_counts = ind.extra_data.get("adsorbate_counts", {})
        ads_str = ", ".join(f"{k}×{v}" for k, v in sorted(ads_counts.items()))
        path = html.escape(ind.geometry_path or "—")
        op = html.escape(ind.operator or "—")
        rows.append(
            f"<tr>"
            f"<td>{rank}</td>"
            f"<td><code>{html.escape(ind.id[:12])}…</code></td>"
            f"<td class='gce'>{gce}</td>"
            f"<td>{raw}</td>"
            f"<td>{ads_str or '—'}</td>"
            f"<td><span class='badge' style='background:#1a3a2a;color:#2ecc71'>{op}</span></td>"
            f"<td class='path'>{path}</td>"
            f"</tr>"
        )
    return (
        "<div style='overflow-x:auto'><table>"
        "<thead><tr>"
        "<th>#</th><th>ID</th><th>G (eV)</th><th>E_raw (eV)</th>"
        "<th>Adsorbates</th><th>Operator</th><th>Path</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table></div>"
    )


def _duplicate_summary(df, threshold: float = 0.90) -> str:
    """HTML for the duplicate clustering section."""
    dup_df = df[df["status"] == "duplicate"].copy()
    conv_df = df[df["status"] == "converged"].copy()
    n_dups = len(dup_df)
    n_unique = len(conv_df)
    n_total_conv = n_unique + n_dups

    if n_dups == 0:
        return (
            f"<p style='color:#7a90a8'>No duplicates detected "
            f"(threshold={threshold:.2f}).</p>"
        )

    ratio = f"{n_unique} / {n_total_conv}"

    clusters: dict[str, list] = {}
    for _, row in dup_df.iterrows():
        extra = row["extra_data"] if isinstance(row["extra_data"], dict) else {}
        dup_of = extra.get("dup_of")
        tanimoto = extra.get("tanimoto")
        if dup_of:
            clusters.setdefault(dup_of, []).append(tanimoto)

    conv_map = {row["id"]: row for _, row in conv_df.iterrows()}
    top = sorted(clusters.items(), key=lambda x: -len(x[1]))[:10]

    rows = []
    for orig_id, tanimoto_list in top:
        orig = conv_map.get(orig_id)
        gce = (
            f"{orig['grand_canonical_energy']:.4f}"
            if orig is not None and orig["grand_canonical_energy"] == orig["grand_canonical_energy"]
            else "—"
        )
        valid = [t for t in tanimoto_list if t is not None]
        avg_sim = f"{sum(valid)/len(valid):.3f}" if valid else "—"
        rows.append(
            f"<tr>"
            f"<td><code>{html.escape(orig_id)}</code></td>"
            f"<td class='gce'>{gce}</td>"
            f"<td>{len(tanimoto_list)}</td>"
            f"<td>{avg_sim}</td>"
            f"</tr>"
        )

    table = (
        "<div style='overflow-x:auto'><table>"
        "<thead><tr>"
        "<th>Original ID</th><th>G (eV)</th>"
        "<th>Duplicates</th><th>Avg Tanimoto</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody>"
        "</table></div>"
    )

    return (
        f"<p style='color:#7a90a8;margin-bottom:12px'>"
        f"Unique converged: <strong style='color:#dde3ec'>{ratio}</strong> total converged"
        f"&nbsp;·&nbsp; {n_dups} duplicate(s) suppressed"
        f"&nbsp;·&nbsp; threshold={threshold:.2f}"
        f"</p>"
        + table
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(
    cfg,
    output_path: Path,
    top_n: int = 20,
    project=None,
    db_path: Path | None = None,
) -> None:
    """
    Write a self-contained HTML report to *output_path*.

    Parameters
    ----------
    cfg         : loaded GaloopConfig
    output_path : where to write the .html file
    top_n       : number of top structures to list in the table
    project     : GaloopProject (signac-based)
    db_path     : deprecated, ignored
    """
    if project is None:
        raise ValueError("Pass a GaloopProject via the project= argument")

    counts      = project.count_by_status()
    best        = project.best(n=top_n)
    all_conv    = project.get_by_status("converged")   # for coverage strip
    df          = project.to_dataframe()
    run_dir     = project.run_dir
    dup_threshold = cfg.fingerprint.duplicate_threshold

    # Slab geometry for heatmap and 3D viewer
    n_slab_atoms = 0
    a_len = b_len = 0.0
    try:
        import numpy as np
        from ase.io import read as ase_read
        slab_path = Path(cfg.slab.geometry)
        if not slab_path.is_absolute():
            slab_path = run_dir / slab_path
        if slab_path.exists():
            _slab = ase_read(str(slab_path), format="vasp")
            n_slab_atoms = len(_slab)
            a_len = float(np.linalg.norm(_slab.cell[0]))
            b_len = float(np.linalg.norm(_slab.cell[1]))
    except Exception as exc:
        log.debug("Could not load slab: %s", exc)

    # Build all SVG / HTML sections
    heatmap_svg      = _svg_heatmap(_collect_adsorbate_xy(df, n_slab_atoms), a_len, b_len)
    operator_svg     = _svg_operator_stats(df)
    coverage_svg     = _svg_coverage_strip(all_conv)
    viewer_html      = _structure_viewer_html(best, run_dir, n_slab_atoms, n=min(top_n, 15))

    total      = sum(counts.values())
    n_conv     = counts.get("converged", 0)

    conv_df = df[(df["status"] == "converged") & df["grand_canonical_energy"].notna()]
    xs: list[float] = list(range(1, len(conv_df) + 1))
    ys: list[float] = conv_df["grand_canonical_energy"].tolist()
    run_min: list[float] = []
    cur_min = float("inf")
    for y in ys:
        cur_min = min(cur_min, y)
        run_min.append(cur_min)

    best_gce     = f"{run_min[-1]:.4f} eV" if run_min else "—"
    success_rate = f"{n_conv / total * 100:.1f} %" if total > 0 else "—"
    gce_svg      = _svg_scatter(xs, ys, run_min)

    import datetime
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name     = html.escape(run_dir.name)
    run_path     = html.escape(str(run_dir))

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>galoop report — {run_name}</title>
<style>{_css()}</style>
</head>
<body>

<h1>galoop run report</h1>
<p class="subtitle">
  Run directory: <code>{run_path}</code>
  &nbsp;·&nbsp; Generated: {generated_at}
</p>

<!-- ── Summary cards ── -->
<div class="section grid grid-4">
  <div class="card">
    <div class="stat-val">{total}</div>
    <div class="stat-lbl">Total structures</div>
  </div>
  <div class="card">
    <div class="stat-val" style="color:#2ecc71">{n_conv}</div>
    <div class="stat-lbl">Converged</div>
  </div>
  <div class="card">
    <div class="stat-val" style="color:#58a6ff">{best_gce}</div>
    <div class="stat-lbl">Best G (eV)</div>
  </div>
  <div class="card">
    <div class="stat-val">{success_rate}</div>
    <div class="stat-lbl">Success rate</div>
  </div>
</div>

<!-- ── Status + conditions ── -->
<div class="section grid grid-2">
  <div class="card">
    <h2>Status breakdown</h2>
    {_status_panel(counts)}
  </div>
  <div class="card">
    <h2>Run conditions</h2>
    {_conditions_table(cfg)}
  </div>
</div>

<!-- ── Chemical potentials ── -->
<div class="section card">
  <h2>Adsorbates &amp; chemical potentials</h2>
  {_chemical_potentials_table(cfg)}
</div>

<!-- ── GCE evolution ── -->
<div class="section card">
  <h2>Grand canonical energy evolution</h2>
  <p style="color:#7a90a8;font-size:0.82rem;margin-bottom:10px">
    Blue dots — individual evaluations &nbsp;·&nbsp; Green line — running minimum
  </p>
  {gce_svg}
</div>

<!-- ── Coverage landscape ── -->
<div class="section card">
  <h2>Coverage landscape</h2>
  <p style="color:#7a90a8;font-size:0.82rem;margin-bottom:10px">
    Each group = one adsorbate stoichiometry &nbsp;·&nbsp;
    Sorted by median G &nbsp;·&nbsp; Bar = median, dots = individual structures
  </p>
  {coverage_svg}
</div>

<!-- ── Operator performance ── -->
<div class="section card">
  <h2>Operator performance</h2>
  <p style="color:#7a90a8;font-size:0.82rem;margin-bottom:10px">
    Stacked bars show outcome distribution per operator &nbsp;·&nbsp; Numbers on right = total spawned
  </p>
  {operator_svg}
</div>

<!-- ── Sampling heatmap ── -->
<div class="section card">
  <h2>Adsorbate sampling coverage</h2>
  <p style="color:#7a90a8;font-size:0.82rem;margin-bottom:10px">
    Fractional x–y positions of all adsorbate atoms (CONTCAR preferred) &nbsp;·&nbsp; viridis = visit count
  </p>
  {heatmap_svg}
</div>

<!-- ── 3D structure viewer ── -->
<div class="section card">
  <h2>Top structures — 3D viewer</h2>
  {viewer_html}
</div>

<!-- ── Duplicate clustering ── -->
<div class="section card">
  <h2>Duplicate clustering</h2>
  {_duplicate_summary(df, threshold=dup_threshold)}
</div>

<!-- ── Top structures table ── -->
<div class="section card">
  <h2>Top {top_n} structures (by grand canonical energy)</h2>
  {_top_structures_table(best, n=top_n)}
</div>

<p style="color:#3a5070;font-size:0.75rem;margin-top:12px;text-align:center">
  galoop &nbsp;·&nbsp; generated {generated_at}
</p>

</body>
</html>
"""

    output_path = Path(output_path)
    output_path.write_text(html_doc, encoding="utf-8")
    log.info("Report written to %s", output_path)
