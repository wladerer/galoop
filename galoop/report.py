"""
galoop/report.py

Generate a self-contained HTML status report for a galoop run.

No external dependencies — everything (CSS, JS, SVG) is inlined so
the file works offline and can be shared as a single document.
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

    # Cell outline
    lines.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        f'fill="none" stroke="#aaa" stroke-width="1"/>'
    )

    # Axis labels
    lines += [
        f'<text x="{pad_l + plot_w / 2}" y="{height - 4}" '
        f'text-anchor="middle" fill="#aaa" font-size="11">a ({a_len:.2f} Å)</text>',
        f'<text x="12" y="{pad_t + plot_h / 2}" text-anchor="middle" fill="#aaa" font-size="11" '
        f'transform="rotate(-90,12,{pad_t + plot_h / 2})">b ({b_len:.2f} Å)</text>',
        f'<text x="{pad_l + plot_w // 2}" y="{pad_t - 6}" '
        f'text-anchor="middle" fill="#7a90a8" font-size="10">'
        f'{len(xys)} adsorbate atom positions</text>',
    ]

    # Colorbar
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


# ---------------------------------------------------------------------------
# SVG helpers
# ---------------------------------------------------------------------------

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
    # Add a small margin around y
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

    # Axis tick values
    n_y_ticks = 5
    y_step = (max_y - min_y) / n_y_ticks
    y_ticks = [min_y + i * y_step for i in range(n_y_ticks + 1)]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'style="background:#1e2a3a;border-radius:6px">',
        # Grid lines
        *(
            f'<line x1="{pad_l}" y1="{py(t):.1f}" x2="{width - pad_r}" y2="{py(t):.1f}" '
            f'stroke="#2c3e50" stroke-width="1"/>'
            for t in y_ticks
        ),
        # Y-axis labels
        *(
            f'<text x="{pad_l - 6}" y="{py(t):.1f}" text-anchor="end" '
            f'fill="#888" font-size="10" dominant-baseline="middle">{t:.2f}</text>'
            for t in y_ticks
        ),
        # X-axis labels (a few evenly spaced)
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

    # Axis labels
    lines += [
        f'<text x="{pad_l + plot_w // 2}" y="{height - 0}" '
        f'text-anchor="middle" fill="#aaa" font-size="11">Evaluation</text>',
        f'<text x="12" y="{pad_t + plot_h // 2}" '
        f'text-anchor="middle" fill="#aaa" font-size="11" '
        f'transform="rotate(-90,12,{pad_t + plot_h // 2})">G (eV)</text>',
    ]

    # Running minimum line
    if len(run_min) > 1:
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, run_min))
        lines.append(
            f'<polyline points="{pts}" fill="none" stroke="#2ecc71" stroke-width="2"/>'
        )

    # Scatter dots
    for x, y in zip(xs, ys):
        lines.append(
            f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" '
            f'fill="#3498db" fill-opacity="0.75"/>'
        )

    lines.append("</svg>")
    return "\n".join(lines)


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


def _chemical_potentials_table(cfg: GaloopConfig) -> str:
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


def _conditions_table(cfg: GaloopConfig) -> str:
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

    # Build cluster map from extra_data
    clusters: dict[str, list] = {}
    for _, row in dup_df.iterrows():
        extra = row["extra_data"] if isinstance(row["extra_data"], dict) else {}
        dup_of = extra.get("dup_of")
        tanimoto = extra.get("tanimoto")
        if dup_of:
            clusters.setdefault(dup_of, []).append(tanimoto)

    # Top-duplicated originals table
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
    db_path: Path,
    cfg: GaloopConfig,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """
    Write a self-contained HTML report to *output_path*.

    Parameters
    ----------
    db_path     : path to galoop.db
    cfg         : loaded GaloopConfig
    output_path : where to write the .html file
    top_n       : number of top structures to list
    """
    from galoop.database import GaloopDB

    with GaloopDB(db_path) as db:
        db.setup()
        counts = db.count_by_status()
        best = db.best(n=top_n)
        df = db.to_dataframe()

    dup_threshold = cfg.fingerprint.duplicate_threshold

    # Load slab to determine n_slab_atoms and cell dimensions for the heatmap
    n_slab_atoms = 0
    a_len = b_len = 0.0
    try:
        import numpy as np
        from ase.io import read as ase_read
        slab_path = Path(cfg.slab.geometry)
        if not slab_path.is_absolute():
            slab_path = db_path.parent / slab_path
        if slab_path.exists():
            _slab = ase_read(str(slab_path), format="vasp")
            n_slab_atoms = len(_slab)
            a_len = float(np.linalg.norm(_slab.cell[0]))
            b_len = float(np.linalg.norm(_slab.cell[1]))
    except Exception as exc:
        log.debug("Could not load slab for heatmap: %s", exc)

    heatmap_svg = _svg_heatmap(
        _collect_adsorbate_xy(df, n_slab_atoms),
        a_len, b_len,
    )

    total = sum(counts.values())
    n_conv = counts.get("converged", 0)
    n_failed = counts.get("failed", 0)
    n_desorbed = counts.get("desorbed", 0)

    # GCE evolution data
    conv_df = df[(df["status"] == "converged") & df["grand_canonical_energy"].notna()]
    xs: list[float] = list(range(1, len(conv_df) + 1))
    ys: list[float] = conv_df["grand_canonical_energy"].tolist()
    run_min: list[float] = []
    cur_min = float("inf")
    for y in ys:
        cur_min = min(cur_min, y)
        run_min.append(cur_min)

    best_gce = f"{ys[-1] if not run_min else run_min[-1]:.4f} eV" if run_min else "—"
    success_rate = f"{n_conv / total * 100:.1f} %" if total > 0 else "—"

    gce_svg = _svg_scatter(xs, ys, run_min)

    import datetime
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>galoop report — {html.escape(str(db_path.parent.name))}</title>
<style>{_css()}</style>
</head>
<body>

<h1>galoop run report</h1>
<p class="subtitle">
  Run directory: <code>{html.escape(str(db_path.parent))}</code>
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
    Blue dots — individual evaluations &nbsp;·&nbsp;
    Green line — running minimum
  </p>
  {gce_svg}
</div>

<!-- ── Sampling heatmap ── -->
<div class="section card">
  <h2>Adsorbate sampling coverage</h2>
  <p style="color:#7a90a8;font-size:0.82rem;margin-bottom:10px">
    Fractional x–y positions of all adsorbate atoms across all structures
    (CONTCAR preferred, POSCAR fallback) &nbsp;·&nbsp; viridis = visit count
  </p>
  {heatmap_svg}
</div>

<!-- ── Duplicate clustering ── -->
<div class="section card">
  <h2>Duplicate clustering</h2>
  {_duplicate_summary(df, threshold=dup_threshold)}
</div>

<!-- ── Top structures ── -->
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
