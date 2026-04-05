#!/usr/bin/env python
"""
Pourbaix and coverage diagrams from a completed galoop run.

Usage
-----
    python scripts/pourbaix.py runs/cu111_corrosion/
    python scripts/pourbaix.py runs/cu111_co_nh3/ --ph-range 0 7 --u-range -1.0 0.5

Reads the GaloopStore database and calibration/reference_energies.txt,
recomputes GCE across a (pH, U) grid, and generates:
  - pourbaix.html  — interactive Pourbaix diagram (plotly)
  - coverage.html  — coverage vs potential at selected pH values
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    """Load converged structures and chemical potentials from a run."""
    from galoop.store import GaloopStore

    store = GaloopStore(run_dir)
    df = store.to_dataframe()
    store.close()

    conv = df[df.status == "converged"].copy()
    if conv.empty:
        raise ValueError(f"No converged structures in {run_dir}")

    # Extract adsorbate counts
    conv["ads_counts"] = conv["extra_data"].apply(
        lambda x: x.get("adsorbate_counts", {})
    )
    # Canonical label for each composition
    conv["composition"] = conv["ads_counts"].apply(_comp_label)

    # Load chemical potentials from calibration or config
    chem_pots = _load_chem_pots(run_dir)

    return conv, chem_pots


def _load_chem_pots(run_dir: Path) -> dict[str, float]:
    """Load adsorbate chemical potentials from config snapshot in DB."""
    from galoop.store import GaloopStore

    store = GaloopStore(run_dir)
    row = store._conn.execute(
        "SELECT value FROM run_params WHERE key = ?", ("config",)
    ).fetchone()
    store.close()

    if row is None:
        raise ValueError("No config snapshot found in database")

    cfg = json.loads(row["value"])
    pots = {}
    for ads in cfg.get("adsorbates", []):
        sym = ads["symbol"]
        mu = ads.get("chemical_potential")
        if mu is not None:
            pots[sym] = mu
    return pots


def _comp_label(counts: dict) -> str:
    """Canonical string label for a composition, e.g. '2CO + 1NH3 + 1O'."""
    parts = []
    for sym in sorted(counts.keys()):
        n = counts[sym]
        if n > 0:
            parts.append(f"{n}{sym}")
    return " + ".join(parts) if parts else "bare"


# ---------------------------------------------------------------------------
# GCE recomputation
# ---------------------------------------------------------------------------

def compute_gce_grid(
    conv: pd.DataFrame,
    chem_pots: dict[str, float],
    ph_range: tuple[float, float] = (0.0, 14.0),
    u_range: tuple[float, float] = (-1.0, 1.0),
    n_ph: int = 100,
    n_u: int = 100,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Recompute GCE for the best structure at each composition across (pH, U).

    Returns
    -------
    ph_grid, u_grid : 1D arrays
    best_per_comp : DataFrame of best structure per composition
    gce_grid : shape (n_comp, n_ph, n_u) — GCE for each composition
    phase_map : shape (n_ph, n_u) — index of lowest-GCE composition
    """
    from galoop.science.energy import grand_canonical_energy_grid

    # Take the best (lowest raw_energy) structure per composition
    best_idx = conv.groupby("composition")["raw_energy"].idxmin()
    best = conv.loc[best_idx].reset_index(drop=True)

    ph_grid = np.linspace(ph_range[0], ph_range[1], n_ph)
    u_grid = np.linspace(u_range[0], u_range[1], n_u)
    pH_2d, U_2d = np.meshgrid(ph_grid, u_grid, indexing="ij")

    n_comp = len(best)
    gce_grid = np.zeros((n_comp + 1, n_ph, n_u))  # +1 for bare slab

    # Bare slab: GCE = 0 everywhere (reference)
    gce_grid[0, :, :] = 0.0

    for i, (_, row) in enumerate(best.iterrows()):
        gce_grid[i + 1] = grand_canonical_energy_grid(
            raw_energy=row["raw_energy"],
            adsorbate_counts=row["ads_counts"],
            chemical_potentials=chem_pots,
            pH=pH_2d,
            potential=U_2d,
            temperature=temperature,
            pressure=pressure,
        )

    # Find lowest-GCE phase at each (pH, U)
    phase_map = np.argmin(gce_grid, axis=0)

    return ph_grid, u_grid, best, gce_grid, phase_map


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _phase_colors(n: int) -> list[str]:
    """Return a list of n distinct colors, gray first for bare slab."""
    palette = [
        "#d3d3d3",  # bare slab = gray
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#a65628", "#f781bf", "#999999",
        "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    ]
    while len(palette) < n:
        palette.append(f"#{np.random.randint(0, 0xFFFFFF):06x}")
    return palette[:n]


# -- Plotly (interactive HTML) ----------------------------------------------

def plot_pourbaix(ph_grid, u_grid, best, phase_map, output: Path):
    """Generate an interactive Pourbaix diagram."""
    import plotly.graph_objects as go

    labels = ["bare slab"] + best["composition"].tolist()
    colors = _phase_colors(len(labels))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=ph_grid, y=u_grid, z=phase_map.T,
        colorscale=[(i / max(len(labels) - 1, 1), colors[i]) for i in range(len(labels))],
        zmin=0, zmax=len(labels) - 1,
        showscale=False,
        hovertemplate="pH=%{x:.1f}<br>U=%{y:.2f} V<extra></extra>",
    ))
    for i, label in enumerate(labels):
        mask = phase_map == i
        if not mask.any():
            continue
        js, ks = np.where(mask)
        fig.add_annotation(
            x=ph_grid[js].mean(), y=u_grid[ks].mean(),
            text=f"<b>{label}</b>", showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1,
        )
    fig.update_layout(
        title="Pourbaix Diagram (CHE)",
        xaxis_title="pH", yaxis_title="Potential (V vs RHE)",
        width=800, height=600,
    )
    fig.write_html(str(output))
    log.info("Pourbaix diagram: %s", output)


def plot_coverage(
    ph_grid, u_grid, best, gce_grid, output: Path,
    ph_values: list[float] | None = None,
):
    """Generate coverage vs potential plots at selected pH values."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if ph_values is None:
        ph_values = [0.0, 3.0, 7.0, 10.0, 14.0]
    ph_values = [p for p in ph_values if ph_grid[0] <= p <= ph_grid[-1]]

    labels = ["bare slab"] + best["composition"].tolist()
    colors = _phase_colors(len(labels))

    fig = make_subplots(
        rows=len(ph_values), cols=1, shared_xaxes=True,
        subplot_titles=[f"pH = {p:.1f}" for p in ph_values],
        vertical_spacing=0.05,
    )
    for row, target_ph in enumerate(ph_values, 1):
        ph_idx = np.argmin(np.abs(ph_grid - target_ph))
        for i, label in enumerate(labels):
            fig.add_trace(
                go.Scatter(
                    x=u_grid, y=gce_grid[i, ph_idx, :],
                    mode="lines", name=label,
                    line=dict(color=colors[i], width=2),
                    showlegend=(row == 1), legendgroup=label,
                ),
                row=row, col=1,
            )
        fig.update_yaxes(title_text="GCE (eV)", row=row, col=1)
    fig.update_xaxes(title_text="Potential (V vs RHE)", row=len(ph_values), col=1)
    fig.update_layout(
        title="Coverage Diagram: GCE vs Potential",
        height=300 * len(ph_values), width=900,
    )
    fig.write_html(str(output))
    log.info("Coverage diagram: %s", output)


# -- Matplotlib (publication PNG) -------------------------------------------

def plot_pourbaix_png(ph_grid, u_grid, best, phase_map, output: Path, dpi: int = 300):
    """Publication-quality Pourbaix diagram as PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    labels = ["bare slab"] + best["composition"].tolist()
    colors = _phase_colors(len(labels))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(labels)), len(labels))

    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    im = ax.pcolormesh(
        ph_grid, u_grid, phase_map.T,
        cmap=cmap, norm=norm, shading="auto", rasterized=True,
    )

    # Phase labels at centroids
    for i, label in enumerate(labels):
        mask = phase_map == i
        if not mask.any():
            continue
        js, ks = np.where(mask)
        ax.text(
            ph_grid[js].mean(), u_grid[ks].mean(), label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="black",
                      linewidth=0.5, boxstyle="round,pad=0.2"),
        )

    ax.set_xlabel("pH")
    ax.set_ylabel("Potential (V vs RHE)")
    ax.set_title("Pourbaix Diagram (CHE)")

    fig.savefig(str(output), dpi=dpi)
    plt.close(fig)
    log.info("Pourbaix PNG: %s", output)


def plot_coverage_png(
    ph_grid, u_grid, best, gce_grid, output: Path,
    ph_values: list[float] | None = None,
    dpi: int = 300,
):
    """Publication-quality coverage diagram as PNG."""
    import matplotlib.pyplot as plt

    if ph_values is None:
        ph_values = [0.0, 3.0, 7.0, 10.0, 14.0]
    ph_values = [p for p in ph_values if ph_grid[0] <= p <= ph_grid[-1]]

    labels = ["bare slab"] + best["composition"].tolist()
    colors = _phase_colors(len(labels))

    fig, axes = plt.subplots(
        len(ph_values), 1, figsize=(6, 2.5 * len(ph_values)),
        sharex=True, constrained_layout=True,
    )
    if len(ph_values) == 1:
        axes = [axes]

    for ax, target_ph in zip(axes, ph_values):
        ph_idx = np.argmin(np.abs(ph_grid - target_ph))
        for i, label in enumerate(labels):
            ax.plot(u_grid, gce_grid[i, ph_idx, :],
                    color=colors[i], linewidth=1.5, label=label)
        ax.set_ylabel("GCE (eV)")
        ax.set_title(f"pH = {target_ph:.1f}", fontsize=10)

    axes[-1].set_xlabel("Potential (V vs RHE)")
    # Single legend outside
    handles, leg_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper right",
               bbox_to_anchor=(1.0, 1.0), fontsize=7, framealpha=0.9)

    fig.savefig(str(output), dpi=dpi)
    plt.close(fig)
    log.info("Coverage PNG: %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Pourbaix and coverage diagrams from a galoop run."
    )
    parser.add_argument("run_dir", type=Path, help="Path to completed galoop run directory")
    parser.add_argument("--ph-range", nargs=2, type=float, default=[0.0, 14.0],
                        metavar=("MIN", "MAX"), help="pH range (default: 0 14)")
    parser.add_argument("--u-range", nargs=2, type=float, default=[-1.0, 1.0],
                        metavar=("MIN", "MAX"), help="Potential range in V vs RHE (default: -1 1)")
    parser.add_argument("--ph-slices", nargs="+", type=float, default=None,
                        help="pH values for coverage slices (default: 0 3 7 10 14)")
    parser.add_argument("--resolution", type=int, default=100,
                        help="Grid resolution (default: 100)")
    parser.add_argument("--pressure", type=float, default=1.0,
                        help="Gas-phase pressure in atm (default: 1.0)")
    parser.add_argument("--png", action="store_true",
                        help="Also generate publication-quality PNGs (matplotlib)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="DPI for PNG output (default: 300)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output directory (default: run_dir)")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or run_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading run data from %s", run_dir)
    conv, chem_pots = load_run(run_dir)

    log.info("Converged structures: %d", len(conv))
    log.info("Unique compositions: %d", conv["composition"].nunique())
    log.info("Chemical potentials: %s", chem_pots)

    log.info("Computing GCE grid (pH %.1f–%.1f, U %.2f–%.2f V) …",
             *args.ph_range, *args.u_range)
    ph_grid, u_grid, best, gce_grid, phase_map = compute_gce_grid(
        conv, chem_pots,
        ph_range=tuple(args.ph_range),
        u_range=tuple(args.u_range),
        n_ph=args.resolution,
        n_u=args.resolution,
        pressure=args.pressure,
    )

    log.info("Best structure per composition:")
    for _, row in best.iterrows():
        log.info("  %-35s  E_raw=%10.4f  GCE=%10.4f  id=%s",
                 row["composition"], row["raw_energy"],
                 row["grand_canonical_energy"], row["id"])

    plot_pourbaix(ph_grid, u_grid, best, phase_map, out_dir / "pourbaix.html")
    plot_coverage(ph_grid, u_grid, best, gce_grid, out_dir / "coverage.html",
                  ph_values=args.ph_slices)

    if args.png:
        plot_pourbaix_png(ph_grid, u_grid, best, phase_map,
                          out_dir / "pourbaix.png", dpi=args.dpi)
        plot_coverage_png(ph_grid, u_grid, best, gce_grid,
                          out_dir / "coverage.png",
                          ph_values=args.ph_slices, dpi=args.dpi)

    log.info("Done. Output in %s", out_dir)


if __name__ == "__main__":
    main()
