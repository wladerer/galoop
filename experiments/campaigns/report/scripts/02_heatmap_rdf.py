"""Adsorbate position heatmaps + radial distribution functions.

Heatmap: for each campaign, collect adsorbate atom positions from all
converged structures, project into fractional (a, b) coordinates of
the slab cell, and 2D-histogram them. This shows which surface
regions the GA actually explored.

RDF: pairwise distance histograms of adsorbate-adsorbate distances,
pooled across converged structures.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

REPO = Path(__file__).resolve().parents[4]
FIG = Path(__file__).resolve().parent.parent / "figures"

CAMPAIGNS = [
    ("cu111_co_camp", "Cu(111)"),
    ("cu100_co_camp", "Cu(100)"),
    ("cu211_co_camp", "Cu(211)"),
    ("pt111_orr_camp", "Pt(111) ORR"),
    ("pt211_nrr_camp", "Pt(211) NRR"),
    ("ag111_co_camp", "Ag(111)"),
    ("pd111_hsat_camp", "Pd(111) H"),
    ("ni111_chx_camp", "Ni(111) CH$_x$"),
    ("au111_co_camp", "Au(111)"),
    ("ru0001_oh_camp", "Ru(0001)"),
]


def load_converged_atoms(run_dir: Path):
    from galoop.store import GaloopStore
    from galoop.individual import STATUS
    out = []
    s = GaloopStore(run_dir)
    try:
        for c in s.get_by_status(STATUS.CONVERGED):
            contcar = (run_dir / "structures" / c.id / "CONTCAR").resolve()
            if contcar.exists():
                try:
                    out.append(read(str(contcar), format="vasp"))
                except Exception:
                    pass
    finally:
        s.close()
    return out


def n_slab_from_config(run_dir: Path) -> int:
    import yaml
    with open(run_dir / "galoop.yaml") as f:
        cfg = yaml.safe_load(f)
    slab = read(cfg["slab"]["geometry"], format="vasp")
    return len(slab)


def collect_ads_fractional(atoms_list, n_slab: int) -> np.ndarray:
    """Return (N, 2) array of fractional (a, b) coordinates of all
    adsorbate atoms across all atoms objects."""
    pts = []
    for a in atoms_list:
        if len(a) <= n_slab:
            continue
        pos = a.get_positions()[n_slab:]
        # Convert to fractional via cell
        frac = np.linalg.solve(a.cell.array.T, pos.T).T
        frac = frac % 1.0
        pts.append(frac[:, :2])
    if not pts:
        return np.zeros((0, 2))
    return np.concatenate(pts, axis=0)


def adsorbate_distances(atoms_list, n_slab: int) -> np.ndarray:
    """Pooled pairwise distances between adsorbate atoms (mic)."""
    from ase.geometry import get_distances
    out = []
    for a in atoms_list:
        if len(a) <= n_slab + 1:
            continue
        ads_pos = a.get_positions()[n_slab:]
        # get_distances with 1 arg returns pairwise (N,N,3) and (N,N).
        _, dmat = get_distances(ads_pos, cell=a.cell, pbc=True)
        dmat = np.asarray(dmat)
        if dmat.ndim != 2 or dmat.shape[0] < 2:
            continue
        iu = np.triu_indices_from(dmat, k=1)
        out.append(dmat[iu])
    if not out:
        return np.zeros((0,))
    return np.concatenate(out)


def plot_heatmap_grid() -> None:
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.2), dpi=150)
    for ax, (tag, title) in zip(axes.flat, CAMPAIGNS):
        rd = REPO / "runs" / tag
        atoms_list = load_converged_atoms(rd)
        n_slab = n_slab_from_config(rd)
        pts = collect_ads_fractional(atoms_list, n_slab)
        if pts.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        else:
            h, xe, ye = np.histogram2d(
                pts[:, 0], pts[:, 1],
                bins=24, range=[[0, 1], [0, 1]],
            )
            ax.imshow(
                h.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="equal", cmap="magma",
                interpolation="nearest",
            )
        ax.set_title(f"{title}\n(N={len(atoms_list)})", fontsize=8)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("frac a", fontsize=7)
        ax.set_ylabel("frac b", fontsize=7)
    fig.suptitle("Adsorbate position heatmaps (all converged structures, projected to ab plane)", fontsize=10)
    fig.tight_layout()
    out = FIG / "heatmap_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_rdf_grid() -> None:
    fig, axes = plt.subplots(2, 5, figsize=(12.5, 5.2), dpi=150, sharex=True, sharey=False)
    bins = np.linspace(0.5, 8.0, 60)
    for ax, (tag, title) in zip(axes.flat, CAMPAIGNS):
        rd = REPO / "runs" / tag
        atoms_list = load_converged_atoms(rd)
        n_slab = n_slab_from_config(rd)
        d = adsorbate_distances(atoms_list, n_slab)
        if d.size == 0:
            ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        else:
            ax.hist(d, bins=bins, color="#2a7aaf", edgecolor="none")
        ax.set_title(f"{title}\n(pairs={d.size})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"$r$ (Å)", fontsize=7)
        ax.set_ylabel("count", fontsize=7)
        ax.axvline(2.0, color="grey", lw=0.5, ls="--")
    fig.suptitle("Adsorbate–adsorbate pair distance histograms (pooled over converged)", fontsize=10)
    fig.tight_layout()
    out = FIG / "rdf_grid.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    plot_heatmap_grid()
    plot_rdf_grid()
