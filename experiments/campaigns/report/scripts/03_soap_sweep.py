"""SOAP cutoff sensitivity analysis for duplicate detection.

For each of three representative campaigns (Cu(111), Pd(111) H, and
Ru(0001)) we load all converged structures, compute SOAP vectors at
a grid of (r_cut, n_max, l_max), form the pairwise Tanimoto
similarity matrix, and measure:

- **dup fraction vs threshold** curves for each SOAP setting
  (how many pairs would be flagged as duplicates under a given
  threshold),
- **uniqueness count vs SOAP r_cut** at a fixed threshold (0.92),
  showing how sensitive the uniqueness-vs-duplicate boundary is to
  cutoff choice.

Output:
  soap_dup_curves.png   — dup fraction curves
  soap_rcut_unique.png  — unique count vs r_cut
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

# Representative subset
TARGETS = [
    ("cu111_co_camp", "Cu(111)", "#2a7aaf"),
    ("pd111_hsat_camp", "Pd(111) H", "#c84a4a"),
    ("ru0001_oh_camp", "Ru(0001)", "#4c9a4a"),
]

SOAP_GRID = [
    # (r_cut, n_max, l_max, label)
    (4.0, 6, 4, "rcut=4 n=6 l=4"),
    (5.0, 6, 4, "rcut=5 n=6 l=4"),
    (6.0, 6, 4, "rcut=6 n=6 l=4"),
    (5.0, 8, 6, "rcut=5 n=8 l=6"),
    (5.0, 4, 2, "rcut=5 n=4 l=2"),
]

RCUT_SCAN = np.arange(3.0, 7.51, 0.5)
THRESH_FIXED = 0.92


def load_atoms(run_dir: Path):
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


def ads_soap_vectors(atoms_list, n_slab: int, r_cut, n_max, l_max) -> np.ndarray:
    from dscribe.descriptors import SOAP
    species = sorted({
        sym for a in atoms_list for sym in a.get_chemical_symbols()
    })
    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        periodic=True,
        sparse=False,
    )
    vecs = []
    for a in atoms_list:
        if len(a) <= n_slab:
            continue
        ads_idx = list(range(n_slab, len(a)))
        # Average SOAP over adsorbate atoms
        v = soap.create(a, centers=ads_idx)
        vecs.append(v.mean(axis=0))
    if not vecs:
        return np.zeros((0, soap.get_number_of_features()))
    return np.stack(vecs)


def tanimoto_matrix(V: np.ndarray) -> np.ndarray:
    """Generalized Jaccard / Tanimoto on non-negative vectors; falls
    back to cosine if negative entries exist."""
    if V.shape[0] == 0:
        return np.zeros((0, 0))
    if (V < 0).any():
        # cosine similarity
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        Vn = V / norms
        return Vn @ Vn.T
    # generalized Tanimoto: sum(min) / sum(max)
    n = V.shape[0]
    M = np.zeros((n, n))
    for i in range(n):
        mn = np.minimum(V[i], V)
        mx = np.maximum(V[i], V)
        s_mn = mn.sum(axis=1)
        s_mx = mx.sum(axis=1) + 1e-12
        M[i] = s_mn / s_mx
    return M


def dup_curve(M: np.ndarray, thresholds):
    """Fraction of upper-triangle pairs with similarity >= thresh."""
    n = M.shape[0]
    if n < 2:
        return np.zeros_like(thresholds, dtype=float)
    iu = np.triu_indices(n, k=1)
    pairs = M[iu]
    return np.array([(pairs >= t).mean() for t in thresholds])


def uniqueness_count(M: np.ndarray, threshold: float) -> int:
    """Greedy uniqueness: count structures that are not >= threshold
    similar to any earlier structure."""
    n = M.shape[0]
    if n == 0:
        return 0
    unique = 1
    for i in range(1, n):
        if M[i, :i].max() < threshold:
            unique += 1
    return unique


def plot_dup_curves() -> None:
    thr = np.linspace(0.5, 1.0, 41)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True, dpi=150)
    for ax, (tag, title, color) in zip(axes, TARGETS):
        rd = REPO / "runs" / tag
        atoms_list = load_atoms(rd)
        n_slab = n_slab_from_config(rd)
        print(f"  {tag}: {len(atoms_list)} converged atoms")
        for (r_cut, n_max, l_max, label) in SOAP_GRID:
            V = ads_soap_vectors(atoms_list, n_slab, r_cut, n_max, l_max)
            M = tanimoto_matrix(V)
            curve = dup_curve(M, thr)
            ax.plot(thr, curve, label=label, lw=1.2)
        ax.set_title(f"{title}  (N={len(atoms_list)})", fontsize=9)
        ax.set_xlabel("similarity threshold", fontsize=8)
        ax.axvline(0.92, color="grey", ls="--", lw=0.6)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("pair fraction above threshold", fontsize=8)
    axes[-1].legend(fontsize=6, loc="upper right")
    fig.suptitle("SOAP-similarity pair-fraction curves under varying cutoff/basis", fontsize=10)
    fig.tight_layout()
    out = FIG / "soap_dup_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def plot_rcut_unique() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=150)
    for (tag, title, color) in TARGETS:
        rd = REPO / "runs" / tag
        atoms_list = load_atoms(rd)
        n_slab = n_slab_from_config(rd)
        ys = []
        for r_cut in RCUT_SCAN:
            V = ads_soap_vectors(atoms_list, n_slab, r_cut, 6, 4)
            M = tanimoto_matrix(V)
            ys.append(uniqueness_count(M, THRESH_FIXED))
        ax.plot(RCUT_SCAN, ys, "o-", color=color, label=f"{title}  ({len(atoms_list)} total)")
    ax.set_xlabel(r"SOAP $r_{\mathrm{cut}}$ (Å)")
    ax.set_ylabel(f"greedy unique count at threshold = {THRESH_FIXED}")
    ax.set_title("Duplicate detection vs SOAP cutoff")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIG / "soap_rcut_unique.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    print("Generating dup curves ...")
    plot_dup_curves()
    print("Generating rcut scan ...")
    plot_rcut_unique()
