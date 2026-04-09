"""Render top-down + side views of the best structure from each campaign.

Output: figures/struct_<tag>.png (1 image, two subplots).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms

REPO = Path(__file__).resolve().parents[4]
FIG = Path(__file__).resolve().parent.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CAMPAIGNS = [
    ("cu111_co_camp", "Cu(111) — CO/H/H$_2$O"),
    ("cu100_co_camp", "Cu(100) — CO/H/H$_2$O"),
    ("cu211_co_camp", "Cu(211) — CO/H/H$_2$O"),
    ("pt111_orr_camp", "Pt(111) — ORR"),
    ("pt211_nrr_camp", "Pt(211) — NRR"),
    ("ag111_co_camp", "Ag(111) — CO/H/H$_2$O"),
    ("pd111_hsat_camp", "Pd(111) — H saturation"),
    ("ni111_chx_camp", "Ni(111) — CH$_x$"),
    ("au111_co_camp", "Au(111) — CO/OH"),
    ("ru0001_oh_camp", "Ru(0001) — O/OH/H$_2$O"),
]


def best_contcar(run_dir: Path) -> Path | None:
    from galoop.store import GaloopStore
    from galoop.individual import STATUS
    s = GaloopStore(run_dir)
    try:
        converged = s.get_by_status(STATUS.CONVERGED)
        if not converged:
            return None
        best = min(
            (c for c in converged if c.grand_canonical_energy is not None),
            key=lambda c: c.grand_canonical_energy,
            default=None,
        )
        if best is None:
            return None
        contcar = (run_dir / "structures" / best.id / "CONTCAR").resolve()
        return contcar if contcar.exists() else None
    finally:
        s.close()


def render(tag: str, title: str) -> None:
    run_dir = REPO / "runs" / tag
    contcar = best_contcar(run_dir)
    if contcar is None:
        print(f"  {tag}: no contcar")
        return
    atoms = read(str(contcar), format="vasp")
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.2))

    # top-down view (+z)
    plot_atoms(atoms, axes[0], rotation="0x,0y,0z", radii=0.6)
    axes[0].set_title("top-down", fontsize=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # side view (a-axis along horizontal, c up)
    plot_atoms(atoms, axes[1], rotation="-90x,0y,0z", radii=0.6)
    axes[1].set_title("side", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out = FIG / f"struct_{tag}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    for tag, title in CAMPAIGNS:
        render(tag, title)
    print("done.")
