"""Build slabs + compute reference energies for the 7 follow-on campaigns.

Loads UMA once, builds each slab (writing slab.vasp into the right run
dir), computes the slab energy, and computes whatever gas-phase
reference molecules each campaign needs. Prints the values; the human
loop (or me) writes them into the yamls.
"""
from __future__ import annotations

import sys
from pathlib import Path

from ase.build import (
    fcc100,
    fcc111,
    fcc211,
    hcp0001,
    molecule,
)
from ase.constraints import FixAtoms
from ase.io import write
from ase import Atoms

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO / "runs" / "cu111_smoke"))
from calc import make_calculator

# Lattice constants (Å), experimental fcc / hcp
A = {
    "Pt": 3.924, "Ag": 4.085, "Pd": 3.891, "Ni": 3.524, "Au": 4.078,
    "Ru_a": 2.706, "Ru_c": 4.282,
}


def fix_bottom_third(slab: Atoms) -> Atoms:
    slab.pbc = [True, True, True]
    zs = slab.get_positions()[:, 2]
    cut = zs.min() + (zs.max() - zs.min()) / 3.0
    slab.set_constraint(
        FixAtoms(indices=[i for i, z in enumerate(zs) if z < cut])
    )
    return slab


def build_OOH() -> Atoms:
    # OOH: O–O–H bent geometry, ~1.45 Å O–O, 0.97 Å O–H, ~100° angle.
    # binding atom = first O (index 0).
    return Atoms(
        symbols=["O", "O", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.45, 0.0, 0.0],
            [1.70, 0.95, 0.0],
        ],
    )


def build_OH() -> Atoms:
    return Atoms(
        symbols=["O", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
    )


def main() -> None:
    print("Building slabs ...")
    slabs = {}

    s = fcc111("Pt", size=(4, 4, 3), a=A["Pt"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/pt111_orr_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Pt(111)_orr"] = ("runs/pt111_orr_camp", s)

    s = fcc211("Pt", size=(3, 4, 3), a=A["Pt"], vacuum=10.0)
    fix_bottom_third(s)
    write("runs/pt211_nrr_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Pt(211)_nrr"] = ("runs/pt211_nrr_camp", s)

    s = fcc111("Ag", size=(4, 4, 3), a=A["Ag"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/ag111_co_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Ag(111)_co"] = ("runs/ag111_co_camp", s)

    s = fcc111("Pd", size=(4, 4, 3), a=A["Pd"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/pd111_hsat_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Pd(111)_hsat"] = ("runs/pd111_hsat_camp", s)

    s = fcc111("Ni", size=(4, 4, 3), a=A["Ni"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/ni111_chx_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Ni(111)_chx"] = ("runs/ni111_chx_camp", s)

    s = fcc111("Au", size=(4, 4, 3), a=A["Au"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/au111_co_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Au(111)_co"] = ("runs/au111_co_camp", s)

    s = hcp0001("Ru", size=(4, 4, 3), a=A["Ru_a"], c=A["Ru_c"], vacuum=10.0, orthogonal=False)
    fix_bottom_third(s)
    write("runs/ru0001_oh_camp/slab.vasp", s, format="vasp", direct=True, vasp5=True)
    slabs["Ru(0001)_oh"] = ("runs/ru0001_oh_camp", s)

    for tag, (d, s) in slabs.items():
        zs = s.get_positions()[:, 2]
        print(f"  {tag:20s}: {len(s):3d} atoms  cell={s.cell.lengths().round(3)}  angles={s.cell.angles().round(2)}  z=[{zs.min():.2f},{zs.max():.2f}]")

    print("\nLoading UMA (one-time) ...")
    calc = make_calculator({"model": "uma-s-1p1", "task": "oc20", "device": "cuda"})

    print("\nGas-phase reference molecules:")
    mol_e: dict[str, float] = {}
    for name in ["CO", "H2", "H2O", "N2", "NH3", "O2"]:
        m = molecule(name)
        m.center(vacuum=6.0)
        m.pbc = True
        m.calc = calc
        e = m.get_potential_energy()
        mol_e[name] = e
        print(f"  E_{name:6s} = {e:.6f}")

    # Custom OOH and OH (for ORR)
    for label, mol in [("OH", build_OH()), ("OOH", build_OOH())]:
        mol.center(vacuum=6.0)
        mol.pbc = True
        mol.calc = calc
        e = mol.get_potential_energy()
        mol_e[label] = e
        print(f"  E_{label:6s} = {e:.6f}  (custom geometry)")

    print("\nSlab energies (clean):")
    slab_e: dict[str, float] = {}
    for tag, (d, s) in slabs.items():
        s.calc = calc
        e = s.get_potential_energy()
        slab_e[tag] = e
        print(f"  E_{tag:20s} = {e:.6f}")

    print("\n--- summary dict for next-step yaml templating ---")
    print("MOL_E =", repr(mol_e))
    print("SLAB_E =", repr(slab_e))


if __name__ == "__main__":
    main()
