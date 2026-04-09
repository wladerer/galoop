"""Build the 7 follow-up slabs only (no calculator). Safe to run while
another galoop campaign is using the GPU."""
from __future__ import annotations

from ase.build import fcc100, fcc111, fcc211, hcp0001
from ase.constraints import FixAtoms
from ase.io import write
from ase import Atoms

A = {"Pt": 3.924, "Ag": 4.085, "Pd": 3.891, "Ni": 3.524, "Au": 4.078,
     "Ru_a": 2.706, "Ru_c": 4.282}


def fix_bottom_third(slab: Atoms) -> Atoms:
    slab.pbc = [True, True, True]
    zs = slab.get_positions()[:, 2]
    cut = zs.min() + (zs.max() - zs.min()) / 3.0
    slab.set_constraint(FixAtoms(indices=[i for i, z in enumerate(zs) if z < cut]))
    return slab


def build(builder, *, path: str, label: str, **kwargs):
    s = builder(**kwargs)
    fix_bottom_third(s)
    write(path, s, format="vasp", direct=True, vasp5=True)
    zs = s.get_positions()[:, 2]
    print(f"  {label:24s}: {len(s):3d} atoms  cell={tuple(s.cell.lengths().round(3))}  "
          f"angles={tuple(s.cell.angles().round(2))}  z=[{zs.min():.2f},{zs.max():.2f}]")
    return s


print("Building slabs ...")
build(fcc111, path="runs/pt111_orr_camp/slab.vasp",  label="Pt(111) 4×4×3",
      symbol="Pt", size=(4, 4, 3), a=A["Pt"], vacuum=10.0, orthogonal=False)
build(fcc211, path="runs/pt211_nrr_camp/slab.vasp",  label="Pt(211) 3×4×3",
      symbol="Pt", size=(3, 4, 3), a=A["Pt"], vacuum=10.0)
build(fcc111, path="runs/ag111_co_camp/slab.vasp",   label="Ag(111) 4×4×3",
      symbol="Ag", size=(4, 4, 3), a=A["Ag"], vacuum=10.0, orthogonal=False)
build(fcc111, path="runs/pd111_hsat_camp/slab.vasp", label="Pd(111) 4×4×3",
      symbol="Pd", size=(4, 4, 3), a=A["Pd"], vacuum=10.0, orthogonal=False)
build(fcc111, path="runs/ni111_chx_camp/slab.vasp",  label="Ni(111) 4×4×3",
      symbol="Ni", size=(4, 4, 3), a=A["Ni"], vacuum=10.0, orthogonal=False)
build(fcc111, path="runs/au111_co_camp/slab.vasp",   label="Au(111) 4×4×3",
      symbol="Au", size=(4, 4, 3), a=A["Au"], vacuum=10.0, orthogonal=False)
build(hcp0001, path="runs/ru0001_oh_camp/slab.vasp", label="Ru(0001) 4×4×3",
      symbol="Ru", size=(4, 4, 3), a=A["Ru_a"], c=A["Ru_c"], vacuum=10.0, orthogonal=False)

print("\nDone.")
