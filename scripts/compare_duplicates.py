#!/usr/bin/env python3
"""
Visually compare duplicate structure clusters.

Usage:
    python scripts/compare_duplicates.py [run_dir]

Loads each duplicate cluster from galoop.db and opens it in the ASE GUI.
Step through structures with the arrow keys. Close the window to advance
to the next cluster.
"""
import sys
from pathlib import Path

from ase.io import read
from ase.visualize import view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from galoop.database import GaloopDB

run_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
db_path = run_dir / "galoop.db"

if not db_path.exists():
    sys.exit(f"No database at {db_path}")

with GaloopDB(db_path) as db:
    converged = {ind.id: ind for ind in db.get_by_status("converged")}
    dups = db.get_by_status("duplicate")

# Group duplicates by original
clusters: dict[str, list] = {}
for dup in dups:
    dup_of = dup.extra_data.get("dup_of")
    if dup_of:
        clusters.setdefault(dup_of, []).append(dup)

if not clusters:
    sys.exit("No duplicate clusters found.")


def load_atoms(ind):
    if not ind.geometry_path:
        return None
    poscar = Path(ind.geometry_path)
    contcar = poscar.parent / "CONTCAR"
    path = contcar if contcar.exists() else poscar
    try:
        return read(str(path), format="vasp")
    except Exception as e:
        print(f"  Could not load {path}: {e}")
        return None


for orig_id, dup_list in sorted(clusters.items(), key=lambda x: -len(x[1])):
    orig = converged.get(orig_id)
    print(f"\nCluster: original={orig_id}  ({len(dup_list)} duplicate(s))")

    to_view = []
    if orig:
        a = load_atoms(orig)
        if a:
            a.info["label"] = f"ORIGINAL {orig_id}  G={orig.grand_canonical_energy}"
            to_view.append(a)

    for dup in sorted(dup_list, key=lambda d: -(d.extra_data.get("tanimoto") or 0)):
        a = load_atoms(dup)
        if a:
            tanimoto = dup.extra_data.get("tanimoto", "?")
            label_score = f"{tanimoto:.3f}" if isinstance(tanimoto, float) else str(tanimoto)
            a.info["label"] = f"DUP {dup.id}  Tanimoto={label_score}"
            to_view.append(a)

    if to_view:
        print(f"  Opening {len(to_view)} structures — use arrow keys to step through.")
        view(to_view)    # blocks until window is closed
    else:
        print("  No loadable structures in this cluster, skipping.")
