"""
galoop/science/reproduce.py

GA operators: splice, merge, add / remove / displace mutations.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Crossover: splice
# ---------------------------------------------------------------------------

def splice(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab: int,
    rng: np.random.Generator | None = None,
) -> Tuple[Atoms, Atoms]:
    """
    Z-cut splice: keep slab + low adsorbates from A, high from B (and vice
    versa).

    Returns two children.  If either parent has no adsorbates the
    corresponding child is a plain copy.
    """
    rng = rng or np.random.default_rng()

    pos_a = parent_a.get_positions()
    pos_b = parent_b.get_positions()
    ads_a = pos_a[n_slab:]
    ads_b = pos_b[n_slab:]
    sym_a = parent_a.get_chemical_symbols()
    sym_b = parent_b.get_chemical_symbols()

    if len(ads_a) == 0 or len(ads_b) == 0:
        return parent_a.copy(), parent_b.copy()

    z_lo = min(np.min(ads_a[:, 2]), np.min(ads_b[:, 2]))
    z_hi = max(np.max(ads_a[:, 2]), np.max(ads_b[:, 2]))
    z_cut = rng.uniform(z_lo, z_hi)

    def _build_child(base: Atoms, keep_below, donor_above_pos, donor_above_sym):
        child = Atoms(
            symbols=base.get_chemical_symbols()[:n_slab],
            positions=base.get_positions()[:n_slab],
            cell=base.get_cell(),
            pbc=base.get_pbc(),
        )
        child.set_constraint(FixAtoms(indices=list(range(n_slab))))
        # Low adsorbates from base
        for idx in np.where(keep_below)[0]:
            child.append(Atoms(base.get_chemical_symbols()[n_slab + idx],
                               positions=[base.get_positions()[n_slab + idx]]))
        # High adsorbates from donor
        for pos, sym in zip(donor_above_pos, donor_above_sym):
            child.append(Atoms(sym, positions=[pos]))
        return child

    mask_a_low = ads_a[:, 2] < z_cut
    mask_b_high = ads_b[:, 2] >= z_cut
    mask_b_low = ads_b[:, 2] < z_cut
    mask_a_high = ads_a[:, 2] >= z_cut

    child1 = _build_child(
        parent_a, mask_a_low,
        ads_b[mask_b_high], [sym_b[n_slab + i] for i in np.where(mask_b_high)[0]],
    )
    child2 = _build_child(
        parent_b, mask_b_low,
        ads_a[mask_a_high], [sym_a[n_slab + i] for i in np.where(mask_a_high)[0]],
    )
    return child1, child2


# ---------------------------------------------------------------------------
# Crossover: merge
# ---------------------------------------------------------------------------

def merge(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab: int,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Combine adsorbates from both parents onto A's slab."""
    child = parent_a.copy()
    child.set_constraint(FixAtoms(indices=list(range(n_slab))))

    ads_pos = parent_b.get_positions()[n_slab:]
    ads_sym = parent_b.get_chemical_symbols()[n_slab:]

    if len(ads_pos) > 0:
        child.extend(Atoms(symbols=ads_sym, positions=ads_pos))
    return child


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def mutate_add(
    atoms: Atoms,
    n_slab: int,
    symbol: str,
    position: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Add one atom of *symbol* at a random position above the slab."""
    rng = rng or np.random.default_rng()
    child = atoms.copy()
    child.set_constraint(FixAtoms(indices=list(range(n_slab))))

    if position is None:
        pos = atoms.get_positions()
        z_top = np.max(pos[:, 2])
        cell = atoms.get_cell()
        xy = rng.uniform(0, 1, size=2) @ cell[:2, :2]
        z = rng.uniform(z_top + 1.0, z_top + 4.0)
        position = np.array([xy[0], xy[1], z])

    child.extend(Atoms(symbols=[symbol], positions=[position]))
    return child


def mutate_remove(
    atoms: Atoms,
    n_slab: int,
    symbol: str | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """Remove one random adsorbate atom.  Returns ``None`` if nothing to remove."""
    rng = rng or np.random.default_rng()
    if len(atoms) <= n_slab:
        return None

    ads_indices = list(range(n_slab, len(atoms)))
    if symbol:
        syms = atoms.get_chemical_symbols()
        ads_indices = [i for i in ads_indices if syms[i] == symbol]
        if not ads_indices:
            return None

    idx = int(rng.choice(ads_indices))
    child = atoms.copy()
    del child[idx]
    return child


def mutate_displace(
    atoms: Atoms,
    n_slab: int,
    symbol: str | None = None,
    displacement: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """Randomly displace one adsorbate atom.  Returns ``None`` if no adsorbates."""
    rng = rng or np.random.default_rng()
    if len(atoms) <= n_slab:
        return None

    child = atoms.copy()
    child.set_constraint(FixAtoms(indices=list(range(n_slab))))

    ads_indices = list(range(n_slab, len(atoms)))
    if symbol:
        syms = child.get_chemical_symbols()
        ads_indices = [i for i in ads_indices if syms[i] == symbol]
        if not ads_indices:
            return None

    idx = int(rng.choice(ads_indices))
    delta = rng.normal(0, displacement / 3, size=3)
    pos = child.get_positions()
    pos[idx] += delta
    child.set_positions(pos)
    return child


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def crossover_operator(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab: int,
    operator_type: str = "splice",
    rng: np.random.Generator | None = None,
) -> Atoms | Tuple[Atoms, Atoms]:
    if operator_type == "splice":
        return splice(parent_a, parent_b, n_slab, rng)
    elif operator_type == "merge":
        return merge(parent_a, parent_b, n_slab, rng)
    raise ValueError(f"Unknown crossover operator: {operator_type}")


def mutation_operator(
    atoms: Atoms,
    n_slab: int,
    operator_type: str = "displace",
    symbol: str | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    if operator_type == "add":
        return mutate_add(atoms, n_slab, symbol or "H", rng=rng)
    elif operator_type == "remove":
        return mutate_remove(atoms, n_slab, symbol, rng=rng)
    elif operator_type == "displace":
        return mutate_displace(atoms, n_slab, symbol, rng=rng)
    raise ValueError(f"Unknown mutation operator: {operator_type}")
