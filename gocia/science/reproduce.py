"""
gocia/science/reproduce.py

GA operators: splice, merge, add/remove/displace mutations.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.geometry import get_distances

log = logging.getLogger(__name__)


def splice(
    slab_a: Atoms,
    slab_b: Atoms,
    n_slab_atoms: int,
    rng: np.random.Generator | None = None,
) -> Tuple[Atoms, Atoms]:
    """
    Splice: cut both structures at random Z, keep bottom from A, top from B.

    Parameters
    ----------
    slab_a : Parent A
    slab_b : Parent B
    n_slab_atoms : Number of fixed slab atoms
    rng : Random generator

    Returns
    -------
    (child1, child2) : Two children (B-top on A-slab, A-top on B-slab)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get adsorbate regions (everything above slab)
    pos_a = slab_a.get_positions()
    pos_b = slab_b.get_positions()

    ads_a = pos_a[n_slab_atoms:]
    ads_b = pos_b[n_slab_atoms:]

    if len(ads_a) == 0 or len(ads_b) == 0:
        # No adsorbates; return copies
        return slab_a.copy(), slab_b.copy()

    # Find cutting plane (Z coordinate)
    z_cut = rng.uniform(
        min(np.min(ads_a[:, 2]), np.min(ads_b[:, 2])),
        max(np.max(ads_a[:, 2]), np.max(ads_b[:, 2])),
    )

    # Splice: keep slab + bottom adsorbates from A, top adsorbates from B
    mask_a_keep = ads_a[:, 2] < z_cut
    mask_b_keep = ads_b[:, 2] >= z_cut

    child1 = slab_a.copy()
    child1.set_constraint(FixAtoms(indices=range(n_slab_atoms)))

    child2 = slab_b.copy()
    child2.set_constraint(FixAtoms(indices=range(n_slab_atoms)))

    # Build child1: slab_a + ads_a[below] + ads_b[above]
    atoms_to_add_1 = []
    positions_to_add_1 = []
    if np.any(mask_b_keep):
        symbols_b_top = [slab_b.get_chemical_symbols()[n_slab_atoms + i] 
                         for i in np.where(mask_b_keep)[0]]
        positions_b_top = ads_b[mask_b_keep]
        atoms_to_add_1.extend(symbols_b_top)
        positions_to_add_1.extend(positions_b_top)

    if atoms_to_add_1:
        child1.extend(Atoms(symbols=atoms_to_add_1, positions=positions_to_add_1))

    # Build child2: slab_b + ads_b[below] + ads_a[above]
    atoms_to_add_2 = []
    positions_to_add_2 = []
    if np.any(mask_a_keep):
        symbols_a_top = [slab_a.get_chemical_symbols()[n_slab_atoms + i] 
                         for i in np.where(mask_a_keep)[0]]
        positions_a_top = ads_a[mask_a_keep]
        atoms_to_add_2.extend(symbols_a_top)
        positions_to_add_2.extend(positions_a_top)

    if atoms_to_add_2:
        child2.extend(Atoms(symbols=atoms_to_add_2, positions=positions_to_add_2))

    return child1, child2


def merge(
    slab_a: Atoms,
    slab_b: Atoms,
    n_slab_atoms: int,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Merge: combine adsorbates from both parents.

    Parameters
    ----------
    slab_a : Parent A
    slab_b : Parent B
    n_slab_atoms : Number of fixed slab atoms
    rng : Random generator

    Returns
    -------
    Atoms : Child with combined adsorbates
    """
    if rng is None:
        rng = np.random.default_rng()

    child = slab_a.copy()
    child.set_constraint(FixAtoms(indices=range(n_slab_atoms)))

    # Get adsorbates from B
    pos_b = slab_b.get_positions()
    ads_b = pos_b[n_slab_atoms:]
    symbols_b_ads = slab_b.get_chemical_symbols()[n_slab_atoms:]

    if len(ads_b) > 0:
        # Add B's adsorbates to A's slab
        ads_atoms = Atoms(symbols=symbols_b_ads, positions=ads_b)
        child.extend(ads_atoms)

    return child


def mutate_add(
    atoms: Atoms,
    n_slab_atoms: int,
    symbol: str,
    position: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Add a single atom to the structure.

    Parameters
    ----------
    atoms : Structure to mutate
    n_slab_atoms : Number of fixed slab atoms
    symbol : Element to add
    position : Position for new atom; random if None
    rng : Random generator

    Returns
    -------
    Atoms : Mutated structure
    """
    if rng is None:
        rng = np.random.default_rng()

    child = atoms.copy()
    child.set_constraint(FixAtoms(indices=range(n_slab_atoms)))

    if position is None:
        # Random position above slab
        positions = atoms.get_positions()
        z_max = np.max(positions[n_slab_atoms:, 2]) if len(atoms) > n_slab_atoms else np.max(positions[:, 2])
        cell = atoms.get_cell()

        xy = rng.uniform(0, 1, size=2) @ cell[:2, :2]
        z = rng.uniform(z_max + 1.0, z_max + 5.0)
        position = np.array([xy[0], xy[1], z])

    new_atom = Atoms(symbol=symbol, positions=[position])
    child.extend(new_atom)

    return child


def mutate_remove(
    atoms: Atoms,
    n_slab_atoms: int,
    symbol: str | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """
    Remove an adsorbate atom from the structure.

    Parameters
    ----------
    atoms : Structure to mutate
    n_slab_atoms : Number of fixed slab atoms
    symbol : Element to remove; random if None
    rng : Random generator

    Returns
    -------
    Atoms : Mutated structure, or None if no adsorbates to remove
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(atoms) <= n_slab_atoms:
        return None  # No adsorbates to remove

    # Pick an adsorbate to remove
    ads_indices = list(range(n_slab_atoms, len(atoms)))
    symbols_ads = [atoms.get_chemical_symbols()[i] for i in ads_indices]

    if symbol:
        # Remove a specific element
        candidates = [i for i, s in zip(ads_indices, symbols_ads) if s == symbol]
        if not candidates:
            return None
        idx_to_remove = rng.choice(candidates)
    else:
        # Remove any adsorbate atom
        idx_to_remove = rng.choice(ads_indices)

    del atoms[idx_to_remove]
    return atoms


def mutate_displace(
    atoms: Atoms,
    n_slab_atoms: int,
    symbol: str | None = None,
    displacement: float = 0.5,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """
    Displace an adsorbate atom randomly.

    Parameters
    ----------
    atoms : Structure to mutate
    n_slab_atoms : Number of fixed slab atoms
    symbol : Element to displace; random if None
    displacement : Max displacement (Å)
    rng : Random generator

    Returns
    -------
    Atoms : Mutated structure, or None if no adsorbates
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(atoms) <= n_slab_atoms:
        return None

    child = atoms.copy()
    child.set_constraint(FixAtoms(indices=range(n_slab_atoms)))

    # Pick an adsorbate
    ads_indices = list(range(n_slab_atoms, len(atoms)))
    symbols_ads = [child.get_chemical_symbols()[i] for i in ads_indices]

    if symbol:
        candidates = [i for i, s in zip(ads_indices, symbols_ads) if s == symbol]
        if not candidates:
            return None
        idx_to_move = rng.choice(candidates)
    else:
        idx_to_move = rng.choice(ads_indices)

    # Random displacement
    delta = rng.normal(0, displacement / 3, size=3)
    pos = child.get_positions()
    pos[idx_to_move] += delta
    child.set_positions(pos)

    return child


def crossover_operator(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab_atoms: int,
    operator_type: str = "splice",
    rng: np.random.Generator | None = None,
) -> Atoms | Tuple[Atoms, Atoms]:
    """
    Generic crossover operator dispatcher.

    Parameters
    ----------
    parent_a, parent_b : Parents
    n_slab_atoms : Slab atoms
    operator_type : "splice" or "merge"
    rng : Random generator

    Returns
    -------
    Atoms or (Atoms, Atoms) depending on operator
    """
    if operator_type.lower() == "splice":
        return splice(parent_a, parent_b, n_slab_atoms, rng)
    elif operator_type.lower() == "merge":
        return merge(parent_a, parent_b, n_slab_atoms, rng)
    else:
        raise ValueError(f"Unknown crossover operator: {operator_type}")


def mutation_operator(
    atoms: Atoms,
    n_slab_atoms: int,
    operator_type: str = "displace",
    symbol: str | None = None,
    rng: np.random.Generator | None = None,
) -> Atoms | None:
    """
    Generic mutation operator dispatcher.

    Parameters
    ----------
    atoms : Structure to mutate
    n_slab_atoms : Slab atoms
    operator_type : "add", "remove", "displace"
    symbol : Element (for add/remove/displace)
    rng : Random generator

    Returns
    -------
    Atoms : Mutated structure, or None if mutation not possible
    """
    if operator_type.lower() == "add":
        return mutate_add(atoms, n_slab_atoms, symbol or "H", rng=rng)
    elif operator_type.lower() == "remove":
        return mutate_remove(atoms, n_slab_atoms, symbol, rng=rng)
    elif operator_type.lower() == "displace":
        return mutate_displace(atoms, n_slab_atoms, symbol, rng=rng)
    else:
        raise ValueError(f"Unknown mutation operator: {operator_type}")
