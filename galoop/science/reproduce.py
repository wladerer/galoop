"""
galoop/science/reproduce.py

GA operators: splice, merge, add / remove / displace / translate mutations.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.data import covalent_radii

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Molecule connectivity helper
# ---------------------------------------------------------------------------

def _group_molecules(atoms: Atoms, n_slab: int, mult: float = 1.25) -> list[list[int]]:
    """Group adsorbate atoms into connected molecules using covalent radii.

    Two adsorbate atoms are considered bonded when their distance is less than
    ``mult * (r_cov_i + r_cov_j)``.  This is element-aware and handles all
    common adsorbates (CO, OH, NH3, hydrocarbons, ...) without a hard-coded
    distance threshold.

    Uses ASE's :class:`NeighborList` with PBC **disabled** for the
    distance check. This is intentional: PBC-aware (MIC) distances
    falsely connect distinct molecules whose periodic images happen to
    be within the covalent cutoff (common on densely packed surfaces
    where inter-molecular O--O or N--N contacts are < 1.8 A). Disabling
    PBC means molecules that genuinely straddle a cell boundary get
    split, but in practice surface adsorbates rarely straddle xy
    boundaries because the slab cell is much larger than any single
    molecule.

    The NeighborList approach is O(N) via cell-list partitioning instead
    of the previous O(N^2) brute-force loop, which matters on large
    supercells at high coverage.

    Returns a list of groups, where each group is a list of absolute atom
    indices (>= n_slab).  Isolated atoms form single-atom groups.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs

    ads_indices = list(range(n_slab, len(atoms)))
    if not ads_indices:
        return []

    # Build a temporary atoms object with PBC disabled so the neighbor
    # list uses Cartesian distances (no MIC).
    work = atoms.copy()
    work.pbc = [False, False, False]

    cutoffs = [mult * covalent_radii[atoms.numbers[i]] for i in range(len(atoms))]
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(work)

    # Build adjacency only among adsorbate atoms
    ads_set = set(ads_indices)
    n = len(ads_indices)
    idx_to_local = {idx: i for i, idx in enumerate(ads_indices)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for idx in ads_indices:
        neighbors, _ = nl.get_neighbors(idx)
        for nbr in neighbors:
            nbr = int(nbr)
            if nbr in ads_set and nbr > idx:
                i_local = idx_to_local[idx]
                j_local = idx_to_local[nbr]
                adj[i_local].append(j_local)
                adj[j_local].append(i_local)

    # Connected-components via DFS
    visited = [False] * n
    molecules: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(ads_indices[node])
            stack.extend(adj[node])
        molecules.append(component)
    return molecules


# ---------------------------------------------------------------------------
# Crossover: splice
# ---------------------------------------------------------------------------

def splice(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab: int,
    rng: np.random.Generator | None = None,
) -> tuple[Atoms, Atoms]:
    """
    Molecule-aware Z-cut splice.

    Groups adsorbate atoms into connected molecules, then assigns each whole
    molecule to "below" or "above" the Z-cut based on its first atom's z.
    This prevents molecules from being split across the two children.

    Returns two children.  If either parent has no adsorbates the
    corresponding child is a plain copy.
    """
    rng = rng or np.random.default_rng()

    mols_a = _group_molecules(parent_a, n_slab)
    mols_b = _group_molecules(parent_b, n_slab)

    if not mols_a or not mols_b:
        return parent_a.copy(), parent_b.copy()

    pos_a = parent_a.get_positions()
    pos_b = parent_b.get_positions()

    # Z-cut is drawn from the range of all adsorbate binding-atom z values
    z_all = (
        [pos_a[mol[0], 2] for mol in mols_a]
        + [pos_b[mol[0], 2] for mol in mols_b]
    )
    z_cut = rng.uniform(min(z_all), max(z_all))

    def _build_child(base: Atoms, keep_mols, donor_mols, donor_atoms: Atoms):
        child = Atoms(
            symbols=base.get_chemical_symbols()[:n_slab],
            positions=pos_a[:n_slab] if base is parent_a else pos_b[:n_slab],
            cell=base.get_cell(),
            pbc=base.get_pbc(),
        )
        child.set_constraint(FixAtoms(indices=list(range(n_slab))))
        syms_base = base.get_chemical_symbols()
        syms_donor = donor_atoms.get_chemical_symbols()
        pos_donor = donor_atoms.get_positions()
        for mol in keep_mols:
            for idx in mol:
                child.extend(Atoms(syms_base[idx], positions=[base.get_positions()[idx]]))
        for mol in donor_mols:
            for idx in mol:
                child.extend(Atoms(syms_donor[idx], positions=[pos_donor[idx]]))
        return child

    mols_a_low  = [m for m in mols_a if pos_a[m[0], 2] <  z_cut]
    mols_a_high = [m for m in mols_a if pos_a[m[0], 2] >= z_cut]
    mols_b_low  = [m for m in mols_b if pos_b[m[0], 2] <  z_cut]
    mols_b_high = [m for m in mols_b if pos_b[m[0], 2] >= z_cut]

    child1 = _build_child(parent_a, mols_a_low,  mols_b_high, parent_b)
    child2 = _build_child(parent_b, mols_b_low,  mols_a_high, parent_a)
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
    """Combine adsorbates from both parents onto A's slab.

    Adsorbates from B are added one whole molecule at a time.  A molecule is
    only accepted if none of its atoms clash with the current child (scale
    0.7× covalent radii); if any atom would clash the entire molecule is
    dropped.  This prevents orphan atoms (e.g. O without C) in the child.
    """
    from galoop.science.surface import check_clash

    child = parent_a.copy()
    child.set_constraint(FixAtoms(indices=list(range(n_slab))))

    syms_b = parent_b.get_chemical_symbols()
    pos_b  = parent_b.get_positions()

    for mol in _group_molecules(parent_b, n_slab):
        trial = child.copy()
        for idx in mol:
            trial.extend(Atoms(symbols=[syms_b[idx]], positions=[pos_b[idx]]))
        if not check_clash(trial, n_slab=len(child), scale=0.7):
            child = trial

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
    """Remove one random adsorbate molecule.  Returns ``None`` if nothing to remove.

    Removes the entire connected molecule that contains the randomly chosen
    atom, preventing orphan atoms (e.g. O left behind after removing C from CO).
    If *symbol* is given, only molecules whose binding atom (first in group)
    matches that symbol are eligible.
    """
    rng = rng or np.random.default_rng()
    if len(atoms) <= n_slab:
        return None

    molecules = _group_molecules(atoms, n_slab)
    if not molecules:
        return None

    if symbol:
        syms = atoms.get_chemical_symbols()
        molecules = [m for m in molecules if syms[m[0]] == symbol]
        if not molecules:
            return None

    mol = molecules[int(rng.integers(len(molecules)))]
    remove_set = set(mol)
    child = atoms.copy()
    # Delete in reverse order to keep indices stable
    for idx in sorted(remove_set, reverse=True):
        del child[idx]
    return child


def mutate_rattle_slab(
    atoms: Atoms,
    n_slab: int,
    amplitude: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Rattle unfixed slab surface atoms AND adsorbates with Gaussian noise.

    Slab atoms get a small amplitude (default 0.1 Å), adsorbates get a
    larger one (3x the slab amplitude) so that the relaxation has a real
    chance to find a different basin.  Adsorbate molecules are perturbed
    as a whole (each molecule gets a single rigid translation) so molecular
    geometry is preserved.
    """
    rng = rng or np.random.default_rng()
    child = atoms.copy()

    fixed: set[int] = set()
    for constraint in child.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.update(int(i) for i in constraint.index)

    pos = child.get_positions()

    # Slab atoms: small per-atom rattle
    for i in range(n_slab):
        if i not in fixed:
            pos[i] += rng.normal(0.0, amplitude, size=3)

    # Adsorbates: rigid molecule rattle with larger amplitude
    ads_amplitude = amplitude * 3
    for mol in _group_molecules(atoms, n_slab):
        delta = np.array([
            rng.normal(0, ads_amplitude),
            rng.normal(0, ads_amplitude),
            rng.normal(0, ads_amplitude * 0.5),
        ])
        for idx in mol:
            pos[idx] += delta

    child.set_positions(pos)
    return child


def mutate_displace(
    atoms: Atoms,
    n_slab: int,
    symbol: str | None = None,
    displacement: float = 1.5,
    rng: np.random.Generator | None = None,
    max_attempts: int = 10,
) -> Atoms | None:
    """Randomly displace one whole adsorbate molecule by a Gaussian step.

    Operates on a connected molecule so molecular geometry is preserved.
    Uses ``displacement`` directly as the sigma in x/y, with a smaller
    sigma in z.  Tries up to *max_attempts* random displacements before
    giving up.  Returns ``None`` if no clash-free displacement is found.
    """
    from galoop.science.surface import check_clash

    rng = rng or np.random.default_rng()
    if len(atoms) <= n_slab:
        return None

    all_molecules = _group_molecules(atoms, n_slab)
    if not all_molecules:
        return None

    eligible = all_molecules
    if symbol:
        syms = atoms.get_chemical_symbols()
        eligible = [m for m in all_molecules if syms[m[0]] == symbol]
        if not eligible:
            return None

    mol = eligible[int(rng.integers(len(eligible)))]
    n_mols = len(all_molecules)

    for _ in range(max_attempts):
        delta = np.array([
            rng.normal(0, displacement),
            rng.normal(0, displacement),
            rng.normal(0, displacement * 0.3),
        ])
        child = atoms.copy()
        child.set_constraint(FixAtoms(indices=list(range(n_slab))))
        pos = child.get_positions()
        for idx in mol:
            pos[idx] += delta
        child.set_positions(pos)

        if check_clash(child, n_slab=n_slab, scale=0.7):
            continue
        if len(_group_molecules(child, n_slab)) != n_mols:
            continue
        return child

    return None


def mutate_translate(
    atoms: Atoms,
    n_slab: int,
    displacement: float = 0.8,
    rng: np.random.Generator | None = None,
    max_attempts: int = 10,
) -> Atoms | None:
    """Translate one adsorbate molecule to a different surface site.

    Picks a random connected molecule and moves it to a randomly chosen
    surface site that is at least 2 Å from its current position.  This
    produces a qualitatively different geometry rather than a small
    perturbation (which always relaxes back to the same basin).

    Tries up to *max_attempts* candidate sites to find one without clashes.
    Falls back to a large Gaussian displacement if no clash-free site
    is found or no surface sites are identified.
    """
    from galoop.science.surface import check_clash

    rng = rng or np.random.default_rng()
    molecules = _group_molecules(atoms, n_slab)
    if not molecules:
        return None

    mol = molecules[int(rng.integers(len(molecules)))]

    from galoop.science.surface import _surface_normal, find_surface_sites

    base_pos = atoms.get_positions()
    n_hat = _surface_normal(atoms)

    # In-plane position of the molecule: project out the surface-normal component
    mol_pos = base_pos[mol]
    current_inplane = (mol_pos - np.outer(mol_pos @ n_hat, n_hat)).mean(axis=0)

    try:
        sites = find_surface_sites(atoms, n_slab, normal=n_hat)
    except Exception as exc:
        log.debug("find_surface_sites failed in mutate_translate: %s", exc)
        sites = []

    def _build_child(delta):
        child = atoms.copy()
        child.set_constraint(FixAtoms(indices=list(range(n_slab))))
        pos = child.get_positions()
        for idx in mol:
            pos[idx] += delta
        child.set_positions(pos)
        return child

    if sites:
        # Project sites to in-plane (strip normal component), compute distances
        sites_arr = np.array(sites)
        sites_inplane = sites_arr - np.outer(sites_arr @ n_hat, n_hat)
        dists = np.linalg.norm(sites_inplane - current_inplane, axis=1)
        far_sites = sites_inplane[dists > 2.0]
        if len(far_sites) == 0:
            far_sites = sites_inplane
        rng.shuffle(far_sites)

        for target_inplane in far_sites[:max_attempts]:
            jitter = rng.normal(0, 0.3, size=3)
            jitter -= (jitter @ n_hat) * n_hat  # keep jitter in-plane
            delta_inplane = (target_inplane - current_inplane) + jitter
            # Small random displacement along the normal (was z jitter before)
            delta = delta_inplane + rng.normal(0, displacement * 0.3) * n_hat
            trial = _build_child(delta)
            # Reject if the translated molecule clashes with anything else
            if not check_clash(trial, n_slab=n_slab, scale=0.7):
                # Also reject if the molecule got merged with another (graph change)
                new_mols = _group_molecules(trial, n_slab)
                if len(new_mols) == len(molecules):
                    return trial

    # Fallback: large Gaussian displacement (no site-aware jump)
    for _ in range(max_attempts):
        delta = np.array([
            rng.normal(0, displacement),
            rng.normal(0, displacement),
            rng.normal(0, displacement * 0.3),
        ])
        trial = _build_child(delta)
        if not check_clash(trial, n_slab=n_slab, scale=0.7):
            new_mols = _group_molecules(trial, n_slab)
            if len(new_mols) == len(molecules):
                return trial

    return None


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def crossover_operator(
    parent_a: Atoms,
    parent_b: Atoms,
    n_slab: int,
    operator_type: str = "splice",
    rng: np.random.Generator | None = None,
) -> Atoms | tuple[Atoms, Atoms]:
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
    elif operator_type == "rattle_slab":
        return mutate_rattle_slab(atoms, n_slab, rng=rng)
    elif operator_type == "translate":
        return mutate_translate(atoms, n_slab, rng=rng)
    raise ValueError(f"Unknown mutation operator: {operator_type}")
