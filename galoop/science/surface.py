"""
galoop/science/surface.py

Surface / adsorbate manipulation: slab loading, adsorbate placement, clash
detection (using ASE's NeighborList), and desorption detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class SlabInfo(NamedTuple):
    """Metadata returned by :func:`load_slab`."""
    atoms: Atoms
    n_slab_atoms: int
    zmin: float
    zmax: float
    symbols: list[str]


# ---------------------------------------------------------------------------
# Slab loading
# ---------------------------------------------------------------------------

def load_slab(
    geometry_path: str | Path,
    zmin: float,
    zmax: float,
) -> SlabInfo:
    """
    Load a slab POSCAR/CONTCAR geometry.

    Atoms with ``z < zmin`` are treated as bulk slab and get ``FixAtoms``
    constraints.

    Parameters
    ----------
    geometry_path : path to POSCAR / CONTCAR
    zmin, zmax : adsorbate placement window (Å)
    """
    geometry_path = Path(geometry_path)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Slab geometry not found: {geometry_path}")

    atoms = read(str(geometry_path), format="vasp")
    n_slab = len(atoms)

    if atoms.constraints:
        from ase.constraints import FixAtoms
        n_fixed = sum(
            len(c.index) for c in atoms.constraints if isinstance(c, FixAtoms)
        )
        log.info("Loaded slab with %d atoms; %d fixed via selective dynamics", n_slab, n_fixed)
    else:
        log.warning(
            "No selective dynamics in slab POSCAR — all slab atoms will be unconstrained. "
            "Add selective dynamics (T/F flags) to freeze bulk layers."
        )

    return SlabInfo(
        atoms=atoms,
        n_slab_atoms=n_slab,
        zmin=zmin,
        zmax=zmax,
        symbols=atoms.get_chemical_symbols(),
    )


# ---------------------------------------------------------------------------
# Adsorbate loading
# ---------------------------------------------------------------------------

def load_adsorbate(
    symbol: str,
    geometry: str | Path | None = None,
    coordinates: list[list[float]] | None = None,
) -> Atoms:
    """
    Load or build an adsorbate :class:`Atoms` object.

    Priority: *geometry* file > inline *coordinates* > single atom (monoatomic only).

    Multi-atom species must supply either *geometry* or *coordinates*; passing
    neither raises a :exc:`ValueError`.
    """
    if geometry is not None:
        p = Path(geometry)
        if not p.exists():
            raise FileNotFoundError(f"Adsorbate geometry not found: {p}")
        return read(str(p))

    if coordinates is not None:
        coords = np.asarray(coordinates, dtype=float)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        symbols = parse_formula(symbol)
        if len(symbols) != len(coords):
            raise ValueError(
                f"Formula '{symbol}' has {len(symbols)} atoms but "
                f"{len(coords)} coordinate rows were given"
            )
        return Atoms(symbols=symbols, positions=coords)

    elems = parse_formula(symbol)
    if len(elems) == 1:
        return Atoms(elems)

    raise ValueError(
        f"Adsorbate '{symbol}' has multiple atoms; "
        "specify 'geometry' (file path) or 'coordinates' (inline positions) in the config"
    )


# ---------------------------------------------------------------------------
# Formula parser
# ---------------------------------------------------------------------------

def parse_formula(formula: str) -> list[str]:
    """
    Parse a chemical formula into an element list.

    Examples
    --------
    >>> parse_formula("H2O")
    ['H', 'H', 'O']
    >>> parse_formula("Fe2O3")
    ['Fe', 'Fe', 'O', 'O', 'O']
    >>> parse_formula("OOH")
    ['O', 'O', 'H']
    """
    symbols: list[str] = []
    i = 0
    while i < len(formula):
        if not formula[i].isupper():
            i += 1
            continue
        elem = formula[i]
        i += 1
        # Lower-case continuation (e.g. "Fe")
        if i < len(formula) and formula[i].islower():
            elem += formula[i]
            i += 1
        # Digit count (e.g. "2")
        count_str = ""
        while i < len(formula) and formula[i].isdigit():
            count_str += formula[i]
            i += 1
        count = int(count_str) if count_str else 1
        symbols.extend([elem] * count)
    return symbols


# ---------------------------------------------------------------------------
# Adsorbate orientation
# ---------------------------------------------------------------------------

def _rotation_to(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix that rotates unit vector *src* onto *dst*."""
    c = float(np.dot(src, dst))
    if c > 1.0 - 1e-10:
        return np.eye(3)
    if c < -1.0 + 1e-10:
        # 180° rotation — pick any axis perpendicular to src
        perp = np.cross(src, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(perp) < 1e-10:
            perp = np.cross(src, np.array([0.0, 1.0, 0.0]))
        perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    v = np.cross(src, dst)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def orient_upright(adsorbate: Atoms, binding_index: int = 0) -> Atoms:
    """
    Return a copy of *adsorbate* rotated so that the atom at *binding_index*
    points toward the surface (−z direction).

    For a single-atom adsorbate this is a no-op.  For multi-atom species
    the molecule is rotated so the binding atom sits at minimum z, ready
    to adsorb.

    Parameters
    ----------
    adsorbate     : molecule to orient
    binding_index : index of the atom that binds to the surface (default 0)

    Examples
    --------
    OH  → ``binding_index=0`` puts O down, H up
    OOH → ``binding_index=0`` puts first O down
    H₂O → ``binding_index=1`` would put O down if the formula is H₂O
    """
    if len(adsorbate) <= 1:
        return adsorbate.copy()

    if binding_index >= len(adsorbate):
        raise ValueError(
            f"binding_index={binding_index} out of range for adsorbate with "
            f"{len(adsorbate)} atoms"
        )

    ads = adsorbate.copy()
    pos = ads.get_positions()
    com = ads.get_center_of_mass()

    v = pos[binding_index] - com
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return ads      # binding atom coincides with COM — nothing to do

    rot = _rotation_to(v / norm, np.array([0.0, 0.0, -1.0]))
    ads.set_positions((pos - com) @ rot.T + com)
    return ads


# ---------------------------------------------------------------------------
# Adsorbate placement
# ---------------------------------------------------------------------------

def find_surface_sites(
    slab: Atoms,
    n_slab: int,
    bond_mult: float = 1.0,
) -> list[np.ndarray]:
    """Identify atop, bridge, and hollow sites on the top surface layer.

    Returns a list of (x, y) positions where adsorbates can bind.

    Strategy: find the top-layer slab atoms, then generate atop (directly
    above each atom), bridge (midpoint of bonded pairs), and hollow
    (centroid of bonded triangles) sites.
    """
    from ase.data import covalent_radii

    pos = slab.get_positions()[:n_slab]
    z_coords = pos[:, 2]
    z_top = z_coords.max()

    # Top-layer atoms: within 1.0 Å of the highest z
    top_mask = z_coords > z_top - 1.0
    top_indices = np.where(top_mask)[0]
    top_pos = pos[top_indices]

    sites: list[np.ndarray] = []

    # Atop sites
    for p in top_pos:
        sites.append(p[:2].copy())

    # Find bonded pairs in the top layer (nearest-neighbor distance)
    cell = slab.get_cell()
    nums = slab.get_atomic_numbers()
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            ii, jj = top_indices[i], top_indices[j]
            cutoff = bond_mult * (covalent_radii[nums[ii]] + covalent_radii[nums[jj]])
            # Use MIC for periodic distance
            d = slab.get_distance(ii, jj, mic=True)
            if d < cutoff * 1.5:  # within ~1.5x bond length = nearest neighbors
                # Bridge site: midpoint (in Cartesian, approximating PBC)
                mid = 0.5 * (pos[ii][:2] + pos[jj][:2])
                sites.append(mid)

    # Hollow sites: centroids of triangles formed by bonded triplets
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            for k in range(j + 1, len(top_indices)):
                ii, jj, kk = top_indices[i], top_indices[j], top_indices[k]
                d_ij = slab.get_distance(ii, jj, mic=True)
                d_jk = slab.get_distance(jj, kk, mic=True)
                d_ik = slab.get_distance(ii, kk, mic=True)
                max_cut = max(
                    covalent_radii[nums[ii]] + covalent_radii[nums[jj]],
                    covalent_radii[nums[jj]] + covalent_radii[nums[kk]],
                    covalent_radii[nums[ii]] + covalent_radii[nums[kk]],
                ) * bond_mult * 1.5
                if d_ij < max_cut and d_jk < max_cut and d_ik < max_cut:
                    centroid = (pos[ii][:2] + pos[jj][:2] + pos[kk][:2]) / 3
                    sites.append(centroid)

    return sites


def place_adsorbate(
    slab: Atoms,
    adsorbate: Atoms,
    zmin: float,
    zmax: float,
    n_orientations: int = 1,
    binding_index: int = 0,
    rng: np.random.Generator | None = None,
    clash_scale: float = 0.85,
    max_attempts: int = 50,
) -> Atoms:
    """
    Place an adsorbate on the slab, preferring surface binding sites.

    First attempts site-aware placement (atop, bridge, hollow sites
    identified from slab geometry).  Falls back to random (x, y, z)
    placement if all sites are occupied.

    A clash check (via ASE :class:`NeighborList`) rejects placements where
    adsorbate atoms are closer than ``clash_scale × sum_of_covalent_radii``
    to any other atom.  Up to *max_attempts* placements are tried.

    Parameters
    ----------
    slab : base slab (may already contain other adsorbates)
    adsorbate : adsorbate molecule to place
    zmin, zmax : vertical window (Å)
    n_orientations : unused (kept for config compat; rotation is always random)
    binding_index : index of atom in adsorbate that binds to surface
    rng : NumPy random generator
    clash_scale : fraction of covalent-radii sum used as clash threshold
    max_attempts : placement retries before accepting with a warning

    Returns
    -------
    Atoms — combined slab + adsorbate
    """
    if rng is None:
        rng = np.random.default_rng()

    # Orient the binding atom toward the surface before placement.
    adsorbate = orient_upright(adsorbate, binding_index=binding_index)

    cell = slab.get_cell()
    n_slab_before = len(slab)

    # Identify surface binding sites (atop, bridge, hollow)
    # Use the bare slab atom count — the first n_slab atoms are always slab
    n_slab_bare = n_slab_before
    # If slab already has adsorbates, count only slab atoms
    # (n_slab_before includes previously placed adsorbates)
    # We need the original slab atom count — infer from constraints
    for c in slab.constraints:
        if isinstance(c, FixAtoms):
            # Slab atoms include fixed + free surface layer
            # Use the max fixed index + the same count as a heuristic
            max_fixed = max(c.index) + 1
            # Rough: slab atoms = at least up to the highest fixed index
            # but usually more (free surface layer above)
            break

    sites = find_surface_sites(slab, n_slab_bare)
    if sites:
        rng.shuffle(sites)

    combined = slab.copy()  # initialise so the variable is always bound

    # Binding height: place binding atom ~1.8-2.2 Å above slab top
    slab_top_z = slab.get_positions()[:n_slab_bare, 2].max()
    from ase.data import covalent_radii as _cov_rad
    binding_z_offset = 1.8  # reasonable default for most adsorbate-metal bonds

    for attempt in range(max_attempts):
        if attempt < len(sites):
            # Site-aware placement: use identified surface site
            xy = sites[attempt]
            z = slab_top_z + binding_z_offset + rng.uniform(-0.2, 0.3)
        else:
            # Fallback: random (x, y) in unit cell, random z in window
            frac_xy = rng.uniform(0, 1, size=2)
            xy = frac_xy @ cell[:2, :2]
            z = rng.uniform(zmin, zmax)

        # Random rotation around the surface normal (z)
        angle = rng.uniform(0, 2 * np.pi)

        ads = adsorbate.copy()
        com = ads.get_center_of_mass()
        ads.translate([xy[0] - com[0], xy[1] - com[1], z - com[2]])

        # Rotate around z through the new centre of mass
        pos = ads.get_positions() - ads.get_center_of_mass()
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
        pos = pos @ rot.T + ads.get_center_of_mass()
        ads.set_positions(pos)

        combined = slab.copy()
        combined.extend(ads)

        if not check_clash(combined, n_slab=n_slab_before, scale=clash_scale):
            return combined

        log.debug("Placement attempt %d/%d clashed — retrying", attempt + 1, max_attempts)

    # Accept last attempt with a warning
    log.warning(
        "Could not find clash-free placement in %d attempts; "
        "accepting last candidate", max_attempts,
    )
    return combined


# ---------------------------------------------------------------------------
# Clash detection  (ASE NeighborList — O(N) with cell-list)
# ---------------------------------------------------------------------------

def check_clash(
    atoms: Atoms,
    n_slab: int,
    scale: float = 0.7,
) -> bool:
    """
    Return ``True`` if any *adsorbate* atom is too close to any other atom.

    Only adsorbate atoms (indices >= *n_slab*) are checked as sources of
    clashes.  Intra-slab contacts are pre-existing and must not be flagged.

    Uses ASE's :class:`NeighborList` backed by a cell-list algorithm, so it
    runs in O(N) rather than the O(N²) naive double loop.

    Parameters
    ----------
    atoms : combined slab + adsorbate structure to check
    n_slab : number of leading atoms that belong to the bare slab
    scale : fraction of covalent-radii sum to use as threshold.
            0.7 is a sensible default for surface adsorbate checks.
    """
    if len(atoms) <= n_slab:
        return False  # no adsorbate atoms present

    cutoffs = natural_cutoffs(atoms, mult=scale)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False, skin=0.0)
    nl.update(atoms)

    for i in range(n_slab, len(atoms)):
        indices, _ = nl.get_neighbors(i)
        if len(indices) > 0:
            return True  # adsorbate atom has a neighbour within threshold

    return False


# ---------------------------------------------------------------------------
# Desorption detection
# ---------------------------------------------------------------------------

def detect_desorption(
    atoms: Atoms,
    slab_info: SlabInfo,
    z_threshold: float | None = None,
) -> bool:
    """
    Return ``True`` if any adsorbate atom has drifted above *z_threshold*.

    Parameters
    ----------
    atoms : final relaxed structure
    slab_info : reference slab metadata
    z_threshold : Å above which an adsorbate is considered desorbed.
                  Defaults to ``slab_info.zmax + 0.5``.
    """
    if z_threshold is None:
        z_threshold = slab_info.zmax + 0.5

    if len(atoms) <= slab_info.n_slab_atoms:
        return False

    ads_z = atoms.get_positions()[slab_info.n_slab_atoms:, 2]
    return bool(np.any(ads_z > z_threshold))


def validate_surface_binding(
    atoms: Atoms,
    n_slab: int,
    bond_mult: float = 1.3,
) -> tuple[bool, list[list[int]]]:
    """Check that every adsorbate molecule is bonded to the surface.

    Groups adsorbate atoms into molecules (via covalent bonding), then
    verifies each molecule has at least one atom within bonding distance
    of a slab atom.  Subsurface atoms count as bound.

    Parameters
    ----------
    atoms : relaxed structure (slab + adsorbates)
    n_slab : number of leading slab atoms
    bond_mult : multiplier on sum of covalent radii for surface bond cutoff

    Returns
    -------
    (all_bound, unbound_molecules) where unbound_molecules is a list of
    atom-index lists for molecules with no surface contact.
    """
    from ase.neighborlist import NeighborList, natural_cutoffs
    from galoop.science.reproduce import _group_molecules

    if len(atoms) <= n_slab:
        return True, []

    molecules = _group_molecules(atoms, n_slab)
    if not molecules:
        return True, []

    # Build neighbor list with bond_mult for detecting surface bonds
    cutoffs = natural_cutoffs(atoms, mult=bond_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)

    slab_indices = set(range(n_slab))
    unbound: list[list[int]] = []

    for mol in molecules:
        has_surface_contact = False
        for atom_idx in mol:
            neighbors, _ = nl.get_neighbors(atom_idx)
            if slab_indices & set(int(n) for n in neighbors):
                has_surface_contact = True
                break
        if not has_surface_contact:
            unbound.append(mol)

    return len(unbound) == 0, unbound
