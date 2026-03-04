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
        log.info("Loaded slab with %d atoms; using selective dynamics from POSCAR", n_slab)
    else:
        z_coords = atoms.get_positions()[:, 2]
        fixed_mask = z_coords < zmin
        n_fixed = int(np.sum(fixed_mask))
        if n_fixed > 0:
            atoms.set_constraint(FixAtoms(indices=np.where(fixed_mask)[0]))
            log.info(
                "No selective dynamics in POSCAR; fixed %d atoms (z < %.1f Å)",
                n_fixed, zmin,
            )
        else:
            log.warning(
                "No selective dynamics and no atoms below zmin=%.1f — slab atoms will be unconstrained",
                zmin,
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
    Randomly place an adsorbate on the slab.

    The adsorbate centre-of-mass is placed at a random (x, y) inside the
    unit cell and a random z in ``[zmin, zmax]``, then rotated around z.
    A clash check (via ASE :class:`NeighborList`) rejects placements where
    adsorbate atoms are closer than ``clash_scale × sum_of_covalent_radii``
    to any other atom.  Up to *max_attempts* placements are tried before
    accepting the last one with a warning.

    Parameters
    ----------
    slab : base slab (may already contain other adsorbates)
    adsorbate : adsorbate molecule to place
    zmin, zmax : vertical window (Å)
    n_orientations : unused (kept for config compat; rotation is always random)
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

    # Capture slab size BEFORE any adsorbate is appended so check_clash
    # knows which atom indices are slab atoms (and should be ignored).
    n_slab_before = len(slab)

    combined = slab.copy()  # initialise so the variable is always bound

    for attempt in range(max_attempts):
        # Random fractional xy, random z in window
        frac_xy = rng.uniform(0, 1, size=2)
        xy_cart = frac_xy @ cell[:2, :2]
        z = rng.uniform(zmin, zmax)

        # Random rotation around the surface normal (z)
        angle = rng.uniform(0, 2 * np.pi)

        ads = adsorbate.copy()
        com = ads.get_center_of_mass()
        ads.translate([xy_cart[0] - com[0], xy_cart[1] - com[1], z - com[2]])

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
