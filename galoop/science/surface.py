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
from ase.io import read, write
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

    z_coords = atoms.get_positions()[:, 2]
    slab_mask = z_coords < zmin
    n_slab = int(np.sum(slab_mask))

    if n_slab == 0:
        log.warning("No atoms below zmin=%.1f — treating all atoms as slab", zmin)
        n_slab = len(atoms)
        slab_mask = np.ones(len(atoms), dtype=bool)

    atoms.set_constraint(FixAtoms(indices=np.where(slab_mask)[0]))
    log.info("Fixed %d slab atoms (z < %.1f Å)", n_slab, zmin)

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

    Priority: *geometry* file > inline *coordinates* > ASE molecule DB > single atom.
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

    # Fall back to ASE molecule database, then single atom
    return _minimal_geometry(symbol)


def _minimal_geometry(symbol: str) -> Atoms:
    """Best-effort geometry from ASE's molecule DB or a lone atom."""
    from ase.build import molecule

    try:
        atoms = molecule(symbol)
    except Exception:
        elems = parse_formula(symbol)
        atoms = Atoms(elems[0] if elems else "H")

    atoms.center(vacuum=3.0)
    return atoms


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
# Adsorbate placement
# ---------------------------------------------------------------------------

def place_adsorbate(
    slab: Atoms,
    adsorbate: Atoms,
    zmin: float,
    zmax: float,
    n_orientations: int = 1,
    rng: np.random.Generator | None = None,
    clash_scale: float = 0.7,
    max_attempts: int = 50,
) -> Atoms:
    """
    Randomly place an adsorbate on the slab.

    The adsorbate centre-of-mass is placed at a random (x, y) inside the
    unit cell and a random z in ``[zmin, zmax]``, then rotated around z.
    A clash check (via ASE :class:`NeighborList`) rejects placements where
    atoms are closer than ``clash_scale × sum_of_covalent_radii``.  Up to
    *max_attempts* placements are tried before accepting the last one with a
    warning.

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

    cell = slab.get_cell()

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

        if not check_clash(combined, scale=clash_scale):
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
    scale: float = 0.7,
) -> bool:
    """
    Return ``True`` if any two atoms are closer than *scale* × sum of their
    natural (covalent) cutoffs.

    Uses ASE's :class:`NeighborList` backed by a cell-list algorithm, so it
    runs in O(N) rather than the O(N²) naive double loop.

    Parameters
    ----------
    atoms : structure to check
    scale : fraction of covalent-radii sum to use as threshold.
            0.7 is a sensible default for surface adsorbate checks.
    """
    cutoffs = natural_cutoffs(atoms, mult=scale)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False, skin=0.0)
    nl.update(atoms)

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        if len(indices) > 0:
            # At least one neighbour within the scaled cutoff → clash
            # But we need to verify the *actual* distance, because the
            # NeighborList bins by cutoff_i + cutoff_j, which is already
            # our scaled covalent sum.  So any hit is a clash.
            return True
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
                  Defaults to ``slab_info.zmax + 5.0``.
    """
    if z_threshold is None:
        z_threshold = slab_info.zmax + 5.0

    if len(atoms) <= slab_info.n_slab_atoms:
        return False

    ads_z = atoms.get_positions()[slab_info.n_slab_atoms:, 2]
    return bool(np.any(ads_z > z_threshold))
