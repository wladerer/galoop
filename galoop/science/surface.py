"""
gocia/science/surface.py

Surface adsorbate manipulation: loading slabs, adsorbates, placement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write

log = logging.getLogger(__name__)


class SlabInfo(NamedTuple):
    """Metadata about a slab."""
    atoms: Atoms
    n_slab_atoms: int
    zmin: float
    zmax: float
    symbols: list[str]


def load_slab(
    geometry_path: str | Path,
    zmin: float,
    zmax: float,
) -> SlabInfo:
    """
    Load a slab geometry and identify fixed atoms.

    Parameters
    ----------
    geometry_path : Path to POSCAR/CONTCAR
    zmin : Min z for adsorbate placement (Å)
    zmax : Max z for adsorbate placement (Å)

    Returns
    -------
    SlabInfo with slab atoms and metadata
    """
    geometry_path = Path(geometry_path)
    if not geometry_path.exists():
        raise FileNotFoundError(f"Slab geometry not found: {geometry_path}")

    try:
        atoms = read(str(geometry_path), format="vasp")
    except Exception as e:
        raise IOError(f"Failed to load slab: {e}")

    # Identify slab atoms: those with z < zmin
    positions = atoms.get_positions()
    z_coords = positions[:, 2]
    slab_mask = z_coords < zmin
    n_slab_atoms = np.sum(slab_mask)

    if n_slab_atoms == 0:
        log.warning("No atoms below zmin; treating all atoms as slab")
        n_slab_atoms = len(atoms)
        slab_mask = np.ones(len(atoms), dtype=bool)

    # Apply FixAtoms constraint to slab
    if n_slab_atoms > 0:
        constraint = FixAtoms(indices=np.where(slab_mask)[0])
        atoms.set_constraint(constraint)
        log.info(f"  Fixed {n_slab_atoms} slab atoms (z < {zmin})")

    symbols = atoms.get_chemical_symbols()
    return SlabInfo(
        atoms=atoms,
        n_slab_atoms=n_slab_atoms,
        zmin=zmin,
        zmax=zmax,
        symbols=symbols,
    )


def load_adsorbate(
    symbol: str,
    geometry: str | Path | None = None,
    coordinates: list[list[float]] | None = None,
) -> Atoms:
    """
    Load an adsorbate geometry.

    Parameters
    ----------
    symbol : Element or formula (e.g., "O", "OH", "OOH")
    geometry : Path to structure file (optional)
    coordinates : Inline XYZ coordinates (optional)

    Returns
    -------
    ASE Atoms object
    """
    if geometry:
        geometry = Path(geometry)
        if not geometry.exists():
            raise FileNotFoundError(f"Adsorbate geometry not found: {geometry}")
        try:
            atoms = read(str(geometry))
        except Exception as e:
            raise IOError(f"Failed to load adsorbate {symbol}: {e}")
        return atoms

    if coordinates:
        # Parse inline coordinates
        # Expected format: list of [x, y, z] or [[x, y, z], ...]
        coords = np.array(coordinates)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        # Symbol could be formula like "OH" or "OOH"
        atoms = _parse_formula_atoms(symbol, coords)
        return atoms

    # Fallback: use ASE's minimal geometries
    atoms = _get_minimal_geometry(symbol)
    return atoms


def _parse_formula_atoms(formula: str, coords: np.ndarray) -> Atoms:
    """Parse a formula like 'OH' and create Atoms with given coordinates."""
    symbols = _parse_formula(formula)
    if len(symbols) != len(coords):
        raise ValueError(
            f"Formula {formula} has {len(symbols)} atoms but {len(coords)} coordinates given"
        )
    atoms = Atoms(symbols=symbols, positions=coords)
    return atoms


def _parse_formula(formula: str) -> list[str]:
    """Parse a formula string into element list."""
    # Simple parser for formulas like "O", "OH", "OOH", "H2O", etc.
    symbols = []
    i = 0
    while i < len(formula):
        if formula[i].isupper():
            elem = formula[i]
            i += 1
            if i < len(formula) and formula[i].islower():
                elem += formula[i]
                i += 1
            # Check for count
            count_str = ""
            while i < len(formula) and formula[i].isdigit():
                count_str += formula[i]
                i += 1
            count = int(count_str) if count_str else 1
            symbols.extend([elem] * count)
        else:
            i += 1
    return symbols


def _get_minimal_geometry(symbol: str) -> Atoms:
    """Get a minimal geometry for an element/formula."""
    from ase.build import molecule

    try:
        # Try ASE's molecule database
        atoms = molecule(symbol)
    except Exception:
        # Fallback: single atom
        symbols = _parse_formula(symbol)
        if symbols:
            atoms = Atoms(symbols=symbols[0])
        else:
            atoms = Atoms("H")

    # Center in box
    atoms.center(vacuum=3.0)
    return atoms


def place_adsorbate(
    slab: Atoms,
    adsorbate: Atoms,
    zmin: float,
    zmax: float,
    n_orientations: int = 1,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """
    Randomly place an adsorbate on a slab.

    Parameters
    ----------
    slab : Slab atoms
    adsorbate : Adsorbate atoms to place
    zmin : Min z for placement (Å)
    zmax : Max z for placement (Å)
    n_orientations : Number of random orientations to try
    rng : NumPy random generator

    Returns
    -------
    Atoms : Combined slab + adsorbate
    """
    if rng is None:
        rng = np.random.default_rng()

    # Get slab surface (highest atom)
    slab_z = slab.get_positions()[:, 2]
    surface_z = np.max(slab_z)

    # Pick a random XY position on the surface
    cell = slab.get_cell()
    xy = rng.uniform(0, 1, size=2)
    xy_pos = xy @ cell[:2, :2]

    # Pick a random Z position in [zmin, zmax]
    z_pos = rng.uniform(zmin, zmax)

    # Pick a random orientation
    angle = rng.uniform(0, 2 * np.pi)

    # Create combined structure
    combined = slab.copy()
    ads_copy = adsorbate.copy()

    # Translate adsorbate
    ads_com = ads_copy.get_center_of_mass()
    ads_copy.translate([xy_pos[0] - ads_com[0], xy_pos[1] - ads_com[1], z_pos - ads_com[2]])

    # Rotate around vertical axis
    rotation = _rotation_matrix_z(angle)
    ads_positions = ads_copy.get_positions()
    ads_positions -= ads_copy.get_center_of_mass()
    ads_positions = ads_positions @ rotation.T
    ads_positions += ads_copy.get_center_of_mass()
    ads_copy.set_positions(ads_positions)

    # Combine
    combined.extend(ads_copy)

    # Check for clashes (optional, for now just warn)
    if _check_clash(combined, threshold=1.5):
        log.debug(f"  Possible steric clash; allowing anyway")

    return combined


def _rotation_matrix_z(angle: float) -> np.ndarray:
    """Rotation matrix around z-axis."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _check_clash(atoms: Atoms, threshold: float = 1.5) -> bool:
    """Check for atomic clashes (atoms closer than threshold)."""
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()

    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            dist = np.linalg.norm(positions[i] - positions[j])
            # Sum of van der Waals radii (rough estimate)
            vdw_sum = 2.0
            if dist < threshold:
                return True
    return False


def detect_desorption(
    atoms: Atoms,
    slab_info: SlabInfo,
    z_threshold: float = 15.0,
) -> bool:
    """
    Check if adsorbates have desorbed (left the surface region).

    Parameters
    ----------
    atoms : Final structure
    slab_info : Reference slab info
    z_threshold : Z-coordinate above which is considered desorbed

    Returns
    -------
    bool : True if desorption detected
    """
    if len(atoms) <= slab_info.n_slab_atoms:
        return False

    adsorbate_positions = atoms.get_positions()[slab_info.n_slab_atoms :]
    z_coords = adsorbate_positions[:, 2]

    if np.any(z_coords > z_threshold):
        return True
    return False
