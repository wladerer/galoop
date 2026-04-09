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
    binding_z_offset: float = 2.0
    snap_z_min_offset: float = 0.8
    snap_z_max_offset: float = 4.0
    placement_clash_scale: float = 0.7
    normal: np.ndarray | None = None  # unit surface-normal vector (along c); None → [0,0,1]


# ---------------------------------------------------------------------------
# Surface normal
# ---------------------------------------------------------------------------

def _surface_normal(atoms: Atoms) -> np.ndarray:
    """Return the unit vector normal to the surface.

    Computed as the normalised cross product of the in-plane lattice vectors
    ``a`` and ``b`` (``cell[0] × cell[1]``), which is perpendicular to the
    surface plane by construction — even for non-orthogonal cells where the
    c-vector is tilted (e.g. ASE ``fcc211``, or any stepped slab built with
    ``ase.build.surface``).

    The sign is chosen so the normal points into the vacuum side (the same
    half-space as the c-vector), giving ``[0, 0, 1]`` for the usual
    orthogonal slab convention. Falls back to ``[0, 0, 1]`` for degenerate
    cells where ``a × b`` vanishes.
    """
    cell = np.asarray(atoms.cell, dtype=float)
    a, b, c = cell[0], cell[1], cell[2]
    n = np.cross(a, b)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    n = n / norm
    # Point toward the vacuum side (same half-space as c).
    if np.dot(n, c) < 0:
        n = -n
    return n


def _surface_normal_from_slab_info(slab_info: "SlabInfo") -> np.ndarray:
    """Return the surface normal, preferring the cached value in *slab_info*."""
    if slab_info.normal is not None:
        return slab_info.normal
    return _surface_normal(slab_info.atoms)


# ---------------------------------------------------------------------------
# Slab loading
# ---------------------------------------------------------------------------

def read_atoms(path: str | Path, **kwargs) -> Atoms:
    """Read a single :class:`Atoms` object from *path*.

    Wraps :func:`ase.io.read` and narrows its ``Atoms | list[Atoms]`` return
    type for static checkers (and for our own sanity — every call site in
    galoop expects exactly one frame).
    """
    result = read(str(path), **kwargs)
    if isinstance(result, list):
        if len(result) != 1:
            raise ValueError(
                f"Expected a single frame in {path}, got {len(result)}"
            )
        single = result[0]
        if not isinstance(single, Atoms):
            raise TypeError(
                f"Expected Atoms in {path}, got {type(single).__name__}"
            )
        return single
    return result


def load_slab_from_config(slab_cfg) -> SlabInfo:
    """Build a :class:`SlabInfo` directly from a :class:`SlabConfig`.

    Convenience wrapper used by every CLI subcommand and the GA loop —
    avoids re-spelling all the SlabConfig fields at every call site.
    """
    return load_slab(
        slab_cfg.geometry,
        zmin=slab_cfg.sampling_zmin,
        zmax=slab_cfg.sampling_zmax,
        binding_z_offset=slab_cfg.binding_z_offset,
        snap_z_min_offset=slab_cfg.snap_z_min_offset,
        snap_z_max_offset=slab_cfg.snap_z_max_offset,
        placement_clash_scale=slab_cfg.placement_clash_scale,
    )


def load_slab(
    geometry_path: str | Path,
    zmin: float,
    zmax: float,
    binding_z_offset: float = 2.0,
    snap_z_min_offset: float = 0.8,
    snap_z_max_offset: float = 4.0,
    placement_clash_scale: float = 0.7,
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

    atoms = read_atoms(geometry_path, format="vasp")
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

    normal = _surface_normal(atoms)
    angle_from_z = np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
    if angle_from_z > 1.0:
        log.info(
            "Surface normal is %.2f° from z-axis — using n̂=%s for height projections",
            angle_from_z, np.round(normal, 4),
        )

    return SlabInfo(
        atoms=atoms,
        n_slab_atoms=n_slab,
        zmin=zmin,
        zmax=zmax,
        symbols=atoms.get_chemical_symbols(),
        binding_z_offset=binding_z_offset,
        snap_z_min_offset=snap_z_min_offset,
        snap_z_max_offset=snap_z_max_offset,
        placement_clash_scale=placement_clash_scale,
        normal=normal,
    )


# ---------------------------------------------------------------------------
# Adsorbate loading
# ---------------------------------------------------------------------------

def load_adsorbate(
    symbol: str,
    geometry: str | Path | None = None,
    coordinates: list[dict[str, list[float]]] | None = None,
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
        return read_atoms(p)

    if coordinates is not None:
        # New schema (validated in config.py): list of single-key dicts
        # {symbol: [x,y,z]}.  Row order defines atom indices.
        symbols: list[str] = []
        positions: list[list[float]] = []
        for row in coordinates:
            (sym, pos), = row.items()
            symbols.append(sym)
            positions.append(pos)
        return Atoms(symbols=symbols, positions=np.asarray(positions, dtype=float))

    elems = parse_formula(symbol)
    if len(elems) == 1:
        return Atoms(elems)

    raise ValueError(
        f"Adsorbate '{symbol}' has multiple atoms; "
        "specify 'geometry' (file path) or 'coordinates' (inline positions) in the config"
    )


def load_ads_template_dict(ads_configs) -> dict[str, Atoms]:
    """Build the ``{symbol: Atoms template}`` dict every spawn site needs.

    Centralised here so callers don't keep re-spelling the comprehension.
    """
    return {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in ads_configs
    }


def build_random_structure(
    slab_info,
    ads_configs,
    ads_atoms: dict[str, Atoms],
    counts: dict[str, int],
    rng: np.random.Generator,
) -> Atoms:
    """Place adsorbates on a fresh copy of the slab according to *counts*.

    Single canonical implementation of the place-each-adsorbate loop. Used by
    the GA's initial population, the ``_place_random`` and ``_spawn_gpr``
    spawners, and the ``galoop sample`` worker.

    Parameters
    ----------
    slab_info : SlabInfo
        Provides ``atoms``, ``zmin``, ``zmax``, ``n_slab_atoms``.
    ads_configs : iterable of AdsorbateConfig
        Used to look up per-species ``binding_index``.
    ads_atoms : dict[str, Atoms]
        Pre-loaded adsorbate templates from ``load_ads_template_dict``.
    counts : dict[str, int]
        How many of each species to place.
    rng : numpy Generator

    Raises
    ------
    Whatever ``place_adsorbate`` raises (placement failures bubble up so
    callers can decide whether to retry or fall back).
    """
    current = slab_info.atoms.copy()
    for sym, cnt in counts.items():
        ads_cfg = next(a for a in ads_configs if a.symbol == sym)
        for _ in range(cnt):
            current = place_adsorbate(
                slab=current,
                adsorbate=ads_atoms[sym],
                zmin=slab_info.zmin,
                zmax=slab_info.zmax,
                binding_index=ads_cfg.binding_index,
                rng=rng,
                n_slab_atoms=slab_info.n_slab_atoms,
                binding_z_offset=slab_info.binding_z_offset,
                clash_scale=slab_info.placement_clash_scale,
            )
    return current


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


def orient_upright(
    adsorbate: Atoms,
    binding_index: int = 0,
    normal: np.ndarray | None = None,
) -> Atoms:
    """
    Return a copy of *adsorbate* rotated so that the atom at *binding_index*
    points toward the surface (along -normal direction).

    For a single-atom adsorbate this is a no-op.  For multi-atom species
    the molecule is rotated so the binding atom sits at the lowest point
    along the surface normal, ready to adsorb.

    Parameters
    ----------
    adsorbate     : molecule to orient
    binding_index : index of the atom that binds to the surface (default 0)
    normal        : unit surface-normal vector; defaults to [0, 0, 1]

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

    n = np.array([0.0, 0.0, 1.0]) if normal is None else np.asarray(normal, dtype=float)

    ads = adsorbate.copy()
    pos = ads.get_positions()
    com = ads.get_center_of_mass()

    v = pos[binding_index] - com
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return ads      # binding atom coincides with COM — nothing to do

    rot = _rotation_to(v / norm, -n)
    ads.set_positions((pos - com) @ rot.T + com)
    return ads


# ---------------------------------------------------------------------------
# Adsorbate placement
# ---------------------------------------------------------------------------

def find_surface_sites(
    slab: Atoms,
    n_slab: int,
    bond_mult: float = 1.0,
    normal: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Identify atop, bridge, and hollow sites on the top surface layer.

    Returns a list of 3D Cartesian positions for candidate adsorption sites.
    Heights are measured along the surface normal (c-vector direction), not
    along the Cartesian z-axis, so this works for tilted/non-orthogonal cells.

    Parameters
    ----------
    slab      : slab Atoms (bare or with previously placed adsorbates)
    n_slab    : number of atoms belonging to the bare slab
    bond_mult : multiplier on covalent radii to detect nearest-neighbour pairs
    normal    : unit surface-normal vector; derived from slab cell if None
    """
    from ase.data import covalent_radii

    n = _surface_normal(slab) if normal is None else np.asarray(normal, dtype=float)

    pos = slab.get_positions()[:n_slab]
    heights = pos @ n
    h_top = heights.max()

    # Top-layer atoms: within 1.0 Å of the highest point along the normal
    top_mask = heights > h_top - 1.0
    top_indices = np.where(top_mask)[0]
    top_pos = pos[top_indices]

    sites: list[np.ndarray] = []

    # Atop sites — full 3D position of each top-layer atom
    for p in top_pos:
        sites.append(p.copy())

    # Find bonded pairs in the top layer (nearest-neighbour distance)
    nums = slab.get_atomic_numbers()
    for i in range(len(top_indices)):
        for j in range(i + 1, len(top_indices)):
            ii, jj = top_indices[i], top_indices[j]
            cutoff = bond_mult * (covalent_radii[nums[ii]] + covalent_radii[nums[jj]])
            d = slab.get_distance(ii, jj, mic=True)
            if d < cutoff * 1.5:
                # Bridge site: midpoint (Cartesian, PBC approximation)
                mid = 0.5 * (pos[ii] + pos[jj])
                sites.append(mid)

    # Hollow sites: centroids of bonded triangles
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
                    centroid = (pos[ii] + pos[jj] + pos[kk]) / 3
                    sites.append(centroid)

    return sites


def place_adsorbate(
    slab: Atoms,
    adsorbate: Atoms,
    zmin: float,
    zmax: float,
    binding_index: int = 0,
    rng: np.random.Generator | None = None,
    clash_scale: float = 0.7,
    max_attempts: int = 50,
    n_slab_atoms: int = 0,
    binding_z_offset: float = 2.0,
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

    # Surface normal: derived from cell, not assumed to be z.
    n = _surface_normal(slab)

    # Orient the binding atom toward the surface (along -n) before placement.
    adsorbate = orient_upright(adsorbate, binding_index=binding_index, normal=n)

    cell = slab.get_cell()
    n_slab_before = len(slab)

    # Use the bare slab atom count for site identification and height reference.
    n_slab_bare = n_slab_atoms if n_slab_atoms > 0 else n_slab_before

    sites = find_surface_sites(slab, n_slab_bare, normal=n)

    # Sort sites so the most isolated ones (farthest from existing adsorbates)
    # are tried first.  This spreads adsorbates across the surface.
    # In-plane distance: project out the normal component, then compute MIC distance.
    if sites and len(slab) > n_slab_bare:
        ads_pos = slab.get_positions()[n_slab_bare:]
        # In-plane positions: subtract normal component
        ads_inplane = ads_pos - np.outer(ads_pos @ n, n)

        def _min_inplane_dist(site_3d: np.ndarray) -> float:
            site_inplane = site_3d - (site_3d @ n) * n
            diffs = ads_inplane - site_inplane
            # Minimum image convention using full 3D cell
            frac = np.linalg.solve(cell.T, diffs.T).T
            frac -= np.round(frac)
            cart = frac @ cell
            return float(np.min(np.linalg.norm(cart, axis=1)))

        sites.sort(key=_min_inplane_dist, reverse=True)
        tier_size = max(3, len(sites) // 5)
        for start in range(0, len(sites), tier_size):
            end = min(start + tier_size, len(sites))
            tier = sites[start:end]
            rng.shuffle(tier)
            sites[start:end] = tier
    elif sites:
        rng.shuffle(sites)

    combined = slab.copy()  # initialise so the variable is always bound

    # Binding height: place COM *binding_z_offset* Å above the actual slab top
    # measured along the surface normal (not Cartesian z).
    slab_pos = slab.get_positions()[:n_slab_bare]
    slab_top_h = float((slab_pos @ n).max())

    for attempt in range(max_attempts):
        if attempt < len(sites):
            # Site-aware: strip the site's normal component and re-add the
            # desired height, so the adsorbate lands at the right height
            # regardless of the site atom's own height.
            site = sites[attempt]
            site_inplane = site - (site @ n) * n
            h = slab_top_h + binding_z_offset + rng.uniform(-0.2, 0.3)
            target = site_inplane + h * n
        else:
            # Fallback: random in-plane position + random height in window
            frac = rng.uniform(0, 1, size=3)
            frac[2] = 0.0
            inplane = frac @ cell  # a random in-plane point
            inplane = inplane - (inplane @ n) * n
            h = rng.uniform(zmin, zmax)
            target = inplane + h * n

        # Random rotation around the surface normal through the adsorbate COM
        angle = rng.uniform(0, 2 * np.pi)
        c_a, s_a = np.cos(angle), np.sin(angle)
        # Rodrigues' rotation matrix around n
        nx, ny, nz = n
        K = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]], dtype=float)
        rot = c_a * np.eye(3) + s_a * K + (1 - c_a) * np.outer(n, n)

        ads = adsorbate.copy()
        com = ads.get_center_of_mass()
        # Translate COM to target, then rotate around n through target
        ads.translate(target - com)
        new_com = ads.get_center_of_mass()
        ads.set_positions((ads.get_positions() - new_com) @ rot.T + new_com)

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
    n = _surface_normal_from_slab_info(slab_info)

    if z_threshold is None:
        z_threshold = slab_info.zmax + 3.0

    if len(atoms) <= slab_info.n_slab_atoms:
        return False

    ads_heights = atoms.get_positions()[slab_info.n_slab_atoms:] @ n
    return bool(np.any(ads_heights > z_threshold))


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
