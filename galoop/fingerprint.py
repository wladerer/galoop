"""
galoop/fingerprint.py

Post-relaxation duplicate detection using SOAP Tanimoto similarity.

Detection uses a multi-stage cascade:
  1. Composition gate          (O(1))
  2. Energy gate               (O(1))
  3. Distance-histogram cosine (cheap pre-filter)
  4. SOAP Tanimoto             (definitive)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class StructRecord:
    """Per-structure data needed for the duplicate-detection cascade."""
    id: str
    soap_vector: np.ndarray
    energy: float | None        # raw DFT energy; None if unavailable
    composition: str            # sorted formula: atoms.get_chemical_formula("metal")
    dist_hist: np.ndarray       # shape (n_bins,), L1-normalised
    prerelax_soap: np.ndarray | None = field(default=None)  # POSCAR SOAP for pre-relax comparison


# ---------------------------------------------------------------------------
# SOAP descriptor
# ---------------------------------------------------------------------------

def compute_soap(
    atoms: Atoms,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    species: list[str] | None = None,
    n_slab_atoms: int = 0,
) -> np.ndarray:
    """
    Compute an averaged SOAP descriptor.

    When *n_slab_atoms* > 0, SOAP is computed only at adsorbate atom
    centres (indices ``n_slab_atoms:``).  The slab atoms are still part
    of the environment so binding-site information is captured, but the
    descriptor is no longer dominated by bulk slab contributions.

    Parameters
    ----------
    atoms : ASE Atoms
    r_cut : SOAP cutoff radius (Å)
    n_max : radial basis functions per species pair
    l_max : maximum angular momentum
    species : element symbols; auto-detected from *atoms* if ``None``
    n_slab_atoms : number of leading atoms that belong to the bare slab.
        When > 0, SOAP is averaged over adsorbate positions only.

    Returns
    -------
    np.ndarray  — flattened SOAP vector

    Raises
    ------
    ImportError if dscribe is not installed.
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise ImportError(
            "SOAP requires dscribe.  Install with: pip install dscribe"
        ) from exc

    if species is None:
        species = sorted(set(atoms.get_chemical_symbols()))
    if not species:
        return np.zeros(100, dtype=float)

    n_ads = len(atoms) - n_slab_atoms
    if n_slab_atoms > 0 and n_ads <= 0:
        return np.zeros(100, dtype=float)

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="off" if n_slab_atoms > 0 else "inner",
        periodic=True,
    )

    if n_slab_atoms > 0:
        # Compute SOAP at adsorbate positions only, then average
        centers = list(range(n_slab_atoms, len(atoms)))
        per_atom = soap.create(atoms, centers=centers)
        return per_atom.mean(axis=0).flatten()
    else:
        return soap.create(atoms).flatten()


# ---------------------------------------------------------------------------
# Similarity metric
# ---------------------------------------------------------------------------

def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Tanimoto (generalised Jaccard) similarity in [0, 1].

    T(a, b) = (a · b) / (‖a‖² + ‖b‖² − a · b)

    Returns 0.0 on shape mismatch or degenerate input.
    """
    if vec1.shape != vec2.shape or len(vec1) == 0:
        return 0.0

    dot = float(np.dot(vec1, vec2))
    norm_sq_1 = float(np.dot(vec1, vec1))
    norm_sq_2 = float(np.dot(vec2, vec2))
    denom = norm_sq_1 + norm_sq_2 - dot

    if denom < 1e-12:
        return 1.0 if dot > 0.5 else 0.0

    return float(dot / denom)


# ---------------------------------------------------------------------------
# Gate helper functions
# ---------------------------------------------------------------------------

def _composition(atoms: Atoms) -> str:
    """Canonical sorted chemical formula for composition gating."""
    return atoms.get_chemical_formula(mode="metal")


def _dist_histogram(atoms: Atoms, n_bins: int = 50, r_max: float = 6.0) -> np.ndarray:
    """
    Bin the upper-triangle of the MIC pairwise distance matrix into n_bins
    over [0, r_max], L1-normalise. Returns zeros array if fewer than 2 atoms.
    """
    if len(atoms) < 2:
        return np.zeros(n_bins)
    dists = atoms.get_all_distances(mic=True)
    upper = dists[np.triu_indices(len(dists), k=1)]
    upper = upper[upper < r_max]
    if upper.size == 0:
        return np.zeros(n_bins)
    hist, _ = np.histogram(upper, bins=n_bins, range=(0.0, r_max))
    total = hist.sum()
    return hist.astype(float) / total if total > 0 else hist.astype(float)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Returns 0 on degenerate input."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _energy_gate_passes(
    e_new: float | None,
    e_existing: float | None,
    tol_pct: float,
) -> bool:
    """True if the two energies are within tol_pct % (relative to existing)."""
    if e_new is None or e_existing is None or tol_pct <= 0:
        return True          # can't gate without energies; let later stages decide
    import math
    if math.isnan(e_new) or math.isnan(e_existing):
        return True
    if abs(e_existing) < 1e-12:
        return True          # avoid divide-by-zero
    return abs(e_new - e_existing) / abs(e_existing) <= tol_pct / 100.0


# ---------------------------------------------------------------------------
# Graph-based chemical environment fingerprinting
# ---------------------------------------------------------------------------

def _grid_iterator(grid: tuple[int, int, int]):
    """Iterate over (x, y, z) integer offsets within ±grid in each dimension."""
    return itertools.product(*(range(-n, n + 1) for n in grid))


def _bond_symbol(atoms: Atoms, a1: int, a2: int) -> str:
    return "{}{}".format(*sorted((atoms[a1].symbol, atoms[a2].symbol)))


def _node_symbol(atom, offset: tuple) -> str:
    return "{}:{}[{},{},{}]".format(atom.symbol, atom.index, offset[0], offset[1], offset[2])


def _connected_component_subgraphs(G):
    import networkx as nx
    for c in nx.connected_components(G):
        yield G.subgraph(c)


def _compare_chem_envs(chem_envs1: list, chem_envs2: list) -> bool:
    """
    Return True if two sets of adsorbate chemical environments are isomorphic.

    Each environment is a networkx Graph.  Two sets match when every graph in
    chem_envs1 has a unique isomorphic partner in chem_envs2 (bond-type edges
    are matched; node identity is not required).
    """
    import networkx.algorithms.isomorphism as iso
    bond_match = iso.categorical_edge_match("bond", "")

    if len(chem_envs1) != len(chem_envs2):
        return False

    envs_copy = list(chem_envs2)
    for env1 in chem_envs1:
        for env2 in envs_copy:
            import networkx as nx
            if nx.is_isomorphic(env1, env2, edge_match=bond_match):
                envs_copy.remove(env2)
                break
        else:
            return False

    return len(envs_copy) == 0


def _unique_adsorbates(chem_envs: list) -> list:
    """
    Remove PBC-duplicate adsorbate graphs (same atom indices, same offsets).
    """
    import networkx as nx
    import networkx.algorithms.isomorphism as iso
    bond_match = iso.categorical_edge_match("bond", "")
    ads_match = iso.categorical_node_match(["index", "ads"], [-1, False])

    unique = []
    for env in chem_envs:
        for unique_env in unique:
            if nx.is_isomorphic(env, unique_env, edge_match=bond_match, node_match=ads_match):
                break
        else:
            unique.append(env)
    return unique


def build_chem_envs(
    atoms: Atoms,
    n_slab_atoms: int,
    radius: int = 2,
    grid: tuple[int, int, int] = (2, 2, 0),
) -> list | None:
    """
    Build adsorbate chemical-environment graphs for *atoms*.

    Returns a list of networkx Graphs (one per unique adsorbate), or ``None``
    if networkx is not installed or there are no adsorbate atoms.

    Parameters
    ----------
    atoms : slab + adsorbates ASE Atoms
    n_slab_atoms : number of leading atoms that belong to the bare slab
    radius : environment radius (graph hops, doubled internally for dist weighting)
    grid : PBC repetitions (X, Y, Z) to avoid loops across periodic boundaries
    """
    try:
        import networkx as nx
        from ase.neighborlist import NeighborList, natural_cutoffs
    except ImportError:
        return None

    adsorbate_atoms = list(range(n_slab_atoms, len(atoms)))
    if not adsorbate_atoms:
        return None

    distances = atoms.get_all_distances(mic=True)
    cutoffs = natural_cutoffs(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    full = nx.Graph()

    # Add all atom nodes for each grid repetition
    for index, atom in enumerate(atoms):
        for x, y, z in _grid_iterator(grid):
            full.add_node(
                _node_symbol(atom, (x, y, z)),
                index=index,
                ads=(index in adsorbate_atoms),
                central_ads=False,
            )

    # Add all bond edges
    for index, atom in enumerate(atoms):
        for x, y, z in _grid_iterator(grid):
            neighbors, offsets = nl.get_neighbors(index)
            for neighbor, offset in zip(neighbors, offsets):
                ox, oy, oz = int(offset[0]), int(offset[1]), int(offset[2])
                if not (-grid[0] <= ox + x <= grid[0]):
                    continue
                if not (-grid[1] <= oy + y <= grid[1]):
                    continue
                if not (-grid[2] <= oz + z <= grid[2]):
                    continue
                # Skip long surface–adsorbate bonds (> 2.5 Å)
                if distances[index][neighbor] > 2.5 and (
                    bool(index in adsorbate_atoms) ^ bool(neighbor in adsorbate_atoms)
                ):
                    continue
                dist_weight = 2 - (1 if index in adsorbate_atoms else 0) - (
                    1 if neighbor in adsorbate_atoms else 0
                )
                ads_only = 0 if (index in adsorbate_atoms and neighbor in adsorbate_atoms) else 2
                full.add_edge(
                    _node_symbol(atom, (x, y, z)),
                    _node_symbol(atoms[neighbor], (x + ox, y + oy, z + oz)),
                    bond=_bond_symbol(atoms, index, neighbor),
                    dist=dist_weight,
                    ads_only=ads_only,
                )

    # Extract one graph per adsorbate (central atoms only, then expand to environment)
    ads_nodes = [_node_symbol(atoms[i], (0, 0, 0)) for i in adsorbate_atoms]
    ads_subgraph = nx.subgraph(full, ads_nodes)

    chem_envs = []
    for component in _connected_component_subgraphs(ads_subgraph):
        initial = list(component.nodes())[0]
        env = nx.ego_graph(full, initial, radius=(radius * 2) + 1, distance="dist")
        env = nx.Graph(nx.subgraph(full, list(env.nodes())))
        for node in component.nodes():
            env.add_node(node, central_ads=True)
        chem_envs.append(env)

    chem_envs = _unique_adsorbates(chem_envs)
    chem_envs.sort(key=lambda x: len(x.edges()))
    return chem_envs


# ---------------------------------------------------------------------------
# Post-relaxation classification
# ---------------------------------------------------------------------------

def classify_postrelax(
    atoms: Atoms,
    energy: float | None,
    struct_cache: dict[str, StructRecord],
    duplicate_threshold: float = 0.95,
    energy_tol_pct: float = 5.0,
    dist_hist_threshold: float = 0.95,
    dist_hist_bins: int = 50,
    dist_hist_rmax: float = 6.0,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    species: list[str] | None = None,
    n_slab_atoms: int = 0,
) -> tuple[str, str | None, np.ndarray]:
    """
    Classify a relaxed structure as ``"duplicate"`` or ``"unique"``.

    Uses a multi-stage cascade (cheapest first):
      1. Composition gate  (O(1))
      2. Energy gate       (O(1))
      3. Distance-histogram cosine pre-filter
      4. SOAP Tanimoto     (definitive)

    Parameters
    ----------
    atoms : relaxed geometry (CONTCAR)
    energy : raw DFT energy; None if unavailable
    struct_cache : ``{struct_id: StructRecord}`` cache
    duplicate_threshold : SOAP Tanimoto cutoff for duplicate classification
    energy_tol_pct : max % energy difference relative to existing (0 = disabled)
    dist_hist_threshold : min cosine similarity for distance histogram pre-filter
    dist_hist_bins : histogram bin count
    dist_hist_rmax : max distance for histogram (Å)
    r_cut, n_max, l_max, species : forwarded to :func:`compute_soap`
    n_slab_atoms : number of leading atoms belonging to the bare slab.
        When > 0, SOAP is averaged over adsorbate positions only.

    Returns
    -------
    (label, closest_id, soap_vector)
        * label — ``"duplicate"`` or ``"unique"``
        * closest_id — id of the most similar existing structure, or ``None``
        * soap_vector — the computed SOAP vector (cache it for future comparisons)
    """
    new_comp = _composition(atoms)
    new_soap = compute_soap(atoms, r_cut, n_max, l_max, species, n_slab_atoms=n_slab_atoms)

    if not struct_cache:
        return "unique", None, new_soap

    new_dhist = _dist_histogram(atoms, dist_hist_bins, dist_hist_rmax)
    best_sim, best_id = 0.0, None

    for rec in struct_cache.values():
        # Gate 1: composition
        if rec.composition != new_comp:
            continue
        # Gate 2: energy
        if not _energy_gate_passes(energy, rec.energy, energy_tol_pct):
            continue
        # Gate 3: distance histogram cosine (cheap pre-filter)
        if _cosine(new_dhist, rec.dist_hist) < dist_hist_threshold:
            continue
        # Gate 4: SOAP Tanimoto (definitive)
        sim = tanimoto_similarity(new_soap, rec.soap_vector)
        if sim > best_sim:
            best_sim, best_id = sim, rec.id

    label = "duplicate" if best_sim >= duplicate_threshold else "unique"
    return label, best_id, new_soap
