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

import logging
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
from ase import Atoms

log = logging.getLogger(__name__)


class SoapKwargs(TypedDict):
    """Keyword arguments for ``compute_soap`` that callers commonly pass as
    a single dict (so all sample / harvest sites stay in sync)."""
    r_cut: float
    n_max: int
    l_max: int
    n_slab_atoms: int


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
