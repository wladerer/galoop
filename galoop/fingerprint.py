"""
galoop/fingerprint.py

SOAP-based post-relaxation duplicate detection.

Duplicates are detected *after* relaxation only — no pre-submission gating.
This avoids false positives from comparing unrelaxed geometries.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
from ase import Atoms

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class StructInfo(NamedTuple):
    """Lightweight record for SOAP comparison."""
    id: str
    soap_vector: np.ndarray


# ---------------------------------------------------------------------------
# SOAP descriptor
# ---------------------------------------------------------------------------

def compute_soap(
    atoms: Atoms,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    species: list[str] | None = None,
) -> np.ndarray:
    """
    Compute a global averaged SOAP descriptor for *atoms*.

    Parameters
    ----------
    atoms : ASE Atoms
    r_cut : SOAP cutoff radius (Å)
    n_max : radial basis functions per species pair
    l_max : maximum angular momentum
    species : element symbols; auto-detected from *atoms* if ``None``

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

    soap = SOAP(
        species=species,
        r_cut=r_cut,
        n_max=n_max,
        l_max=l_max,
        average="inner",
        periodic=True,
    )
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
# Post-relaxation classification
# ---------------------------------------------------------------------------

def classify_postrelax(
    atoms: Atoms,
    existing_soaps: dict[str, np.ndarray],
    duplicate_threshold: float = 0.90,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    species: list[str] | None = None,
) -> tuple[str, str | None, np.ndarray]:
    """
    Classify a relaxed structure as ``"duplicate"`` or ``"unique"``.

    Parameters
    ----------
    atoms : relaxed geometry (CONTCAR)
    existing_soaps : ``{struct_id: soap_vector}`` cache
    duplicate_threshold : Tanimoto similarity cutoff
    r_cut, n_max, l_max, species : forwarded to :func:`compute_soap`

    Returns
    -------
    (label, closest_id, soap_vector)
        * label — ``"duplicate"`` or ``"unique"``
        * closest_id — id of the most similar existing structure, or ``None``
        * soap_vector — the computed vector (cache it for future comparisons)
    """
    soap_vec = compute_soap(
        atoms, r_cut=r_cut, n_max=n_max, l_max=l_max, species=species,
    )

    if not existing_soaps:
        return "unique", None, soap_vec

    best_sim = 0.0
    best_id: str | None = None

    for eid, evec in existing_soaps.items():
        sim = tanimoto_similarity(soap_vec, evec)
        if sim > best_sim:
            best_sim = sim
            best_id = eid

    if best_sim >= duplicate_threshold:
        return "duplicate", best_id, soap_vec

    return "unique", best_id, soap_vec
