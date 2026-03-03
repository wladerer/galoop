"""
gocia/fingerprint.py

SOAP-only post-relaxation duplicate detection. No pre-submission gating.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from ase import Atoms


class StructInfo(NamedTuple):
    """Lightweight record for SOAP comparison."""
    id: str
    soap_vector: np.ndarray


def compute_soap(
    atoms: Atoms,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
    species: list[str] | None = None,
) -> np.ndarray:
    """
    Compute global averaged SOAP descriptor.

    Parameters
    ----------
    atoms : ASE Atoms
    r_cut : cutoff radius (Å)
    n_max : radial basis functions per species
    l_max : max angular momentum
    species : list of element symbols; auto-detected if None

    Returns
    -------
    np.ndarray
        SOAP descriptor vector (flattened).

    Raises
    ------
    ImportError
        If dscribe is not installed.
    """
    try:
        from dscribe.descriptors import SOAP
    except ImportError as exc:
        raise ImportError(
            "SOAP requires dscribe. Install with: pip install dscribe"
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


def tanimoto_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute Tanimoto similarity between two SOAP vectors in [0, 1].

    Higher = more similar. Returns 0 on error (mismatched shapes, etc.).
    """
    if vec1.shape != vec2.shape:
        return 0.0
    if len(vec1) == 0:
        return 0.0

    dot = float(np.dot(vec1, vec2))
    norm1 = float(np.linalg.norm(vec1) ** 2)
    norm2 = float(np.linalg.norm(vec2) ** 2)

    denom = norm1 + norm2 - dot
    if denom < 1e-12:
        return 1.0 if dot > 0.5 else 0.0

    return float(dot / denom)


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
    Classify a relaxed structure as duplicate or unique based on SOAP.

    Parameters
    ----------
    atoms : relaxed ASE Atoms (CONTCAR)
    existing_soaps : dict of {struct_id: soap_vector}
    duplicate_threshold : Tanimoto similarity above which is a duplicate
    r_cut, n_max, l_max, species : SOAP parameters

    Returns
    -------
    (label, closest_id, soap_vector)
        label : "duplicate" or "unique"
        closest_id : id of the most similar structure, or None
        soap_vector : the computed SOAP vector for caching
    """
    soap_vec = compute_soap(atoms, r_cut=r_cut, n_max=n_max, l_max=l_max, species=species)

    if not existing_soaps:
        return "unique", None, soap_vec

    best_sim = 0.0
    best_id = None

    for existing_id, existing_soap in existing_soaps.items():
        sim = tanimoto_similarity(soap_vec, existing_soap)
        if sim > best_sim:
            best_sim = sim
            best_id = existing_id

    if best_sim >= duplicate_threshold:
        return "duplicate", best_id, soap_vec

    return "unique", None, soap_vec
