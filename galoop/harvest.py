"""
galoop/harvest.py

Post-relax classification: turn a finished Parsl future into one of the
terminal states (converged / duplicate / failed / desorbed / unbound) and
update the SOAP cache when a new unique structure lands.

Split out of galoop/galoop.py during the Phase 2 refactor; the loop in
loop.py drives the harvest pipeline by calling the helpers here.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms

from galoop.fingerprint import (
    StructRecord,
    _composition,
    _dist_histogram,
    classify_postrelax,
    compute_soap,
    tanimoto_similarity,
)
from galoop.individual import STATUS
from galoop.science.surface import read_atoms

if TYPE_CHECKING:  # avoid circular import at runtime
    from galoop.store import GaloopStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main classification entry point
# ---------------------------------------------------------------------------

def handle_converged(
    ind,
    struct_dir: Path,
    store: GaloopStore,
    struct_cache: dict[str, StructRecord],
    chem_pots: Mapping[str, float],
    config,
    total_evals: int,
    best_gce: float,
    n_slab_atoms: int = 0,
):
    """Classify a relaxed structure and compute GCE if unique.

    Returns the (possibly updated) Individual, the new total_evals counter,
    and the new best_gce. The Individual is also persisted to the store.
    """
    from galoop.science.energy import grand_canonical_energy

    contcar = struct_dir / "CONTCAR"
    if not contcar.exists():
        log.warning("  %s: CONTCAR missing after relaxation", ind.id)
        ind = ind.with_status(STATUS.FAILED)
        store.update(ind)
        return ind, total_evals, best_gce

    try:
        atoms = read_atoms(contcar, format="vasp")
        raw_e = read_final_energy(struct_dir)

        # Sanity check: reject structures with atom-atom overlap
        if has_atom_overlap(atoms, min_dist=0.5):
            log.warning("  %s: atom overlap detected (d < 0.5 Å) — failed", ind.id)
            ind = ind.with_status(STATUS.FAILED)
            ind.extra_data = {**ind.extra_data, "fail_reason": "atom_overlap"}
            store.update(ind)
            return ind, total_evals, best_gce

        label, dup_id, soap_vec = classify_postrelax(
            atoms,
            energy=raw_e,
            struct_cache=struct_cache,
            duplicate_threshold=config.fingerprint.duplicate_threshold,
            energy_tol_pct=config.fingerprint.energy_tol_pct,
            dist_hist_threshold=config.fingerprint.dist_hist_threshold,
            dist_hist_bins=config.fingerprint.dist_hist_bins,
            dist_hist_rmax=config.fingerprint.r_cut,
            r_cut=config.fingerprint.r_cut,
            n_max=config.fingerprint.n_max,
            l_max=config.fingerprint.l_max,
            n_slab_atoms=n_slab_atoms,
        )

        if label == "duplicate":
            sim = best_similarity(soap_vec, struct_cache)
            log.info("  %s: duplicate of %s  (Tanimoto=%.3f)", ind.id, dup_id, sim)
            ind = ind.mark_duplicate()
            ind.extra_data = {**ind.extra_data, "dup_of": dup_id, "tanimoto": float(sim)}
            store.update(ind)
        else:
            sim = best_similarity(soap_vec, struct_cache) if struct_cache else 0.0
            log.debug("  %s: unique  (best Tanimoto=%.3f)", ind.id, sim)

            gce = grand_canonical_energy(
                raw_energy=raw_e,
                adsorbate_counts=ind.extra_data.get("adsorbate_counts", {}),
                chemical_potentials=chem_pots,
                potential=config.conditions.potential,
                pH=config.conditions.pH,
                temperature=config.conditions.temperature,
                pressure=config.conditions.pressure,
            )
            ind = ind.with_energy(raw=raw_e, grand_canonical=gce)
            store.update(ind)

            # Compute pre-relax SOAP from POSCAR for pre-relax duplicate comparison
            prerelax_soap = None
            poscar = struct_dir / "POSCAR"
            if poscar.exists():
                try:
                    pre_atoms = read_atoms(poscar, format="vasp")
                    prerelax_soap = compute_soap(
                        pre_atoms,
                        r_cut=config.fingerprint.r_cut,
                        n_max=config.fingerprint.n_max,
                        l_max=config.fingerprint.l_max,
                        n_slab_atoms=n_slab_atoms,
                    )
                except Exception as exc:
                    log.debug("  %s: pre-relax SOAP unavailable (%s)", ind.id, exc)

            struct_cache[ind.id] = StructRecord(
                id=ind.id,
                soap_vector=soap_vec,
                energy=raw_e,
                composition=_composition(atoms),
                dist_hist=_dist_histogram(
                    atoms,
                    n_bins=config.fingerprint.dist_hist_bins,
                    r_max=config.fingerprint.r_cut,
                ),
                prerelax_soap=prerelax_soap,
            )

            if gce < best_gce - 1e-6:
                best_gce = gce

            total_evals += 1
            log.info("  %s: converged  G=%.4f eV", ind.id, gce)

    except Exception as exc:
        # Log full traceback so we can spot systemic bugs (the MACE shape
        # mismatch hunt would have been hours shorter with this in place).
        log.exception("  %s: post-relax evaluation failed (%s)", ind.id, exc)
        ind = ind.with_status(STATUS.FAILED)
        ind.extra_data = {**ind.extra_data, "fail_reason": f"harvest_exception: {exc}"}
        store.update(ind)

    return ind, total_evals, best_gce


# ---------------------------------------------------------------------------
# SOAP cache rebuild (resume path)
# ---------------------------------------------------------------------------

def rebuild_struct_cache(
    store: GaloopStore,
    struct_cache: dict[str, StructRecord],
    config,
    n_slab_atoms: int = 0,
) -> None:
    """Rebuild the in-memory SOAP cache by re-reading every converged CONTCAR.

    Run once at startup so duplicate detection has the full history.
    """
    for ind in store.all_converged_unique():
        struct_dir = store.individual_dir(ind.id)
        contcar = struct_dir / "CONTCAR"
        if not contcar.exists():
            continue
        try:
            atoms = read_atoms(contcar, format="vasp")
            soap_vec = compute_soap(
                atoms,
                r_cut=config.fingerprint.r_cut,
                n_max=config.fingerprint.n_max,
                l_max=config.fingerprint.l_max,
                n_slab_atoms=n_slab_atoms,
            )
            prerelax_soap = None
            poscar = struct_dir / "POSCAR"
            if poscar.exists():
                try:
                    pre_atoms = read_atoms(poscar, format="vasp")
                    prerelax_soap = compute_soap(
                        pre_atoms,
                        r_cut=config.fingerprint.r_cut,
                        n_max=config.fingerprint.n_max,
                        l_max=config.fingerprint.l_max,
                        n_slab_atoms=n_slab_atoms,
                    )
                except Exception as exc:
                    log.debug("  %s: pre-relax SOAP unavailable (%s)", ind.id, exc)

            struct_cache[ind.id] = StructRecord(
                id=ind.id,
                soap_vector=soap_vec,
                energy=ind.raw_energy,
                composition=_composition(atoms),
                dist_hist=_dist_histogram(
                    atoms,
                    n_bins=config.fingerprint.dist_hist_bins,
                    r_max=config.fingerprint.r_cut,
                ),
                prerelax_soap=prerelax_soap,
            )
        except Exception as exc:
            log.debug("Could not rebuild cache for %s: %s", ind.id, exc)


# ---------------------------------------------------------------------------
# Pre-relax duplicate gate (used by spawn.py before submitting a child)
# ---------------------------------------------------------------------------

def is_prerelax_duplicate(
    atoms: Atoms,
    struct_cache: dict[str, StructRecord],
    n_slab_atoms: int,
    threshold: float = 0.95,
    r_cut: float = 6.0,
    n_max: int = 8,
    l_max: int = 6,
) -> bool:
    """Check if *atoms* is similar to any cached converged structure.

    Compares the candidate's SOAP against the cached pre-relax SOAP
    (from POSCAR) for like-with-like comparison. Falls back to the
    post-relax SOAP if pre-relax is unavailable.
    """
    new_comp = _composition(atoms)
    new_soap = compute_soap(atoms, r_cut=r_cut, n_max=n_max, l_max=l_max,
                            n_slab_atoms=n_slab_atoms)

    for rec in struct_cache.values():
        if rec.composition != new_comp:
            continue
        ref_soap = rec.prerelax_soap if rec.prerelax_soap is not None else rec.soap_vector
        if tanimoto_similarity(new_soap, ref_soap) >= threshold:
            return True

    return False


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def read_final_energy(struct_dir: Path | str) -> float:
    """Read the final energy from a struct dir's FINAL_ENERGY file."""
    ef = Path(struct_dir) / "FINAL_ENERGY"
    if ef.exists():
        try:
            return float(ef.read_text().strip())
        except ValueError:
            pass
    return float("nan")


def has_atom_overlap(atoms: Atoms, min_dist: float = 0.5) -> bool:
    """True if any two atoms are closer than *min_dist* Å."""
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 999.0)
    return float(dists.min()) < min_dist


def best_similarity(soap_vec: np.ndarray, struct_cache: dict) -> float:
    """Return the highest Tanimoto similarity to any cached structure."""
    if not struct_cache:
        return 0.0
    return max(tanimoto_similarity(soap_vec, rec.soap_vector)
               for rec in struct_cache.values())
