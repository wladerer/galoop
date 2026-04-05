"""
galoop/galoop.py

Steady-state GA loop backed by a signac workspace.

Job lifecycle
-------------
  pending   — spawned, POSCAR written to job dir
  submitted — row picked it up, submitted to cluster
  relaxed   — _relax command finished (CONTCAR + FINAL_ENERGY written)
  converged — unique; GCE computed
  duplicate — dup of an existing converged structure
  failed    — pipeline error
  desorbed  — adsorbate left the surface

Row handles relax submission.  This loop handles:
  1. Evaluating relaxed → converged / duplicate
  2. Spawning offspring (creates new pending jobs)
  3. Convergence check
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from galoop.individual import Individual, STATUS, OPERATOR
from galoop.project import GaloopProject, STATUS_RELAXED
from galoop.fingerprint import (
    classify_postrelax, compute_soap, tanimoto_similarity,
    StructRecord, _dist_histogram, _composition, build_chem_envs,
)
from galoop.science.reproduce import (
    splice, merge, mutate_add, mutate_remove,
    mutate_displace, mutate_rattle_slab, mutate_translate,
)

log = logging.getLogger(__name__)

POLL_INTERVAL = 15  # seconds between loop iterations


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run(
    config,
    run_dir: Path,
    slab_info,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Steady-state GA loop.

    Parameters
    ----------
    config    : validated GaloopConfig
    run_dir   : root run directory (contains galoop.yaml; workspace/ created here)
    slab_info : SlabInfo from load_slab()
    rng       : NumPy random generator
    """
    rng = rng or np.random.default_rng()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Initialising signac workspace in %s/workspace", run_dir)
    project = GaloopProject(run_dir)

    chem_pots = {a.symbol: a.chemical_potential for a in config.adsorbates}

    # Build initial population on first run
    if not any(True for _ in project._project):
        log.info("Building initial population …")
        _build_initial_population(config, slab_info, project, rng)

    struct_cache: dict[str, StructRecord] = {}
    total_evals = len(project.get_by_status(STATUS.CONVERGED))

    log.info("Resuming with %d converged structures.  Rebuilding struct cache …",
             total_evals)
    _rebuild_struct_cache(project, struct_cache, config, slab_info.n_slab_atoms)

    best_gce = float("inf")
    stall_count = 0
    # Initialise best_gce from existing converged pool
    for ind in project.get_by_status(STATUS.CONVERGED):
        if ind.grand_canonical_energy is not None and ind.grand_canonical_energy < best_gce:
            best_gce = ind.grand_canonical_energy

    log.info(
        "Start: evals=%d  best=%.4f eV  "
        "Run `row run` in this directory to submit relax jobs.",
        total_evals, best_gce,
    )

    # pair_counts tracks (frozenset of parent IDs) → uses this cycle
    # Reset each poll cycle to prevent over-mating between the same parents
    pair_counts: dict[frozenset, int] = {}

    # ── main loop ──────────────────────────────────────────────────────────
    while True:
        if _stop_requested(run_dir):
            log.info("Stop file detected — exiting.")
            break

        # Evaluate any structures whose pipeline has finished
        relaxed = project.get_by_status(STATUS_RELAXED)
        if relaxed:
            prev_best = best_gce
            for ind in relaxed:
                job = project.get_job_by_id(ind.id)
                if job is None:
                    continue
                struct_dir = Path(job.path)
                ind, total_evals, best_gce = _handle_converged(
                    ind, struct_dir, project, struct_cache, chem_pots,
                    config, total_evals, best_gce,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )
            # Stall increments once per poll cycle, not once per structure
            if total_evals > 0:
                if best_gce < prev_best - 1e-6:
                    stall_count = 0
                else:
                    stall_count += 1

        # Reset pair usage tracking for this spawn batch
        pair_counts = {}

        # Spawn offspring to keep the pending pool topped up
        _fill_workers(project, config, slab_info, rng, total_evals, pair_counts)

        # Convergence check
        n_active = (
            len(project.get_by_status(STATUS.PENDING))
            + len(project.get_by_status(STATUS.SUBMITTED))
            + len(project.get_by_status(STATUS_RELAXED))
        )
        if not n_active and _should_stop(total_evals, stall_count, config):
            log.info("Convergence criteria met.")
            break

        log.info(
            "Evals=%d  Best=%.4f eV  Stall=%d/%d  Active=%d",
            total_evals, best_gce, stall_count, config.ga.max_stall, n_active,
        )
        time.sleep(POLL_INTERVAL)

    log.info("Run complete.  Total evaluations: %d", total_evals)


# ═══════════════════════════════════════════════════════════════════════════
# Post-relax evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _handle_converged(
    ind, struct_dir: Path, project: GaloopProject,
    struct_cache, chem_pots, config,
    total_evals, best_gce,
    n_slab_atoms: int = 0,
):
    """Classify a relaxed structure and compute GCE if unique."""
    from galoop.science.energy import grand_canonical_energy

    contcar = struct_dir / "CONTCAR"
    if not contcar.exists():
        log.warning("  %s: CONTCAR missing after relaxation", ind.id)
        ind = ind.with_status(STATUS.FAILED)
        project.update(ind)
        return ind, total_evals, best_gce

    try:
        atoms = read(str(contcar), format="vasp")
        raw_e = _read_energy(struct_dir)

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
            sim = _best_similarity(soap_vec, struct_cache)
            log.info("  %s: duplicate of %s  (Tanimoto=%.3f)", ind.id, dup_id, sim)
            ind = ind.mark_duplicate()
            ind.extra_data = {**ind.extra_data, "dup_of": dup_id, "tanimoto": float(sim)}
            project.update(ind)
        else:
            sim = _best_similarity(soap_vec, struct_cache) if struct_cache else 0.0
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
            project.update(ind)

            record = StructRecord(
                id=ind.id,
                soap_vector=soap_vec,
                energy=raw_e,
                composition=_composition(atoms),
                dist_hist=_dist_histogram(
                    atoms,
                    n_bins=config.fingerprint.dist_hist_bins,
                    r_max=config.fingerprint.r_cut,
                ),
                chem_envs=build_chem_envs(atoms, n_slab_atoms) if n_slab_atoms > 0 else None,
            )
            struct_cache[ind.id] = record

            if gce < best_gce - 1e-6:
                best_gce = gce

            total_evals += 1
            log.info("  %s: converged  G=%.4f eV", ind.id, gce)

    except Exception as exc:
        log.warning("  %s: post-relax evaluation failed (%s)", ind.id, exc)

    return ind, total_evals, best_gce


# ═══════════════════════════════════════════════════════════════════════════
# Initial population
# ═══════════════════════════════════════════════════════════════════════════

def _build_initial_population(config, slab_info, project: GaloopProject, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in config.adsorbates
    }

    for i in range(config.ga.population_size):
        counts = _random_stoichiometry(
            config.adsorbates, rng,
            config.ga.min_adsorbates, config.ga.max_adsorbates,
        )

        current = slab_info.atoms.copy()
        try:
            for sym, cnt in counts.items():
                ads_cfg = next(a for a in config.adsorbates if a.symbol == sym)
                for _ in range(cnt):
                    current = place_adsorbate(
                        slab=current,
                        adsorbate=ads_atoms[sym],
                        zmin=slab_info.zmin,
                        zmax=slab_info.zmax,
                        n_orientations=ads_cfg.n_orientations,
                        binding_index=ads_cfg.binding_index,
                        rng=rng,
                    )
        except Exception as exc:
            log.warning("structure_%05d: placement failed (%s) — falling back", i, exc)
            first = next(iter(ads_atoms))
            counts = {first: 1}
            current = slab_info.atoms.copy()
            current = place_adsorbate(
                current, ads_atoms[first],
                slab_info.zmin, slab_info.zmax, rng=rng,
            )

        ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
        job = project.create_job(ind)
        poscar = Path(job.path) / "POSCAR"
        write(str(poscar), current, format="vasp")
        job.doc["geometry_path"] = str(poscar)
        log.debug("initial structure %05d: %s", i, dict(counts))

    log.info("Initial population: %d structures", config.ga.population_size)


# ═══════════════════════════════════════════════════════════════════════════
# Offspring spawning
# ═══════════════════════════════════════════════════════════════════════════

def _fill_workers(project: GaloopProject, config, slab_info, rng, total_evals,
                  pair_counts: dict | None = None):
    """Spawn offspring until we have enough pending jobs."""
    n_pending = len(project.get_by_status(STATUS.PENDING))
    limit = config.scheduler.nworkers

    while n_pending < limit:
        if total_evals >= config.ga.max_structures:
            break
        new_ind = _spawn_one(project, config, slab_info, rng, pair_counts)
        if new_ind is None:
            break
        n_pending += 1


def _spawn_one(project: GaloopProject, config, slab_info, rng,
               pair_counts: dict | None = None):
    selectable = project.selectable_pool()

    if not selectable:
        return _place_random(project, config, slab_info, rng)

    try:
        op = _sample_operator(rng, config)
        n_parents = 2 if op in (OPERATOR.SPLICE, OPERATOR.MERGE) else 1

        if len(selectable) < n_parents:
            return _place_random(project, config, slab_info, rng)

        energies = np.array([p.grand_canonical_energy or 0.0 for p in selectable])
        shifted = energies - energies.min()
        weights = np.exp(-shifted / config.ga.boltzmann_temperature)
        weights /= weights.sum()
        indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
        parents = [selectable[i] for i in indices]

        # Over-mating penalty: if this pair was already used this cycle,
        # try one re-draw before proceeding
        if n_parents == 2 and pair_counts is not None:
            pair_key = frozenset(p.id for p in parents)
            if pair_counts.get(pair_key, 0) >= 1:
                indices2 = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
                parents = [selectable[i] for i in indices2]

        parent_atoms = []
        for p in parents:
            parent_job = project.get_job_by_id(p.id)
            if parent_job is None:
                return _place_random(project, config, slab_info, rng)
            contcar = Path(parent_job.path) / "CONTCAR"
            if not contcar.exists():
                return _place_random(project, config, slab_info, rng)
            parent_atoms.append(read(str(contcar), format="vasp"))

        if op == OPERATOR.SPLICE:
            from galoop.science.surface import check_clash
            child, _ = splice(parent_atoms[0], parent_atoms[1],
                               slab_info.n_slab_atoms, rng=rng)
            if check_clash(child, n_slab=slab_info.n_slab_atoms, scale=0.7):
                return _place_random(project, config, slab_info, rng)
        elif op == OPERATOR.MERGE:
            child = merge(parent_atoms[0], parent_atoms[1],
                          slab_info.n_slab_atoms, rng=rng)
        elif op == OPERATOR.MUTATE_ADD:
            from galoop.science.surface import load_adsorbate, place_adsorbate

            parent_counts = parents[0].extra_data.get("adsorbate_counts", {})
            if sum(parent_counts.values()) >= config.ga.max_adsorbates:
                return _place_random(project, config, slab_info, rng)

            addable = [
                a.symbol for a in config.adsorbates
                if parent_counts.get(a.symbol, 0) < a.max_count
            ]
            if not addable:
                return _place_random(project, config, slab_info, rng)
            sym = str(rng.choice(addable))
            ads_cfg = next(a for a in config.adsorbates if a.symbol == sym)
            ads_mol = load_adsorbate(
                symbol=sym,
                geometry=getattr(ads_cfg, "geometry", None),
                coordinates=getattr(ads_cfg, "coordinates", None),
            )
            child = place_adsorbate(
                parent_atoms[0], ads_mol,
                slab_info.zmin, slab_info.zmax,
                n_orientations=ads_cfg.n_orientations,
                binding_index=ads_cfg.binding_index,
                rng=rng,
            )
        elif op == OPERATOR.MUTATE_REMOVE:
            result = mutate_remove(parent_atoms[0], slab_info.n_slab_atoms, rng=rng)
            if result is None:
                return _place_random(project, config, slab_info, rng)
            child = result
        elif op == OPERATOR.MUTATE_DISPLACE:
            result = mutate_displace(parent_atoms[0], slab_info.n_slab_atoms,
                                     displacement=0.5, rng=rng)
            if result is None:
                return _place_random(project, config, slab_info, rng)
            child = result
        elif op == OPERATOR.MUTATE_RATTLE_SLAB:
            child = mutate_rattle_slab(parent_atoms[0], slab_info.n_slab_atoms,
                                       amplitude=config.ga.rattle_amplitude, rng=rng)
        elif op == OPERATOR.MUTATE_TRANSLATE:
            result = mutate_translate(parent_atoms[0], slab_info.n_slab_atoms,
                                      displacement=0.8, rng=rng)
            if result is None:
                return _place_random(project, config, slab_info, rng)
            child = result
        else:
            return _place_random(project, config, slab_info, rng)

        # Record pair usage for over-mating penalty
        if n_parents == 2 and pair_counts is not None:
            pair_key = frozenset(p.id for p in parents)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        if op in (OPERATOR.SPLICE, OPERATOR.MERGE):
            trial_counts = _infer_adsorbate_counts(
                child.get_chemical_symbols()[slab_info.n_slab_atoms:],
                config.adsorbates,
            )
            total = sum(trial_counts.values())
            if total > config.ga.max_adsorbates or total < config.ga.min_adsorbates:
                return _place_random(project, config, slab_info, rng)

        if op in (OPERATOR.MUTATE_DISPLACE, OPERATOR.MUTATE_RATTLE_SLAB, OPERATOR.MUTATE_TRANSLATE):
            ads_counts = parents[0].extra_data.get("adsorbate_counts", {})
        else:
            ads_counts = _infer_adsorbate_counts(
                child.get_chemical_symbols()[slab_info.n_slab_atoms:],
                config.adsorbates,
            )
        ind = Individual.from_parents(
            parents=parents,
            operator=op,
            extra_data={"adsorbate_counts": ads_counts},
        )
        job = project.create_job(ind)
        poscar = Path(job.path) / "POSCAR"
        write(str(poscar), child, format="vasp")
        job.doc["geometry_path"] = str(poscar)
        log.debug("Spawned %s via %s", ind.id, op)
        return ind

    except Exception as exc:
        log.debug("Operator failed: %s — falling back to random", exc)
        return _place_random(project, config, slab_info, rng)


def _place_random(project: GaloopProject, config, slab_info, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in config.adsorbates
    }
    counts = _random_stoichiometry(
        config.adsorbates, rng,
        config.ga.min_adsorbates, config.ga.max_adsorbates,
    )

    current = slab_info.atoms.copy()
    for sym, cnt in counts.items():
        ads_cfg = next(a for a in config.adsorbates if a.symbol == sym)
        for _ in range(cnt):
            current = place_adsorbate(
                current, ads_atoms[sym],
                slab_info.zmin, slab_info.zmax,
                n_orientations=ads_cfg.n_orientations,
                binding_index=ads_cfg.binding_index,
                rng=rng,
            )

    ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
    job = project.create_job(ind)
    poscar = Path(job.path) / "POSCAR"
    write(str(poscar), current, format="vasp")
    job.doc["geometry_path"] = str(poscar)
    log.debug("Spawned %s via random", ind.id)
    return ind


# ═══════════════════════════════════════════════════════════════════════════
# Struct cache helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rebuild_struct_cache(project: GaloopProject, struct_cache, config,
                          n_slab_atoms: int = 0):
    for job in project.all_converged_unique_jobs():
        contcar = Path(job.path) / "CONTCAR"
        if not contcar.exists():
            continue
        try:
            atoms = read(str(contcar), format="vasp")
            soap_vec = compute_soap(
                atoms,
                r_cut=config.fingerprint.r_cut,
                n_max=config.fingerprint.n_max,
                l_max=config.fingerprint.l_max,
            )
            ind_id = job.statepoint["id"]
            struct_cache[ind_id] = StructRecord(
                id=ind_id,
                soap_vector=soap_vec,
                energy=job.doc.get("raw_energy"),
                composition=_composition(atoms),
                dist_hist=_dist_histogram(
                    atoms,
                    n_bins=config.fingerprint.dist_hist_bins,
                    r_max=config.fingerprint.r_cut,
                ),
                chem_envs=build_chem_envs(atoms, n_slab_atoms) if n_slab_atoms > 0 else None,
            )
        except Exception as exc:
            log.debug("Could not rebuild cache for %s: %s",
                      job.statepoint.get("id"), exc)


# ═══════════════════════════════════════════════════════════════════════════
# Small helpers
# ═══════════════════════════════════════════════════════════════════════════

def _random_stoichiometry(ads_configs, rng, min_total, max_total):
    counts = {
        a.symbol: int(rng.integers(a.min_count, a.max_count + 1))
        for a in ads_configs
    }
    while sum(counts.values()) > max_total:
        shrinkable = [
            s for s in counts
            if counts[s] > next(a.min_count for a in ads_configs if a.symbol == s)
        ]
        if not shrinkable:
            break
        counts[str(rng.choice(shrinkable))] -= 1
    while sum(counts.values()) < min_total:
        growable = [
            s for s in counts
            if counts[s] < next(a.max_count for a in ads_configs if a.symbol == s)
        ]
        if not growable:
            break
        counts[str(rng.choice(growable))] += 1
    return counts


def _sample_operator(rng, config) -> str:
    w = config.ga.operator_weights
    ops = [
        OPERATOR.SPLICE, OPERATOR.MERGE,
        OPERATOR.MUTATE_ADD, OPERATOR.MUTATE_REMOVE,
        OPERATOR.MUTATE_DISPLACE, OPERATOR.MUTATE_RATTLE_SLAB,
        OPERATOR.MUTATE_TRANSLATE,
    ]
    probs = np.array([
        w.splice, w.merge, w.mutate_add, w.mutate_remove,
        w.mutate_displace, w.mutate_rattle_slab, w.mutate_translate,
    ], dtype=float)
    probs /= probs.sum()
    return str(rng.choice(ops, p=probs))


def _should_stop(total_evals, stall_count, config):
    if total_evals < config.ga.min_structures:
        return False
    if total_evals >= config.ga.max_structures:
        return True
    return stall_count >= config.ga.max_stall


def _stop_requested(run_dir):
    return (Path(run_dir) / "galoopstop").exists()


def _read_energy(struct_dir):
    ef = Path(struct_dir) / "FINAL_ENERGY"
    if ef.exists():
        try:
            return float(ef.read_text().strip())
        except ValueError:
            pass
    return float("nan")


def _infer_adsorbate_counts(element_symbols, adsorbate_configs) -> dict[str, int]:
    from galoop.science.surface import parse_formula

    remaining = Counter(element_symbols)
    counts: dict[str, int] = {}

    for ads in sorted(adsorbate_configs, key=lambda a: -len(parse_formula(a.symbol))):
        formula_elems = Counter(parse_formula(ads.symbol))
        if not formula_elems:
            continue
        n = min(remaining[e] // c for e, c in formula_elems.items())
        if n > 0:
            counts[ads.symbol] = n
            for e, c in formula_elems.items():
                remaining[e] -= n * c

    return counts


def _best_similarity(soap_vec: np.ndarray, struct_cache: dict) -> float:
    if not struct_cache:
        return 0.0
    return max(tanimoto_similarity(soap_vec, rec.soap_vector)
               for rec in struct_cache.values())
