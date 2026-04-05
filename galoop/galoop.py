"""
galoop/galoop.py

Steady-state GA loop backed by SQLite (GaloopStore) and Parsl futures.

Job lifecycle
-------------
  pending   — spawned, POSCAR written to structures/{id}/
  submitted — Parsl future dispatched
  converged — unique; GCE computed
  duplicate — dup of an existing converged structure
  failed    — pipeline error or unreasonable result
  desorbed  — adsorbate left the surface

Single-process: the GA loop submits relaxation jobs via Parsl and harvests
them as futures complete.  No external ``row run`` process needed.
"""

from __future__ import annotations

import logging
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from galoop.individual import Individual, STATUS, OPERATOR
from galoop.store import GaloopStore
from galoop.fingerprint import (
    classify_postrelax, compute_soap, tanimoto_similarity,
    StructRecord, _dist_histogram, _composition, build_chem_envs,
)
from galoop.science.reproduce import (
    splice, merge, mutate_add, mutate_remove,
    mutate_displace, mutate_rattle_slab, mutate_translate,
)

log = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds between loop iterations
KB_EV = 8.617333e-5  # eV / K


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
    run_dir   : root run directory
    slab_info : SlabInfo from load_slab()
    rng       : NumPy random generator
    """
    import parsl
    from galoop.engine.scheduler import build_parsl_config, relax_structure

    rng = rng or np.random.default_rng()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Initialising GaloopStore in %s", run_dir)
    store = GaloopStore(run_dir)

    # Initialise Parsl
    parsl_config = build_parsl_config(config.scheduler, run_dir=run_dir)
    parsl.load(parsl_config)

    chem_pots = {a.symbol: a.chemical_potential for a in config.adsorbates}

    # Resolve MACE model path relative to config
    mace_model = config.mace_model
    _candidate = run_dir / mace_model
    if _candidate.exists():
        mace_model = str(_candidate)

    stage_configs = [s.model_dump() for s in config.calculator_stages]

    # Build initial population on first run
    if store.is_empty():
        log.info("Building initial population …")
        _build_initial_population(config, slab_info, store, rng)

    struct_cache: dict[str, StructRecord] = {}
    total_evals = len(store.get_by_status(STATUS.CONVERGED))

    log.info("Resuming with %d converged structures.  Rebuilding struct cache …",
             total_evals)
    _rebuild_struct_cache(store, struct_cache, config, slab_info.n_slab_atoms)

    best_gce = float("inf")
    stall_count = 0
    for ind in store.get_by_status(STATUS.CONVERGED):
        if ind.grand_canonical_energy is not None and ind.grand_canonical_energy < best_gce:
            best_gce = ind.grand_canonical_energy

    # Active Parsl futures: ind_id -> Future
    active_futures: dict[str, object] = {}

    # Re-submit any structures that were submitted but not completed (crash recovery)
    for ind in store.get_by_status(STATUS.SUBMITTED):
        struct_dir = store.individual_dir(ind.id)
        if (struct_dir / "POSCAR").exists():
            fut = relax_structure(
                str(struct_dir), stage_configs,
                mace_model=mace_model,
                mace_device=config.mace_device,
                mace_dtype=config.mace_dtype,
                n_slab_atoms=slab_info.n_slab_atoms,
            )
            active_futures[ind.id] = fut

    log.info(
        "Start: evals=%d  best=%.4f eV  active=%d",
        total_evals, best_gce, len(active_futures),
    )

    pair_counts: dict[frozenset, int] = {}

    # ── main loop ──────────────────────────────────────────────────────────
    try:
        while True:
            if _stop_requested(run_dir):
                log.info("Stop file detected — exiting.")
                break

            # Harvest completed futures
            prev_best = best_gce
            done_ids = [
                ind_id for ind_id, fut in active_futures.items()
                if fut.done()
            ]
            for ind_id in done_ids:
                fut = active_futures.pop(ind_id)
                ind = store.get(ind_id)
                if ind is None:
                    continue
                struct_dir = store.individual_dir(ind_id)

                try:
                    result = fut.result()
                except Exception as exc:
                    log.warning("  %s: relaxation failed (%s)", ind_id, exc)
                    ind = ind.with_status(STATUS.FAILED)
                    store.update(ind)
                    continue

                if not result["converged"] or math.isnan(float(result["final_energy"])):
                    ind = ind.with_status(STATUS.FAILED)
                    store.update(ind)
                    continue

                # Check desorption
                from galoop.science.surface import detect_desorption
                if detect_desorption(result["final_atoms"], slab_info):
                    ind = ind.with_status(STATUS.DESORBED)
                    store.update(ind)
                    continue

                ind, total_evals, best_gce = _handle_converged(
                    ind, struct_dir, store, struct_cache, chem_pots,
                    config, total_evals, best_gce,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )

            # Stall tracking (once per poll cycle)
            if done_ids and total_evals > 0:
                if best_gce < prev_best - 1e-6:
                    stall_count = 0
                else:
                    stall_count += 1

            # Convergence check — stop spawning if stalled, drain active futures
            if _should_stop(total_evals, stall_count, config):
                if not active_futures:
                    log.info("Convergence criteria met.")
                    break
                # Don't spawn more — just wait for active futures to drain
            else:
                # Reset pair usage tracking for this spawn batch
                pair_counts = {}

                # Spawn offspring to fill worker pool
                _fill_workers(
                    store, config, slab_info, rng, total_evals,
                    active_futures, relax_structure, stage_configs,
                    mace_model, slab_info.n_slab_atoms,
                    pair_counts, struct_cache,
                )

            log.info(
                "Evals=%d  Best=%.4f eV  Stall=%d/%d  Active=%d",
                total_evals, best_gce, stall_count, config.ga.max_stall,
                len(active_futures),
            )
            time.sleep(POLL_INTERVAL)

    finally:
        parsl.clear()
        store.close()

    log.info("Run complete.  Total evaluations: %d", total_evals)


# ═══════════════════════════════════════════════════════════════════════════
# Post-relax evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _handle_converged(
    ind, struct_dir: Path, store: GaloopStore,
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
        store.update(ind)
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
            store.update(ind)
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
            store.update(ind)

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

def _build_initial_population(config, slab_info, store: GaloopStore, rng):
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

        # Simulated annealing: settle adsorbates into binding sites
        if config.ga.anneal_initial:
            current = _anneal_structure(
                current, config, slab_info.n_slab_atoms,
            )

        ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
        struct_dir = store.insert(ind)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), current, format="vasp")
        ind.geometry_path = str(poscar)
        store.update(ind)
        log.debug("initial structure %05d: %s", i, dict(counts))

    log.info("Initial population: %d structures", config.ga.population_size)


def _anneal_structure(
    atoms: Atoms,
    config,
    n_slab_atoms: int,
) -> Atoms:
    """Simulated annealing via basin-hopping: rattle adsorbates and quench.

    Alternates between random perturbation and relaxation with decreasing
    amplitude, using Metropolis acceptance.  The slab is fully fixed so
    only adsorbate degrees of freedom are explored.

    Parameters
    ----------
    atoms : slab + adsorbates
    config : GaloopConfig
    n_slab_atoms : number of leading slab atoms

    Returns
    -------
    Atoms with adsorbates settled into better binding positions
    """
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from pathlib import Path as _Path

    T_start = config.ga.anneal_temp_start
    T_end = config.ga.anneal_temp_end
    total_steps = config.ga.anneal_steps

    best = atoms.copy()

    # Build MACE calculator
    model = config.mace_model
    model_path = _Path(model)
    if model_path.exists():
        from mace.calculators import MACECalculator
        calc = MACECalculator(
            model_paths=str(model_path),
            device=config.mace_device,
            default_dtype=config.mace_dtype,
        )
    else:
        from mace.calculators import mace_mp
        calc = mace_mp(
            model=model,
            device=config.mace_device,
            default_dtype=config.mace_dtype,
        )

    best.calc = calc

    # Fix entire slab for fast adsorbate-only optimization
    best.set_constraint(FixAtoms(indices=list(range(n_slab_atoms))))

    ads_indices = list(range(n_slab_atoms, len(best)))
    if not ads_indices:
        return best

    # Initial relaxation to nearest minimum
    try:
        dyn = BFGS(best, logfile=None)
        dyn.run(fmax=0.1, steps=50)
        best_energy = best.get_potential_energy()
    except Exception:
        return atoms

    # Basin-hopping with cooling schedule
    temps = np.linspace(T_start, T_end, total_steps)

    for step, T in enumerate(temps):
        trial = best.copy()
        trial.calc = calc
        trial.set_constraint(best.constraints)

        # Perturb adsorbate positions — amplitude scales with temperature
        amplitude = 0.5 * (T / T_start)
        pos = trial.get_positions()
        slab_top_z = pos[:n_slab_atoms, 2].max()
        z_min = slab_top_z + 0.5   # don't go subsurface
        z_max = slab_top_z + 4.0   # don't fly into vacuum
        for idx in ads_indices:
            pos[idx] += np.random.randn(3) * amplitude
            pos[idx, 2] = np.clip(pos[idx, 2], z_min, z_max)
        trial.set_positions(pos)

        try:
            dyn = BFGS(trial, logfile=None)
            dyn.run(fmax=0.1, steps=30)
            trial_energy = trial.get_potential_energy()
        except Exception:
            continue

        # Metropolis acceptance
        dE = trial_energy - best_energy
        if dE < 0 or np.random.random() < np.exp(-dE / (KB_EV * max(T, 1.0))):
            best = trial
            best_energy = trial_energy

    # Final quench to tighter tolerance
    try:
        dyn = BFGS(best, logfile=None)
        dyn.run(fmax=0.05, steps=50)
    except Exception:
        pass

    # Restore original constraints and strip calculator
    result = best.copy()
    result.set_constraint(atoms.constraints)
    result.calc = None
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Offspring spawning
# ═══════════════════════════════════════════════════════════════════════════

def _fill_workers(store: GaloopStore, config, slab_info, rng, total_evals,
                  active_futures, relax_fn, stage_configs,
                  mace_model, n_slab_atoms,
                  pair_counts: dict | None = None,
                  struct_cache: dict | None = None):
    """Spawn offspring until the worker pool is full, submit via Parsl."""
    limit = config.scheduler.nworkers
    max_attempts = limit * 5  # avoid infinite loop
    attempts = 0

    while len(active_futures) < limit:
        if total_evals >= config.ga.max_structures:
            break
        if attempts >= max_attempts:
            log.debug("Hit max spawn attempts (%d), moving on", max_attempts)
            break
        result = _spawn_one(store, config, slab_info, rng, pair_counts)
        if result is None:
            attempts += 1
            continue  # operator failed, retry with new draw
        new_ind, child_atoms = result

        # Pre-relaxation duplicate check: graph isomorphism on unrelaxed structure
        if struct_cache and n_slab_atoms > 0:
            if _is_prerelax_duplicate(child_atoms, struct_cache, n_slab_atoms):
                log.debug("  %s: pre-relax duplicate — skipped", new_ind.id)
                new_ind = new_ind.mark_duplicate()
                new_ind.extra_data = {**new_ind.extra_data, "prerelax_dup": True}
                store.update(new_ind)
                attempts += 1
                continue

        # Write POSCAR and submit
        struct_dir = store.individual_dir(new_ind.id)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), child_atoms, format="vasp")
        new_ind.geometry_path = str(poscar)
        new_ind.status = STATUS.SUBMITTED
        store.update(new_ind)

        fut = relax_fn(
            str(struct_dir), stage_configs,
            mace_model=mace_model,
            mace_device=config.mace_device,
            mace_dtype=config.mace_dtype,
            n_slab_atoms=n_slab_atoms,
        )
        active_futures[new_ind.id] = fut


def _spawn_one(store: GaloopStore, config, slab_info, rng,
               pair_counts: dict | None = None):
    """Create one offspring.  Returns (Individual, Atoms) or None.

    Returns None when the operator fails — the caller should retry.
    Only falls back to random placement when there are no selectable parents.
    """
    selectable = store.selectable_pool()

    if not selectable:
        return _place_random(store, config, slab_info, rng)

    try:
        op = _sample_operator(rng, config)
        n_parents = 2 if op in (OPERATOR.SPLICE, OPERATOR.MERGE) else 1

        if len(selectable) < n_parents:
            return _place_random(store, config, slab_info, rng)

        energies = np.array([p.grand_canonical_energy or 0.0 for p in selectable])
        shifted = energies - energies.min()
        weights = np.exp(-shifted / config.ga.boltzmann_temperature)
        weights /= weights.sum()
        indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
        parents = [selectable[i] for i in indices]

        # Over-mating penalty
        if n_parents == 2 and pair_counts is not None:
            pair_key = frozenset(p.id for p in parents)
            if pair_counts.get(pair_key, 0) >= 1:
                indices2 = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
                parents = [selectable[i] for i in indices2]

        parent_atoms = []
        for p in parents:
            struct_dir = store.individual_dir(p.id)
            contcar = struct_dir / "CONTCAR"
            if not contcar.exists():
                return None  # parent geometry missing
            parent_atoms.append(read(str(contcar), format="vasp"))

        if op == OPERATOR.SPLICE:
            from galoop.science.surface import check_clash
            child, _ = splice(parent_atoms[0], parent_atoms[1],
                               slab_info.n_slab_atoms, rng=rng)
            if check_clash(child, n_slab=slab_info.n_slab_atoms, scale=0.7):
                return None  # clash after splice
        elif op == OPERATOR.MERGE:
            child = merge(parent_atoms[0], parent_atoms[1],
                          slab_info.n_slab_atoms, rng=rng)
        elif op == OPERATOR.MUTATE_ADD:
            from galoop.science.surface import load_adsorbate, place_adsorbate

            parent_counts = parents[0].extra_data.get("adsorbate_counts", {})
            if sum(parent_counts.values()) >= config.ga.max_adsorbates:
                return None  # already at max

            addable = [
                a.symbol for a in config.adsorbates
                if parent_counts.get(a.symbol, 0) < a.max_count
            ]
            if not addable:
                return None  # nothing to add
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
                return None  # nothing to remove
            child = result
        elif op == OPERATOR.MUTATE_DISPLACE:
            result = mutate_displace(parent_atoms[0], slab_info.n_slab_atoms,
                                     displacement=config.ga.displace_amplitude, rng=rng)
            if result is None:
                return None
            child = result
        elif op == OPERATOR.MUTATE_RATTLE_SLAB:
            child = mutate_rattle_slab(parent_atoms[0], slab_info.n_slab_atoms,
                                       amplitude=config.ga.rattle_amplitude, rng=rng)
        elif op == OPERATOR.MUTATE_TRANSLATE:
            result = mutate_translate(parent_atoms[0], slab_info.n_slab_atoms,
                                      displacement=config.ga.translate_amplitude, rng=rng)
            if result is None:
                return None
            child = result
        else:
            return None

        # Record pair usage
        if n_parents == 2 and pair_counts is not None:
            pair_key = frozenset(p.id for p in parents)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        # Bounds check for crossover operators
        if op in (OPERATOR.SPLICE, OPERATOR.MERGE):
            trial_counts = _infer_adsorbate_counts(
                child.get_chemical_symbols()[slab_info.n_slab_atoms:],
                config.adsorbates,
            )
            total = sum(trial_counts.values())
            if total > config.ga.max_adsorbates or total < config.ga.min_adsorbates:
                return None  # out of bounds

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
        store.insert(ind)
        log.debug("Spawned %s via %s", ind.id, op)
        return ind, child

    except Exception as exc:
        log.debug("Operator %s failed: %s", op if 'op' in dir() else '?', exc)
        return None


def _place_random(store: GaloopStore, config, slab_info, rng):
    """Generate a random structure.  Returns (Individual, Atoms)."""
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
    store.insert(ind)
    log.debug("Spawned %s via random", ind.id)
    return ind, current


# ═══════════════════════════════════════════════════════════════════════════
# Struct cache helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rebuild_struct_cache(store: GaloopStore, struct_cache, config,
                          n_slab_atoms: int = 0):
    for ind in store.all_converged_unique():
        struct_dir = store.individual_dir(ind.id)
        contcar = struct_dir / "CONTCAR"
        if not contcar.exists():
            continue
        try:
            atoms = read(str(contcar), format="vasp")
            soap_vec = compute_soap(
                atoms,
                r_cut=config.fingerprint.r_cut,
                n_max=config.fingerprint.n_max,
                l_max=config.fingerprint.l_max,
                n_slab_atoms=n_slab_atoms,
            )
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
                chem_envs=build_chem_envs(atoms, n_slab_atoms) if n_slab_atoms > 0 else None,
            )
        except Exception as exc:
            log.debug("Could not rebuild cache for %s: %s", ind.id, exc)


def _is_prerelax_duplicate(
    atoms: Atoms,
    struct_cache: dict[str, StructRecord],
    n_slab_atoms: int,
) -> bool:
    """Check if *atoms* is topologically identical to any cached structure.

    Uses composition gate + graph isomorphism only (no energy gate since
    the structure hasn't been relaxed yet).  This is cheap enough to run
    before submitting to Parsl.
    """
    from galoop.fingerprint import build_chem_envs, _compare_chem_envs, _composition

    new_comp = _composition(atoms)
    new_envs = build_chem_envs(atoms, n_slab_atoms)
    if new_envs is None:
        return False

    for rec in struct_cache.values():
        if rec.composition != new_comp:
            continue
        if rec.chem_envs is None:
            continue
        if _compare_chem_envs(new_envs, rec.chem_envs):
            return True

    return False


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
