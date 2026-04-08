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
    StructRecord, _dist_histogram, _composition,
)
from galoop.science.reproduce import (
    splice, merge, mutate_add, mutate_remove,
    mutate_displace, mutate_rattle_slab, mutate_translate,
)

log = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds between loop iterations


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
    try:
        parsl.load(parsl_config)
    except Exception as exc:
        log.error("Failed to initialise Parsl: %s", exc)
        store.close()
        raise

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

    # GPR surrogate (optional)
    gpr = None
    if config.ga.gpr_guided:
        from galoop.gpr import CompositionGPR
        gpr = CompositionGPR(
            species=[a.symbol for a in config.adsorbates],
            ads_configs=config.adsorbates,
            min_total=config.ga.min_adsorbates,
            max_total=config.ga.max_adsorbates,
        )
        _retrain_gpr(gpr, store)

    best_gce = float("inf")
    stall_count = 0
    spawn_stall_count = 0  # consecutive poll cycles where _fill_workers couldn't spawn anything
    prev_evals = total_evals  # track evals to detect new unique arrivals
    for ind in store.get_by_status(STATUS.CONVERGED):
        if ind.grand_canonical_energy is not None and ind.grand_canonical_energy < best_gce:
            best_gce = ind.grand_canonical_energy

    # Active Parsl futures: ind_id -> Future
    active_futures: dict[str, object] = {}

    # Crash recovery: handle structures orphaned by a previous crash
    orphans = store.get_by_status(STATUS.SUBMITTED)
    if orphans:
        n_resubmit = 0
        n_reset = 0
        for ind in orphans:
            struct_dir = store.individual_dir(ind.id)
            contcar = struct_dir / "CONTCAR"
            poscar = struct_dir / "POSCAR"

            if contcar.exists():
                # Relaxation completed but wasn't harvested — will be
                # picked up by _handle_converged on the next poll.
                # Re-submit to get a result dict (cheap if already done).
                fut = relax_structure(
                    str(struct_dir), stage_configs,
                    mace_model=mace_model,
                    mace_device=config.mace_device,
                    mace_dtype=config.mace_dtype,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )
                active_futures[ind.id] = fut
                n_resubmit += 1
            elif poscar.exists():
                # Has POSCAR but relaxation never finished — re-submit
                fut = relax_structure(
                    str(struct_dir), stage_configs,
                    mace_model=mace_model,
                    mace_device=config.mace_device,
                    mace_dtype=config.mace_dtype,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )
                active_futures[ind.id] = fut
                n_resubmit += 1
            else:
                # No geometry files at all — reset to pending
                ind = ind.with_status(STATUS.PENDING)
                store.update(ind)
                n_reset += 1

        if n_resubmit:
            log.info("Crash recovery: re-submitted %d orphaned structures", n_resubmit)
        if n_reset:
            log.info("Crash recovery: reset %d structures with no geometry to pending", n_reset)

    log.info(
        "Start: evals=%d  best=%.4f eV  active=%d",
        total_evals, best_gce, len(active_futures),
    )

    pair_counts: dict[frozenset, int] = {}

    # Register signal handler for graceful shutdown
    import signal
    _shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal _shutdown_requested
        sig_name = signal.Signals(signum).name
        log.info("Received %s — shutting down gracefully after current cycle", sig_name)
        _shutdown_requested = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # ── main loop ──────────────────────────────────────────────────────────
    try:
        while True:
            if _shutdown_requested or _stop_requested(run_dir):
                log.info("Shutdown requested — exiting.")
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

                # Check surface binding: every adsorbate molecule must
                # contact the slab (subsurface atoms count as bound)
                from galoop.science.surface import validate_surface_binding
                is_bound, unbound_mols = validate_surface_binding(
                    result["final_atoms"], slab_info.n_slab_atoms,
                )
                if not is_bound:
                    log.info("  %s: %d unbound molecule(s) — marking UNBOUND",
                             ind_id, len(unbound_mols))
                    ind = ind.with_status(STATUS.UNBOUND)
                    ind.extra_data = {**ind.extra_data,
                                      "n_unbound_molecules": len(unbound_mols)}
                    store.update(ind)
                    continue

                ind, total_evals, best_gce = _handle_converged(
                    ind, struct_dir, store, struct_cache, chem_pots,
                    config, total_evals, best_gce,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )

            # Stall tracking: only count when new unique evaluations arrive
            if total_evals > prev_evals:
                if best_gce < prev_best - 1e-6:
                    stall_count = 0
                else:
                    stall_count += (total_evals - prev_evals)
                prev_evals = total_evals

                # Retrain GPR when new data arrives
                if gpr is not None and done_ids:
                    _retrain_gpr(gpr, store)

            # Convergence check — stop spawning if stalled, drain active futures
            if _should_stop(total_evals, stall_count, spawn_stall_count, config):
                if not active_futures:
                    log.info("Convergence criteria met.")
                    break
                # Don't spawn more — just wait for active futures to drain
            else:
                # Reset pair usage tracking for this spawn batch
                pair_counts = {}

                # Spawn offspring to fill worker pool
                spawned = _fill_workers(
                    store, config, slab_info, rng, total_evals,
                    active_futures, relax_structure, stage_configs,
                    mace_model, slab_info.n_slab_atoms,
                    pair_counts, struct_cache, gpr,
                )
                # Track spawn exhaustion: workers are free but nothing novel can be generated
                if not spawned and len(active_futures) < config.scheduler.nworkers:
                    spawn_stall_count += 1
                else:
                    spawn_stall_count = 0

            log.info(
                "Evals=%d  Best=%.4f eV  Stall=%d/%d  SpawnStall=%d/%d  Active=%d",
                total_evals, best_gce, stall_count, config.ga.max_stall,
                spawn_stall_count, config.ga.max_spawn_stall,
                len(active_futures),
            )
            time.sleep(POLL_INTERVAL)

    finally:
        # Mark any still-active futures' structures back to pending
        # so they get re-submitted on restart
        for ind_id in list(active_futures.keys()):
            fut = active_futures[ind_id]
            if not fut.done():
                ind = store.get(ind_id)
                if ind and ind.status == STATUS.SUBMITTED:
                    ind = ind.with_status(STATUS.PENDING)
                    store.update(ind)

        try:
            parsl.clear()
        except Exception as exc:
            log.debug("Parsl cleanup: %s", exc)

        try:
            store.close()
        except Exception as exc:
            log.debug("Store cleanup: %s", exc)

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

        # Sanity check: reject structures with atom-atom overlap
        if _has_atom_overlap(atoms, min_dist=0.5):
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

            # Compute pre-relax SOAP from POSCAR for pre-relax duplicate comparison
            prerelax_soap = None
            poscar = struct_dir / "POSCAR"
            if poscar.exists():
                try:
                    pre_atoms = read(str(poscar), format="vasp")
                    prerelax_soap = compute_soap(
                        pre_atoms,
                        r_cut=config.fingerprint.r_cut,
                        n_max=config.fingerprint.n_max,
                        l_max=config.fingerprint.l_max,
                        n_slab_atoms=n_slab_atoms,
                    )
                except Exception:
                    pass

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
                prerelax_soap=prerelax_soap,
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

    # Stratified seeding: cycle target totals across [min_adsorbates, max_adsorbates]
    # so the initial population spans every coverage level instead of biasing
    # random draws toward the middle of the range.
    lo, hi = config.ga.min_adsorbates, config.ga.max_adsorbates
    coverage_levels = list(range(lo, hi + 1))
    for i in range(config.ga.population_size):
        target_total = coverage_levels[i % len(coverage_levels)]
        counts = _random_stoichiometry(
            config.adsorbates, rng,
            target_total, target_total,
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
                        n_slab_atoms=slab_info.n_slab_atoms,
                    )
        except Exception as exc:
            log.warning("structure_%05d: placement failed (%s) — falling back", i, exc)
            first = next(iter(ads_atoms))
            counts = {first: 1}
            current = slab_info.atoms.copy()
            current = place_adsorbate(
                current, ads_atoms[first],
                slab_info.zmin, slab_info.zmax, rng=rng,
                n_slab_atoms=slab_info.n_slab_atoms,
            )

        # Quick constrained pre-relax: let adsorbates settle toward surface
        current = _snap_to_surface(
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


def _snap_to_surface(
    atoms: Atoms,
    config,
    n_slab_atoms: int,
) -> Atoms:
    """Quick constrained pre-relax to settle adsorbates toward the surface.

    Fixes the slab and runs a short BFGS so adsorbates fall into reasonable
    binding positions.  Then clamps adsorbate z-coordinates to a safe window
    (0.8–4.0 Å above slab top) to prevent burial or desorption.

    Much cheaper than full simulated annealing (~1s vs ~2.5 min per structure).
    """
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS
    from pathlib import Path as _Path

    work = atoms.copy()

    ads_indices = list(range(n_slab_atoms, len(work)))
    if not ads_indices:
        return work

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

    work.calc = calc
    work.set_constraint(FixAtoms(indices=list(range(n_slab_atoms))))

    # Short BFGS: adsorbates relax toward surface binding sites
    try:
        dyn = BFGS(work, logfile=None)
        dyn.run(fmax=0.2, steps=30)
    except Exception:
        # If BFGS fails, still clamp z and return
        pass

    # Clamp adsorbate z to safe window above slab
    pos = work.get_positions()
    slab_top_z = pos[:n_slab_atoms, 2].max()
    z_min = slab_top_z + 0.8   # don't bury into slab
    z_max = slab_top_z + 4.0   # don't float into vacuum
    for idx in ads_indices:
        pos[idx, 2] = np.clip(pos[idx, 2], z_min, z_max)
    work.set_positions(pos)

    # Strip calculator and constraints for clean POSCAR
    result = work.copy()
    result.calc = None
    result.set_constraint(atoms.constraints)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Offspring spawning
# ═══════════════════════════════════════════════════════════════════════════

def _fill_workers(store: GaloopStore, config, slab_info, rng, total_evals,
                  active_futures, relax_fn, stage_configs,
                  mace_model, n_slab_atoms,
                  pair_counts: dict | None = None,
                  struct_cache: dict | None = None,
                  gpr=None) -> bool:
    """Spawn offspring until the worker pool is full, submit via Parsl.

    Returns True if any worker slot was filled; False if spawn exhausted
    (hit max attempts with all candidates rejected as pre-relax duplicates
    or operator failures).
    """
    limit = config.scheduler.nworkers
    max_attempts = limit * 5  # avoid infinite loop
    attempts = 0
    submitted = 0
    initial_active = len(active_futures)

    while len(active_futures) < limit:
        if total_evals >= config.ga.max_structures:
            break
        if attempts >= max_attempts:
            log.debug("Hit max spawn attempts (%d), moving on", max_attempts)
            break

        # GPR-guided spawning: with gpr_fraction probability, use GPR
        use_gpr = (
            gpr is not None
            and gpr.is_ready
            and total_evals >= config.ga.gpr_min_samples
            and rng.random() < config.ga.gpr_fraction
        )

        if use_gpr:
            result = _spawn_gpr(
                gpr, store, config, slab_info, rng,
            )
        else:
            result = _spawn_one(store, config, slab_info, rng, pair_counts)

        if result is None:
            attempts += 1
            continue
        new_ind, child_atoms = result

        # Pre-relaxation duplicate check: SOAP similarity on unrelaxed structure
        if struct_cache:
            if _is_prerelax_duplicate(child_atoms, struct_cache, n_slab_atoms,
                                      threshold=config.fingerprint.prerelax_duplicate_threshold,
                                      r_cut=config.fingerprint.r_cut,
                                      n_max=config.fingerprint.n_max,
                                      l_max=config.fingerprint.l_max):
                log.debug("  %s: pre-relax duplicate — skipped", new_ind.id)
                new_ind = new_ind.mark_duplicate()
                new_ind.extra_data = {**new_ind.extra_data, "prerelax_dup": True}
                store.update(new_ind)
                attempts += 1
                continue

        # Write POSCAR and submit — ensure correct slab constraints
        child_atoms.set_constraint(slab_info.atoms.constraints)
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
        submitted += 1

    return submitted > 0


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
                n_slab_atoms=slab_info.n_slab_atoms,
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
                n_slab_atoms=slab_info.n_slab_atoms,
            )

    ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
    store.insert(ind)
    log.debug("Spawned %s via random", ind.id)
    return ind, current


def _spawn_gpr(gpr, store: GaloopStore, config, slab_info, rng):
    """Generate a structure with a GPR-suggested composition.

    Returns (Individual, Atoms) or None.
    """
    from galoop.science.surface import load_adsorbate, place_adsorbate

    counts = gpr.suggest(rng, kappa=config.ga.gpr_kappa)

    if sum(counts.values()) == 0:
        return None

    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in config.adsorbates
    }

    current = slab_info.atoms.copy()
    try:
        for sym, cnt in counts.items():
            if cnt <= 0:
                continue
            ads_cfg = next(a for a in config.adsorbates if a.symbol == sym)
            for _ in range(cnt):
                current = place_adsorbate(
                    current, ads_atoms[sym],
                    slab_info.zmin, slab_info.zmax,
                    n_orientations=ads_cfg.n_orientations,
                    binding_index=ads_cfg.binding_index,
                    rng=rng,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )
    except Exception as exc:
        log.debug("GPR spawn placement failed: %s", exc)
        return None

    ind = Individual.from_init(
        extra_data={"adsorbate_counts": dict(counts)},
    )
    ind.operator = OPERATOR.GPR
    store.insert(ind)
    log.debug("Spawned %s via GPR: %s", ind.id, counts)
    return ind, current


def _retrain_gpr(gpr, store: GaloopStore) -> None:
    """Retrain the GPR on all converged structures."""
    converged = store.get_by_status(STATUS.CONVERGED)
    if len(converged) < 2:
        return
    compositions = [ind.extra_data.get("adsorbate_counts", {}) for ind in converged]
    energies = [ind.grand_canonical_energy for ind in converged
                if ind.grand_canonical_energy is not None]
    if len(energies) < 2:
        return
    try:
        gpr.fit(compositions[:len(energies)], energies)
    except Exception as exc:
        log.debug("GPR training failed: %s", exc)


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
            # Pre-relax SOAP from POSCAR
            prerelax_soap = None
            poscar = struct_dir / "POSCAR"
            if poscar.exists():
                try:
                    pre_atoms = read(str(poscar), format="vasp")
                    prerelax_soap = compute_soap(
                        pre_atoms,
                        r_cut=config.fingerprint.r_cut,
                        n_max=config.fingerprint.n_max,
                        l_max=config.fingerprint.l_max,
                        n_slab_atoms=n_slab_atoms,
                    )
                except Exception:
                    pass

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


def _is_prerelax_duplicate(
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
    (from POSCAR) for like-with-like comparison.  Falls back to the
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


def _should_stop(total_evals, stall_count, spawn_stall_count, config):
    if total_evals >= config.ga.max_structures:
        return True
    # Spawn exhaustion: can't generate novel candidates anymore.
    # This can fire before min_structures if the composition/geometry space is small.
    if spawn_stall_count >= config.ga.max_spawn_stall:
        return True
    if total_evals < config.ga.min_structures:
        return False
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


def _has_atom_overlap(atoms: Atoms, min_dist: float = 0.5) -> bool:
    """True if any two atoms are closer than *min_dist* Å."""
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 999.0)
    return float(dists.min()) < min_dist
