"""
galoop/loop.py

Main GA event loop. Submits relaxation jobs via Parsl and harvests them
as futures complete. Crash recovery, signal handling, and stall tracking
live here; the actual classification (harvest.py) and spawning (spawn.py)
are imported from sibling modules.

Split out of galoop/galoop.py during the Phase 2 refactor.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from galoop.fingerprint import StructRecord
from galoop.harvest import handle_converged, rebuild_struct_cache
from galoop.individual import STATUS
from galoop.spawn import build_initial_population, fill_workers, retrain_gpr
from galoop.store import GaloopStore

log = logging.getLogger(__name__)

POLL_INTERVAL = 10  # seconds between loop iterations


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

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

    # By the time `run()` reaches here, calibrate() has either filled in
    # every chemical_potential or raised. Drop the Optional from the type
    # so the harvest pipeline doesn't have to second-guess.
    chem_pots: dict[str, float] = {
        a.symbol: float(a.chemical_potential)
        for a in config.adsorbates
        if a.chemical_potential is not None
    }
    if len(chem_pots) != len(config.adsorbates):
        missing = [a.symbol for a in config.adsorbates if a.chemical_potential is None]
        raise RuntimeError(
            f"Adsorbates missing chemical_potential: {missing}. "
            "Either set them in the config or let calibrate() compute them."
        )

    stage_configs = [s.model_dump() for s in config.calculator_stages]

    # Build initial population on first run
    if store.is_empty():
        log.info("Building initial population …")
        build_initial_population(config, slab_info, store, rng)

    struct_cache: dict[str, StructRecord] = {}
    total_evals = len(store.get_by_status(STATUS.CONVERGED))

    log.info("Resuming with %d converged structures.  Rebuilding struct cache …",
             total_evals)
    rebuild_struct_cache(store, struct_cache, config, slab_info.n_slab_atoms)

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
        retrain_gpr(gpr, store)

    best_gce = float("inf")
    stall_count = 0
    spawn_stall_count = 0
    prev_evals = total_evals
    for ind in store.get_by_status(STATUS.CONVERGED):
        if ind.grand_canonical_energy is not None and ind.grand_canonical_energy < best_gce:
            best_gce = ind.grand_canonical_energy

    # Parsl AppFuture; using Any so we don't take a hard type dep on parsl
    # internals (which lack stubs anyway).
    active_futures: dict[str, Any] = {}

    # Crash recovery: handle structures orphaned by a previous crash
    _recover_orphans(store, active_futures, stage_configs,
                     slab_info, relax_structure)

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

                ind, total_evals, best_gce = handle_converged(
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
                # Reset spawn_stall_count on real forward progress (a new
                # converged evaluation arrived), not merely on fill_workers
                # managing to sneak a structure past the dup filter.
                spawn_stall_count = 0
                prev_evals = total_evals

                # Retrain GPR when new data arrives
                if gpr is not None and done_ids:
                    retrain_gpr(gpr, store)

            # Convergence check — stop spawning if stalled, drain active futures
            if _should_stop(total_evals, stall_count, spawn_stall_count, config):
                if not active_futures:
                    log.info("Convergence criteria met.")
                    break
            else:
                pair_counts = {}

                spawned = fill_workers(
                    store, config, slab_info, rng, total_evals,
                    active_futures, relax_structure, stage_configs,
                    slab_info.n_slab_atoms,
                    pair_counts, struct_cache, gpr,
                )
                if not spawned and len(active_futures) < config.scheduler.nworkers:
                    spawn_stall_count += 1
                # Note: do NOT reset spawn_stall_count on spawn success here;
                # it's reset only when a new converged evaluation arrives
                # (see the total_evals > prev_evals block above).

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


# ---------------------------------------------------------------------------
# Crash recovery
# ---------------------------------------------------------------------------

def _recover_orphans(store, active_futures, stage_configs,
                     slab_info, relax_fn) -> None:
    """Submit any pending/orphaned structures with geometry on disk.

    Two cases handled here:

    - **Initial population**: ``build_initial_population`` writes POSCARs and
      leaves the rows ``PENDING``. We submit them on the first loop tick so
      they actually get evaluated. (Without this, the GA discards the
      stratified initial seeds and bootstraps via the ``_place_random``
      fallback inside ``_fill_workers``, wasting the 15 carefully spawned
      starting points.)
    - **Crash recovery**: structures left ``SUBMITTED`` from a previous
      run also get re-dispatched if their POSCAR/CONTCAR is still on disk.

    Anything missing geometry is reset back to ``PENDING`` so the spawn
    loop can pick it up via a fresh attempt.
    """
    candidates = (
        store.get_by_status(STATUS.SUBMITTED)
        + store.get_by_status(STATUS.PENDING)
    )
    if not candidates:
        return

    n_resubmit = 0
    n_reset = 0
    for ind in candidates:
        struct_dir = store.individual_dir(ind.id)
        contcar = struct_dir / "CONTCAR"
        poscar = struct_dir / "POSCAR"

        if contcar.exists() or poscar.exists():
            fut = relax_fn(
                str(struct_dir), stage_configs,
                n_slab_atoms=slab_info.n_slab_atoms,
            )
            active_futures[ind.id] = fut
            ind = ind.with_status(STATUS.SUBMITTED)
            store.update(ind)
            n_resubmit += 1
        elif ind.status == STATUS.SUBMITTED:
            ind = ind.with_status(STATUS.PENDING)
            store.update(ind)
            n_reset += 1

    if n_resubmit:
        log.info("Submitted %d pending/orphaned structures", n_resubmit)
    if n_reset:
        log.info("Reset %d structures with no geometry to pending", n_reset)


# ---------------------------------------------------------------------------
# Stop checks
# ---------------------------------------------------------------------------

def _should_stop(total_evals: int, stall_count: int, spawn_stall_count: int, config) -> bool:
    if total_evals >= config.ga.max_structures:
        return True
    # Spawn exhaustion: can't generate novel candidates anymore.
    if spawn_stall_count >= config.ga.max_spawn_stall:
        return True
    if total_evals < config.ga.min_structures:
        return False
    return stall_count >= config.ga.max_stall


def _stop_requested(run_dir) -> bool:
    return (Path(run_dir) / "galoopstop").exists()
