"""
galoop/galoop.py

Async steady-state GA loop.  Spawns and harvests structures independently.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from galoop.individual import Individual, STATUS, OPERATOR
from galoop.database import GaloopDB, row_to_individual
from galoop.fingerprint import classify_postrelax, compute_soap

log = logging.getLogger(__name__)

POLL_INTERVAL = 15  # seconds between scheduler polls


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run(
    config,
    run_dir: Path,
    slab_info,
    pipeline,
    scheduler,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Steady-state async GA loop.

    Parameters
    ----------
    config : validated GaloopConfig
    run_dir : root run directory
    slab_info : SlabInfo from load_slab()
    pipeline : Pipeline from build_pipeline()  (currently unused in the loop
               because jobs run externally — but kept for future in-process mode)
    scheduler : Scheduler instance
    rng : NumPy random generator
    """
    rng = rng or np.random.default_rng()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting async GA loop in %s", run_dir)

    chem_pots = {a.symbol: a.chemical_potential for a in config.adsorbates}

    with GaloopDB(run_dir / "galoop.db") as db:
        db.setup()

        # Build initial population if this is a fresh run
        if not (run_dir / "gen_000").exists():
            log.info("Building initial population …")
            _build_initial_population(config, slab_info, db, run_dir, rng)

        active_jobs: dict[str, str] = {}   # struct_key → job_id
        soap_cache: dict[str, np.ndarray] = {}
        best_gce = float("inf")
        stall_count = 0
        total_evals = len(db.get_by_status(STATUS.CONVERGED))

        log.info("Resuming with %d converged structures.  Rebuilding SOAP cache …", total_evals)
        _rebuild_soap_cache(run_dir, db, soap_cache, config)

        # ── main loop ─────────────────────────────────────────────────────
        while True:
            if _stop_requested(run_dir):
                log.info("Stop file detected — exiting.")
                break

            # Poll finished jobs
            if active_jobs:
                statuses = scheduler.status(list(active_jobs.values()))

                for struct_key, job_id in list(active_jobs.items()):
                    js = statuses.get(job_id)
                    if js is None or js.value not in ("done", "failed"):
                        continue

                    struct_dir = _key_to_dir(struct_key, run_dir)
                    ind = _find_ind_for_dir(struct_dir, db)
                    if ind is None:
                        log.warning("No DB record for %s", struct_key)
                        del active_jobs[struct_key]
                        continue

                    # Sync status from filesystem sentinel
                    sentinel = _read_sentinel(struct_dir)
                    if sentinel and sentinel != ind.status:
                        ind = ind.with_status(sentinel)
                        db.update(ind)

                    if ind.status == STATUS.CONVERGED:
                        ind, total_evals, best_gce, stall_count = _handle_converged(
                            ind, struct_dir, db, soap_cache, chem_pots,
                            config, total_evals, best_gce, stall_count,
                        )
                    elif STATUS.is_terminal(ind.status):
                        total_evals += 1
                        log.info("  %s: %s", struct_key, ind.status)

                    del active_jobs[struct_key]

                    # Spawn a replacement
                    new_ind = _spawn_one(run_dir, db, config, slab_info, rng, total_evals)
                    if new_ind:
                        _submit_one(new_ind, run_dir, db, scheduler, active_jobs, config)

            # Top up worker pool
            _fill_workers(run_dir, db, scheduler, active_jobs, config)

            # Convergence check
            if not active_jobs and not db.get_by_status(STATUS.PENDING):
                if _should_stop(total_evals, stall_count, config):
                    log.info("Convergence criteria met.")
                    break

            log.info(
                "Evals=%d  Best=%.4f eV  Stall=%d/%d  Active=%d/%d",
                total_evals, best_gce, stall_count,
                config.ga.max_stall_generations,
                len(active_jobs), scheduler.nworkers,
            )
            time.sleep(POLL_INTERVAL)

    log.info("Run complete.  Total evaluations: %d", total_evals)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _handle_converged(
    ind, struct_dir, db, soap_cache, chem_pots,
    config, total_evals, best_gce, stall_count,
):
    """Post-relax classification + energy for a converged structure."""
    from galoop.science.energy import grand_canonical_energy

    contcar = struct_dir / "CONTCAR"
    if not contcar.exists():
        log.warning("  %s: CONTCAR missing after convergence", ind.id)
        return ind, total_evals, best_gce, stall_count

    try:
        atoms = read(str(contcar), format="vasp")
        label, dup_id, soap_vec = classify_postrelax(
            atoms, soap_cache,
            duplicate_threshold=config.fingerprint.duplicate_threshold,
            r_cut=config.fingerprint.r_cut,
            n_max=config.fingerprint.n_max,
            l_max=config.fingerprint.l_max,
        )

        if label == "duplicate":
            log.info("  %s: duplicate of %s", ind.id, dup_id)
            ind = ind.mark_duplicate()
            db.update(ind)
        else:
            raw_e = _read_energy(struct_dir)
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
            db.update(ind)
            soap_cache[ind.id] = soap_vec

            if gce < best_gce - 1e-6:
                best_gce = gce
                stall_count = 0
            else:
                stall_count += 1

            total_evals += 1
            log.info("  %s: converged  G=%.4f eV", ind.id, gce)

    except Exception as exc:
        log.warning("  %s: post-relax check failed (%s)", ind.id, exc)

    return ind, total_evals, best_gce, stall_count


# ── initial population ────────────────────────────────────────────────────

def _build_initial_population(config, slab_info, db, run_dir, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    gen_dir = run_dir / "gen_000"
    gen_dir.mkdir(exist_ok=True)

    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in config.adsorbates
    }

    for i in range(config.ga.population_size):
        struct_id = f"{i:04d}"
        struct_dir = gen_dir / f"struct_{struct_id}"
        struct_dir.mkdir(exist_ok=True)

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
                        rng=rng,
                    )
        except Exception as exc:
            log.warning("struct_%s: placement failed (%s) — falling back", struct_id, exc)
            first = next(iter(ads_atoms))
            counts = {first: 1}
            current = slab_info.atoms.copy()
            current = place_adsorbate(
                current, ads_atoms[first],
                slab_info.zmin, slab_info.zmax, rng=rng,
            )

        poscar = struct_dir / "POSCAR"
        write(str(poscar), current, format="vasp")

        ind = Individual.from_init(
            generation=0,
            geometry_path=str(poscar),
            extra_data={"adsorbate_counts": dict(counts)},
        )
        _write_sentinel(struct_dir, STATUS.PENDING)
        db.insert(ind)
        log.debug("struct_%s: %s", struct_id, dict(counts))

    log.info("Initial population: %d structures", config.ga.population_size)


# ── offspring spawning ────────────────────────────────────────────────────

def _spawn_one(run_dir, db, config, slab_info, rng, total_evals):
    from galoop.science.reproduce import splice, merge
    from galoop.science.surface import load_adsorbate, place_adsorbate

    selectable = db.selectable_pool()

    bucket_n = max(1, (total_evals + config.ga.population_size) // config.ga.population_size)
    bucket_dir = run_dir / f"gen_{bucket_n:03d}"

    if not selectable:
        return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

    try:
        op = _sample_operator(rng)
        n_parents = 2 if op in (OPERATOR.SPLICE, OPERATOR.MERGE) else 1

        if len(selectable) < n_parents:
            return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

        # Boltzmann-weighted selection
        energies = np.array([p.grand_canonical_energy or 0.0 for p in selectable])
        weights = np.exp(-energies / 0.1)
        weights /= weights.sum()
        indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
        parents = [selectable[i] for i in indices]

        parent_atoms = []
        for p in parents:
            if not p.geometry_path:
                return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)
            contcar = Path(p.geometry_path).parent / "CONTCAR"
            if not contcar.exists():
                return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)
            parent_atoms.append(read(str(contcar), format="vasp"))

        if op == OPERATOR.SPLICE:
            child, _ = splice(parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng)
        elif op == OPERATOR.MERGE:
            child = merge(parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng)
        else:
            return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

        bucket_dir.mkdir(parents=True, exist_ok=True)
        struct_n = len(sorted(bucket_dir.glob("struct_????"))) + 1
        struct_dir = bucket_dir / f"struct_{struct_n:04d}"
        struct_dir.mkdir(exist_ok=True)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), child, format="vasp")

        ads_counts = dict(Counter(child.get_chemical_symbols()[slab_info.n_slab_atoms:]))
        ind = Individual.from_parents(
            generation=int(bucket_dir.name.split("_")[1]),
            parents=parents,
            operator=op,
            geometry_path=str(poscar),
            extra_data={"adsorbate_counts": ads_counts},
        )
        _write_sentinel(struct_dir, STATUS.PENDING)
        db.insert(ind)
        log.debug("Spawned %s via %s", struct_dir.name, op)
        return ind

    except Exception as exc:
        log.debug("Operator failed: %s — falling back to random", exc)
        return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)


def _place_random(run_dir, bucket_dir, db, config, slab_info, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    bucket_dir.mkdir(parents=True, exist_ok=True)
    struct_n = len(sorted(bucket_dir.glob("struct_????"))) + 1
    struct_dir = bucket_dir / f"struct_{struct_n:04d}"
    struct_dir.mkdir(exist_ok=True)

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
                n_orientations=ads_cfg.n_orientations, rng=rng,
            )

    poscar = struct_dir / "POSCAR"
    write(str(poscar), current, format="vasp")

    ind = Individual.from_init(
        generation=int(bucket_dir.name.split("_")[1]),
        geometry_path=str(poscar),
        extra_data={"adsorbate_counts": dict(counts)},
    )
    _write_sentinel(struct_dir, STATUS.PENDING)
    db.insert(ind)
    log.debug("Spawned %s via random", struct_dir.name)
    return ind


# ── job submission ────────────────────────────────────────────────────────

def _fill_workers(run_dir, db, scheduler, active_jobs, config):
    n_active = len(active_jobs)
    limit = scheduler.nworkers

    if n_active >= limit:
        return

    for ind in db.get_by_status(STATUS.PENDING):
        if n_active >= limit:
            break
        if not ind.geometry_path:
            continue
        struct_dir = Path(ind.geometry_path).parent
        if not struct_dir.exists():
            ind = ind.with_status(STATUS.FAILED)
            db.update(ind)
            continue
        key = _dir_to_key(struct_dir)
        if key in active_jobs:
            continue
        _submit_one(ind, run_dir, db, scheduler, active_jobs, config)
        n_active += 1


def _submit_one(ind, run_dir, db, scheduler, active_jobs, config):
    if not ind.geometry_path:
        return
    struct_dir = Path(ind.geometry_path).parent
    config_path = (run_dir / "galoop.yaml").resolve()
    body = (
        f"cd {struct_dir.resolve()}\n"
        f"galoop _run-pipeline {struct_dir.resolve()} --config {config_path}\n"
    )
    job_name = f"galoop_{struct_dir.parent.name}_{struct_dir.name}"

    try:
        job_id = scheduler.submit(job_name, body, struct_dir)
    except Exception as exc:
        log.warning("Submit failed for %s: %s", struct_dir.name, exc)
        return

    ind = ind.with_status(STATUS.SUBMITTED)
    db.update(ind)
    _write_sentinel(struct_dir, STATUS.SUBMITTED)
    active_jobs[_dir_to_key(struct_dir)] = job_id
    log.info("Submitted %s/%s → %s", struct_dir.parent.name, struct_dir.name, job_id)


# ── small helpers ─────────────────────────────────────────────────────────

def _random_stoichiometry(ads_configs, rng, min_total, max_total):
    counts = {
        a.symbol: int(rng.integers(a.min_count, a.max_count + 1))
        for a in ads_configs
    }
    # Trim down
    while sum(counts.values()) > max_total:
        shrinkable = [
            s for s in counts
            if counts[s] > next(a.min_count for a in ads_configs if a.symbol == s)
        ]
        if not shrinkable:
            break
        counts[str(rng.choice(shrinkable))] -= 1
    # Pad up
    while sum(counts.values()) < min_total:
        growable = [
            s for s in counts
            if counts[s] < next(a.max_count for a in ads_configs if a.symbol == s)
        ]
        if not growable:
            break
        counts[str(rng.choice(growable))] += 1
    return counts


def _sample_operator(rng) -> str:
    ops = [OPERATOR.SPLICE, OPERATOR.MERGE, OPERATOR.MUTATE_ADD, OPERATOR.MUTATE_REMOVE]
    probs = [0.4, 0.3, 0.2, 0.1]
    return str(rng.choice(ops, p=probs))


def _rebuild_soap_cache(run_dir, db, soap_cache, config):
    for ind in db.get_by_status(STATUS.CONVERGED):
        if not ind.geometry_path:
            continue
        contcar = Path(ind.geometry_path).parent / "CONTCAR"
        if contcar.exists():
            try:
                atoms = read(str(contcar), format="vasp")
                soap_cache[ind.id] = compute_soap(
                    atoms,
                    r_cut=config.fingerprint.r_cut,
                    n_max=config.fingerprint.n_max,
                    l_max=config.fingerprint.l_max,
                )
            except Exception as exc:
                log.debug("Could not load SOAP for %s: %s", ind.id, exc)


def _should_stop(total_evals, stall_count, config):
    min_evals = config.ga.min_generations * config.ga.population_size
    max_evals = config.ga.max_generations * config.ga.population_size
    if total_evals < min_evals:
        return False
    if total_evals >= max_evals:
        return True
    return stall_count >= config.ga.max_stall_generations


def _stop_requested(run_dir):
    return (Path(run_dir) / "gociastop").exists()


def _dir_to_key(struct_dir):
    return f"{struct_dir.parent.name}/{struct_dir.name}"


def _key_to_dir(key, run_dir):
    return Path(run_dir) / key


def _find_ind_for_dir(struct_dir, db):
    return db.find_by_geometry_path_substring(struct_dir.name)


def _read_energy(struct_dir):
    ef = Path(struct_dir) / "FINAL_ENERGY"
    if ef.exists():
        try:
            return float(ef.read_text().strip())
        except ValueError:
            pass
    return float("nan")


def _write_sentinel(struct_dir, status):
    struct_dir = Path(struct_dir)
    for name in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED", "DUPLICATE", "DESORBED"):
        (struct_dir / name).unlink(missing_ok=True)
    (struct_dir / status.upper()).touch()


def _read_sentinel(struct_dir):
    struct_dir = Path(struct_dir)
    for name in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED", "DUPLICATE", "DESORBED"):
        if (struct_dir / name).exists():
            return name.lower()
    return None
