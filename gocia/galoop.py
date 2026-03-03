"""
gocia/galoop.py

Lean async steady-state GA loop. Spawn and process independently.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from individual import Individual, STATUS, OPERATOR
from database import GociaDB
from fingerprint import classify_postrelax, compute_soap

log = logging.getLogger(__name__)

POLL_INTERVAL = 15  # seconds between scheduler polls


def run(
    config,
    run_dir: Path,
    slab_info,
    stages,
    scheduler,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Steady-state async GA loop.

    Parameters
    ----------
    config : Validated GociaConfig
    run_dir : Root run directory
    slab_info : SlabInfo from load_slab()
    stages : Calculator stages from build_pipeline()
    scheduler : Scheduler instance
    rng : NumPy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    log.info(f"Starting async GA loop in {run_dir}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    chemical_potentials = {
        ads.symbol: ads.chemical_potential for ads in config.adsorbates
    }

    with GociaDB(run_dir / "gocia.db") as db:
        db.setup()

        # Build or load initial population
        if not (run_dir / "gen_000").exists():
            log.info("Building initial population...")
            _build_initial_population(
                config, slab_info, db, run_dir, rng
            )

        active_jobs = {}  # {struct_key: job_id}
        soap_cache = {}  # {struct_id: soap_vector}
        best_gce = float("inf")
        stall_count = 0
        total_evals = _count_converged(db)

        log.info(
            f"Resuming with {total_evals} converged structures. "
            f"Building SOAP cache..."
        )
        _rebuild_soap_cache(run_dir, db, soap_cache, config)

        # Main loop
        while True:
            if _stop_requested(run_dir):
                log.info("gociastop detected — exiting.")
                break

            # ── Poll and process finished jobs ──
            if active_jobs:
                statuses = scheduler.status(list(active_jobs.values()))

                for struct_key, job_id in list(active_jobs.items()):
                    status = statuses.get(job_id)
                    if status is None or status.value not in ("done", "failed"):
                        continue

                    struct_dir = _key_to_dir(struct_key, run_dir)
                    ind = _find_ind_for_dir(struct_dir, db)

                    if ind is None:
                        log.warning(f"  Cannot find DB record for {struct_key}")
                        del active_jobs[struct_key]
                        continue

                    # Sync status from filesystem
                    sentinel = _read_sentinel(struct_dir)
                    if sentinel and sentinel != ind.status:
                        ind = ind.with_status(sentinel)
                        db.update(ind)

                    # If fully converged, run post-relaxation check
                    if ind.status == STATUS.CONVERGED:
                        contcar = struct_dir / "CONTCAR"
                        if contcar.exists():
                            try:
                                atoms = read(str(contcar), format="vasp")
                                label, dup_id, soap_vec = classify_postrelax(
                                    atoms,
                                    soap_cache,
                                    duplicate_threshold=config.fingerprint.duplicate_threshold,
                                    r_cut=config.fingerprint.r_cut,
                                    n_max=config.fingerprint.n_max,
                                    l_max=config.fingerprint.l_max,
                                )

                                if label == "duplicate":
                                    log.info(
                                        f"  {struct_key}: duplicate of {dup_id}"
                                    )
                                    ind = ind.mark_duplicate()
                                    db.update(ind)
                                else:
                                    # Compute fitness
                                    raw_e = _read_energy(struct_dir)
                                    from gocia.science.energy import (
                                        grand_canonical_energy,
                                    )

                                    gce = grand_canonical_energy(
                                        raw_energy=raw_e,
                                        adsorbate_counts=ind.extra_data.get(
                                            "adsorbate_counts", {}
                                        ),
                                        chemical_potentials=chemical_potentials,
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
                                    log.info(
                                        f"  {struct_key}: converged "
                                        f"G={gce:.4f} eV"
                                    )
                            except Exception as e:
                                log.warning(
                                    f"  {struct_key}: post-relax check failed ({e})"
                                )
                        else:
                            log.warning(f"  {struct_key}: CONTCAR missing")

                    elif STATUS.is_terminal(ind.status):
                        total_evals += 1
                        log.info(f"  {struct_key}: {ind.status}")

                    del active_jobs[struct_key]

                    # Spawn replacement
                    new_ind = _spawn_one(run_dir, db, config, slab_info, rng, total_evals)
                    if new_ind:
                        _submit_one(new_ind, run_dir, db, scheduler, active_jobs, config)

            # ── Top up worker pool ──
            _fill_workers(run_dir, db, scheduler, active_jobs, config)

            # ── Check stop criteria ──
            n_active = len(active_jobs)
            n_pending = len(db.get_by_status(STATUS.PENDING))

            if n_active == 0 and n_pending == 0:
                if _should_stop(total_evals, stall_count, config):
                    log.info("Convergence criteria met.")
                    break

            log.info(
                f"Evals={total_evals}  Best={best_gce:.4f} eV  "
                f"Stall={stall_count}/{config.ga.max_stall_generations}  "
                f"Active={n_active}/{config.scheduler.nworkers}"
            )

            time.sleep(POLL_INTERVAL)

    log.info(f"Run complete. Total evaluations: {total_evals}")


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _build_initial_population(config, slab_info, db, run_dir, rng):
    """Build gen_000 via random placement."""
    from gocia.science.surface import load_adsorbate, place_adsorbate

    gen_dir = run_dir / "gen_000"
    gen_dir.mkdir(exist_ok=True)

    adsorbate_atoms = {}
    for ads in config.adsorbates:
        adsorbate_atoms[ads.symbol] = load_adsorbate(
            symbol=ads.symbol,
            geometry=getattr(ads, "geometry", None),
            coordinates=getattr(ads, "coordinates", None),
        )

    for i in range(config.ga.population_size):
        struct_id = f"{i:04d}"
        struct_dir = gen_dir / f"struct_{struct_id}"
        struct_dir.mkdir(exist_ok=True)

        # Random stoichiometry
        adsorbate_counts = _random_stoichiometry(
            config.adsorbates,
            rng,
            min_total=config.ga.min_adsorbates,
            max_total=config.ga.max_adsorbates,
        )

        # Place adsorbates
        current = slab_info.atoms.copy()
        try:
            for symbol, count in adsorbate_counts.items():
                ads_cfg = next(a for a in config.adsorbates if a.symbol == symbol)
                for _ in range(count):
                    current = place_adsorbate(
                        slab=current,
                        adsorbate=adsorbate_atoms[symbol],
                        zmin=slab_info.zmin,
                        zmax=slab_info.zmax,
                        n_orientations=ads_cfg.n_orientations,
                        rng=rng,
                    )
        except Exception as e:
            log.warning(f"  struct_{struct_id}: placement failed ({e})")
            first_sym = next(iter(adsorbate_atoms))
            adsorbate_counts = {first_sym: 1}
            current = slab_info.atoms.copy()
            current = place_adsorbate(
                slab=current,
                adsorbate=adsorbate_atoms[first_sym],
                zmin=slab_info.zmin,
                zmax=slab_info.zmax,
                n_orientations=1,
                rng=rng,
            )

        poscar = struct_dir / "POSCAR"
        write(str(poscar), current, format="vasp")

        ind = Individual.from_init(
            generation=0,
            geometry_path=str(poscar),
            extra_data={"adsorbate_counts": dict(adsorbate_counts)},
        )
        _write_sentinel(struct_dir, STATUS.PENDING)
        db.insert(ind)
        log.debug(f"  struct_{struct_id}: {dict(adsorbate_counts)}")

    log.info(f"Initial population created: {config.ga.population_size} structures")


def _random_stoichiometry(ads_configs, rng, min_total, max_total):
    """Draw random stoichiometry respecting bounds."""
    counts = {
        a.symbol: int(rng.integers(a.min_count, a.max_count + 1))
        for a in ads_configs
    }
    while sum(counts.values()) > max_total:
        candidates = [
            s
            for s in counts
            if counts[s] > next(a.min_count for a in ads_configs if a.symbol == s)
        ]
        if not candidates:
            break
        counts[str(rng.choice(candidates))] -= 1
    while sum(counts.values()) < min_total:
        candidates = [
            s
            for s in counts
            if counts[s] < next(a.max_count for a in ads_configs if a.symbol == s)
        ]
        if not candidates:
            break
        counts[str(rng.choice(candidates))] += 1
    return counts


def _rebuild_soap_cache(run_dir, db, soap_cache, config):
    """Populate SOAP cache from existing converged structures."""
    for ind in db.get_by_status(STATUS.CONVERGED):
        if not ind.geometry_path:
            continue
        struct_dir = Path(ind.geometry_path).parent
        contcar = struct_dir / "CONTCAR"
        if contcar.exists():
            try:
                atoms = read(str(contcar), format="vasp")
                soap_vec = compute_soap(
                    atoms,
                    r_cut=config.fingerprint.r_cut,
                    n_max=config.fingerprint.n_max,
                    l_max=config.fingerprint.l_max,
                )
                soap_cache[ind.id] = soap_vec
            except Exception as e:
                log.debug(f"  Could not load SOAP for {ind.id}: {e}")


def _spawn_one(run_dir, db, config, slab_info, rng, total_evals):
    """Spawn one offspring via crossover, mutation, or random placement."""
    from gocia.science.reproduce import splice, merge, mutate_add, mutate_remove
    from gocia.science.surface import load_adsorbate

    selectable = db.selectable_pool()

    bucket_n = max(1, (total_evals + config.ga.population_size) // config.ga.population_size)
    bucket_dir = run_dir / f"gen_{bucket_n:03d}"

    if not selectable:
        # Fallback: random placement
        return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

    try:
        op = _sample_operator(config, rng)
        n_parents = {"splice": 2, "merge": 2}.get(op, 1)

        if len(selectable) < n_parents:
            return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

        # Weighted selection
        weights = np.exp(
            -np.array([p.grand_canonical_energy or 0 for p in selectable]) / 0.1
        )
        weights /= weights.sum()
        indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
        parents = [selectable[i] for i in indices]

        # Load parent geometries
        parent_atoms = []
        for p in parents:
            if not p.geometry_path:
                return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)
            contcar = Path(p.geometry_path).parent / "CONTCAR"
            if not contcar.exists():
                return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)
            parent_atoms.append(read(str(contcar), format="vasp"))

        # Apply operator
        if op == "splice":
            children, _ = splice(
                parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng
            )
        elif op == "merge":
            children = merge(
                parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng
            )
        else:
            return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)

        # Write structure
        bucket_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(bucket_dir.glob("struct_????"))
        struct_n = len(existing) + 1
        struct_dir = bucket_dir / f"struct_{struct_n:04d}"
        struct_dir.mkdir(exist_ok=True)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), children, format="vasp")

        adsorbate_counts = dict(
            Counter(children.get_chemical_symbols()[slab_info.n_slab_atoms :])
        )
        ind = Individual.from_parents(
            generation=int(bucket_dir.name.split("_")[1]),
            parents=parents,
            operator=op,
            geometry_path=str(poscar),
            extra_data={"adsorbate_counts": adsorbate_counts},
        )
        _write_sentinel(struct_dir, STATUS.PENDING)
        db.insert(ind)
        log.debug(f"  Spawned {struct_dir.name} via {op}")
        return ind

    except Exception as e:
        log.debug(f"  Operator failed: {e} — falling back to random")
        return _place_random(run_dir, bucket_dir, db, config, slab_info, rng)


def _place_random(run_dir, bucket_dir, db, config, slab_info, rng):
    """Place a random structure."""
    from gocia.science.surface import load_adsorbate, place_adsorbate

    bucket_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(bucket_dir.glob("struct_????"))
    struct_n = len(existing) + 1
    struct_dir = bucket_dir / f"struct_{struct_n:04d}"
    struct_dir.mkdir(exist_ok=True)

    adsorbate_atoms = {}
    for ads in config.adsorbates:
        adsorbate_atoms[ads.symbol] = load_adsorbate(
            symbol=ads.symbol,
            geometry=getattr(ads, "geometry", None),
            coordinates=getattr(ads, "coordinates", None),
        )

    adsorbate_counts = _random_stoichiometry(
        config.adsorbates, rng, config.ga.min_adsorbates, config.ga.max_adsorbates
    )

    current = slab_info.atoms.copy()
    for symbol, count in adsorbate_counts.items():
        ads_cfg = next(a for a in config.adsorbates if a.symbol == symbol)
        for _ in range(count):
            current = place_adsorbate(
                slab=current,
                adsorbate=adsorbate_atoms[symbol],
                zmin=slab_info.zmin,
                zmax=slab_info.zmax,
                n_orientations=ads_cfg.n_orientations,
                rng=rng,
            )

    poscar = struct_dir / "POSCAR"
    write(str(poscar), current, format="vasp")

    ind = Individual.from_init(
        generation=int(bucket_dir.name.split("_")[1]),
        geometry_path=str(poscar),
        extra_data={"adsorbate_counts": dict(adsorbate_counts)},
    )
    _write_sentinel(struct_dir, STATUS.PENDING)
    db.insert(ind)
    log.debug(f"  Spawned {struct_dir.name} via random")
    return ind


def _sample_operator(config, rng):
    """Sample a GA operator by probability."""
    ops = [OPERATOR.SPLICE, OPERATOR.MERGE, OPERATOR.MUTATE_ADD, OPERATOR.MUTATE_REMOVE]
    probs = [0.4, 0.3, 0.2, 0.1]
    return str(rng.choice(ops, p=probs))


def _fill_workers(run_dir, db, scheduler, active_jobs, config):
    """Submit pending structures until worker pool is full."""
    n_active = len(active_jobs)
    limit = config.scheduler.nworkers

    if n_active >= limit:
        return

    pending = db.get_by_status(STATUS.PENDING)
    for ind in pending:
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
    """Submit a single structure to the scheduler."""
    if not ind.geometry_path:
        return
    struct_dir = Path(ind.geometry_path).parent
    config_path = (run_dir / "gocia.yaml").resolve()
    body = f"cd {struct_dir.resolve()}\ngocia _run-pipeline {struct_dir.resolve()} --config {config_path}\n"
    job_name = f"gocia_{struct_dir.parent.name}_{struct_dir.name}"

    try:
        job_id = scheduler.submit_structure(job_name, body, struct_dir)
    except Exception as e:
        log.warning(f"  Submit failed for {struct_dir.name}: {e}")
        return

    ind = ind.with_status(STATUS.SUBMITTED)
    db.update(ind)
    _write_sentinel(struct_dir, STATUS.SUBMITTED)
    active_jobs[_dir_to_key(struct_dir)] = job_id
    log.info(f"  Submitted {struct_dir.parent.name}/{struct_dir.name} → {job_id}")


def _count_converged(db):
    """Count converged structures."""
    return len(db.get_by_status(STATUS.CONVERGED))


def _should_stop(total_evals, stall_count, config):
    """Check stop criteria."""
    min_evals = config.ga.min_generations * config.ga.population_size
    max_evals = config.ga.max_generations * config.ga.population_size

    if total_evals < min_evals:
        return False
    if total_evals >= max_evals:
        return True
    return stall_count >= config.ga.max_stall_generations


def _stop_requested(run_dir):
    """Check if gociastop file exists."""
    return (run_dir / "gociastop").exists()


def _dir_to_key(struct_dir):
    """Create stable key from struct dir."""
    return f"{struct_dir.parent.name}/{struct_dir.name}"


def _key_to_dir(key, run_dir):
    """Reconstruct dir from key."""
    return run_dir / key


def _find_ind_for_dir(struct_dir, db):
    """Find Individual whose geometry_path is in struct_dir."""
    try:
        gen_n = int(struct_dir.parent.name.split("_")[1])
    except (ValueError, IndexError):
        return None

    for ind in db._conn.execute(
        "SELECT * FROM structures WHERE geometry_path LIKE ?",
        (f"%{struct_dir.name}%",),
    ).fetchall():
        return _row_to_individual(ind)
    return None


def _read_energy(struct_dir):
    """Read FINAL_ENERGY file."""
    ef = struct_dir / "FINAL_ENERGY"
    if ef.exists():
        try:
            return float(ef.read_text().strip())
        except ValueError:
            pass
    return float("nan")


def _write_sentinel(struct_dir, status):
    """Write status sentinel file."""
    struct_dir = Path(struct_dir)
    # Remove old sentinels
    for f in struct_dir.glob("*"):
        if f.name.upper() in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED"):
            f.unlink(missing_ok=True)
    # Write new
    (struct_dir / status.upper()).touch()


def _read_sentinel(struct_dir):
    """Read status sentinel file."""
    struct_dir = Path(struct_dir)
    for name in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED", "DUPLICATE", "DESORBED"):
        if (struct_dir / name).exists():
            return name.lower()
    return None


def _row_to_individual(row):
    """Convert DB row to Individual."""
    from database import _row_to_individual as convert

    return convert(row)
