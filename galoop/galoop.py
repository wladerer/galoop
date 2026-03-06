"""
galoop/galoop.py

Async steady-state GA loop.  Spawns and harvests structures independently.
All structures live under a single flat directory: <run_dir>/gcga/structure_NNNNN/
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ase.io import read, write

from galoop.individual import Individual, STATUS, OPERATOR
from galoop.database import GaloopDB
from galoop.fingerprint import (
    classify_postrelax, compute_soap, tanimoto_similarity,
    StructRecord, _dist_histogram, _composition, build_chem_envs,
)
from galoop.science.reproduce import splice, merge, mutate_add, mutate_remove

log = logging.getLogger(__name__)

POLL_INTERVAL = 15  # seconds between future polls


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
    Steady-state async GA loop.

    Parameters
    ----------
    config    : validated GaloopConfig
    run_dir   : root run directory
    slab_info : SlabInfo from load_slab()
    rng       : NumPy random generator
    """
    rng = rng or np.random.default_rng()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    gcga_dir = run_dir / "gcga"

    log.info("Starting async GA loop in %s", run_dir)

    chem_pots = {a.symbol: a.chemical_potential for a in config.adsorbates}

    with GaloopDB(run_dir / "galoop.db") as db:
        db.setup()

        # Build initial population if this is a fresh run
        if not gcga_dir.exists():
            log.info("Building initial population …")
            _build_initial_population(config, slab_info, db, run_dir, rng)

        active_futures: dict[str, object] = {}   # struct_key → parsl AppFuture
        struct_cache: dict[str, StructRecord] = {}
        best_gce = float("inf")
        stall_count = 0
        total_evals = len(db.get_by_status(STATUS.CONVERGED))

        log.info("Resuming with %d converged structures.  Rebuilding struct cache …", total_evals)
        _rebuild_struct_cache(run_dir, db, struct_cache, config, slab_info.n_slab_atoms)

        # Re-sync any structures left in 'submitted' from a previous session
        total_evals, best_gce, stall_count = _reconcile_submitted_orphans(
            run_dir, db, struct_cache, chem_pots, config,
            total_evals, best_gce, stall_count,
            n_slab_atoms=slab_info.n_slab_atoms,
        )

        # ── main loop ─────────────────────────────────────────────────────
        while True:
            if _stop_requested(run_dir):
                log.info("Stop file detected — exiting.")
                break

            # Harvest completed futures
            for struct_key, future in list(active_futures.items()):
                if not future.done():
                    continue

                struct_dir = _key_to_dir(struct_key, run_dir)
                ind = _find_ind_for_dir(struct_dir, db)
                if ind is None:
                    log.warning("No DB record for %s", struct_key)
                    del active_futures[struct_key]
                    continue

                # Check for execution-level exception
                try:
                    future.result()
                except Exception as exc:
                    log.warning("  %s: task raised: %s", struct_key, exc)
                    ind = ind.with_status(STATUS.FAILED)
                    db.update(ind)
                    del active_futures[struct_key]
                    total_evals += 1
                    continue

                # Sync status from filesystem sentinel
                sentinel = _read_sentinel(struct_dir)
                if sentinel and sentinel != ind.status:
                    ind = ind.with_status(sentinel)
                    db.update(ind)

                if ind.status == STATUS.CONVERGED:
                    ind, total_evals, best_gce, stall_count = _handle_converged(
                        ind, struct_dir, db, struct_cache, chem_pots,
                        config, total_evals, best_gce, stall_count,
                        n_slab_atoms=slab_info.n_slab_atoms,
                    )
                elif STATUS.is_terminal(ind.status):
                    total_evals += 1
                    log.info("  %s: %s", struct_key, ind.status)

                del active_futures[struct_key]

            # Top up worker pool
            _fill_workers(run_dir, db, active_futures, config, slab_info, rng, total_evals)

            # Convergence check
            if not active_futures and not db.get_by_status(STATUS.PENDING):
                if _should_stop(total_evals, stall_count, config):
                    log.info("Convergence criteria met.")
                    break

            log.info(
                "Evals=%d  Best=%.4f eV  Stall=%d/%d  Active=%d/%d",
                total_evals, best_gce, stall_count,
                config.ga.max_stall,
                len(active_futures), config.scheduler.nworkers,
            )
            time.sleep(POLL_INTERVAL)

    log.info("Run complete.  Total evaluations: %d", total_evals)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _handle_converged(
    ind, struct_dir, db, struct_cache, chem_pots,
    config, total_evals, best_gce, stall_count,
    n_slab_atoms: int = 0,
):
    """Post-relax classification + energy for a converged structure."""
    from galoop.science.energy import grand_canonical_energy

    contcar = struct_dir / "CONTCAR"
    if not contcar.exists():
        log.warning("  %s: CONTCAR missing after convergence", ind.id)
        return ind, total_evals, best_gce, stall_count

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
            db.update(ind)
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
            db.update(ind)
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

    gcga_dir = run_dir / "gcga"
    gcga_dir.mkdir(exist_ok=True)

    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in config.adsorbates
    }

    for i in range(config.ga.population_size):
        struct_dir = gcga_dir / f"structure_{i:05d}"
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

        poscar = struct_dir / "POSCAR"
        write(str(poscar), current, format="vasp")

        ind = Individual.from_init(
            geometry_path=str(poscar),
            extra_data={"adsorbate_counts": dict(counts)},
        )
        _write_sentinel(struct_dir, STATUS.PENDING)
        db.insert(ind)
        log.debug("structure_%05d: %s", i, dict(counts))

    log.info("Initial population: %d structures", config.ga.population_size)


# ── offspring spawning ────────────────────────────────────────────────────

def _next_struct_dir(gcga_dir: Path) -> Path:
    """Create and return the next available structure directory."""
    gcga_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(gcga_dir.glob("structure_*")))
    struct_dir = gcga_dir / f"structure_{n:05d}"
    struct_dir.mkdir(exist_ok=True)
    return struct_dir


def _spawn_one(run_dir, db, config, slab_info, rng):
    gcga_dir = run_dir / "gcga"
    selectable = db.selectable_pool()

    if not selectable:
        return _place_random(run_dir, db, config, slab_info, rng)

    try:
        op = _sample_operator(rng)
        n_parents = 2 if op in (OPERATOR.SPLICE, OPERATOR.MERGE) else 1

        if len(selectable) < n_parents:
            return _place_random(run_dir, db, config, slab_info, rng)

        # Boltzmann-weighted selection (shift by min for numerical stability)
        energies = np.array([p.grand_canonical_energy or 0.0 for p in selectable])
        shifted = energies - energies.min()
        weights = np.exp(-shifted / 0.1)
        weights /= weights.sum()
        indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
        parents = [selectable[i] for i in indices]

        parent_atoms = []
        for p in parents:
            if not p.geometry_path:
                return _place_random(run_dir, db, config, slab_info, rng)
            contcar = Path(p.geometry_path).parent / "CONTCAR"
            if not contcar.exists():
                return _place_random(run_dir, db, config, slab_info, rng)
            parent_atoms.append(read(str(contcar), format="vasp"))

        if op == OPERATOR.SPLICE:
            child, _ = splice(parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng)
        elif op == OPERATOR.MERGE:
            child = merge(parent_atoms[0], parent_atoms[1], slab_info.n_slab_atoms, rng=rng)
        elif op == OPERATOR.MUTATE_ADD:
            from galoop.science.surface import load_adsorbate, place_adsorbate

            parent_counts = parents[0].extra_data.get("adsorbate_counts", {})
            if sum(parent_counts.values()) >= config.ga.max_adsorbates:
                return _place_random(run_dir, db, config, slab_info, rng)

            addable = [
                a.symbol for a in config.adsorbates
                if parent_counts.get(a.symbol, 0) < a.max_count
            ]
            if not addable:
                return _place_random(run_dir, db, config, slab_info, rng)
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
                return _place_random(run_dir, db, config, slab_info, rng)
            child = result
        else:
            return _place_random(run_dir, db, config, slab_info, rng)

        # Reject offspring that violate total adsorbate bounds
        if op in (OPERATOR.SPLICE, OPERATOR.MERGE):
            trial_counts = _infer_adsorbate_counts(
                child.get_chemical_symbols()[slab_info.n_slab_atoms:],
                config.adsorbates,
            )
            total = sum(trial_counts.values())
            if total > config.ga.max_adsorbates or total < config.ga.min_adsorbates:
                return _place_random(run_dir, db, config, slab_info, rng)

        struct_dir = _next_struct_dir(gcga_dir)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), child, format="vasp")

        ads_counts = _infer_adsorbate_counts(
            child.get_chemical_symbols()[slab_info.n_slab_atoms:],
            config.adsorbates,
        )
        ind = Individual.from_parents(
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
        return _place_random(run_dir, db, config, slab_info, rng)


def _place_random(run_dir, db, config, slab_info, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    gcga_dir = run_dir / "gcga"
    struct_dir = _next_struct_dir(gcga_dir)

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

    poscar = struct_dir / "POSCAR"
    write(str(poscar), current, format="vasp")

    ind = Individual.from_init(
        geometry_path=str(poscar),
        extra_data={"adsorbate_counts": dict(counts)},
    )
    _write_sentinel(struct_dir, STATUS.PENDING)
    db.insert(ind)
    log.debug("Spawned %s via random", struct_dir.name)
    return ind


# ── job submission ────────────────────────────────────────────────────────

def _fill_workers(run_dir, db, active_futures, config, slab_info, rng, total_evals):
    n_active = len(active_futures)
    limit = config.scheduler.nworkers

    if n_active >= limit:
        return

    # Submit existing PENDING structures before spawning any offspring
    for ind in db.get_by_status(STATUS.PENDING):
        if n_active >= limit:
            return
        if not ind.geometry_path:
            continue
        struct_dir = Path(ind.geometry_path).resolve().parent
        if not struct_dir.exists():
            ind = ind.with_status(STATUS.FAILED)
            db.update(ind)
            continue
        key = _dir_to_key(struct_dir)
        if key in active_futures:
            continue
        _submit_one(ind, run_dir, db, active_futures, config)
        n_active += 1

    # Only spawn new offspring once the PENDING pool is drained
    while n_active < limit:
        new_ind = _spawn_one(run_dir, db, config, slab_info, rng)
        if new_ind is None:
            break
        _submit_one(new_ind, run_dir, db, active_futures, config)
        n_active += 1


def _submit_one(ind, run_dir, db, active_futures, config):
    from galoop.engine.scheduler import relax_structure

    if not ind.geometry_path:
        return
    struct_dir = Path(ind.geometry_path).parent
    config_path = (run_dir / "galoop.yaml").resolve()

    try:
        future = relax_structure(
            str(struct_dir.resolve()),
            str(config_path),
            stdout=str(struct_dir / "parsl.out"),
            stderr=str(struct_dir / "parsl.err"),
        )
    except Exception as exc:
        log.warning("Submit failed for %s: %s", struct_dir.name, exc)
        return

    ind = ind.with_status(STATUS.SUBMITTED)
    db.update(ind)
    _write_sentinel(struct_dir, STATUS.SUBMITTED)
    active_futures[_dir_to_key(struct_dir)] = future
    log.info("Submitted %s", struct_dir.name)


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


def _reconcile_submitted_orphans(run_dir, db, struct_cache, chem_pots, config,
                                  total_evals, best_gce, stall_count,
                                  n_slab_atoms: int = 0):
    """
    On resume, any structure still in 'submitted' state is an orphan — parsl
    no longer knows about it.  Re-read the filesystem sentinel and:
      - CONVERGED  → run through _handle_converged (compute GCE, dup-check)
      - FAILED / DESORBED / DUPLICATE → update DB status
      - anything else → reset to PENDING so it gets resubmitted
    """
    orphans = db.get_by_status(STATUS.SUBMITTED)
    if not orphans:
        return total_evals, best_gce, stall_count

    log.info("Reconciling %d submitted orphan(s) from previous session …", len(orphans))
    for ind in orphans:
        if not ind.geometry_path:
            continue
        struct_dir = Path(ind.geometry_path).parent
        sentinel = _read_sentinel(struct_dir)
        if sentinel == STATUS.CONVERGED:
            ind, total_evals, best_gce, stall_count = _handle_converged(
                ind, struct_dir, db, struct_cache, chem_pots,
                config, total_evals, best_gce, stall_count,
                n_slab_atoms=n_slab_atoms,
            )
        elif sentinel in (STATUS.FAILED, STATUS.DESORBED, STATUS.DUPLICATE):
            ind = ind.with_status(sentinel)
            db.update(ind)
            total_evals += 1
            log.debug("  Reconciled %s → %s", struct_dir.name, sentinel)
        else:
            ind = ind.with_status(STATUS.PENDING)
            _write_sentinel(struct_dir, STATUS.PENDING)
            db.update(ind)
            log.debug("  Reset %s → pending (orphaned submitted)", struct_dir.name)

    return total_evals, best_gce, stall_count


def _rebuild_struct_cache(run_dir, db, struct_cache, config, n_slab_atoms: int = 0):
    for ind in db.get_by_status(STATUS.CONVERGED):
        if not ind.geometry_path:
            continue
        contcar = Path(ind.geometry_path).parent / "CONTCAR"
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
            log.debug("Could not rebuild record for %s: %s", ind.id, exc)


def _should_stop(total_evals, stall_count, config):
    if total_evals < config.ga.min_structures:
        return False
    if total_evals >= config.ga.max_structures:
        return True
    return stall_count >= config.ga.max_stall


def _stop_requested(run_dir):
    return (Path(run_dir) / "gociastop").exists()


def _dir_to_key(struct_dir):
    # e.g. "gcga/structure_00000"
    return f"{struct_dir.parent.name}/{struct_dir.name}"


def _key_to_dir(key, run_dir):
    return Path(run_dir) / key


def _find_ind_for_dir(struct_dir, db):
    return db.find_by_geometry_path_substring(
        f"{struct_dir.parent.name}/{struct_dir.name}"
    )


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


def _infer_adsorbate_counts(
    element_symbols: list[str],
    adsorbate_configs,
) -> dict[str, int]:
    """
    Reconstruct molecular-level {symbol: count} from a flat element list.

    Greedy, largest-formula-first, so 'OOH' is matched before 'O'.
    Leftover atoms that don't complete a molecule are silently discarded.
    """
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
    """Return the highest Tanimoto similarity between soap_vec and anything in cache."""
    if not struct_cache:
        return 0.0
    return max(tanimoto_similarity(soap_vec, rec.soap_vector)
               for rec in struct_cache.values())
