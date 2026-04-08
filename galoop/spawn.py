"""
galoop/spawn.py

Everything related to producing new candidate structures: the initial
population, the operator dispatch (`spawn_one`), the GPR-guided spawner,
the random fallback, and the worker-pool refill loop that hands work to Parsl.

Split out of galoop/galoop.py during the Phase 2 refactor.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from ase import Atoms
from ase.io import write

from galoop.harvest import is_prerelax_duplicate
from galoop.individual import OPERATOR, STATUS, Individual
from galoop.science.reproduce import (
    merge,
    mutate_displace,
    mutate_rattle_slab,
    mutate_remove,
    mutate_translate,
    splice,
)
from galoop.science.surface import (
    build_random_structure,
    load_ads_template_dict,
    read_atoms,
)

if TYPE_CHECKING:
    from galoop.store import GaloopStore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

def build_initial_population(config, slab_info, store: GaloopStore, rng) -> None:
    """Spawn the initial GA population, stratified across coverage levels.

    Honors the ``galoopstop`` sentinel file between structures for fast
    abandonment: if the user touches galoopstop mid-init-pop, we stop at the
    next iteration boundary and return whatever was built so far. Structures
    whose ``snap_to_surface`` BFGS call exceeds ``config.slab.snap_timeout_s``
    are skipped and not inserted into the store (so the population may end up
    smaller than ``population_size``, which is fine — the main loop proceeds
    with whatever it gets).
    """
    from galoop.loop import _stop_requested

    run_dir = store.run_dir

    # Resolve the snap backend once, up front — users pay the model-load cost
    # a single time regardless of population_size. Defaults to the first
    # calculator stage's backend/params if config.snap_stage is unset.
    snap_stage_cfg = _resolve_snap_stage(config)

    ads_atoms = load_ads_template_dict(config.adsorbates)

    # Stratified seeding: cycle target totals across [min_adsorbates, max_adsorbates]
    # so the initial population spans every coverage level instead of biasing
    # random draws toward the middle of the range.
    lo, hi = config.ga.min_adsorbates, config.ga.max_adsorbates
    coverage_levels = list(range(lo, hi + 1))
    built = 0
    for i in range(config.ga.population_size):
        if _stop_requested(run_dir):
            log.info(
                "Stop requested during init-pop at structure %d/%d — abandoning",
                i, config.ga.population_size,
            )
            break

        target_total = coverage_levels[i % len(coverage_levels)]
        counts = random_stoichiometry(
            config.adsorbates, rng, target_total, target_total,
        )

        try:
            current = build_random_structure(
                slab_info, config.adsorbates, ads_atoms, counts, rng,
            )
        except Exception as exc:
            log.warning("structure_%05d: placement failed (%s) — falling back", i, exc)
            first = next(iter(ads_atoms))
            counts = {first: 1}
            current = build_random_structure(
                slab_info, config.adsorbates, ads_atoms, counts, rng,
            )

        # Quick constrained pre-relax: let adsorbates settle toward surface.
        # Wrapped in a per-structure wall-clock timeout so a single hung MACE
        # BFGS call can't freeze the whole init-pop phase (see FINDINGS.md
        # cycle 4a).
        try:
            current = snap_to_surface(
                current, config, slab_info.n_slab_atoms,
                snap_stage=snap_stage_cfg,
            )
        except SnapTimeoutError as exc:
            log.warning(
                "structure_%05d: snap_to_surface timed out after %.1fs — skipping",
                i, exc.timeout_s,
            )
            continue

        ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
        struct_dir = store.insert(ind)
        poscar = struct_dir / "POSCAR"
        write(str(poscar), current, format="vasp")
        ind.geometry_path = str(poscar)
        store.update(ind)
        built += 1
        log.debug("initial structure %05d: %s", i, dict(counts))

    log.info("Initial population: %d structures", built)


class SnapTimeoutError(RuntimeError):
    """Raised when snap_to_surface's BFGS call exceeds the per-structure timeout."""

    def __init__(self, timeout_s: float):
        super().__init__(f"snap_to_surface exceeded {timeout_s:.1f}s timeout")
        self.timeout_s = timeout_s


import contextlib
import signal
import threading


@contextlib.contextmanager
def _snap_timeout(timeout_s: float):
    """SIGALRM-based wall-clock timeout for snap_to_surface's BFGS call.

    Only active on the main thread of the main interpreter — SIGALRM cannot
    be delivered elsewhere. On non-main threads (e.g. under pytest-xdist or
    from a worker) this silently no-ops rather than raising, since the
    expected deployment path is synchronous init-pop on the main thread.
    """
    on_main_thread = threading.current_thread() is threading.main_thread()
    if not on_main_thread or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):
        raise SnapTimeoutError(timeout_s)

    # itimer lets us pass float seconds; SIGALRM's plain alarm() is integer-only.
    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Snap-to-surface (cheap constrained pre-relax)
# ---------------------------------------------------------------------------

def _resolve_snap_stage(config) -> dict:
    """Return a stage-config dict for snap_to_surface.

    If ``config.snap_stage`` is set, use it. Otherwise default to the first
    entry in ``config.calculator_stages`` with fmax/max_steps overridden to
    the snap defaults (0.2 / 30). This lets users opt in to a cheaper
    snap-time backend without forcing them to duplicate config.
    """
    if config.snap_stage is not None:
        return config.snap_stage.model_dump()
    first = config.calculator_stages[0]
    d = first.model_dump()
    d["fmax"] = 0.2
    d["max_steps"] = 30
    # Disable prescan for snap — snap itself IS the prescan.
    d["fix_slab_first"] = False
    d["prescan_fmax"] = None
    return d


def snap_to_surface(
    atoms: Atoms,
    config,
    n_slab_atoms: int,
    snap_stage: dict | None = None,
) -> Atoms:
    """Quick constrained pre-relax to settle adsorbates toward the surface.

    Fixes the slab and runs a short BFGS so adsorbates fall into reasonable
    binding positions. Then clamps adsorbate z-coordinates to a safe window
    (0.8–4.0 Å above slab top) to prevent burial or desorption.

    Routes through :mod:`galoop.engine.backends` so any registered backend
    (MACE, fairchem, Orb, user's own MLIP via import path) can drive snap.
    ``snap_stage`` is a resolved stage-config dict; if ``None``, it's
    derived from ``config.snap_stage`` or the first calculator_stages entry.
    """
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    from galoop.engine import backends

    work = atoms.copy()

    ads_indices = list(range(n_slab_atoms, len(work)))
    if not ads_indices:
        return work

    import contextlib

    if snap_stage is None:
        snap_stage = _resolve_snap_stage(config)

    factory, _drives = backends.resolve(snap_stage["type"])
    calc = factory(dict(snap_stage.get("params", {})))
    # Drop any stale results from a prior atoms-of-different-size call.
    with contextlib.suppress(AttributeError):
        calc.results.clear()

    work.calc = calc
    work.set_constraint(FixAtoms(indices=list(range(n_slab_atoms))))

    # Short BFGS: adsorbates relax toward surface binding sites.
    # Guarded by a SIGALRM-based wall-clock timeout so a single hung MACE
    # forward pass (e.g. CUDA stall) can't freeze init-pop. The timeout
    # default (120s) assumes MACE; raise config.slab.snap_timeout_s if you
    # ever point snap at DFT.
    timeout_s = float(getattr(config.slab, "snap_timeout_s", 120.0))
    try:
        with _snap_timeout(timeout_s):
            dyn = BFGS(work, logfile=None)
            dyn.run(fmax=0.2, steps=30)
    except SnapTimeoutError:
        # Propagate to caller so build_initial_population can skip this
        # structure entirely rather than silently returning clamp-only
        # geometry for something that hung.
        raise
    except Exception as exc:
        # If BFGS fails, still clamp z and return — but log so we can spot
        # systemic failures (e.g. CUDA OOM) rather than silently degrading
        # to clamp-only initial geometries.
        log.warning("snap_to_surface BFGS failed (%s) — falling back to z-clamp only", exc)

    # Clamp adsorbate z to safe window above slab. Offsets come from
    # SlabConfig (default 0.8–4.0 Å) so users can tune for unusual systems.
    pos = work.get_positions()
    slab_top_z = pos[:n_slab_atoms, 2].max()
    z_min = slab_top_z + config.slab.snap_z_min_offset
    z_max = slab_top_z + config.slab.snap_z_max_offset
    for idx in ads_indices:
        pos[idx, 2] = np.clip(pos[idx, 2], z_min, z_max)
    work.set_positions(pos)

    # Strip calculator and constraints for clean POSCAR
    result = work.copy()
    result.calc = None
    result.set_constraint(atoms.constraints)
    return result


# ---------------------------------------------------------------------------
# Worker pool refill
# ---------------------------------------------------------------------------

def fill_workers(
    store: GaloopStore,
    config,
    slab_info,
    rng,
    total_evals: int,
    active_futures: dict,
    relax_fn: Callable,
    stage_configs: list[dict],
    n_slab_atoms: int,
    pair_counts: dict | None = None,
    struct_cache: dict | None = None,
    gpr=None,
) -> bool:
    """Spawn offspring until the worker pool is full, submit via Parsl.

    Returns True if fill_workers produced a healthy batch, False if spawn
    was exhausted — i.e. hit max_attempts before the pool was filled, meaning
    most candidates were rejected as pre-relax duplicates or operator failures.
    A "trickle" (submitted 1 slot but couldn't fill the rest) counts as
    exhaustion so that spawn_stall_count in the main loop can accumulate and
    eventually terminate runs stuck in a high-duplicate-rate regime.
    """
    limit = config.scheduler.nworkers
    max_attempts = limit * 5  # avoid infinite loop
    attempts = 0
    submitted = 0
    exhausted = False

    while len(active_futures) < limit:
        if total_evals >= config.ga.max_structures:
            break
        if attempts >= max_attempts:
            log.debug("Hit max spawn attempts (%d), moving on", max_attempts)
            exhausted = True
            break

        # GPR-guided spawning: with gpr_fraction probability, use GPR
        use_gpr = (
            gpr is not None
            and gpr.is_ready
            and total_evals >= config.ga.gpr_min_samples
            and rng.random() < config.ga.gpr_fraction
        )

        if use_gpr:
            result = spawn_via_gpr(gpr, store, config, slab_info, rng)
        else:
            result = spawn_one(store, config, slab_info, rng, pair_counts)

        if result is None:
            attempts += 1
            continue
        new_ind, child_atoms = result

        # Pre-relaxation duplicate check: SOAP similarity on unrelaxed structure
        if struct_cache and is_prerelax_duplicate(
            child_atoms, struct_cache, n_slab_atoms,
            threshold=config.fingerprint.prerelax_duplicate_threshold,
            r_cut=config.fingerprint.r_cut,
            n_max=config.fingerprint.n_max,
            l_max=config.fingerprint.l_max,
        ):
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
            n_slab_atoms=n_slab_atoms,
        )
        active_futures[new_ind.id] = fut
        submitted += 1

    # Treat "hit max attempts before filling pool" as a stall signal, even if
    # one or two slots trickled through. This lets the main loop's
    # spawn_stall_count increment and eventually terminate runs where the
    # duplicate rate is overwhelming forward progress.
    return not exhausted and submitted > 0


# ---------------------------------------------------------------------------
# Operator dispatch table
# ---------------------------------------------------------------------------
#
# Each operator is a small function that takes (parents, parent_atoms, config,
# slab_info, rng) and returns the child Atoms or None to signal failure.
# This replaces the long if/elif ladder that lived in spawn_one.

def _op_splice(parents, parent_atoms, config, slab_info, rng):
    from galoop.science.surface import check_clash

    child, _ = splice(parent_atoms[0], parent_atoms[1],
                      slab_info.n_slab_atoms, rng=rng)
    if check_clash(child, n_slab=slab_info.n_slab_atoms, scale=0.7):
        return None
    return child


def _op_merge(parents, parent_atoms, config, slab_info, rng):
    return merge(parent_atoms[0], parent_atoms[1],
                 slab_info.n_slab_atoms, rng=rng)


def _op_mutate_add(parents, parent_atoms, config, slab_info, rng):
    from galoop.science.surface import load_adsorbate, place_adsorbate

    parent_counts = parents[0].extra_data.get("adsorbate_counts", {})
    if sum(parent_counts.values()) >= config.ga.max_adsorbates:
        return None  # already at max

    addable = [
        a.symbol for a in config.adsorbates
        if parent_counts.get(a.symbol, 0) < a.max_count
    ]
    if not addable:
        return None
    sym = str(rng.choice(addable))
    ads_cfg = next(a for a in config.adsorbates if a.symbol == sym)
    ads_mol = load_adsorbate(
        symbol=sym,
        geometry=getattr(ads_cfg, "geometry", None),
        coordinates=getattr(ads_cfg, "coordinates", None),
    )
    return place_adsorbate(
        parent_atoms[0], ads_mol,
        slab_info.zmin, slab_info.zmax,
        binding_index=ads_cfg.binding_index,
        rng=rng,
        n_slab_atoms=slab_info.n_slab_atoms,
    )


def _op_mutate_remove(parents, parent_atoms, config, slab_info, rng):
    return mutate_remove(parent_atoms[0], slab_info.n_slab_atoms, rng=rng)


def _op_mutate_displace(parents, parent_atoms, config, slab_info, rng):
    return mutate_displace(
        parent_atoms[0], slab_info.n_slab_atoms,
        displacement=config.ga.displace_amplitude, rng=rng,
    )


def _op_mutate_rattle_slab(parents, parent_atoms, config, slab_info, rng):
    return mutate_rattle_slab(
        parent_atoms[0], slab_info.n_slab_atoms,
        amplitude=config.ga.rattle_amplitude, rng=rng,
    )


def _op_mutate_translate(parents, parent_atoms, config, slab_info, rng):
    return mutate_translate(
        parent_atoms[0], slab_info.n_slab_atoms,
        displacement=config.ga.translate_amplitude, rng=rng,
    )


_OPERATOR_DISPATCH: dict[str, Callable] = {
    OPERATOR.SPLICE: _op_splice,
    OPERATOR.MERGE: _op_merge,
    OPERATOR.MUTATE_ADD: _op_mutate_add,
    OPERATOR.MUTATE_REMOVE: _op_mutate_remove,
    OPERATOR.MUTATE_DISPLACE: _op_mutate_displace,
    OPERATOR.MUTATE_RATTLE_SLAB: _op_mutate_rattle_slab,
    OPERATOR.MUTATE_TRANSLATE: _op_mutate_translate,
}

_PRESERVE_PARENT_COUNTS_OPS = frozenset({
    OPERATOR.MUTATE_DISPLACE,
    OPERATOR.MUTATE_RATTLE_SLAB,
    OPERATOR.MUTATE_TRANSLATE,
})

_TWO_PARENT_OPS = frozenset({OPERATOR.SPLICE, OPERATOR.MERGE})


def _select_parents(selectable, n_parents: int, config, rng, pair_counts):
    """Boltzmann-weighted parent draw with a one-shot over-mating penalty."""
    energies = np.array([p.grand_canonical_energy or 0.0 for p in selectable])
    shifted = energies - energies.min()
    weights = np.exp(-shifted / config.ga.boltzmann_temperature)
    weights /= weights.sum()
    indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
    parents = [selectable[i] for i in indices]

    if n_parents == 2 and pair_counts is not None:
        pair_key = frozenset(p.id for p in parents)
        if pair_counts.get(pair_key, 0) >= 1:
            indices = rng.choice(len(selectable), size=n_parents, replace=False, p=weights)
            parents = [selectable[i] for i in indices]
    return parents


def spawn_one(store: GaloopStore, config, slab_info, rng,
              pair_counts: dict | None = None):
    """Create one offspring. Returns (Individual, Atoms) or None.

    Returns None when the operator fails — the caller should retry.
    Only falls back to random placement when there are no selectable parents.
    """
    selectable = store.selectable_pool()

    if not selectable:
        return spawn_random(store, config, slab_info, rng)

    op: str | None = None
    try:
        op = sample_operator(rng, config)
        n_parents = 2 if op in _TWO_PARENT_OPS else 1

        if len(selectable) < n_parents:
            return spawn_random(store, config, slab_info, rng)

        parents = _select_parents(selectable, n_parents, config, rng, pair_counts)

        parent_atoms = []
        for p in parents:
            struct_dir = store.individual_dir(p.id)
            contcar = struct_dir / "CONTCAR"
            if not contcar.exists():
                return None  # parent geometry missing
            parent_atoms.append(read_atoms(contcar, format="vasp"))

        handler = _OPERATOR_DISPATCH.get(op)
        if handler is None:
            return None
        child = handler(parents, parent_atoms, config, slab_info, rng)
        if child is None:
            return None

        # Record pair usage
        if n_parents == 2 and pair_counts is not None:
            pair_key = frozenset(p.id for p in parents)
            pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

        # Bounds check for crossover operators
        if op in _TWO_PARENT_OPS:
            trial_counts = infer_adsorbate_counts(
                child.get_chemical_symbols()[slab_info.n_slab_atoms:],
                config.adsorbates,
            )
            total = sum(trial_counts.values())
            if total > config.ga.max_adsorbates or total < config.ga.min_adsorbates:
                return None

        if op in _PRESERVE_PARENT_COUNTS_OPS:
            ads_counts = parents[0].extra_data.get("adsorbate_counts", {})
        else:
            ads_counts = infer_adsorbate_counts(
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
        log.debug("Operator %s failed: %s", op or "?", exc)
        return None


# ---------------------------------------------------------------------------
# Random + GPR fallback spawners
# ---------------------------------------------------------------------------

def spawn_random(store: GaloopStore, config, slab_info, rng):
    """Generate a random structure. Returns (Individual, Atoms)."""
    ads_atoms = load_ads_template_dict(config.adsorbates)
    counts = random_stoichiometry(
        config.adsorbates, rng,
        config.ga.min_adsorbates, config.ga.max_adsorbates,
    )

    current = build_random_structure(
        slab_info, config.adsorbates, ads_atoms, counts, rng,
    )

    ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
    store.insert(ind)
    log.debug("Spawned %s via random", ind.id)
    return ind, current


def spawn_via_gpr(gpr, store: GaloopStore, config, slab_info, rng):
    """Generate a structure with a GPR-suggested composition.

    Returns (Individual, Atoms) or None.
    """
    counts = gpr.suggest(rng, kappa=config.ga.gpr_kappa)

    if sum(counts.values()) == 0:
        return None

    counts = {sym: cnt for sym, cnt in counts.items() if cnt > 0}
    ads_atoms = load_ads_template_dict(config.adsorbates)

    try:
        current = build_random_structure(
            slab_info, config.adsorbates, ads_atoms, counts, rng,
        )
    except Exception as exc:
        log.debug("GPR spawn placement failed: %s", exc)
        return None

    ind = Individual.from_init(extra_data={"adsorbate_counts": dict(counts)})
    ind.operator = OPERATOR.GPR
    store.insert(ind)
    log.debug("Spawned %s via GPR: %s", ind.id, counts)
    return ind, current


def retrain_gpr(gpr, store: GaloopStore) -> None:
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


# ---------------------------------------------------------------------------
# Small helpers (kept here because they're spawn-time concerns)
# ---------------------------------------------------------------------------

def random_stoichiometry(ads_configs, rng, min_total: int, max_total: int) -> dict[str, int]:
    """Sample per-species adsorbate counts within [min_total, max_total]."""
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


def sample_operator(rng, config) -> str:
    """Pick a GA operator weighted by config.ga.operator_weights."""
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


def infer_adsorbate_counts(element_symbols, adsorbate_configs) -> dict[str, int]:
    """Reconstruct {symbol: count} from a slab+adsorbate atoms list.

    Used after crossover/merge operators to figure out which adsorbate
    species are present in the child.
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
