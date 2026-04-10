"""
galoop/cli.py

Command-line interface.

Usage
-----
galoop run --config galoop.yaml   # Start / resume the GA loop
galoop sample -c galoop.yaml -n 200 -o samples.xyz   # Generate N unique structures
galoop status -d .                # Print structure counts
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import numpy as np

from galoop.config import load_config
from galoop.store import GaloopStore


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """galoop — lean genetic algorithm for adsorbate structure search."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--seed", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True)
def run(config: str, run_dir: str, seed: int | None, verbose: bool):
    """Start or resume the GA loop."""
    from galoop.galoop import run as run_ga
    from galoop.science.surface import load_slab_from_config

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    log = logging.getLogger(__name__)

    config_path = Path(config)
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    # Clean up stale stop file from previous run
    stop_file = run_dir_path / "galoopstop"
    if stop_file.exists():
        log.info("Removing stale stop file from previous run")
        stop_file.unlink()

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        log.error("Config validation failed: %s", exc)
        sys.exit(1)

    try:
        # Auto-calibrate missing energies before loading slab
        needs_cal = (
            cfg.slab.energy is None
            or any(a.chemical_potential is None for a in cfg.adsorbates)
        )
        if needs_cal:
            from galoop.calibrate import calibrate
            calibrate(cfg, run_dir=run_dir_path)

        slab_info = load_slab_from_config(cfg.slab)

        store = GaloopStore(run_dir_path)
        current_cfg = cfg.model_dump()
        diffs = store.diff_config(current_cfg)
        if diffs:
            log.warning("Config has changed since the last run:")
            for d in diffs:
                tag = "  [ENERGY CRITICAL]" if d["energy_critical"] else ""
                log.warning("  %-55s  %s  →  %s%s",
                            d["field"], d["old"], d["new"], tag)
        store.save_config_snapshot(current_cfg)
        store.close()

        rng = np.random.default_rng(seed)
        run_ga(cfg, run_dir_path, slab_info, rng)

    except Exception:
        log.exception("Fatal error in GA loop")
        sys.exit(1)


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", default="galoop.yaml",
              type=click.Path(exists=True),
              help="Path to galoop.yaml")
@click.option("--run-dir", "-d", default=".", type=click.Path(),
              help="Where to cache the calibration work (CONTCARs, ref energies)")
@click.option("--write-yaml", is_flag=True,
              help="After calibration, write slab.energy and each adsorbate's "
                   "chemical_potential back into the source yaml in place. "
                   "Subsequent `galoop run` will skip calibration.")
@click.option("--force", is_flag=True,
              help="Re-run calibration even if all values are already set.")
@click.option("--verbose", "-v", is_flag=True)
def calibrate(config: str, run_dir: str, write_yaml: bool, force: bool,
              verbose: bool):
    """Compute slab energy + adsorbate chemical potentials for a config.

    By default, calibration runs every time `galoop run` starts if any
    reference is missing. Calling this subcommand once with ``--write-yaml``
    persists the computed values into the yaml, after which `galoop run`
    will skip re-calibration.
    """
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    log = logging.getLogger(__name__)

    config_path = Path(config)
    run_dir_path = Path(run_dir)
    run_dir_path.mkdir(parents=True, exist_ok=True)

    try:
        cfg = load_config(str(config_path))
    except Exception as exc:
        log.error("Config validation failed: %s", exc)
        sys.exit(1)

    has_all = (
        cfg.slab.energy is not None
        and all(a.chemical_potential is not None for a in cfg.adsorbates)
    )
    if has_all and not force:
        log.info("Config already has slab.energy and every chemical_potential set.")
        log.info("  slab.energy = %.6f eV", cfg.slab.energy)
        for a in cfg.adsorbates:
            log.info("  mu(%s) = %.6f eV", a.symbol, a.chemical_potential)
        log.info("Nothing to do. Use --force to re-run.")
        return

    if force:
        cfg.slab.energy = None
        for a in cfg.adsorbates:
            a.chemical_potential = None

    from galoop.calibrate import calibrate as _calibrate
    try:
        results = _calibrate(cfg, run_dir=run_dir_path)
    except Exception:
        log.exception("Calibration failed")
        sys.exit(1)

    log.info("Calibration results:")
    log.info("  slab.energy = %.6f eV", cfg.slab.energy)
    for a in cfg.adsorbates:
        log.info("  mu(%s) = %.6f eV", a.symbol, a.chemical_potential)

    if write_yaml:
        try:
            _write_calibrated_yaml(config_path, cfg)
        except Exception:
            log.exception("Failed to write yaml; calibration results are in "
                          "%s/reference_energies.txt", run_dir_path / "calibration")
            sys.exit(1)
        log.info("Wrote calibrated values back to %s", config_path)
    else:
        log.info("Use --write-yaml to persist these values into %s", config_path)


def _write_calibrated_yaml(yaml_path: Path, cfg) -> None:
    """Splice the calibrated slab.energy and per-adsorbate chemical_potential
    into an existing yaml file, preserving the rest of its structure.

    Comments and key order are preserved as best as possible via PyYAML's
    safe_load/safe_dump pair. This is not a comment-preserving round trip —
    if that becomes important, swap for ruamel.yaml.
    """
    import yaml

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if cfg.slab.energy is not None:
        data.setdefault("slab", {})
        data["slab"]["energy"] = float(cfg.slab.energy)

    ads_by_sym = {a.symbol: a for a in cfg.adsorbates}
    for entry in data.get("adsorbates", []):
        sym = entry.get("symbol")
        a = ads_by_sym.get(sym)
        if a is not None and a.chemical_potential is not None:
            entry["chemical_potential"] = float(a.chemical_potential)

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Sweep yaml: a list of run directories under 'runs:'")
@click.option("--stop-on-failure", is_flag=True,
              help="Abort the whole sweep if any single run raises.")
@click.option("--verbose", "-v", is_flag=True)
def sweep(config: str, stop_on_failure: bool, verbose: bool):
    """Run a chained sequence of GA campaigns.

    The sweep yaml lists run directories; each must contain its own
    ``galoop.yaml``. Each campaign runs in the SAME Python process
    sequentially — Parsl is initialised, torn down, and re-initialised
    around each run, so backend caches (e.g. MLIP model weights) can
    survive across runs within the same session for speed.

    Example sweep.yaml:

    \b
        runs:
          - runs/cu111_co_camp
          - runs/cu100_co_camp
          - runs/cu211_co_camp
        stop_on_failure: false    # optional, default false
    """
    import time

    import yaml

    from galoop.galoop import run as run_ga
    from galoop.science.surface import load_slab_from_config

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    log = logging.getLogger(__name__)

    with open(config) as f:
        sweep_cfg = yaml.safe_load(f) or {}
    runs = sweep_cfg.get("runs", [])
    if not runs:
        log.error("Sweep config has no 'runs:' list")
        sys.exit(1)
    stop_on_failure = stop_on_failure or bool(sweep_cfg.get("stop_on_failure", False))

    summary = []
    t0_sweep = time.time()

    for idx, run_dir_str in enumerate(runs, start=1):
        run_dir = Path(run_dir_str).resolve()
        yaml_path = run_dir / "galoop.yaml"
        log.info("=" * 60)
        log.info("sweep [%d/%d]  %s", idx, len(runs), run_dir)
        log.info("=" * 60)

        t0 = time.time()
        status_str = "ok"
        error_msg = ""

        if not yaml_path.exists():
            log.error("  no galoop.yaml in %s", run_dir)
            summary.append((run_dir.name, "missing-config", 0.0, ""))
            if stop_on_failure:
                break
            continue

        try:
            cfg = load_config(str(yaml_path))
            needs_cal = (
                cfg.slab.energy is None
                or any(a.chemical_potential is None for a in cfg.adsorbates)
            )
            if needs_cal:
                log.info("  auto-calibrating (consider `galoop calibrate "
                         "--write-yaml` to persist)")
                from galoop.calibrate import calibrate as _calibrate
                _calibrate(cfg, run_dir=run_dir)

            slab_info = load_slab_from_config(cfg.slab)
            rng = np.random.default_rng()
            run_ga(cfg, run_dir, slab_info, rng)
        except Exception as exc:
            log.exception("  run failed")
            status_str = "failed"
            error_msg = f"{type(exc).__name__}: {exc}"
            summary.append((run_dir.name, status_str, time.time() - t0, error_msg))
            if stop_on_failure:
                break
            continue

        summary.append((run_dir.name, status_str, time.time() - t0, ""))

    total = time.time() - t0_sweep
    log.info("=" * 60)
    log.info("sweep complete  total %.1fs  (%d/%d ok)",
             total,
             sum(1 for _, s, _, _ in summary if s == "ok"),
             len(summary))
    for name, s, dt, err in summary:
        log.info("  %-30s  %-8s  %6.1fs  %s", name, s, dt, err)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--run-dir", "-d", default=".", type=click.Path())
def status(run_dir: str):
    """Print run status."""
    run_dir_path = Path(run_dir)
    db_path = run_dir_path / "galoop.db"

    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    store = GaloopStore(run_dir_path)
    counts = store.count_by_status()
    total = sum(counts.values())
    n_converged = counts.get("converged", 0)
    n_dup = counts.get("duplicate", 0)
    dup_rate = f"{n_dup / (n_converged + n_dup) * 100:.0f}%" if (n_converged + n_dup) > 0 else "N/A"

    click.echo("\nStructure counts by status:")
    for st, n in sorted(counts.items()):
        click.echo(f"  {st:<20} {n}")
    click.echo(f"\n  total evaluated:     {total}")
    click.echo(f"  duplicate rate:      {dup_rate}")

    best = store.best(n=5)
    if best:
        click.echo("\nTop 5 by grand canonical energy:")
        for i, ind in enumerate(best, 1):
            g = (f"{ind.grand_canonical_energy:.4f}"
                 if ind.grand_canonical_energy is not None else "N/A")
            ads = ind.extra_data.get("adsorbate_counts", {})
            click.echo(f"  {i}. G={g} eV  {ind.id}  op={ind.operator}  ads={ads}")

    store.close()


# ---------------------------------------------------------------------------
# sample — generate N unique starting structures (no GA loop)
# ---------------------------------------------------------------------------

@dataclass
class _SamplerWorkerState:
    """Per-process state for the parallel placement-only sampler.

    Stored module-level so worker processes inherit it via the
    ``initializer`` argument to ``ProcessPoolExecutor``. Wrapped in a
    dataclass so static checkers see real attribute types instead of
    ``dict[str, object]``.
    """
    cfg: Any
    slab_info: Any
    ads_atoms: dict
    soap_kwargs: Any   # SoapKwargs (TypedDict; lazy import to avoid cycle)


_W: _SamplerWorkerState | None = None


def _sample_worker_init(config_path: str):
    """Initialise a sampler worker process: load config, slab, adsorbates."""
    global _W
    from galoop.config import load_config
    from galoop.fingerprint import SoapKwargs
    from galoop.science.surface import load_ads_template_dict, load_slab_from_config

    cfg = load_config(Path(config_path))
    slab_info = load_slab_from_config(cfg.slab)
    _W = _SamplerWorkerState(
        cfg=cfg,
        slab_info=slab_info,
        ads_atoms=load_ads_template_dict(cfg.adsorbates),
        soap_kwargs=SoapKwargs(
            r_cut=cfg.fingerprint.r_cut,
            n_max=cfg.fingerprint.n_max,
            l_max=cfg.fingerprint.l_max,
            n_slab_atoms=slab_info.n_slab_atoms,
        ),
    )


def _sample_worker_attempt(target_total: int, seed: int) -> dict | None:
    """Generate one candidate structure. Returns a dict or None on failure.

    Pure CPU: random placement with site-aware logic, validation, SOAP.
    No MACE — coworkers' downstream DFT/MACE pass will relax these.
    """
    import numpy as _np

    from galoop.fingerprint import compute_soap
    from galoop.galoop import _random_stoichiometry
    from galoop.science.surface import (
        build_random_structure,
        detect_desorption,
        validate_surface_binding,
    )

    state = _W
    assert state is not None, "_sample_worker_init must run before _sample_worker_attempt"
    cfg = state.cfg
    slab_info = state.slab_info
    ads_atoms = state.ads_atoms
    soap_kwargs = state.soap_kwargs

    rng = _np.random.default_rng(seed)

    try:
        counts = _random_stoichiometry(
            cfg.adsorbates, rng, target_total, target_total,
        )
        current = build_random_structure(
            slab_info, cfg.adsorbates, ads_atoms, counts, rng,
        )
    except Exception as exc:
        return {"reject": "placement", "error": str(exc)}

    if detect_desorption(current, slab_info):
        return {"reject": "desorbed"}
    bound, _ = validate_surface_binding(current, slab_info.n_slab_atoms)
    if not bound:
        return {"reject": "unbound"}

    try:
        vec = compute_soap(current, **soap_kwargs)
    except Exception as exc:
        return {"reject": "fingerprint", "error": str(exc)}

    return {
        "atoms": current,
        "soap": vec,
        "counts": dict(counts),
        "composition": current.get_chemical_formula(),
    }


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--output", "-o", default="samples.xyz", type=click.Path(),
              help="Output extended-XYZ file (a sibling manifest.csv is also written)")
@click.option("-n", "--n-structures", default=200, type=int,
              help="Target number of unique, validated structures")
@click.option("--relax", is_flag=True,
              help="Run a full MACE relaxation on each accepted structure "
                   "before writing it. Slower but produces locally minimised "
                   "geometries with energies. Default: placement only.")
@click.option("--workers", "-w", type=int, default=None,
              help="Number of parallel sampler workers (default: cpu_count // 2). "
                   "Ignored when --relax is set.")
@click.option("--max-attempts", type=int, default=None,
              help="Stop after this many attempts even if N is not reached "
                   "(default: 20 × N)")
@click.option("--soap-threshold", type=float, default=None,
              help="SOAP Tanimoto similarity above which two structures are "
                   "treated as duplicates. Higher (e.g. 0.98) → stricter notion "
                   "of 'same', so more structures are accepted as unique. "
                   "Lower (e.g. 0.85) → coarser dedup. "
                   "(default: fingerprint.duplicate_threshold from config)")
@click.option("--seed", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True)
def sample(config: str, output: str, n_structures: int, relax: bool,
           workers: int | None, max_attempts: int | None,
           soap_threshold: float | None, seed: int | None, verbose: bool):
    """Generate N unique adsorbate configurations as a screening dataset.

    Stratifies coverage across [min_adsorbates, max_adsorbates] from the config,
    places adsorbates with the same placement engine the GA uses, validates
    surface binding and desorption, and deduplicates with SOAP Tanimoto.
    Writes one POSCAR per accepted structure plus a manifest CSV.
    """
    import csv
    import json
    import os
    import shutil
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    from ase.io import write as ase_write

    from galoop.config import load_config
    from galoop.fingerprint import compute_soap, tanimoto_similarity
    from galoop.science.surface import (
        detect_desorption,
        load_slab_from_config,
        validate_surface_binding,
    )

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )
    log = logging.getLogger("galoop.sample")

    cfg = load_config(Path(config))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    fp_cfg = cfg.fingerprint
    dup_threshold = (soap_threshold if soap_threshold is not None
                     else fp_cfg.duplicate_threshold)

    lo, hi = cfg.ga.min_adsorbates, cfg.ga.max_adsorbates
    coverage_levels = list(range(lo, hi + 1))

    if max_attempts is None:
        max_attempts = max(50 * n_structures, 200)

    # Composition-bucketed fingerprint table: only structures with the same
    # chemical formula need a SOAP comparison, which collapses the linear
    # scan from O(N) to O(N / n_buckets).
    buckets: dict[str, list[np.ndarray]] = {}
    manifest: list[dict] = []
    rejected = {"placement": 0, "relax": 0, "desorbed": 0,
                "unbound": 0, "duplicate": 0, "fingerprint": 0}

    if seed is not None:
        base_seed = int(seed)
    else:
        # secrets.randbits is correctly typed (-> int) so ty stays happy.
        import secrets
        base_seed = secrets.randbits(32)

    # ====================================================================
    # Path A: --relax (sequential, MACE in-process). Heavy, low throughput.
    # ====================================================================
    if relax:
        from galoop.engine.calculator import build_pipeline
        from galoop.fingerprint import SoapKwargs
        from galoop.galoop import _random_stoichiometry, _snap_to_surface
        from galoop.science.surface import build_random_structure, load_ads_template_dict

        slab_info = load_slab_from_config(cfg.slab)
        ads_atoms = load_ads_template_dict(cfg.adsorbates)
        soap_kwargs: SoapKwargs = SoapKwargs(
            r_cut=fp_cfg.r_cut,
            n_max=fp_cfg.n_max,
            l_max=fp_cfg.l_max,
            n_slab_atoms=slab_info.n_slab_atoms,
        )

        pipeline = build_pipeline([s.model_dump() for s in cfg.calculator_stages])
        work_root = out_path.parent / f".{out_path.stem}_work"
        work_root.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(base_seed)

        log.info("Sampling %d unique structures (relax=on) → %s",
                 n_structures, out_path)

        next_id = 1
        attempt = 0
        while len(manifest) < n_structures and attempt < max_attempts:
            attempt += 1
            target = coverage_levels[(attempt - 1) % len(coverage_levels)]
            try:
                counts = _random_stoichiometry(cfg.adsorbates, rng, target, target)
                current = build_random_structure(
                    slab_info, cfg.adsorbates, ads_atoms, counts, rng,
                )
            except Exception as exc:
                log.debug("attempt %d: placement failed (%s)", attempt, exc)
                rejected["placement"] += 1
                continue

            try:
                current = _snap_to_surface(current, cfg, slab_info.n_slab_atoms)
            except Exception as exc:
                log.debug("attempt %d: snap failed (%s)", attempt, exc)
                rejected["relax"] += 1
                continue

            tmp_dir = work_root / f"attempt_{attempt:06d}"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                result = pipeline.run(
                    current, tmp_dir,
                    n_slab_atoms=slab_info.n_slab_atoms,
                )
            except Exception as exc:
                log.debug("attempt %d: relax raised (%s)", attempt, exc)
                shutil.rmtree(tmp_dir, ignore_errors=True)
                rejected["relax"] += 1
                continue
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not result.get("converged") or result.get("final_atoms") is None:
                rejected["relax"] += 1
                continue
            current = result["final_atoms"]
            energy = float(result["final_energy"])

            if detect_desorption(current, slab_info):
                rejected["desorbed"] += 1
                continue
            bound, _unbound = validate_surface_binding(current, slab_info.n_slab_atoms)
            if not bound:
                rejected["unbound"] += 1
                continue

            try:
                vec = compute_soap(current, **soap_kwargs)
            except Exception as exc:
                log.debug("attempt %d: SOAP failed (%s)", attempt, exc)
                rejected["fingerprint"] += 1
                continue

            comp = current.get_chemical_formula()
            bucket = buckets.setdefault(comp, [])
            if any(tanimoto_similarity(vec, prev) >= dup_threshold for prev in bucket):
                rejected["duplicate"] += 1
                continue
            bucket.append(vec)

            tagged = current.copy()
            tagged.info["sample_id"] = next_id
            tagged.info["adsorbate_counts"] = json.dumps(dict(counts))
            tagged.info["n_slab_atoms"] = slab_info.n_slab_atoms
            tagged.info["energy"] = energy
            ase_write(str(out_path), tagged, format="extxyz", append=True)
            manifest.append({
                "id": next_id, "n_atoms": len(current),
                "energy_eV": energy, "adsorbate_counts": dict(counts),
            })
            log.info("  [%d/%d] sample_%05d  counts=%s  E=%.3f eV",
                     next_id, n_structures, next_id, dict(counts), energy)
            next_id += 1

        # Clean up the relax scratch directory if it's empty (best-effort).
        import contextlib
        with contextlib.suppress(OSError):
            work_root.rmdir()

    # ====================================================================
    # Path B: placement-only, parallel via ProcessPoolExecutor. Default.
    # ====================================================================
    else:
        n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) // 2)
        log.info("Sampling %d unique structures (placement-only, %d workers) → %s",
                 n_structures, n_workers, out_path)

        # Cheap re-load on the parent for the dedup hash and metadata.
        # n_slab_atoms is needed when stamping the xyz comments.
        slab_info = load_slab_from_config(cfg.slab)

        # Each task gets a unique attempt_id; the worker derives a deterministic
        # per-attempt RNG from base_seed + attempt_id so the run is reproducible.
        next_attempt_id = 0
        next_sample_id = 1
        in_flight = {}    # future -> attempt_id

        def _submit(executor):
            nonlocal next_attempt_id
            target = coverage_levels[next_attempt_id % len(coverage_levels)]
            attempt_seed = (base_seed + next_attempt_id) & 0xFFFFFFFF
            fut = executor.submit(_sample_worker_attempt, target, attempt_seed)
            in_flight[fut] = next_attempt_id
            next_attempt_id += 1

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_sample_worker_init,
            initargs=(str(Path(config).resolve()),),
        ) as executor:
            # Prime the queue: 2× workers in flight to keep them busy.
            for _ in range(min(2 * n_workers, max_attempts)):
                _submit(executor)

            while in_flight and len(manifest) < n_structures:
                done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    in_flight.pop(fut, None)
                    try:
                        result = fut.result()
                    except Exception as exc:
                        log.debug("worker raised: %s", exc)
                        rejected["placement"] += 1
                        result = None

                    if result is None or "reject" in result:
                        if result and "reject" in result:
                            rejected[result["reject"]] = rejected.get(result["reject"], 0) + 1
                        if next_attempt_id < max_attempts and len(manifest) < n_structures:
                            _submit(executor)
                        continue

                    # Composition-bucketed dedup (parent-side, single-threaded)
                    comp = result["composition"]
                    vec = result["soap"]
                    bucket = buckets.setdefault(comp, [])
                    if any(tanimoto_similarity(vec, prev) >= dup_threshold for prev in bucket):
                        rejected["duplicate"] += 1
                        if next_attempt_id < max_attempts and len(manifest) < n_structures:
                            _submit(executor)
                        continue
                    bucket.append(vec)

                    atoms = result["atoms"]
                    counts = result["counts"]
                    tagged = atoms.copy()
                    tagged.info["sample_id"] = next_sample_id
                    tagged.info["adsorbate_counts"] = json.dumps(counts)
                    tagged.info["n_slab_atoms"] = slab_info.n_slab_atoms
                    ase_write(str(out_path), tagged, format="extxyz", append=True)
                    manifest.append({
                        "id": next_sample_id, "n_atoms": len(atoms),
                        "energy_eV": None, "adsorbate_counts": counts,
                    })
                    log.info("  [%d/%d] sample_%05d  counts=%s",
                             next_sample_id, n_structures, next_sample_id, counts)
                    next_sample_id += 1

                    if next_attempt_id < max_attempts and len(manifest) < n_structures:
                        _submit(executor)

            # Cancel any still-in-flight futures (best effort)
            for fut in list(in_flight):
                fut.cancel()

    # ---- Manifest CSV (sibling of the xyz) ----
    manifest_path = out_path.with_suffix(".manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "n_atoms", "energy_eV", "adsorbate_counts"])
        for s in manifest:
            w.writerow([
                s["id"], s["n_atoms"],
                f"{s['energy_eV']:.6f}" if s["energy_eV"] is not None else "",
                json.dumps(s["adsorbate_counts"]),
            ])

    log.info("")
    log.info("Done: %d/%d unique structures written", len(manifest), n_structures)
    log.info("Rejected: %s", rejected)
    log.info("XYZ:      %s", out_path)
    log.info("Manifest: %s", manifest_path)

    if len(manifest) < n_structures:
        log.warning(
            "Only generated %d of %d requested structures — "
            "the configuration space may be saturated. "
            "Try widening sampling_zmin/zmax, raising max_adsorbates, "
            "or lowering fingerprint.duplicate_threshold.",
            len(manifest), n_structures,
        )


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def stop(run_dir: str):
    """Request graceful stop."""
    stop_file = Path(run_dir) / "galoopstop"
    stop_file.touch()
    click.echo(f"Stop requested: {stop_file}")


# ---------------------------------------------------------------------------
# init — scaffold a new campaign directory
# ---------------------------------------------------------------------------

_MINIMAL_YAML_TEMPLATE = """\
# Minimal galoop campaign config. Edit the placeholders below (<LIKE THIS>)
# before launching. Full schema reference: docs/example.yaml and README.md.

slab:
  geometry: {slab_path}
  sampling_zmin: <TOP_Z + 0.5>   # lower bound of adsorbate placement window (Å)
  sampling_zmax: <TOP_Z + 3.5>   # upper bound (Å)

adsorbates:
  # Mono-atomic example — for polyatomics add binding_index + coordinates.
  - symbol: H
    min_count: 0
    max_count: 4

calculator_stages:
  - name: preopt
    type: {backend_type}
    fmax: 0.05
    max_steps: 300
    fix_slab_first: true
    prescan_fmax: 0.10
{backend_params}

scheduler:
  type: local                    # local | slurm | pbs
  nworkers: 2

ga:
  population_size: 20
  min_structures: 40
  max_structures: 500
  max_stall: 20
  min_adsorbates: 1
  max_adsorbates: 4

conditions:
  potential: 0.0                 # V vs RHE
  pH: 0.0
  temperature: 298.15

fingerprint:
  r_cut: 5.0
  n_max: 6
  l_max: 4
  duplicate_threshold: 0.92
"""

_MACE_PARAMS_BLOCK = """\
    params:
      model: small               # small | medium | large | path/to/custom.pt
      device: cuda               # cpu | cuda | auto
      dtype: float32
"""

_CUSTOM_PARAMS_BLOCK = """\
    params:
      # Passed straight to your factory in calc.py.
      # Add whatever keys your factory reads — checkpoint, device, task, …
      device: cuda
"""

_CALC_TEMPLATE = '''\
"""
calc.py — user-space factory for a custom calculator backend.

Reference this from galoop.yaml as:

    calculator_stages:
      - name: refine
        type: calc:make_calculator
        fmax: 0.03
        max_steps: 300
        params:
          # whatever keys make_calculator reads
          device: cuda

As long as the directory containing calc.py is on PYTHONPATH (e.g. launch
galoop from this directory, or set PYTHONPATH=.), galoop will import and
call this factory once per stage build.

The factory must return any ASE-compatible Calculator. If your calculator
drives its own internal relaxation (VASP-style), return the tuple
    (make_calculator, True)
at module level instead, and galoop will call get_potential_energy() once
rather than running BFGS on the Python side.
"""

from __future__ import annotations

import threading
from typing import Any

# Process-local cache so repeated stage builds (e.g. by Parsl workers)
# reuse the loaded model. Keyed on whatever params distinguish one
# instance from another.
_CACHE: dict[tuple, Any] = {}
_LOCK = threading.Lock()


def _cache_key(params: dict) -> tuple:
    return (
        params.get("model", "default"),
        params.get("device", "cpu"),
    )


def make_calculator(params: dict):
    """Return an ASE Calculator configured from *params*.

    Replace the body of this function with real model loading code. The
    example below is a placeholder that deliberately raises so you don't
    accidentally launch a campaign against a no-op calculator.
    """
    key = _cache_key(params)
    if key in _CACHE:
        return _CACHE[key]

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        # ------------------------------------------------------------
        # Example: fairchem UMA (Meta FAIR, OMat24 + OC20 + more)
        #
        #   pip install fairchem-core
        #
        # from fairchem.core import pretrained_mlip
        # from fairchem.core.calculate.ase_calculator import FAIRChemCalculator
        # predictor = pretrained_mlip.get_predict_unit(
        #     params.get("model", "uma-s-1p1"),
        #     device=params.get("device", "cuda"),
        # )
        # calc = FAIRChemCalculator(predictor, task_name=params.get("task", "oc20"))
        # ------------------------------------------------------------
        # Example: Orb-v3 (OMat24 checkpoint)
        #
        #   pip install orb-models
        #
        # from orb_models.forcefield import pretrained
        # from orb_models.forcefield.calculator import ORBCalculator
        # device = params.get("device", "cuda")
        # orbff = pretrained.orb_v3_direct_20_omat(device=device)
        # calc = ORBCalculator(orbff, device=device)
        # ------------------------------------------------------------

        raise NotImplementedError(
            "calc.py:make_calculator is a placeholder. Edit it to return "
            "a real ASE Calculator before launching a campaign."
        )

        _CACHE[key] = calc
        return calc
'''


@cli.command()
@click.argument("target_dir", type=click.Path(), default=".")
@click.option("--slab", "-s", type=click.Path(exists=True), default=None,
              help="Path to an existing slab geometry file (POSCAR/CIF/xyz). "
                   "Will be copied into TARGET_DIR as slab.vasp if it's not "
                   "already there.")
@click.option("--backend", "-b",
              type=click.Choice(["mace", "vasp", "custom"], case_sensitive=False),
              default="mace",
              help="Which calculator backend to scaffold. 'custom' pairs with "
                   "--calc-template to generate a calc.py you fill in.")
@click.option("--calc-template", is_flag=True,
              help="Also write a calc.py factory template alongside the yaml.")
@click.option("--force", is_flag=True,
              help="Overwrite galoop.yaml / calc.py if they already exist.")
def init(target_dir: str, slab: str | None, backend: str,
         calc_template: bool, force: bool):
    """Scaffold a new galoop campaign in TARGET_DIR.

    Writes a minimal galoop.yaml with placeholder values you edit before
    launching. Pass --slab to link an existing geometry file, and
    --calc-template to generate a calc.py factory for a custom MLIP.

    Examples
    --------
    galoop init runs/pt_nrr -s my_slab.vasp
    galoop init runs/uma_test -s slab.vasp -b custom --calc-template
    """
    import shutil

    target = Path(target_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)

    # --- slab geometry ---------------------------------------------------
    if slab is not None:
        slab_src = Path(slab).resolve()
        slab_dst = target / "slab.vasp"
        if slab_src != slab_dst:
            if slab_dst.exists() and not force:
                click.echo(
                    f"error: {slab_dst} already exists (use --force to overwrite)",
                    err=True,
                )
                sys.exit(1)
            shutil.copy(slab_src, slab_dst)
            click.echo(f"copied {slab_src} -> {slab_dst}")
        slab_path_for_yaml = str(slab_dst)
    else:
        slab_path_for_yaml = "<PATH_TO_YOUR_SLAB_FILE>"
        click.echo(
            "warning: no --slab provided; you must edit slab.geometry "
            "before running `galoop run`",
            err=True,
        )

    # --- backend params block -------------------------------------------
    backend_lower = backend.lower()
    if backend_lower == "mace":
        backend_type = "mace"
        backend_params = _MACE_PARAMS_BLOCK.rstrip()
    elif backend_lower == "vasp":
        backend_type = "vasp"
        backend_params = (
            "    params:\n"
            "      incar:\n"
            "        ENCUT: 520\n"
            "        EDIFF: 1.0e-6\n"
            "        ISMEAR: 0\n"
            "        SIGMA: 0.05"
        )
    else:  # custom
        backend_type = "calc:make_calculator"
        backend_params = _CUSTOM_PARAMS_BLOCK.rstrip()

    yaml_text = _MINIMAL_YAML_TEMPLATE.format(
        slab_path=slab_path_for_yaml,
        backend_type=backend_type,
        backend_params=backend_params,
    )

    # --- write galoop.yaml -----------------------------------------------
    yaml_path = target / "galoop.yaml"
    if yaml_path.exists() and not force:
        click.echo(
            f"error: {yaml_path} already exists (use --force to overwrite)",
            err=True,
        )
        sys.exit(1)
    yaml_path.write_text(yaml_text)
    click.echo(f"wrote {yaml_path}")

    # --- optionally write calc.py ---------------------------------------
    if calc_template or backend_lower == "custom":
        calc_path = target / "calc.py"
        if calc_path.exists() and not force:
            click.echo(
                f"error: {calc_path} already exists (use --force to overwrite)",
                err=True,
            )
            sys.exit(1)
        calc_path.write_text(_CALC_TEMPLATE)
        click.echo(f"wrote {calc_path}")
        if backend_lower == "custom":
            click.echo(
                "\nNote: calc.py contains a placeholder that raises on use. "
                "Edit make_calculator() before launching, and make sure the "
                "directory is on PYTHONPATH when you run galoop."
            )

    click.echo(f"\nNext steps:\n  1. edit {yaml_path}\n  2. galoop run -c {yaml_path} -d {target} -v")


if __name__ == "__main__":
    cli()
