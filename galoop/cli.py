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
                    mace_model=cfg.mace_model,
                    mace_device=cfg.mace_device,
                    mace_dtype=cfg.mace_dtype,
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


if __name__ == "__main__":
    cli()
