"""
galoop/cli.py

Command-line interface.

Usage
-----
galoop run --config galoop.yaml   # Start / resume the GA loop
galoop sample -c galoop.yaml -n 200 -o samples/   # Generate N unique structures
galoop status -d .                # Print structure counts
galoop report -d .                # Generate HTML report
"""

import logging
import sys
from pathlib import Path

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
    from galoop.science.surface import load_slab

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

        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )

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
# report
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--config", "-c", default=None, type=click.Path())
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output HTML path (default: <run-dir>/report.html)")
@click.option("--top", default=20, type=int, help="Number of top structures to list")
def report(run_dir: str, config: str | None, output: str | None, top: int):
    """Generate a self-contained HTML status report."""
    from galoop.report import generate

    run_dir_path = Path(run_dir).resolve()
    db_path = run_dir_path / "galoop.db"

    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    cfg_path = Path(config).resolve() if config else run_dir_path / "galoop.yaml"
    if not cfg_path.exists():
        click.echo(f"Config not found: {cfg_path}  (pass --config to specify)", err=True)
        sys.exit(1)

    try:
        cfg = load_config(cfg_path)
    except Exception as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    out = Path(output) if output else run_dir_path / "report.html"
    store = GaloopStore(run_dir_path)

    # Backfill auto-calibrated values from stored config snapshot
    import json
    row = store._conn.execute(
        "SELECT value FROM run_params WHERE key = ?", ("config",)
    ).fetchone()
    if row is not None:
        stored = json.loads(row["value"])
        if cfg.slab.energy is None and "slab" in stored:
            cfg.slab.energy = stored["slab"].get("energy")
        for ads in cfg.adsorbates:
            if ads.chemical_potential is None:
                for stored_ads in stored.get("adsorbates", []):
                    if stored_ads.get("symbol") == ads.symbol:
                        ads.chemical_potential = stored_ads.get("chemical_potential")
                        break

    generate(project=store, cfg=cfg, output_path=out, top_n=top)
    store.close()
    click.echo(f"Report written to {out}")


# ---------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------

@cli.command("graph")
@click.option("--run-dir", "-d", default=".", type=click.Path(exists=True))
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
@click.option("--converged", is_flag=True,
              help="Show all unique converged structures instead of duplicate clusters.")
@click.option("--output", "-o", default=None,
              help="Output HTML path (default: <run-dir>/graphs.html)")
@click.option("--radius", default=2, type=int, show_default=True,
              help="Graph environment radius.")
def graph(run_dir: str, config: str, converged: bool, output: str | None, radius: int):
    """Visualise adsorbate chemical-environment graphs in the browser."""
    import webbrowser

    from ase.io import read as ase_read

    from galoop import graph_viz
    from galoop.fingerprint import build_chem_envs
    from galoop.individual import STATUS
    from galoop.science.surface import load_slab

    run_dir_path = Path(run_dir).resolve()
    cfg_path = Path(config).resolve()
    db_path = run_dir_path / "galoop.db"

    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    try:
        cfg = load_config(cfg_path)
    except Exception as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    slab_path = cfg_path.parent / cfg.slab.geometry
    try:
        slab_info = load_slab(slab_path, cfg.slab.sampling_zmin, cfg.slab.sampling_zmax)
    except Exception as exc:
        click.echo(f"Could not load slab: {exc}", err=True)
        sys.exit(1)

    store = GaloopStore(run_dir_path)
    all_converged = store.get_by_status(STATUS.CONVERGED)
    all_dups = store.get_by_status(STATUS.DUPLICATE)

    out_path = Path(output) if output else run_dir_path / "graphs.html"
    pages: list[dict] = []

    def _load_atoms(ind):
        struct_dir = store.individual_dir(ind.id)
        for name in ("CONTCAR", "POSCAR"):
            candidate = struct_dir / name
            if candidate.exists():
                try:
                    return ase_read(str(candidate), format="vasp")
                except Exception:
                    return None
        return None

    def _build_page_for(ind, title: str) -> dict | None:
        atoms = _load_atoms(ind)
        if atoms is None:
            return None
        envs = build_chem_envs(atoms, slab_info.n_slab_atoms, radius=radius)
        if not envs:
            return None
        return graph_viz.build_page(title, envs)

    if converged:
        sorted_conv = sorted(
            all_converged,
            key=lambda x: (x.grand_canonical_energy is None,
                           x.grand_canonical_energy or 0.0),
        )
        total = len(sorted_conv)
        for i, ind in enumerate(sorted_conv, 1):
            gce = (f"G={ind.grand_canonical_energy:.4f} eV"
                   if ind.grand_canonical_energy is not None else "G=N/A")
            page = _build_page_for(ind, f"[{i}/{total}] {ind.id}  {gce}")
            if page:
                pages.append(page)
    else:
        clusters: dict[str, list] = {}
        for dup in all_dups:
            dup_of = dup.extra_data.get("dup_of")
            if dup_of:
                clusters.setdefault(dup_of, []).append(dup)

        sorted_conv = sorted(
            all_converged,
            key=lambda x: (x.grand_canonical_energy is None,
                           x.grand_canonical_energy or 0.0),
        )
        total_unique = len(sorted_conv)
        for rank, orig in enumerate(sorted_conv, 1):
            n_dups = len(clusters.get(orig.id, []))
            gce = (f"G={orig.grand_canonical_energy:.4f} eV"
                   if orig.grand_canonical_energy is not None else "G=N/A")
            dup_tag = f"  +{n_dups} dup{'s' if n_dups != 1 else ''}" if n_dups else ""
            page = _build_page_for(orig,
                                   f"[{rank}/{total_unique}] {orig.id}  {gce}{dup_tag}")
            if page:
                pages.append(page)

            for dup in clusters.get(orig.id, []):
                tanimoto = dup.extra_data.get("tanimoto")
                sim_str = (f"Tanimoto={tanimoto:.3f}" if tanimoto is not None
                           else "Tanimoto=N/A")
                page = _build_page_for(dup,
                                       f"  DUP {dup.id}  {sim_str}  (orig {orig.id})")
                if page:
                    pages.append(page)

    store.close()

    if not pages:
        click.echo("No graphs to display.", err=True)
        sys.exit(1)

    graph_viz.generate_html(pages, out_path)
    click.echo(f"Graph viewer written to {out_path}")
    webbrowser.open(out_path.as_uri())


# ---------------------------------------------------------------------------
# sample — generate N unique starting structures (no GA loop)
# ---------------------------------------------------------------------------

# Per-process worker globals, populated by _sample_worker_init.
_W: dict = {}


def _sample_worker_init(config_path: str):
    """Initialise a sampler worker process: load config, slab, adsorbates."""
    from galoop.config import load_config
    from galoop.science.surface import load_adsorbate, load_slab

    cfg = load_config(Path(config_path))
    slab_info = load_slab(
        cfg.slab.geometry,
        zmin=cfg.slab.sampling_zmin,
        zmax=cfg.slab.sampling_zmax,
    )
    ads_atoms = {
        a.symbol: load_adsorbate(
            symbol=a.symbol,
            geometry=getattr(a, "geometry", None),
            coordinates=getattr(a, "coordinates", None),
        )
        for a in cfg.adsorbates
    }
    _W["cfg"] = cfg
    _W["slab_info"] = slab_info
    _W["ads_atoms"] = ads_atoms
    _W["soap_kwargs"] = dict(
        r_cut=cfg.fingerprint.r_cut,
        n_max=cfg.fingerprint.n_max,
        l_max=cfg.fingerprint.l_max,
        n_slab_atoms=slab_info.n_slab_atoms,
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
        detect_desorption,
        place_adsorbate,
        validate_surface_binding,
    )

    cfg = _W["cfg"]
    slab_info = _W["slab_info"]
    ads_atoms = _W["ads_atoms"]
    soap_kwargs = _W["soap_kwargs"]

    rng = _np.random.default_rng(seed)

    try:
        counts = _random_stoichiometry(
            cfg.adsorbates, rng, target_total, target_total,
        )
        current = slab_info.atoms.copy()
        for sym, cnt in counts.items():
            ads_cfg = next(a for a in cfg.adsorbates if a.symbol == sym)
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
        load_adsorbate,
        load_slab,
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

    base_seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy & 0xFFFFFFFF)

    # ====================================================================
    # Path A: --relax (sequential, MACE in-process). Heavy, low throughput.
    # ====================================================================
    if relax:
        from galoop.engine.calculator import build_pipeline
        from galoop.galoop import _random_stoichiometry, _snap_to_surface
        from galoop.science.surface import place_adsorbate

        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )
        ads_atoms = {
            a.symbol: load_adsorbate(
                symbol=a.symbol,
                geometry=getattr(a, "geometry", None),
                coordinates=getattr(a, "coordinates", None),
            )
            for a in cfg.adsorbates
        }
        soap_kwargs = dict(
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
                current = slab_info.atoms.copy()
                for sym, cnt in counts.items():
                    ads_cfg = next(a for a in cfg.adsorbates if a.symbol == sym)
                    for _ in range(cnt):
                        current = place_adsorbate(
                            slab=current, adsorbate=ads_atoms[sym],
                            zmin=slab_info.zmin, zmax=slab_info.zmax,
                            n_orientations=ads_cfg.n_orientations,
                            binding_index=ads_cfg.binding_index, rng=rng,
                            n_slab_atoms=slab_info.n_slab_atoms,
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

        try:
            work_root.rmdir()
        except OSError:
            pass

    # ====================================================================
    # Path B: placement-only, parallel via ProcessPoolExecutor. Default.
    # ====================================================================
    else:
        n_workers = workers if workers is not None else max(1, (os.cpu_count() or 2) // 2)
        log.info("Sampling %d unique structures (placement-only, %d workers) → %s",
                 n_structures, n_workers, out_path)

        # Cheap re-load on the parent for the dedup hash and metadata.
        # n_slab_atoms is needed when stamping the xyz comments.
        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )

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
