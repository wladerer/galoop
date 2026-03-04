"""
galoop/cli.py

Command-line interface.
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

from galoop.config import load_config
from galoop.database import GaloopDB
from galoop.galoop import run as run_ga


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """galoop — lean genetic algorithm for adsorbate structure search."""


@cli.command()
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--seed", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True)
def run(config: str, run_dir: str, seed: int | None, verbose: bool):
    """Start or resume the GA loop."""
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
        cfg = load_config(config_path)
    except Exception as exc:
        log.error("Config validation failed: %s", exc)
        sys.exit(1)

    from galoop.engine.calculator import build_pipeline
    from galoop.engine.scheduler import build_scheduler
    from galoop.science.surface import load_slab

    try:
        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )
        pipeline = build_pipeline(cfg.calculator_stages)
        scheduler = build_scheduler(cfg.scheduler)

        with GaloopDB(run_dir_path / "galoop.db") as db:
            db.setup()
            current_cfg = cfg.model_dump()
            diffs = db.diff_run_params(current_cfg)
            if diffs:
                log.warning("Config has changed since the last run:")
                for d in diffs:
                    tag = "  [ENERGY CRITICAL]" if d["energy_critical"] else ""
                    log.warning("  %-55s  %s  →  %s%s",
                                d["field"], d["old"], d["new"], tag)
            db.save_run_params(current_cfg)

        rng = np.random.default_rng(seed)
        run_ga(cfg, run_dir_path, slab_info, pipeline, scheduler, rng)

    except Exception:
        log.exception("Fatal error in GA loop")
        sys.exit(1)


@cli.command()
@click.option("--run-dir", "-d", default=".", type=click.Path())
def status(run_dir: str):
    """Print run status."""
    db_path = Path(run_dir) / "galoop.db"

    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    with GaloopDB(db_path) as db:
        counts = db.count_by_status()
        click.echo("\nStructure counts by status:")
        for st, n in sorted(counts.items()):
            click.echo(f"  {st:<20} {n}")

        best = db.best(n=5)
        if best:
            click.echo("\nTop 5 structures:")
            for i, ind in enumerate(best, 1):
                g = f"{ind.grand_canonical_energy:.4f}" if ind.grand_canonical_energy is not None else "N/A"
                click.echo(f"  {i}. gen={ind.generation}  G={g} eV  {ind.id}")


@cli.command("config-diff")
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
def config_diff(run_dir: str, config: str):
    """Show what has changed between the stored and current config."""
    from galoop.database import GaloopDB, diff_configs

    db_path = Path(run_dir) / "galoop.db"
    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    try:
        cfg = load_config(Path(config))
    except Exception as exc:
        click.echo(f"Config error: {exc}", err=True)
        sys.exit(1)

    with GaloopDB(db_path) as db:
        diffs = db.diff_run_params(cfg.model_dump())

    if not diffs:
        click.echo("No changes detected.")
        return

    energy_diffs = [d for d in diffs if d["energy_critical"]]
    other_diffs  = [d for d in diffs if not d["energy_critical"]]

    if energy_diffs:
        click.echo(click.style("\n  ENERGY-CRITICAL CHANGES", fg="red", bold=True))
        click.echo(click.style("  (these affect grand-canonical energy comparisons)", fg="red"))
        for d in energy_diffs:
            click.echo(f"    {d['field']}")
            click.echo(click.style(f"      was:  {d['old']}", fg="yellow"))
            click.echo(click.style(f"      now:  {d['new']}", fg="green"))

    if other_diffs:
        click.echo(click.style("\n  OTHER CHANGES", fg="cyan", bold=True))
        for d in other_diffs:
            click.echo(f"    {d['field']}")
            click.echo(f"      was:  {d['old']}")
            click.echo(f"      now:  {d['new']}")

    click.echo()


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
    generate(db_path=db_path, cfg=cfg, output_path=out, top_n=top)
    click.echo(f"Report written to {out}")


@cli.command()
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--threshold", type=float, default=None,
              help="Show only duplicates with Tanimoto >= THRESHOLD (default: from config).")
def duplicates(run_dir: str, threshold: float | None):
    """Show duplicate clustering from the database."""
    db_path = Path(run_dir) / "galoop.db"
    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    with GaloopDB(db_path) as db:
        converged = db.get_by_status("converged")
        dups = db.get_by_status("duplicate")

    # Try to read the stored threshold from config
    cfg_threshold = 0.90
    cfg_path = Path(run_dir) / "galoop.yaml"
    if cfg_path.exists():
        try:
            cfg = load_config(cfg_path)
            cfg_threshold = cfg.fingerprint.duplicate_threshold
        except Exception:
            pass

    display_threshold = threshold if threshold is not None else cfg_threshold

    # Group duplicates by original: orig_id -> [(dup_ind, tanimoto), ...]
    clusters: dict[str, list] = {}
    unlinked = 0
    for dup in dups:
        dup_of = dup.extra_data.get("dup_of")
        tanimoto = dup.extra_data.get("tanimoto")
        if dup_of is None:
            unlinked += 1
            continue
        if tanimoto is not None and tanimoto < display_threshold:
            continue
        clusters.setdefault(dup_of, []).append((dup, tanimoto))

    conv_map = {ind.id: ind for ind in converged}
    n_shown_dups = sum(len(v) for v in clusters.values())
    n_total_dups = len(dups)

    click.echo(
        f"\nDuplicates ({n_shown_dups} shown"
        + (f", {n_total_dups} total" if n_shown_dups != n_total_dups else "")
        + f", threshold={display_threshold:.2f})\n"
    )

    if not clusters and not unlinked:
        click.echo("  No duplicates found.")
    else:
        for orig_id, dup_list in sorted(clusters.items(), key=lambda x: -len(x[1])):
            orig = conv_map.get(orig_id)
            if orig is not None:
                gce = (
                    f"G={orig.grand_canonical_energy:.4f} eV"
                    if orig.grand_canonical_energy is not None
                    else "G=N/A"
                )
                click.echo(f"  Original {orig_id}  {gce}  gen={orig.generation}")
            else:
                click.echo(f"  Original {orig_id}  (not in converged pool)")
            for dup_ind, tanimoto in sorted(dup_list, key=lambda x: -(x[1] or 0.0)):
                sim_str = f"Tanimoto={tanimoto:.3f}" if tanimoto is not None else "Tanimoto=N/A"
                click.echo(f"    └─ {dup_ind.id}  {sim_str}  gen={dup_ind.generation}")

        if unlinked:
            click.echo(f"\n  ({unlinked} duplicate(s) have no dup_of record — pre-date this feature)")

    n_unique = len(converged)
    n_total_conv = n_unique + n_total_dups
    click.echo(f"\nUnique converged: {n_unique} / {n_total_conv} total converged\n")


@cli.command("rebuild-db")
@click.option("--run-dir", "-d", default=".", type=click.Path(exists=True))
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
def rebuild_db(run_dir: str, config: str, force: bool) -> None:
    """Reconstitute galoop.db from filesystem output files.

    Walks all gen_NNN/struct_NNNN directories, reads sentinel files and
    FINAL_ENERGY, re-runs duplicate detection on converged structures, and
    writes a fresh database.  The existing DB is backed up to galoop.db.bak.

    Operator and parent lineage cannot be recovered and are set to 'unknown'.
    """
    import math
    import shutil

    from ase.io import read as ase_read

    from galoop.database import GaloopDB
    from galoop.fingerprint import (
        StructRecord, classify_postrelax,
        _composition, _dist_histogram, build_chem_envs,
    )
    from galoop.individual import Individual, OPERATOR, STATUS
    from galoop.science.energy import grand_canonical_energy
    from galoop.science.surface import load_slab

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    log = logging.getLogger(__name__)

    run_dir_path = Path(run_dir).resolve()
    cfg_path = Path(config).resolve()
    db_path = run_dir_path / "galoop.db"

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

    if db_path.exists():
        if not force:
            click.confirm(
                f"This will replace {db_path} (backup → galoop.db.bak). Continue?",
                abort=True,
            )
        backup = db_path.with_suffix(".db.bak")
        shutil.copy2(db_path, backup)
        click.echo(f"Backed up existing DB → {backup}")
        db_path.unlink()

    gen_dirs = sorted(run_dir_path.glob("gen_???"))
    if not gen_dirs:
        click.echo("No gen_NNN directories found.", err=True)
        sys.exit(1)

    chem_pots = {a.symbol: a.chemical_potential for a in cfg.adsorbates}
    struct_cache: dict[str, StructRecord] = {}
    counts = {s: 0 for s in ("converged", "duplicate", "failed", "desorbed", "other")}

    def _sentinel(d: Path) -> str | None:
        for name in ("CONVERGED", "FAILED", "DUPLICATE", "DESORBED", "SUBMITTED", "PENDING"):
            if (d / name).exists():
                return name.lower()
        return None

    def _energy(d: Path) -> float:
        ef = d / "FINAL_ENERGY"
        try:
            return float(ef.read_text().strip()) if ef.exists() else float("nan")
        except ValueError:
            return float("nan")

    def _ads_counts(struct_dir: Path) -> dict:
        from galoop.galoop import _infer_adsorbate_counts
        geom = struct_dir / "CONTCAR"
        if not geom.exists():
            geom = struct_dir / "POSCAR"
        if not geom.exists():
            return {}
        try:
            atoms = ase_read(str(geom), format="vasp")
            return _infer_adsorbate_counts(
                atoms.get_chemical_symbols()[slab_info.n_slab_atoms:],
                cfg.adsorbates,
            )
        except Exception:
            return {}

    with GaloopDB(db_path) as db:
        db.setup()
        db.save_run_params(cfg.model_dump())

        for gen_dir in gen_dirs:
            gen_num = int(gen_dir.name.split("_")[1])
            for struct_dir in sorted(gen_dir.glob("struct_????")):
                poscar = struct_dir / "POSCAR"
                contcar = struct_dir / "CONTCAR"
                if not poscar.exists():
                    continue

                sentinel = _sentinel(struct_dir)
                op = OPERATOR.INIT if gen_num == 0 else "unknown"

                ind = Individual(
                    generation=gen_num,
                    operator=op,
                    status=sentinel or STATUS.PENDING,
                    geometry_path=str(poscar),
                    extra_data={"adsorbate_counts": _ads_counts(struct_dir)},
                )
                db.insert(ind)

                if sentinel != STATUS.CONVERGED:
                    if sentinel in (STATUS.FAILED, STATUS.DESORBED):
                        counts["failed"] += 1
                    else:
                        counts["other"] += 1
                    continue

                # CONVERGED: re-run fingerprint + GCE
                if not contcar.exists():
                    log.warning("%s: CONVERGED sentinel but no CONTCAR — skipping fingerprint",
                                struct_dir)
                    counts["other"] += 1
                    continue

                raw_e = _energy(struct_dir)
                try:
                    atoms = ase_read(str(contcar), format="vasp")
                    label, dup_id, soap_vec = classify_postrelax(
                        atoms,
                        energy=raw_e,
                        struct_cache=struct_cache,
                        duplicate_threshold=cfg.fingerprint.duplicate_threshold,
                        energy_tol_pct=cfg.fingerprint.energy_tol_pct,
                        dist_hist_threshold=cfg.fingerprint.dist_hist_threshold,
                        dist_hist_bins=cfg.fingerprint.dist_hist_bins,
                        dist_hist_rmax=cfg.fingerprint.r_cut,
                        r_cut=cfg.fingerprint.r_cut,
                        n_max=cfg.fingerprint.n_max,
                        l_max=cfg.fingerprint.l_max,
                        n_slab_atoms=slab_info.n_slab_atoms,
                    )

                    if label == "duplicate":
                        ind = ind.mark_duplicate()
                        ind.extra_data = {**ind.extra_data, "dup_of": dup_id}
                        db.update(ind)
                        counts["duplicate"] += 1
                    else:
                        gce = grand_canonical_energy(
                            raw_energy=raw_e,
                            adsorbate_counts=ind.extra_data.get("adsorbate_counts", {}),
                            chemical_potentials=chem_pots,
                            potential=cfg.conditions.potential,
                            pH=cfg.conditions.pH,
                            temperature=cfg.conditions.temperature,
                            pressure=cfg.conditions.pressure,
                        )
                        ind = ind.with_energy(raw=raw_e, grand_canonical=gce)
                        db.update(ind)
                        struct_cache[ind.id] = StructRecord(
                            id=ind.id,
                            soap_vector=soap_vec,
                            energy=raw_e,
                            composition=_composition(atoms),
                            dist_hist=_dist_histogram(
                                atoms,
                                n_bins=cfg.fingerprint.dist_hist_bins,
                                r_max=cfg.fingerprint.r_cut,
                            ),
                            chem_envs=build_chem_envs(atoms, slab_info.n_slab_atoms),
                        )
                        counts["converged"] += 1
                        gce_str = f"{gce:.4f}" if not math.isnan(gce) else "nan"
                        log.info("  %s/%s  G=%s eV", gen_dir.name, struct_dir.name, gce_str)

                except Exception as exc:
                    log.warning("  %s/%s: fingerprint failed (%s) — marked converged without GCE",
                                gen_dir.name, struct_dir.name, exc)
                    ind = ind.with_status(STATUS.CONVERGED)
                    db.update(ind)
                    counts["converged"] += 1

    total = sum(counts.values())
    click.echo(
        f"\nRebuilt DB from {total} structures:\n"
        f"  Converged (unique) : {counts['converged']}\n"
        f"  Duplicate          : {counts['duplicate']}\n"
        f"  Failed / desorbed  : {counts['failed']}\n"
        f"  Pending / other    : {counts['other']}\n"
        f"\nWritten to {db_path}"
    )


@cli.command("graph")
@click.option("--run-dir", "-d", default=".", type=click.Path(exists=True))
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
@click.option("--converged", is_flag=True,
              help="Show all unique converged structures instead of duplicate clusters.")
@click.option("--output", "-o", default=None,
              help="Output HTML path (default: <run-dir>/graphs.html)")
@click.option("--radius", default=2, type=int, show_default=True,
              help="Graph environment radius.")
def graph(run_dir: str, config: str, converged: bool, output: str | None, radius: int) -> None:
    """Visualise adsorbate chemical-environment graphs in the browser."""
    import webbrowser

    from ase.io import read as ase_read

    from galoop import graph_viz
    from galoop.fingerprint import build_chem_envs
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

    with GaloopDB(db_path) as db:
        all_converged = db.get_by_status("converged")
        all_dups = db.get_by_status("duplicate")

    out_path = Path(output) if output else run_dir_path / "graphs.html"
    pages: list[dict] = []

    def _load_atoms(ind):
        """Load CONTCAR (fallback POSCAR) for an Individual."""
        geom_path = Path(ind.geometry_path) if ind.geometry_path else None
        if geom_path:
            base = geom_path.parent
        else:
            # Try to infer from geometry_path pattern
            return None
        for name in ("CONTCAR", "POSCAR"):
            candidate = base / name
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
        # One page per unique converged structure, sorted by GCE
        sorted_conv = sorted(
            all_converged,
            key=lambda x: (x.grand_canonical_energy is None,
                           x.grand_canonical_energy or 0.0),
        )
        total = len(sorted_conv)
        for i, ind in enumerate(sorted_conv, 1):
            gce = (f"G={ind.grand_canonical_energy:.4f} eV"
                   if ind.grand_canonical_energy is not None else "G=N/A")
            title = f"[{i}/{total}] {ind.id}  {gce}  gen={ind.generation}"
            page = _build_page_for(ind, title)
            if page:
                pages.append(page)
    else:
        # Default mode: all unique converged sorted by GCE, each followed by its duplicates.
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
            orig_title = (f"[{rank}/{total_unique}] {orig.id}  {gce}"
                          f"  gen={orig.generation}{dup_tag}")
            page = _build_page_for(orig, orig_title)
            if page:
                pages.append(page)

            for dup in clusters.get(orig.id, []):
                tanimoto = dup.extra_data.get("tanimoto")
                sim_str = (f"Tanimoto={tanimoto:.3f}" if tanimoto is not None
                           else "Tanimoto=N/A")
                dup_title = (f"  DUP {dup.id}  {sim_str}  gen={dup.generation}"
                             f"  (orig {orig.id})")
                page = _build_page_for(dup, dup_title)
                if page:
                    pages.append(page)

    if not pages:
        click.echo("No graphs to display (no structures with geometry found).", err=True)
        sys.exit(1)

    graph_viz.generate_html(pages, out_path)
    click.echo(f"Graph viewer written to {out_path}")
    webbrowser.open(out_path.as_uri())


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def stop(run_dir: str):
    """Request graceful stop."""
    stop_file = Path(run_dir) / "gociastop"
    stop_file.touch()
    click.echo(f"Stop requested: {stop_file}")


@cli.command("_run-pipeline", hidden=True)
@click.argument("struct_dir", type=click.Path(exists=True))
@click.option("--config", "-c", default="galoop.yaml", type=click.Path(exists=True))
def run_pipeline(struct_dir: str, config: str) -> None:
    """Relax a single structure through the full calculator pipeline."""
    import math
    from ase import Atoms
    from ase.io import read
    from galoop.engine.calculator import Pipeline, build_pipeline
    from galoop.science.surface import SlabInfo, detect_desorption, load_slab

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    log = logging.getLogger(__name__)

    struct_path = Path(struct_dir)

    try:
        cfg = load_config(Path(config))
    except Exception as exc:
        log.error("Config load failed: %s", exc)
        _drop_sentinel(struct_path, "FAILED")
        sys.exit(1)

    poscar = struct_path / "POSCAR"
    if not poscar.exists():
        log.error("POSCAR not found: %s", poscar)
        _drop_sentinel(struct_path, "FAILED")
        sys.exit(1)

    try:
        atoms: Atoms = read(str(poscar), format="vasp")
        slab_geometry = Path(config).resolve().parent / cfg.slab.geometry
        slab_info: SlabInfo = load_slab(slab_geometry, cfg.slab.sampling_zmin, cfg.slab.sampling_zmax)
        pipeline: Pipeline = build_pipeline(cfg.calculator_stages)
    except Exception as exc:
        log.exception("Setup failed: %s", exc)
        _drop_sentinel(struct_path, "FAILED")
        sys.exit(1)

    config_dir = Path(config).resolve().parent
    mace_model = cfg.mace_model
    _candidate = config_dir / mace_model
    if _candidate.exists():
        mace_model = str(_candidate)

    result: dict[str, object] = pipeline.run(
        atoms, struct_path,
        mace_model=mace_model,
        mace_device=cfg.mace_device,
        mace_dtype=cfg.mace_dtype,
        n_slab_atoms=slab_info.n_slab_atoms,
    )

    if not result["converged"] or math.isnan(float(result["final_energy"])):  # type: ignore[arg-type]
        _drop_sentinel(struct_path, "FAILED")
        sys.exit(0)

    if detect_desorption(result["final_atoms"], slab_info):  # type: ignore[arg-type]
        _drop_sentinel(struct_path, "DESORBED")
        sys.exit(0)

    _drop_sentinel(struct_path, "CONVERGED")
    sys.exit(0)


def _drop_sentinel(struct_dir: Path, state: str) -> None:
    """Remove all existing sentinels and write the new one."""
    for name in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED", "DUPLICATE", "DESORBED"):
        (struct_dir / name).unlink(missing_ok=True)
    (struct_dir / state).touch()

if __name__ == "__main__":
    cli()
