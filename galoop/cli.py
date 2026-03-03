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

    result: dict[str, object] = pipeline.run(atoms, struct_path, mace_model=cfg.mace_model, mace_device=cfg.mace_device)

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
