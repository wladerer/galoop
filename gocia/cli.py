"""
gocia/cli.py

Command-line interface.
"""

import logging
import sys
from pathlib import Path

import click
import numpy as np

from config import load_config
from database import GociaDB
from galoop import run as run_ga


@click.group()
@click.version_option()
def cli():
    """GOCIA — lean genetic algorithm for adsorbate structure search."""
    pass


@cli.command()
@click.option("--config", "-c", default="gocia.yaml", type=click.Path(exists=True))
@click.option("--run-dir", "-d", default=".", type=click.Path())
@click.option("--seed", type=int, default=None)
@click.option("--verbose", "-v", is_flag=True)
def run(config: str, run_dir: str, seed: int, verbose: bool):
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
    except Exception as e:
        log.error(f"Config validation failed: {e}")
        sys.exit(1)

    # Import here to avoid circular deps
    from gocia.engine.calculator import build_pipeline
    from gocia.engine.scheduler import build_scheduler
    from gocia.science.surface import load_slab

    try:
        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )
        stages = build_pipeline(cfg.calculator_stages)
        scheduler = build_scheduler(cfg.scheduler)

        with GociaDB(run_dir_path / "gocia.db") as db:
            db.setup()

        rng = np.random.default_rng(seed)
        run_ga(cfg, run_dir_path, slab_info, stages, scheduler, rng)

    except Exception as e:
        log.exception("Fatal error in GA loop")
        sys.exit(1)


@cli.command()
@click.option("--run-dir", "-d", default=".", type=click.Path())
def status(run_dir: str):
    """Print run status."""
    run_dir_path = Path(run_dir)
    db_path = run_dir_path / "gocia.db"

    if not db_path.exists():
        click.echo(f"No database at {db_path}", err=True)
        sys.exit(1)

    with GociaDB(db_path) as db:
        counts = db.count_by_status()
        click.echo("\nStructure counts by status:")
        for status, n in sorted(counts.items()):
            click.echo(f"  {status:<20} {n}")

        best = db.best(n=5)
        if best:
            click.echo("\nTop 5 structures:")
            for i, ind in enumerate(best, 1):
                g = f"{ind.grand_canonical_energy:.4f}" if ind.grand_canonical_energy else "N/A"
                click.echo(f"  {i}. gen={ind.generation}  G={g} eV  {ind.id}")


@cli.command()
@click.argument("run_dir", type=click.Path(exists=True))
def stop(run_dir: str):
    """Request graceful stop."""
    stop_file = Path(run_dir) / "gociastop"
    stop_file.touch()
    click.echo(f"Stop requested: {stop_file}")


if __name__ == "__main__":
    cli()
