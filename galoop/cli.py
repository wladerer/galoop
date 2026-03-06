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
    import parsl
    from galoop.engine.scheduler import build_parsl_config
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

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        log.error("Config validation failed: %s", exc)
        sys.exit(1)

    try:
        slab_info = load_slab(
            cfg.slab.geometry,
            zmin=cfg.slab.sampling_zmin,
            zmax=cfg.slab.sampling_zmax,
        )

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

        parsl_cfg = build_parsl_config(cfg.scheduler, run_dir=run_dir_path)
        parsl.load(parsl_cfg)

        rng = np.random.default_rng(seed)
        try:
            run_ga(cfg, run_dir_path, slab_info, rng)
        finally:
            parsl.dfk().cleanup()
            parsl.clear()

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
                click.echo(f"  {i}. G={g} eV  {ind.id}")


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
        geom_path = Path(ind.geometry_path) if ind.geometry_path else None
        if not geom_path:
            return None
        base = geom_path.parent
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
        sorted_conv = sorted(
            all_converged,
            key=lambda x: (x.grand_canonical_energy is None,
                           x.grand_canonical_energy or 0.0),
        )
        total = len(sorted_conv)
        for i, ind in enumerate(sorted_conv, 1):
            gce = (f"G={ind.grand_canonical_energy:.4f} eV"
                   if ind.grand_canonical_energy is not None else "G=N/A")
            title = f"[{i}/{total}] {ind.id}  {gce}"
            page = _build_page_for(ind, title)
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
            orig_title = f"[{rank}/{total_unique}] {orig.id}  {gce}{dup_tag}"
            page = _build_page_for(orig, orig_title)
            if page:
                pages.append(page)

            for dup in clusters.get(orig.id, []):
                tanimoto = dup.extra_data.get("tanimoto")
                sim_str = (f"Tanimoto={tanimoto:.3f}" if tanimoto is not None
                           else "Tanimoto=N/A")
                dup_title = f"  DUP {dup.id}  {sim_str}  (orig {orig.id})"
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
