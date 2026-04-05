"""
galoop/cli.py

Command-line interface.

Usage
-----
galoop run --config galoop.yaml   # Start / resume the GA controller loop
row run                            # (separate terminal) Submit pending jobs

The `run` command manages spawning and evaluation; `row run` handles cluster
submission of the relax operation.  Both read/write the same signac workspace.
"""

import logging
import shutil
import sys
from pathlib import Path

import click
import numpy as np

from galoop.config import load_config
from galoop.project import GaloopProject


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
    """Start or resume the GA controller loop.

    Tip: run `row run` in the same directory to have row submit pending
    relax jobs to your cluster as this loop spawns them.
    """
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

    # Ensure workflow.toml is present in the run directory so `row run` works
    run_toml = run_dir_path / "workflow.toml"
    if not run_toml.exists():
        # Look for it next to the package or next to this config file
        candidates = [
            Path(__file__).parent.parent / "workflow.toml",  # repo root
            config_path.parent / "workflow.toml",
        ]
        for src in candidates:
            if src.exists():
                shutil.copy(src, run_toml)
                log.info("Copied workflow.toml to %s", run_dir_path)
                break
        else:
            log.warning(
                "workflow.toml not found — copy it to %s before running `row run`",
                run_dir_path,
            )

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

        project = GaloopProject(run_dir_path)
        current_cfg = cfg.model_dump()
        diffs = project.diff_config(current_cfg)
        if diffs:
            log.warning("Config has changed since the last run:")
            for d in diffs:
                tag = "  [ENERGY CRITICAL]" if d["energy_critical"] else ""
                log.warning("  %-55s  %s  →  %s%s",
                            d["field"], d["old"], d["new"], tag)
        project.save_config_snapshot(current_cfg)

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
    workspace = run_dir_path / "workspace"

    if not workspace.exists():
        click.echo(f"No workspace at {workspace}", err=True)
        sys.exit(1)

    project = GaloopProject(run_dir_path)
    counts = project.count_by_status()
    total = sum(counts.values())
    n_converged = counts.get("converged", 0)
    n_dup = counts.get("duplicate", 0)
    dup_rate = f"{n_dup / (n_converged + n_dup) * 100:.0f}%" if (n_converged + n_dup) > 0 else "N/A"

    click.echo("\nStructure counts by status:")
    for st, n in sorted(counts.items()):
        click.echo(f"  {st:<20} {n}")
    click.echo(f"\n  total evaluated:     {total}")
    click.echo(f"  duplicate rate:      {dup_rate}")

    best = project.best(n=5)
    if best:
        click.echo("\nTop 5 by grand canonical energy:")
        for i, ind in enumerate(best, 1):
            g = (f"{ind.grand_canonical_energy:.4f}"
                 if ind.grand_canonical_energy is not None else "N/A")
            ads = ind.extra_data.get("adsorbate_counts", {})
            click.echo(f"  {i}. G={g} eV  {ind.id}  op={ind.operator}  ads={ads}")


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
    workspace = run_dir_path / "workspace"

    if not workspace.exists():
        click.echo(f"No workspace at {workspace}", err=True)
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
    project = GaloopProject(run_dir_path)
    generate(project=project, cfg=cfg, output_path=out, top_n=top)
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
    workspace = run_dir_path / "workspace"

    if not workspace.exists():
        click.echo(f"No workspace at {workspace}", err=True)
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

    project = GaloopProject(run_dir_path)
    all_converged = project.get_by_status(STATUS.CONVERGED)
    all_dups = project.get_by_status(STATUS.DUPLICATE)

    out_path = Path(output) if output else run_dir_path / "graphs.html"
    pages: list[dict] = []

    def _load_atoms(ind):
        job = project.get_job_by_id(ind.id)
        if job is None:
            return None
        for name in ("CONTCAR", "POSCAR"):
            candidate = Path(job.path) / name
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

    if not pages:
        click.echo("No graphs to display.", err=True)
        sys.exit(1)

    graph_viz.generate_html(pages, out_path)
    click.echo(f"Graph viewer written to {out_path}")
    webbrowser.open(out_path.as_uri())


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
# _relax  (called by row for each pending job)
# ---------------------------------------------------------------------------

@cli.command("_relax", hidden=True)
@click.argument("job_id")
@click.option("--run-dir", "-d", required=True, type=click.Path(exists=True))
def relax(job_id: str, run_dir: str) -> None:
    """Run the calculator pipeline for a single structure (called by row)."""
    import math
    from ase.io import read
    from galoop.engine.calculator import build_pipeline
    from galoop.project import STATUS_RELAXED
    from galoop.individual import STATUS
    from galoop.science.surface import detect_desorption, load_slab

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    log = logging.getLogger(__name__)

    run_dir_path = Path(run_dir).resolve()
    config_path = run_dir_path / "galoop.yaml"

    project = GaloopProject(run_dir_path)
    job = project.get_job_by_id(job_id)

    if job is None:
        log.error("No job found for id %s", job_id)
        sys.exit(1)

    # Mark as submitted while pipeline runs
    job.doc["status"] = STATUS.SUBMITTED

    struct_dir = Path(job.path)

    try:
        cfg = load_config(config_path)
    except Exception as exc:
        log.error("Config load failed: %s", exc)
        job.doc["status"] = STATUS.FAILED
        sys.exit(1)

    poscar = struct_dir / "POSCAR"
    if not poscar.exists():
        log.error("POSCAR not found: %s", poscar)
        job.doc["status"] = STATUS.FAILED
        sys.exit(1)

    try:
        atoms = read(str(poscar), format="vasp")
        slab_geometry = config_path.parent / cfg.slab.geometry
        slab_info = load_slab(slab_geometry,
                              cfg.slab.sampling_zmin,
                              cfg.slab.sampling_zmax)
        pipeline = build_pipeline(cfg.calculator_stages)
    except Exception as exc:
        log.exception("Setup failed: %s", exc)
        job.doc["status"] = STATUS.FAILED
        sys.exit(1)

    mace_model = cfg.mace_model
    _candidate = config_path.parent / mace_model
    if _candidate.exists():
        mace_model = str(_candidate)

    result = pipeline.run(
        atoms, struct_dir,
        mace_model=mace_model,
        mace_device=cfg.mace_device,
        mace_dtype=cfg.mace_dtype,
        n_slab_atoms=slab_info.n_slab_atoms,
    )

    if not result["converged"] or math.isnan(float(result["final_energy"])):
        job.doc["status"] = STATUS.FAILED
        sys.exit(0)

    if detect_desorption(result["final_atoms"], slab_info):
        job.doc["status"] = STATUS.DESORBED
        sys.exit(0)

    job.doc["status"] = STATUS_RELAXED
    sys.exit(0)


if __name__ == "__main__":
    cli()
