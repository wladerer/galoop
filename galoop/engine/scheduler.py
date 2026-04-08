"""
galoop/engine/scheduler.py

Parsl-based job execution.  Provides build_parsl_config() and the
relax_structure python_app used by the GA loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from parsl.app.app import python_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider, SlurmProvider, TorqueProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsl app — runs the multi-stage calculator pipeline on a worker
# ---------------------------------------------------------------------------

@python_app
def relax_structure(
    struct_dir: str,
    stage_configs: list[dict],
    mace_model: str = "medium",
    mace_device: str = "cpu",
    mace_dtype: str = "float32",
    n_slab_atoms: int = 0,
) -> dict:
    """Run the calculator pipeline for one structure.

    Executed on a Parsl worker.  Returns the pipeline result dict with
    keys: converged, final_energy, stage_results, final_atoms.
    """
    from pathlib import Path

    from galoop.engine.calculator import build_pipeline
    from galoop.science.surface import read_atoms

    struct_path = Path(struct_dir)
    poscar = struct_path / "POSCAR"
    atoms = read_atoms(poscar, format="vasp")

    pipeline = build_pipeline(stage_configs)
    return pipeline.run(
        atoms,
        struct_path,
        mace_model=mace_model,
        mace_device=mace_device,
        mace_dtype=mace_dtype,
        n_slab_atoms=n_slab_atoms,
    )


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_parsl_config(config, run_dir=None) -> Config:
    """
    Build a Parsl :class:`Config` from a :class:`SchedulerConfig`.

    Supported types
    ---------------
    local  — ThreadPoolExecutor, up to *nworkers* concurrent processes
    slurm  — HighThroughputExecutor with SlurmProvider
    pbs    — HighThroughputExecutor with TorqueProvider

    Parameters
    ----------
    config   : SchedulerConfig or dict
    run_dir  : Root run directory.  Parsl's runinfo/ and executor working
               directories are placed inside it so nothing escapes to the CWD.
    """
    if hasattr(config, "model_dump"):
        cfg = config.model_dump()
    else:
        cfg = dict(config)

    sched_type = cfg.get("type", "local").lower()
    nworkers   = cfg.get("nworkers", 4)
    walltime   = cfg.get("walltime", "01:00:00")
    resources  = cfg.get("resources", {})

    if run_dir:
        log_dir = Path(run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        parsl_run_dir = str(log_dir)
        worker_log_dir = str(log_dir)
    else:
        parsl_run_dir = "logs"
        worker_log_dir = None

    if sched_type == "slurm":
        scheduler_options = ""
        if "partition" in resources:
            scheduler_options += f"#SBATCH --partition={resources['partition']}\n"
        if "account" in resources:
            scheduler_options += f"#SBATCH --account={resources['account']}\n"
        for extra in resources.get("extra", []):
            scheduler_options += f"#SBATCH {extra}\n"

        worker_init_lines = []
        for mod in resources.get("modules", []):
            worker_init_lines.append(f"module load {mod}")
        for key, val in resources.get("env", {}).items():
            worker_init_lines.append(f"export {key}={val}")
        worker_init = "\n".join(worker_init_lines)

        executor = HighThroughputExecutor(
            label="slurm",
            worker_logdir_root=worker_log_dir,
            provider=SlurmProvider(
                partition=resources.get("partition", "default"),
                walltime=walltime,
                nodes_per_block=1,
                max_blocks=nworkers,
                min_blocks=0,
                scheduler_options=scheduler_options,
                worker_init=worker_init if worker_init else "",
            ),
        )
        log.info("Parsl executor: SLURM  partition=%s  max_blocks=%d",
                 resources.get("partition", "default"), nworkers)

    elif sched_type == "pbs":
        executor = HighThroughputExecutor(
            label="pbs",
            worker_logdir_root=worker_log_dir,
            provider=TorqueProvider(
                queue=resources.get("queue", "default"),
                walltime=walltime,
                nodes_per_block=1,
                max_blocks=nworkers,
                min_blocks=0,
            ),
        )
        log.info("Parsl executor: PBS  queue=%s  max_blocks=%d",
                 resources.get("queue", "default"), nworkers)

    else:
        # HTEX over LocalProvider: each worker is its own process, so the
        # MACE calculator (which is not thread-safe) gets a fresh per-process
        # instance instead of being shared across threads.
        executor = HighThroughputExecutor(
            label="local",
            worker_logdir_root=worker_log_dir,
            max_workers_per_node=nworkers,
            provider=LocalProvider(
                init_blocks=1,
                min_blocks=1,
                max_blocks=1,
            ),
        )
        log.info("Parsl executor: local HTEX  workers=%d", nworkers)

    return Config(executors=[executor], run_dir=parsl_run_dir)
