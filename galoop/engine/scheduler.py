"""
galoop/engine/scheduler.py

Parsl-based job execution.  Provides build_parsl_config() and the
relax_structure bash_app used by the GA loop.
"""

from __future__ import annotations

import logging

import parsl
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider, SlurmProvider, TorqueProvider

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsl app
# ---------------------------------------------------------------------------

@bash_app
def relax_structure(
    struct_dir: str,
    config_path: str,
    stdout: str = parsl.AUTO_LOGNAME,
    stderr: str = parsl.AUTO_LOGNAME,
) -> str:
    """Run the single-structure relaxation pipeline as a bash command."""
    return f"galoop _run-pipeline {struct_dir} --config {config_path}"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_parsl_config(config, run_dir=None) -> Config:
    """
    Build a parsl :class:`Config` from a :class:`SchedulerConfig`.

    Supported types
    ---------------
    local  — ThreadPoolExecutor, up to *nworkers* concurrent processes
    slurm  — HighThroughputExecutor with SlurmProvider (one worker per block)
    pbs    — HighThroughputExecutor with TorqueProvider (one worker per block)

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

    parsl_run_dir = str(run_dir) if run_dir else "runinfo"
    worker_log_dir = str(run_dir) if run_dir else None

    if sched_type == "slurm":
        executor = HighThroughputExecutor(
            label="slurm",
            worker_logdir_root=worker_log_dir,
            provider=SlurmProvider(
                partition=resources.get("partition", "default"),
                walltime=walltime,
                nodes_per_block=1,
                max_blocks=nworkers,
                min_blocks=0,
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
        executor = ThreadPoolExecutor(max_threads=nworkers, label="local")
        log.info("Parsl executor: local  max_threads=%d", nworkers)

    return Config(executors=[executor], run_dir=parsl_run_dir)
