"""
galoop/engine/scheduler.py

Job scheduler abstraction: local, SLURM, PBS.
"""

from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Scheduler(ABC):
    """Interface that all schedulers must implement."""

    nworkers: int

    @abstractmethod
    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        """Submit *script* and return a job ID string."""
        ...

    @abstractmethod
    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Poll statuses for a batch of job IDs."""
        ...

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a job.  Return True on success."""
        ...


# ---------------------------------------------------------------------------
# Local (subprocess) scheduler
# ---------------------------------------------------------------------------

class LocalScheduler(Scheduler):
    """Run jobs as local sub-processes — no HPC queue needed."""

    def __init__(self, nworkers: int = 4):
        self.nworkers = nworkers
        self._jobs: dict[str, tuple[subprocess.Popen, float]] = {}
        self._next_id = 0

    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        job_id = f"local_{self._next_id:06d}"
        self._next_id += 1

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "run_job.sh"
        script_file.write_text(f"#!/bin/bash\nset -e\n{script}\n")
        script_file.chmod(0o755)

        proc = subprocess.Popen(
            ["bash", str(script_file)],
            cwd=str(workdir),
            stdout=open(workdir / "stdout.txt", "w"),
            stderr=open(workdir / "stderr.txt", "w"),
        )
        self._jobs[job_id] = (proc, time.time())
        log.info("Submitted local job %s (PID %d)", job_id, proc.pid)
        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        result: dict[str, JobStatus] = {}
        for jid in job_ids:
            if jid not in self._jobs:
                result[jid] = JobStatus.UNKNOWN
                continue
            proc, _ = self._jobs[jid]
            rc = proc.poll()
            if rc is None:
                result[jid] = JobStatus.RUNNING
            elif rc == 0:
                result[jid] = JobStatus.DONE
                del self._jobs[jid]
            else:
                result[jid] = JobStatus.FAILED
                del self._jobs[jid]
        return result

    def cancel(self, job_id: str) -> bool:
        if job_id not in self._jobs:
            return False
        proc, _ = self._jobs[job_id]
        try:
            proc.terminate()
            del self._jobs[job_id]
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

class SlurmScheduler(Scheduler):

    def __init__(
        self,
        nworkers: int = 4,
        walltime: str = "01:00:00",
        partition: str = "default",
        **resources,
    ):
        self.nworkers = nworkers
        self.walltime = walltime
        self.partition = partition
        self.resources = resources

    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "job_script.slurm"
        script_file.write_text(
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={job_name}\n"
            f"#SBATCH --time={self.walltime}\n"
            f"#SBATCH --partition={self.partition}\n"
            f"#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task=4\n"
            f"#SBATCH --output={workdir}/slurm.out\n"
            f"#SBATCH --error={workdir}/slurm.err\n\n"
            f"set -e\ncd {workdir}\n{script}\n"
        )
        script_file.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_file)],
            capture_output=True, text=True, check=True,
        )
        job_id = result.stdout.strip().split()[-1]
        log.info("Submitted SLURM job %s", job_id)
        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        result: dict[str, JobStatus] = {}
        for jid in job_ids:
            try:
                out = subprocess.run(
                    ["sacct", "-j", jid, "-n", "-o", "State"],
                    capture_output=True, text=True, timeout=5,
                ).stdout.strip()
                state = out.split()[0].upper() if out else "UNKNOWN"
                mapping = {
                    "RUNNING": JobStatus.RUNNING,
                    "COMPLETED": JobStatus.DONE,
                    "COMPLETING": JobStatus.DONE,
                    "FAILED": JobStatus.FAILED,
                    "TIMEOUT": JobStatus.FAILED,
                    "CANCELLED": JobStatus.FAILED,
                    "PENDING": JobStatus.PENDING,
                    "CONFIGURING": JobStatus.PENDING,
                }
                result[jid] = mapping.get(state, JobStatus.UNKNOWN)
            except Exception:
                result[jid] = JobStatus.UNKNOWN
        return result

    def cancel(self, job_id: str) -> bool:
        try:
            subprocess.run(["scancel", job_id], check=True, timeout=5)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# PBS / Torque
# ---------------------------------------------------------------------------

class PbsScheduler(Scheduler):

    def __init__(
        self,
        nworkers: int = 4,
        walltime: str = "01:00:00",
        queue: str = "default",
        **resources,
    ):
        self.nworkers = nworkers
        self.walltime = walltime
        self.queue = queue
        self.resources = resources

    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "job_script.pbs"
        script_file.write_text(
            f"#!/bin/bash\n"
            f"#PBS -N {job_name}\n"
            f"#PBS -l walltime={self.walltime}\n"
            f"#PBS -q {self.queue}\n"
            f"#PBS -o {workdir}/pbs.out\n"
            f"#PBS -e {workdir}/pbs.err\n\n"
            f"set -e\ncd {workdir}\n{script}\n"
        )
        script_file.chmod(0o755)

        result = subprocess.run(
            ["qsub", str(script_file)],
            capture_output=True, text=True, check=True,
        )
        job_id = result.stdout.strip()
        log.info("Submitted PBS job %s", job_id)
        return job_id

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        result: dict[str, JobStatus] = {}
        for jid in job_ids:
            try:
                out = subprocess.run(
                    ["qstat", jid],
                    capture_output=True, text=True, timeout=5,
                ).stdout
                if "unknown Job" in out or not out:
                    result[jid] = JobStatus.DONE
                elif " R " in out:
                    result[jid] = JobStatus.RUNNING
                elif " Q " in out:
                    result[jid] = JobStatus.PENDING
                else:
                    result[jid] = JobStatus.UNKNOWN
            except Exception:
                result[jid] = JobStatus.UNKNOWN
        return result

    def cancel(self, job_id: str) -> bool:
        try:
            subprocess.run(["qdel", job_id], check=True, timeout=5)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Builder — extracts only what each scheduler needs
# ---------------------------------------------------------------------------

def build_scheduler(config) -> Scheduler:
    """
    Build a scheduler from a :class:`SchedulerConfig` (or plain dict).
    """
    if hasattr(config, "model_dump"):
        cfg = config.model_dump()
    elif hasattr(config, "items"):
        cfg = dict(config)
    else:
        cfg = dict(config)

    sched_type = cfg.get("type", "local").lower()
    nworkers = cfg.get("nworkers", 4)
    walltime = cfg.get("walltime", "01:00:00")
    resources = cfg.get("resources", {})

    if sched_type == "slurm":
        return SlurmScheduler(nworkers=nworkers, walltime=walltime, **resources)
    elif sched_type == "pbs":
        return PbsScheduler(nworkers=nworkers, walltime=walltime, **resources)
    else:
        return LocalScheduler(nworkers=nworkers)
