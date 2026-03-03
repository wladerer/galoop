"""
gocia/engine/scheduler.py

Job scheduler abstraction: local, SLURM, PBS.
Submits jobs, polls status, retrieves results.
"""

from __future__ import annotations

import logging
import subprocess
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    UNKNOWN = "unknown"


class JobInfo(NamedTuple):
    """Job status information."""
    job_id: str
    status: JobStatus
    exit_code: int | None


class Scheduler(ABC):
    """Base scheduler interface."""

    @abstractmethod
    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        """
        Submit a job.

        Parameters
        ----------
        job_name : Job identifier
        script : Shell script to run
        workdir : Working directory (cwd for job)

        Returns
        -------
        str : Job ID
        """
        pass

    @abstractmethod
    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """
        Get status of jobs.

        Parameters
        ----------
        job_ids : List of job IDs

        Returns
        -------
        dict : {job_id: JobStatus}
        """
        pass

    @abstractmethod
    def cancel(self, job_id: str) -> bool:
        """Cancel a job."""
        pass


class LocalScheduler(Scheduler):
    """
    Local (non-HPC) scheduler.
    Spawns processes via subprocess; no actual job queuing.
    """

    def __init__(self, nworkers: int = 4):
        self.nworkers = nworkers
        self.jobs = {}  # {job_id: (process, start_time)}
        self.next_job_id = 0

    def submit(self, job_name: str, script: str, workdir: Path) -> str:
        """Submit a local job."""
        job_id = f"local_{self.next_job_id:06d}"
        self.next_job_id += 1

        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "run_job.sh"
        script_file.write_text(f"#!/bin/bash\nset -e\n{script}\n")
        script_file.chmod(0o755)

        try:
            proc = subprocess.Popen(
                ["bash", str(script_file)],
                cwd=str(workdir),
                stdout=open(workdir / "stdout.txt", "w"),
                stderr=open(workdir / "stderr.txt", "w"),
            )
            self.jobs[job_id] = (proc, time.time())
            log.info(f"  Submitted local job {job_id} (PID {proc.pid})")
            return job_id
        except Exception as e:
            log.error(f"  Failed to submit job: {e}")
            raise

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Poll job statuses."""
        result = {}
        for job_id in job_ids:
            if job_id not in self.jobs:
                result[job_id] = JobStatus.UNKNOWN
                continue

            proc, _ = self.jobs[job_id]
            poll = proc.poll()

            if poll is None:
                result[job_id] = JobStatus.RUNNING
            elif poll == 0:
                result[job_id] = JobStatus.DONE
                del self.jobs[job_id]
            else:
                result[job_id] = JobStatus.FAILED
                del self.jobs[job_id]

        return result

    def cancel(self, job_id: str) -> bool:
        """Cancel a local job."""
        if job_id not in self.jobs:
            return False
        proc, _ = self.jobs[job_id]
        try:
            proc.terminate()
            del self.jobs[job_id]
            return True
        except Exception:
            return False


class SlurmScheduler(Scheduler):
    """SLURM job scheduler."""

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
        """Submit a SLURM job."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "job_script.slurm"
        slurm_header = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={self.walltime}
#SBATCH --partition={self.partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output={workdir}/slurm.out
#SBATCH --error={workdir}/slurm.err

set -e
cd {workdir}
{script}
"""
        script_file.write_text(slurm_header)
        script_file.chmod(0o755)

        try:
            result = subprocess.run(
                ["sbatch", str(script_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = result.stdout.strip().split()[-1]
            log.info(f"  Submitted SLURM job {job_id}")
            return job_id
        except Exception as e:
            log.error(f"  Failed to submit SLURM job: {e}")
            raise

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Poll SLURM job statuses via sacct."""
        result = {}
        for job_id in job_ids:
            try:
                output = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-o", "State"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()

                state = output.split()[0].upper() if output else "UNKNOWN"

                if state in ("RUNNING",):
                    result[job_id] = JobStatus.RUNNING
                elif state in ("COMPLETED", "COMPLETING"):
                    result[job_id] = JobStatus.DONE
                elif state in ("FAILED", "TIMEOUT", "CANCELLED"):
                    result[job_id] = JobStatus.FAILED
                elif state in ("PENDING", "CONFIGURING"):
                    result[job_id] = JobStatus.PENDING
                else:
                    result[job_id] = JobStatus.UNKNOWN
            except Exception:
                result[job_id] = JobStatus.UNKNOWN

        return result

    def cancel(self, job_id: str) -> bool:
        """Cancel a SLURM job."""
        try:
            subprocess.run(["scancel", job_id], check=True, timeout=5)
            return True
        except Exception:
            return False


class PbsScheduler(Scheduler):
    """PBS/Torque job scheduler."""

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
        """Submit a PBS job."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        script_file = workdir / "job_script.pbs"
        pbs_header = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l walltime={self.walltime}
#PBS -q {self.queue}
#PBS -o {workdir}/pbs.out
#PBS -e {workdir}/pbs.err

set -e
cd {workdir}
{script}
"""
        script_file.write_text(pbs_header)
        script_file.chmod(0o755)

        try:
            result = subprocess.run(
                ["qsub", str(script_file)],
                capture_output=True,
                text=True,
                check=True,
            )
            job_id = result.stdout.strip()
            log.info(f"  Submitted PBS job {job_id}")
            return job_id
        except Exception as e:
            log.error(f"  Failed to submit PBS job: {e}")
            raise

    def status(self, job_ids: list[str]) -> dict[str, JobStatus]:
        """Poll PBS job statuses via qstat."""
        result = {}
        for job_id in job_ids:
            try:
                output = subprocess.run(
                    ["qstat", job_id],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout

                if "unknown Job" in output or not output:
                    result[job_id] = JobStatus.DONE
                elif "R" in output:
                    result[job_id] = JobStatus.RUNNING
                elif "Q" in output:
                    result[job_id] = JobStatus.PENDING
                else:
                    result[job_id] = JobStatus.UNKNOWN
            except Exception:
                result[job_id] = JobStatus.UNKNOWN

        return result

    def cancel(self, job_id: str) -> bool:
        """Cancel a PBS job."""
        try:
            subprocess.run(["qdel", job_id], check=True, timeout=5)
            return True
        except Exception:
            return False


def build_scheduler(config) -> Scheduler:
    """
    Build a scheduler from config.

    Parameters
    ----------
    config : SchedulerConfig with keys:
        - type: "local", "slurm", or "pbs"
        - nworkers: int
        - walltime: str (HH:MM:SS)
        - resources: dict (extra params)

    Returns
    -------
    Scheduler subclass
    """
    cfg_dict = dict(config) if hasattr(config, "items") else config.model_dump()
    sched_type = cfg_dict.pop("type", "local").lower()

    if sched_type == "slurm":
        return SlurmScheduler(**cfg_dict)
    elif sched_type == "pbs":
        return PbsScheduler(**cfg_dict)
    else:  # local
        return LocalScheduler(**cfg_dict)
