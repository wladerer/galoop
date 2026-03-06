"""
tests/test_scheduler.py

Tests for the parsl-based scheduler: config building and app execution.
"""

from __future__ import annotations

import pytest
import parsl
from parsl.config import Config


# ---------------------------------------------------------------------------
# build_parsl_config
# ---------------------------------------------------------------------------

class TestBuildParslConfig:

    def test_local_returns_config(self, minimal_config):
        from galoop.engine.scheduler import build_parsl_config
        cfg = build_parsl_config(minimal_config.scheduler)
        assert isinstance(cfg, Config)

    def test_local_executor_label(self, minimal_config):
        from galoop.engine.scheduler import build_parsl_config
        cfg = build_parsl_config(minimal_config.scheduler)
        assert cfg.executors[0].label == "local"

    def test_local_nworkers(self):
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.executors import ThreadPoolExecutor
        sched = SchedulerConfig(type="local", nworkers=3)
        cfg = build_parsl_config(sched)
        executor = cfg.executors[0]
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor.max_threads == 3

    def test_slurm_executor_label(self):
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        sched = SchedulerConfig(type="slurm", nworkers=8, resources={"partition": "gpu"})
        cfg = build_parsl_config(sched)
        assert cfg.executors[0].label == "slurm"

    def test_pbs_executor_label(self):
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        sched = SchedulerConfig(type="pbs", nworkers=4, resources={"queue": "batch"})
        cfg = build_parsl_config(sched)
        assert cfg.executors[0].label == "pbs"

    def test_accepts_plain_dict(self):
        from galoop.engine.scheduler import build_parsl_config
        cfg = build_parsl_config({"type": "local", "nworkers": 2})
        assert isinstance(cfg, Config)

    def test_single_executor(self, minimal_config):
        from galoop.engine.scheduler import build_parsl_config
        cfg = build_parsl_config(minimal_config.scheduler)
        assert len(cfg.executors) == 1


# ---------------------------------------------------------------------------
# relax_structure bash_app
# ---------------------------------------------------------------------------

class TestRelaxStructureApp:

    def test_is_callable(self):
        from galoop.engine.scheduler import relax_structure
        assert callable(relax_structure)

    def test_command_template(self):
        """The underlying function should produce the expected shell command."""
        from galoop.engine.scheduler import relax_structure
        # Access the unwrapped function to inspect the command template
        cmd = relax_structure.func("/path/to/struct", "/path/to/config.yaml")
        assert "galoop _run-pipeline" in cmd
        assert "/path/to/struct" in cmd
        assert "/path/to/config.yaml" in cmd

    def test_config_path_in_command(self):
        from galoop.engine.scheduler import relax_structure
        cmd = relax_structure.func("/run/gcga/structure_00000", "/run/galoop.yaml")
        assert "--config /run/galoop.yaml" in cmd


# ---------------------------------------------------------------------------
# Parsl local execution
# ---------------------------------------------------------------------------

class TestParslLocalExecution:

    def test_trivial_bash_app_runs(self, tmp_path):
        """Submit a trivial bash job through a local parsl executor."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import bash_app

        sched = SchedulerConfig(type="local", nworkers=1)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @bash_app
            def echo_hello(stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
                return "echo hello_from_parsl"

            future = echo_hello(
                stdout=str(tmp_path / "out.txt"),
                stderr=str(tmp_path / "err.txt"),
            )
            future.result(timeout=30)
            assert (tmp_path / "out.txt").read_text().strip() == "hello_from_parsl"
        finally:
            parsl.clear()

    def test_failed_bash_app_raises(self, tmp_path):
        """A bash app that exits non-zero should raise on .result()."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import bash_app

        sched = SchedulerConfig(type="local", nworkers=1)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @bash_app
            def fail_app(stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
                return "exit 1"

            future = fail_app(
                stdout=str(tmp_path / "out.txt"),
                stderr=str(tmp_path / "err.txt"),
            )
            with pytest.raises(Exception):
                future.result(timeout=30)
        finally:
            parsl.clear()

    def test_multiple_futures_complete(self, tmp_path):
        """Multiple bash apps submitted to a local executor all complete."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import bash_app

        sched = SchedulerConfig(type="local", nworkers=2)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @bash_app
            def write_n(n: int, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
                return f"echo {n}"

            futures = [
                write_n(i,
                        stdout=str(tmp_path / f"out_{i}.txt"),
                        stderr=str(tmp_path / f"err_{i}.txt"))
                for i in range(4)
            ]
            for f in futures:
                f.result(timeout=30)
        finally:
            parsl.clear()
