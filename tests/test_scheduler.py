"""
tests/test_scheduler.py

Tests for the Parsl-based scheduler: config building and app execution.
"""

from __future__ import annotations

import pytest

parsl = pytest.importorskip("parsl", reason="parsl not installed; scheduler tests skipped")
Config = pytest.importorskip("parsl.config", reason="parsl not installed").Config


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
        from parsl.executors import HighThroughputExecutor
        sched = SchedulerConfig(type="local", nworkers=3)
        cfg = build_parsl_config(sched)
        executor = cfg.executors[0]
        # Local now uses HTEX-over-LocalProvider so each MACE worker is its
        # own process — ThreadPoolExecutor caused MACE calc state races.
        assert isinstance(executor, HighThroughputExecutor)
        assert executor.max_workers_per_node == 3

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
# relax_structure python_app
# ---------------------------------------------------------------------------

class TestRelaxStructureApp:

    def test_is_callable(self):
        from galoop.engine.scheduler import relax_structure
        assert callable(relax_structure)


# ---------------------------------------------------------------------------
# Parsl local execution
# ---------------------------------------------------------------------------

class TestParslLocalExecution:

    def test_trivial_python_app_runs(self, tmp_path):
        """Submit a trivial python job through a local parsl executor."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import python_app

        sched = SchedulerConfig(type="local", nworkers=1)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @python_app
            def add(a, b):
                return a + b

            future = add(2, 3)
            assert future.result(timeout=30) == 5
        finally:
            parsl.clear()

    def test_failed_python_app_raises(self, tmp_path):
        """A python app that raises should raise on .result()."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import python_app

        sched = SchedulerConfig(type="local", nworkers=1)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @python_app
            def fail_app():
                raise ValueError("intentional failure")

            future = fail_app()
            with pytest.raises(Exception):
                future.result(timeout=30)
        finally:
            parsl.clear()

    def test_multiple_futures_complete(self, tmp_path):
        """Multiple python apps submitted to a local executor all complete."""
        from galoop.config import SchedulerConfig
        from galoop.engine.scheduler import build_parsl_config
        from parsl.app.app import python_app

        sched = SchedulerConfig(type="local", nworkers=2)
        cfg = build_parsl_config(sched, run_dir=tmp_path)
        dfk = parsl.load(cfg)
        try:
            @python_app
            def square(n):
                return n * n

            futures = [square(i) for i in range(4)]
            results = [f.result(timeout=30) for f in futures]
            assert results == [0, 1, 4, 9]
        finally:
            parsl.clear()
