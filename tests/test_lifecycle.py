"""
tests/test_lifecycle.py

Tests for job status transitions and persistence across project re-opens.
Sentinels are gone — status lives in job.doc.  Restart = open the same
signac workspace again.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from galoop.individual import OPERATOR, STATUS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pop(config, slab_info, project, seed=42):
    from galoop.galoop import _build_initial_population
    rng = np.random.default_rng(seed)
    _build_initial_population(config, slab_info, project, rng)


def _fake_converge(ind, project, energy: float = -1.0):
    """Copy POSCAR→CONTCAR, write FINAL_ENERGY, and update project to CONVERGED."""
    job = project.get_job_by_id(ind.id)
    struct_dir = Path(job.path)
    poscar = struct_dir / "POSCAR"
    contcar = struct_dir / "CONTCAR"
    shutil.copy(poscar, contcar)
    (struct_dir / "FINAL_ENERGY").write_text(f"{energy:.10f}\n")
    updated = ind.with_energy(raw=energy * 100, grand_canonical=energy)
    project.update(updated)
    return updated


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:

    def test_pending_to_submitted(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        updated = ind.with_status(STATUS.SUBMITTED)
        temp_db.update(updated)
        stored = temp_db.get(ind.id)
        assert stored.status == STATUS.SUBMITTED

    def test_submitted_to_converged_stores_energy(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        updated = ind.with_energy(raw=-500.0, grand_canonical=-1.23)
        temp_db.update(updated)
        stored = temp_db.get(ind.id)
        assert stored.status == STATUS.CONVERGED
        assert stored.raw_energy == pytest.approx(-500.0)
        assert stored.grand_canonical_energy == pytest.approx(-1.23)

    def test_converged_is_selectable(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_energy(raw=-500.0, grand_canonical=-1.0))
        pool = temp_db.selectable_pool()
        assert any(p.id == ind.id for p in pool)

    def test_desorbed_excluded_from_selectable_pool(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_status(STATUS.DESORBED))
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_failed_excluded_from_selectable_pool(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_status(STATUS.FAILED))
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_duplicate_excluded_from_selectable_pool(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.mark_duplicate())
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_full_lifecycle_counts(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        pending = temp_db.get_by_status(STATUS.PENDING)
        assert len(pending) == minimal_config.ga.population_size

        for ind in pending[:2]:
            temp_db.update(ind.with_energy(raw=-500.0, grand_canonical=-1.0))
        temp_db.update(pending[2].with_status(STATUS.FAILED))
        temp_db.update(pending[3].with_status(STATUS.DESORBED))

        counts = temp_db.count_by_status()
        assert counts[STATUS.CONVERGED] == 2
        assert counts[STATUS.FAILED] == 1
        assert counts[STATUS.DESORBED] == 1
        assert counts[STATUS.PENDING] == minimal_config.ga.population_size - 4

    def test_job_doc_status_updated_on_converge(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        _fake_converge(ind, temp_db, energy=-1.5)
        job = temp_db.get_job_by_id(ind.id)
        assert job.doc["status"] == STATUS.CONVERGED


# ---------------------------------------------------------------------------
# Restart (re-open the same signac workspace)
# ---------------------------------------------------------------------------

class TestRestart:

    def test_population_not_rebuilt_if_workspace_exists(
        self, tmp_path, minimal_config, slab_info
    ):
        """The run() gate: population is not rebuilt if workspace already has jobs."""
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        project = GaloopProject(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, project, rng)
        n_before = len(project.get_by_status(STATUS.PENDING))

        # Simulate the restart gate from galoop.run()
        if not any(True for _ in project._project):
            _build_initial_population(minimal_config, slab_info, project, rng)

        assert len(project.get_by_status(STATUS.PENDING)) == n_before

    def test_jobs_persist_after_reopen(self, tmp_path, minimal_config, slab_info):
        """All structures survive a project close/reopen cycle."""
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        rng = np.random.default_rng(42)
        project = GaloopProject(tmp_path)
        _build_initial_population(minimal_config, slab_info, project, rng)
        n_written = len(project.get_by_status(STATUS.PENDING))

        # Re-open from same path
        project2 = GaloopProject(tmp_path)
        n_read = len(project2.get_by_status(STATUS.PENDING))
        assert n_read == n_written

    def test_converged_count_survives_restart(self, tmp_path, minimal_config, slab_info):
        """Converged structures are still queryable after reopening."""
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        project = GaloopProject(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, project, rng)
        pending = project.get_by_status(STATUS.PENDING)
        for ind in pending[:3]:
            _fake_converge(ind, project, energy=-1.0)
        n_converged = len(project.get_by_status(STATUS.CONVERGED))

        project2 = GaloopProject(tmp_path)
        assert len(project2.get_by_status(STATUS.CONVERGED)) == n_converged

    def test_pending_structures_preserved_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        project = GaloopProject(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, project, rng)
        pending = project.get_by_status(STATUS.PENDING)
        for ind in pending[: len(pending) // 2]:
            _fake_converge(ind, project)
        n_still_pending = len(project.get_by_status(STATUS.PENDING))

        project2 = GaloopProject(tmp_path)
        assert len(project2.get_by_status(STATUS.PENDING)) == n_still_pending

    def test_extra_data_survives_restart(self, tmp_path, minimal_config, slab_info):
        """Adsorbate counts round-trip through a project reopen."""
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        project = GaloopProject(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, project, rng)
        before = {
            ind.id: ind.extra_data["adsorbate_counts"]
            for ind in project.get_by_status(STATUS.PENDING)
        }

        project2 = GaloopProject(tmp_path)
        after = {
            ind.id: ind.extra_data["adsorbate_counts"]
            for ind in project2.get_by_status(STATUS.PENDING)
        }
        assert before == after

    def test_selectable_pool_rebuilt_correctly_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        from galoop.galoop import _build_initial_population
        from galoop.project import GaloopProject

        project = GaloopProject(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, project, rng)

        converged_ids = set()
        for ind in project.get_by_status(STATUS.PENDING)[:4]:
            updated = _fake_converge(ind, project,
                                     energy=-1.0 - len(converged_ids))
            converged_ids.add(updated.id)

        project2 = GaloopProject(tmp_path)
        pool_ids = {p.id for p in project2.selectable_pool()}
        assert converged_ids == pool_ids
