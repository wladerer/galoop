"""
tests/test_lifecycle.py

Tests for job status transitions, DB consistency through a run, and restart logic.
Uses no actual calculator — structures are advanced by writing sentinels manually.
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

def _write_sentinel(struct_dir: Path, state: str) -> None:
    for name in ("PENDING", "SUBMITTED", "CONVERGED", "FAILED", "DUPLICATE", "DESORBED"):
        (struct_dir / name).unlink(missing_ok=True)
    (struct_dir / state).touch()


def _build_pop(config, slab_info, db, run_dir, seed=42):
    from galoop.galoop import _build_initial_population
    rng = np.random.default_rng(seed)
    _build_initial_population(config, slab_info, db, run_dir, rng)


def _fake_converge(ind, db, struct_dir: Path, energy: float = -1.0):
    """Copy POSCAR→CONTCAR, write FINAL_ENERGY, and update DB to CONVERGED."""
    poscar = struct_dir / "POSCAR"
    contcar = struct_dir / "CONTCAR"
    shutil.copy(poscar, contcar)
    (struct_dir / "FINAL_ENERGY").write_text(f"{energy:.10f}\n")
    _write_sentinel(struct_dir, "CONVERGED")
    updated = ind.with_energy(raw=energy * 100, grand_canonical=energy)
    db.update(updated)
    return updated


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:

    def test_pending_to_submitted(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        updated = ind.with_status(STATUS.SUBMITTED)
        temp_db.update(updated)
        stored = temp_db.get(ind.id)
        assert stored.status == STATUS.SUBMITTED

    def test_submitted_to_converged_stores_energy(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        updated = ind.with_energy(raw=-500.0, grand_canonical=-1.23)
        temp_db.update(updated)
        stored = temp_db.get(ind.id)
        assert stored.status == STATUS.CONVERGED
        assert stored.raw_energy == pytest.approx(-500.0)
        assert stored.grand_canonical_energy == pytest.approx(-1.23)

    def test_converged_is_selectable(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_energy(raw=-500.0, grand_canonical=-1.0))
        pool = temp_db.selectable_pool()
        assert any(p.id == ind.id for p in pool)

    def test_desorbed_excluded_from_selectable_pool(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_status(STATUS.DESORBED))
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_failed_excluded_from_selectable_pool(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.with_status(STATUS.FAILED))
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_duplicate_excluded_from_selectable_pool(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        temp_db.update(ind.mark_duplicate())
        assert all(p.id != ind.id for p in temp_db.selectable_pool())

    def test_full_lifecycle_counts(self, tmp_path, minimal_config, slab_info, temp_db):
        """Simulate a mixed run: some converged, some failed, some pending."""
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        pending = temp_db.get_by_status(STATUS.PENDING)
        assert len(pending) == minimal_config.ga.population_size

        # Converge two
        for ind in pending[:2]:
            temp_db.update(ind.with_energy(raw=-500.0, grand_canonical=-1.0))
        # Fail one
        temp_db.update(pending[2].with_status(STATUS.FAILED))
        # Desorp one
        temp_db.update(pending[3].with_status(STATUS.DESORBED))

        counts = temp_db.count_by_status()
        assert counts[STATUS.CONVERGED] == 2
        assert counts[STATUS.FAILED] == 1
        assert counts[STATUS.DESORBED] == 1
        assert counts[STATUS.PENDING] == minimal_config.ga.population_size - 4

    def test_sentinel_written_on_converge(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        pending = temp_db.get_by_status(STATUS.PENDING)
        ind = pending[0]
        struct_dir = Path(ind.geometry_path).parent
        _fake_converge(ind, temp_db, struct_dir, energy=-1.5)
        assert (struct_dir / "CONVERGED").exists()
        assert not (struct_dir / "PENDING").exists()


# ---------------------------------------------------------------------------
# Restart logic
# ---------------------------------------------------------------------------

class TestRestart:

    def test_gcga_gate_prevents_rebuild(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        """The galoop.run() gate: population is not rebuilt if gcga dir exists."""
        from galoop.galoop import _build_initial_population

        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, temp_db, tmp_path, rng)
        n_before = len(temp_db.get_by_status(STATUS.PENDING))

        # Simulate the restart gate from galoop.run()
        if not (tmp_path / "gcga").exists():
            _build_initial_population(minimal_config, slab_info, temp_db, tmp_path, rng)

        assert len(temp_db.get_by_status(STATUS.PENDING)) == n_before

    def test_db_persists_after_close_and_reopen(
        self, tmp_path, minimal_config, slab_info
    ):
        """All structures survive a DB close/reopen cycle."""
        from galoop.database import GaloopDB
        from galoop.galoop import _build_initial_population

        db_path = tmp_path / "galoop.db"
        rng = np.random.default_rng(42)

        with GaloopDB(db_path) as db:
            db.setup()
            _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)
            n_written = len(db.get_by_status(STATUS.PENDING))

        with GaloopDB(db_path) as db:
            n_read = len(db.get_by_status(STATUS.PENDING))

        assert n_read == n_written

    def test_converged_count_survives_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        """Converged structures are still queryable after reopening."""
        from galoop.database import GaloopDB
        from galoop.galoop import _build_initial_population

        db_path = tmp_path / "galoop.db"
        rng = np.random.default_rng(42)

        with GaloopDB(db_path) as db:
            db.setup()
            _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)
            pending = db.get_by_status(STATUS.PENDING)
            for ind in pending[:3]:
                struct_dir = Path(ind.geometry_path).parent
                _fake_converge(ind, db, struct_dir, energy=-1.0)
            n_converged = len(db.get_by_status(STATUS.CONVERGED))

        with GaloopDB(db_path) as db:
            assert len(db.get_by_status(STATUS.CONVERGED)) == n_converged

    def test_pending_structures_preserved_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        """Pending structures are still available for submission after restart."""
        from galoop.database import GaloopDB
        from galoop.galoop import _build_initial_population

        db_path = tmp_path / "galoop.db"
        rng = np.random.default_rng(42)

        with GaloopDB(db_path) as db:
            db.setup()
            _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)
            pending = db.get_by_status(STATUS.PENDING)
            # Converge half
            for ind in pending[: len(pending) // 2]:
                struct_dir = Path(ind.geometry_path).parent
                _fake_converge(ind, db, struct_dir)
            n_still_pending = len(db.get_by_status(STATUS.PENDING))

        with GaloopDB(db_path) as db:
            assert len(db.get_by_status(STATUS.PENDING)) == n_still_pending

    def test_extra_data_survives_restart(self, tmp_path, minimal_config, slab_info):
        """Adsorbate counts round-trip through a DB close/reopen."""
        from galoop.database import GaloopDB
        from galoop.galoop import _build_initial_population

        db_path = tmp_path / "galoop.db"
        rng = np.random.default_rng(42)

        with GaloopDB(db_path) as db:
            db.setup()
            _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)
            before = {
                ind.id: ind.extra_data["adsorbate_counts"]
                for ind in db.get_by_status(STATUS.PENDING)
            }

        with GaloopDB(db_path) as db:
            after = {
                ind.id: ind.extra_data["adsorbate_counts"]
                for ind in db.get_by_status(STATUS.PENDING)
            }

        assert before == after

    def test_selectable_pool_rebuilt_correctly_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        """Converged, weighted structures are in the selectable pool on restart."""
        from galoop.database import GaloopDB
        from galoop.galoop import _build_initial_population

        db_path = tmp_path / "galoop.db"
        rng = np.random.default_rng(42)

        converged_ids = set()
        with GaloopDB(db_path) as db:
            db.setup()
            _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)
            for ind in db.get_by_status(STATUS.PENDING)[:4]:
                struct_dir = Path(ind.geometry_path).parent
                updated = _fake_converge(ind, db, struct_dir, energy=-1.0 - len(converged_ids))
                converged_ids.add(updated.id)

        with GaloopDB(db_path) as db:
            pool_ids = {p.id for p in db.selectable_pool()}
            assert converged_ids == pool_ids
