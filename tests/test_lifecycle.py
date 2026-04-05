"""
tests/test_lifecycle.py

Tests for job status transitions and persistence across store re-opens.
Status lives in the SQLite database.  Restart = open the same DB again.
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

def _build_pop(config, slab_info, store, seed=42):
    from galoop.galoop import _build_initial_population
    rng = np.random.default_rng(seed)
    _build_initial_population(config, slab_info, store, rng)


def _fake_converge(ind, store, energy: float = -1.0):
    """Copy POSCAR->CONTCAR, write FINAL_ENERGY, and update store to CONVERGED."""
    struct_dir = store.individual_dir(ind.id)
    poscar = struct_dir / "POSCAR"
    contcar = struct_dir / "CONTCAR"
    shutil.copy(poscar, contcar)
    (struct_dir / "FINAL_ENERGY").write_text(f"{energy:.10f}\n")
    updated = ind.with_energy(raw=energy * 100, grand_canonical=energy)
    store.update(updated)
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

    def test_status_updated_on_converge(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ind = temp_db.get_by_status(STATUS.PENDING)[0]
        _fake_converge(ind, temp_db, energy=-1.5)
        stored = temp_db.get(ind.id)
        assert stored.status == STATUS.CONVERGED


# ---------------------------------------------------------------------------
# Restart (re-open the same SQLite database)
# ---------------------------------------------------------------------------

class TestRestart:

    def test_population_not_rebuilt_if_db_exists(
        self, tmp_path, minimal_config, slab_info
    ):
        """The run() gate: population is not rebuilt if store already has records."""
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        store = GaloopStore(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, store, rng)
        n_before = len(store.get_by_status(STATUS.PENDING))

        # Simulate the restart gate from galoop.run()
        if store.is_empty():
            _build_initial_population(minimal_config, slab_info, store, rng)

        assert len(store.get_by_status(STATUS.PENDING)) == n_before
        store.close()

    def test_records_persist_after_reopen(self, tmp_path, minimal_config, slab_info):
        """All structures survive a store close/reopen cycle."""
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        rng = np.random.default_rng(42)
        store = GaloopStore(tmp_path)
        _build_initial_population(minimal_config, slab_info, store, rng)
        n_written = len(store.get_by_status(STATUS.PENDING))
        store.close()

        # Re-open from same path
        store2 = GaloopStore(tmp_path)
        n_read = len(store2.get_by_status(STATUS.PENDING))
        assert n_read == n_written
        store2.close()

    def test_converged_count_survives_restart(self, tmp_path, minimal_config, slab_info):
        """Converged structures are still queryable after reopening."""
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        store = GaloopStore(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, store, rng)
        pending = store.get_by_status(STATUS.PENDING)
        for ind in pending[:3]:
            _fake_converge(ind, store, energy=-1.0)
        n_converged = len(store.get_by_status(STATUS.CONVERGED))
        store.close()

        store2 = GaloopStore(tmp_path)
        assert len(store2.get_by_status(STATUS.CONVERGED)) == n_converged
        store2.close()

    def test_pending_structures_preserved_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        store = GaloopStore(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, store, rng)
        pending = store.get_by_status(STATUS.PENDING)
        for ind in pending[: len(pending) // 2]:
            _fake_converge(ind, store)
        n_still_pending = len(store.get_by_status(STATUS.PENDING))
        store.close()

        store2 = GaloopStore(tmp_path)
        assert len(store2.get_by_status(STATUS.PENDING)) == n_still_pending
        store2.close()

    def test_extra_data_survives_restart(self, tmp_path, minimal_config, slab_info):
        """Adsorbate counts round-trip through a store reopen."""
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        store = GaloopStore(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, store, rng)
        before = {
            ind.id: ind.extra_data["adsorbate_counts"]
            for ind in store.get_by_status(STATUS.PENDING)
        }
        store.close()

        store2 = GaloopStore(tmp_path)
        after = {
            ind.id: ind.extra_data["adsorbate_counts"]
            for ind in store2.get_by_status(STATUS.PENDING)
        }
        assert before == after
        store2.close()

    def test_selectable_pool_rebuilt_correctly_after_restart(
        self, tmp_path, minimal_config, slab_info
    ):
        from galoop.galoop import _build_initial_population
        from galoop.store import GaloopStore

        store = GaloopStore(tmp_path)
        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, store, rng)

        converged_ids = set()
        for ind in store.get_by_status(STATUS.PENDING)[:4]:
            updated = _fake_converge(ind, store,
                                     energy=-1.0 - len(converged_ids))
            converged_ids.add(updated.id)
        store.close()

        store2 = GaloopStore(tmp_path)
        pool_ids = {p.id for p in store2.selectable_pool()}
        assert converged_ids == pool_ids
        store2.close()
