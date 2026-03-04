"""
tests/test_population.py

Tests for initial population building and offspring spawning.
Covers structure files, DB entries, stoichiometry variation, and operator tracking.
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

def _build_pop(minimal_config, slab_info, db, tmp_path, seed=42):
    from galoop.galoop import _build_initial_population
    rng = np.random.default_rng(seed)
    _build_initial_population(minimal_config, slab_info, db, tmp_path, rng)


def _spawn_n(n, run_dir, db, config, slab_info, seed=7):
    from galoop.galoop import _spawn_one
    rng = np.random.default_rng(seed)
    results = []
    for i in range(n):
        ind = _spawn_one(run_dir, db, config, slab_info, rng, total_evals=i)
        if ind is not None:
            results.append(ind)
    return results


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

class TestInitialPopulation:

    def test_structure_count(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        pending = temp_db.get_by_status(STATUS.PENDING)
        assert len(pending) == minimal_config.ga.population_size

    def test_poscar_files_written(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.geometry_path is not None
            assert Path(ind.geometry_path).exists()

    def test_pending_sentinels_written(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        gen_dir = tmp_path / "gen_000"
        struct_dirs = sorted(gen_dir.glob("struct_????"))
        assert len(struct_dirs) == minimal_config.ga.population_size
        for d in struct_dirs:
            assert (d / "PENDING").exists()

    def test_all_db_entries_pending(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.status == STATUS.PENDING
            assert ind.raw_energy is None
            assert ind.grand_canonical_energy is None

    def test_all_use_init_operator(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.operator == OPERATOR.INIT

    def test_adsorbate_counts_stored(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            counts = ind.extra_data.get("adsorbate_counts", {})
            assert isinstance(counts, dict)
            assert len(counts) > 0

    def test_adsorbate_total_within_bounds(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            total = sum(ind.extra_data["adsorbate_counts"].values())
            assert minimal_config.ga.min_adsorbates <= total <= minimal_config.ga.max_adsorbates

    def test_stoichiometry_varies_across_population(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        stoichiometries = {
            frozenset(ind.extra_data["adsorbate_counts"].items())
            for ind in temp_db.get_by_status(STATUS.PENDING)
        }
        assert len(stoichiometries) > 1, "all structures have identical stoichiometry"

    def test_no_duplicate_ids(self, tmp_path, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        ids = [ind.id for ind in temp_db.get_by_status(STATUS.PENDING)]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Offspring spawning
# ---------------------------------------------------------------------------

class TestSpawnOffspring:

    def test_spawn_creates_poscar(self, tmp_path, minimal_config, slab_info, converged_population):
        from galoop.galoop import _spawn_one
        rng = np.random.default_rng(1)
        # Use same temp_db implicitly via converged_population fixture
        from galoop.database import GaloopDB
        db_path = tmp_path / "test.db"
        with GaloopDB(db_path) as db:
            ind = _spawn_one(tmp_path, db, minimal_config, slab_info, rng, total_evals=0)
        assert ind is not None
        assert Path(ind.geometry_path).exists()

    def test_spawn_adds_db_entry(self, tmp_path, minimal_config, slab_info, temp_db, converged_population):
        offspring = _spawn_n(3, tmp_path, temp_db, minimal_config, slab_info)
        assert len(offspring) == 3
        for ind in offspring:
            stored = temp_db.get(ind.id)
            assert stored is not None
            assert stored.status == STATUS.PENDING

    def test_spawn_starts_as_pending(self, tmp_path, minimal_config, slab_info, temp_db, converged_population):
        offspring = _spawn_n(1, tmp_path, temp_db, minimal_config, slab_info)
        assert offspring[0].status == STATUS.PENDING

    def test_spawn_records_parent_ids_for_ga_operators(
        self, tmp_path, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(20, tmp_path, temp_db, minimal_config, slab_info, seed=99)
        ga_offspring = [o for o in offspring if o.operator != OPERATOR.INIT]
        assert ga_offspring, "expected at least one GA operator to succeed"
        for ind in ga_offspring:
            assert len(ind.parent_ids) >= 1

    def test_spawn_operator_variety(
        self, tmp_path, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(30, tmp_path, temp_db, minimal_config, slab_info, seed=5)
        operators_used = {o.operator for o in offspring}
        assert len(operators_used) >= 2, f"only one operator seen: {operators_used}"

    def test_spawn_falls_back_to_random_when_pool_empty(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        # Empty DB — no selectable pool
        from galoop.galoop import _spawn_one
        _build_pop(minimal_config, slab_info, temp_db, tmp_path)
        rng = np.random.default_rng(42)
        ind = _spawn_one(tmp_path, temp_db, minimal_config, slab_info, rng, total_evals=0)
        assert ind is not None
        assert ind.operator == OPERATOR.INIT

    def test_spawn_stoichiometry_variation(
        self, tmp_path, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(15, tmp_path, temp_db, minimal_config, slab_info, seed=13)
        stoichiometries = {
            frozenset(o.extra_data.get("adsorbate_counts", {}).items())
            for o in offspring
        }
        assert len(stoichiometries) > 1, "all offspring have identical stoichiometry"

    def test_spawn_generation_increments(
        self, tmp_path, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(3, tmp_path, temp_db, minimal_config, slab_info)
        for ind in offspring:
            assert ind.generation >= 1

    def test_spawn_boltzmann_weights_realistic_energies(
        self, tmp_path, minimal_config, slab_info, temp_db
    ):
        """Boltzmann selection must not overflow with DFT-scale energies (~-300 eV)."""
        import shutil
        import warnings
        from galoop.galoop import _build_initial_population, _spawn_one

        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, temp_db, tmp_path, rng)

        # Converge structures with realistic MACE/DFT total energies
        for i, ind in enumerate(temp_db.get_by_status(STATUS.PENDING)):
            poscar = Path(ind.geometry_path)
            shutil.copy(poscar, poscar.parent / "CONTCAR")
            energy = -320.0 - i * 5.0          # e.g. -320, -325, -330 … eV
            temp_db.update(ind.with_energy(raw=energy, grand_canonical=energy))

        rng2 = np.random.default_rng(7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # overflow → test failure
            ind = _spawn_one(tmp_path, temp_db, minimal_config, slab_info, rng2, total_evals=0)

        assert ind is not None


# ---------------------------------------------------------------------------
# orient_upright
# ---------------------------------------------------------------------------

class TestOrientUpright:

    def test_single_atom_unchanged(self):
        from ase import Atoms
        from galoop.science.surface import orient_upright
        atom = Atoms("O", positions=[[1.0, 2.0, 3.0]])
        result = orient_upright(atom)
        np.testing.assert_allclose(result.get_positions(), atom.get_positions())

    def test_atom0_below_com_after_orient(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        oh = orient_upright(molecule("OH"))
        com_z = oh.get_center_of_mass()[2]
        o_z = oh.get_positions()[0, 2]
        assert o_z < com_z, f"O should be below COM: O_z={o_z:.3f}, COM_z={com_z:.3f}"

    def test_atom1_above_com_after_orient(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        oh = orient_upright(molecule("OH"))
        com_z = oh.get_center_of_mass()[2]
        h_z = oh.get_positions()[1, 2]
        assert h_z > com_z, f"H should be above COM: H_z={h_z:.3f}, COM_z={com_z:.3f}"

    def test_bond_length_preserved(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        oh = molecule("OH")
        before = np.linalg.norm(oh.get_positions()[0] - oh.get_positions()[1])
        after_pos = orient_upright(oh).get_positions()
        after = np.linalg.norm(after_pos[0] - after_pos[1])
        np.testing.assert_allclose(after, before, atol=1e-6)

    def test_com_preserved(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        oh = molecule("OH")
        com_before = oh.get_center_of_mass()
        com_after = orient_upright(oh).get_center_of_mass()
        np.testing.assert_allclose(com_after, com_before, atol=1e-6)

    def test_already_upright_unchanged(self):
        """A molecule already oriented correctly should not be rotated."""
        from ase import Atoms
        from galoop.science.surface import orient_upright
        # O below, H above — already correct
        oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
        result = orient_upright(oh)
        o_z = result.get_positions()[0, 2]
        assert o_z < result.get_center_of_mass()[2]

    def test_nonzero_binding_index(self):
        """binding_index selects which atom faces the surface."""
        from ase import Atoms
        from galoop.science.surface import orient_upright
        # H2O: H H O — bind via O at index 2
        h2o = Atoms("HHO", positions=[[0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]])
        result = orient_upright(h2o, binding_index=2)
        o_z = result.get_positions()[2, 2]
        assert o_z < result.get_center_of_mass()[2]

    def test_out_of_range_binding_index_raises(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        import pytest
        with pytest.raises(ValueError, match="binding_index"):
            orient_upright(molecule("OH"), binding_index=5)

    def test_placed_adsorbate_o_below_h(self, slab_info):
        """After placement, O should be closer to the surface than H."""
        from ase.build import molecule
        from galoop.science.surface import place_adsorbate
        oh = molecule("OH")
        rng = np.random.default_rng(0)
        combined = place_adsorbate(
            slab_info.atoms.copy(), oh,
            slab_info.zmin, slab_info.zmax, rng=rng,
        )
        ads = combined.get_positions()[slab_info.n_slab_atoms:]
        o_z, h_z = ads[0, 2], ads[1, 2]
        assert o_z < h_z, f"O (z={o_z:.3f}) should be below H (z={h_z:.3f}) after placement"
