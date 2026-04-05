"""
tests/test_population.py

Tests for initial population building and offspring spawning.
Covers structure files, job entries, stoichiometry variation, and operator tracking.
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

def _build_pop(minimal_config, slab_info, store, seed=42):
    from galoop.galoop import _build_initial_population
    rng = np.random.default_rng(seed)
    _build_initial_population(minimal_config, slab_info, store, rng)


def _spawn_n(n, store, config, slab_info, seed=7):
    from galoop.galoop import _spawn_one
    rng = np.random.default_rng(seed)
    results = []
    attempts = 0
    max_attempts = n * 10  # operators may fail, allow retries
    while len(results) < n and attempts < max_attempts:
        result = _spawn_one(store, config, slab_info, rng)
        attempts += 1
        if result is not None:
            ind, atoms = result
            # Write POSCAR so subsequent spawns can read parent geometry
            struct_dir = store.individual_dir(ind.id)
            from ase.io import write
            write(str(struct_dir / "POSCAR"), atoms, format="vasp")
            write(str(struct_dir / "CONTCAR"), atoms, format="vasp")
            results.append(ind)
    return results


# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

class TestInitialPopulation:

    def test_structure_count(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        pending = temp_db.get_by_status(STATUS.PENDING)
        assert len(pending) == minimal_config.ga.population_size

    def test_poscar_files_written(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.geometry_path is not None
            assert Path(ind.geometry_path).exists()

    def test_all_db_entries_pending(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.status == STATUS.PENDING
            assert ind.raw_energy is None
            assert ind.grand_canonical_energy is None

    def test_all_use_init_operator(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            assert ind.operator == OPERATOR.INIT

    def test_adsorbate_counts_stored(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            counts = ind.extra_data.get("adsorbate_counts", {})
            assert isinstance(counts, dict)
            assert len(counts) > 0

    def test_adsorbate_total_within_bounds(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        for ind in temp_db.get_by_status(STATUS.PENDING):
            total = sum(ind.extra_data["adsorbate_counts"].values())
            assert minimal_config.ga.min_adsorbates <= total <= minimal_config.ga.max_adsorbates

    def test_stoichiometry_varies_across_population(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        stoichiometries = {
            frozenset(ind.extra_data["adsorbate_counts"].items())
            for ind in temp_db.get_by_status(STATUS.PENDING)
        }
        assert len(stoichiometries) > 1, "all structures have identical stoichiometry"

    def test_no_duplicate_ids(self, minimal_config, slab_info, temp_db):
        _build_pop(minimal_config, slab_info, temp_db)
        ids = [ind.id for ind in temp_db.get_by_status(STATUS.PENDING)]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Offspring spawning
# ---------------------------------------------------------------------------

class TestSpawnOffspring:

    def test_spawn_creates_poscar(self, minimal_config, slab_info, converged_population, temp_db):
        from galoop.galoop import _spawn_one
        rng = np.random.default_rng(1)
        # Operators can fail and return None; retry a few times
        result = None
        for _ in range(20):
            result = _spawn_one(temp_db, minimal_config, slab_info, rng)
            if result is not None:
                break
        assert result is not None, "spawn failed after 20 attempts"
        ind, atoms = result
        struct_dir = temp_db.individual_dir(ind.id)
        from ase.io import write
        write(str(struct_dir / "POSCAR"), atoms, format="vasp")
        assert (struct_dir / "POSCAR").exists()

    def test_spawn_adds_db_entry(self, minimal_config, slab_info, temp_db, converged_population):
        offspring = _spawn_n(3, temp_db, minimal_config, slab_info)
        assert len(offspring) >= 1, "expected at least one successful spawn"
        for ind in offspring:
            stored = temp_db.get(ind.id)
            assert stored is not None

    def test_spawn_records_parent_ids_for_ga_operators(
        self, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(20, temp_db, minimal_config, slab_info, seed=99)
        ga_offspring = [o for o in offspring if o.operator != OPERATOR.INIT]
        assert ga_offspring, "expected at least one GA operator to succeed"
        for ind in ga_offspring:
            assert len(ind.parent_ids) >= 1

    def test_spawn_operator_variety(
        self, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(30, temp_db, minimal_config, slab_info, seed=5)
        operators_used = {o.operator for o in offspring}
        assert len(operators_used) >= 2, f"only one operator seen: {operators_used}"

    def test_spawn_falls_back_to_random_when_pool_empty(
        self, minimal_config, slab_info, temp_db
    ):
        from galoop.galoop import _spawn_one
        _build_pop(minimal_config, slab_info, temp_db)
        rng = np.random.default_rng(42)
        result = _spawn_one(temp_db, minimal_config, slab_info, rng)
        assert result is not None
        ind, atoms = result
        assert ind.operator == OPERATOR.INIT

    def test_spawn_stoichiometry_variation(
        self, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(15, temp_db, minimal_config, slab_info, seed=13)
        stoichiometries = {
            frozenset(o.extra_data.get("adsorbate_counts", {}).items())
            for o in offspring
        }
        assert len(stoichiometries) > 1, "all offspring have identical stoichiometry"

    def test_spawn_non_init_has_parent_ids(
        self, minimal_config, slab_info, temp_db, converged_population
    ):
        offspring = _spawn_n(20, temp_db, minimal_config, slab_info, seed=99)
        ga_offspring = [o for o in offspring if o.operator != OPERATOR.INIT]
        assert ga_offspring, "expected at least one GA operator offspring"
        for ind in ga_offspring:
            assert len(ind.parent_ids) >= 1

    def test_spawn_boltzmann_weights_realistic_energies(
        self, minimal_config, slab_info, temp_db
    ):
        """Boltzmann selection must not overflow with DFT-scale energies (~-300 eV)."""
        import warnings
        from galoop.galoop import _build_initial_population, _spawn_one

        rng = np.random.default_rng(42)
        _build_initial_population(minimal_config, slab_info, temp_db, rng)

        for i, ind in enumerate(temp_db.get_by_status(STATUS.PENDING)):
            struct_dir = temp_db.individual_dir(ind.id)
            poscar = struct_dir / "POSCAR"
            shutil.copy(poscar, struct_dir / "CONTCAR")
            energy = -320.0 - i * 5.0
            temp_db.update(ind.with_energy(raw=energy, grand_canonical=energy))

        rng2 = np.random.default_rng(7)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _spawn_one(temp_db, minimal_config, slab_info, rng2)

        assert result is not None


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
        from ase import Atoms
        from galoop.science.surface import orient_upright
        oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]])
        result = orient_upright(oh)
        o_z = result.get_positions()[0, 2]
        assert o_z < result.get_center_of_mass()[2]

    def test_nonzero_binding_index(self):
        from ase import Atoms
        from galoop.science.surface import orient_upright
        h2o = Atoms("HHO", positions=[[0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.0, 0.0, 0.0]])
        result = orient_upright(h2o, binding_index=2)
        o_z = result.get_positions()[2, 2]
        assert o_z < result.get_center_of_mass()[2]

    def test_out_of_range_binding_index_raises(self):
        from ase.build import molecule
        from galoop.science.surface import orient_upright
        with pytest.raises(ValueError, match="binding_index"):
            orient_upright(molecule("OH"), binding_index=5)

    def test_placed_adsorbate_o_below_h(self, slab_info):
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
