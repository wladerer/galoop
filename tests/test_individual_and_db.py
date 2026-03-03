"""
tests/test_individual_and_db.py

Individual data model + Database CRUD.
"""

import pytest

from galoop.individual import Individual, STATUS, OPERATOR


class TestStatus:

    def test_terminal(self):
        for s in (STATUS.CONVERGED, STATUS.FAILED, STATUS.DUPLICATE, STATUS.DESORBED):
            assert STATUS.is_terminal(s)

    def test_not_terminal(self):
        for s in (STATUS.PENDING, STATUS.SUBMITTED):
            assert not STATUS.is_terminal(s)

    def test_active(self):
        assert STATUS.is_active(STATUS.PENDING)
        assert STATUS.is_active(STATUS.SUBMITTED)
        assert not STATUS.is_active(STATUS.CONVERGED)


class TestIndividual:

    def test_from_init(self):
        ind = Individual.from_init(generation=0, geometry_path="/tmp/t.vasp")
        assert ind.operator == OPERATOR.INIT
        assert ind.status == STATUS.PENDING
        assert ind.weight == 1.0
        assert len(ind.id) == 8

    def test_from_parents(self):
        p1 = Individual.from_init(generation=0)
        p2 = Individual.from_init(generation=0)
        child = Individual.from_parents(generation=1, parents=[p1, p2], operator=OPERATOR.SPLICE)
        assert len(child.parent_ids) == 2
        assert p1.id in child.parent_ids

    def test_with_status_copies(self):
        ind = Individual.from_init(generation=0)
        updated = ind.with_status(STATUS.CONVERGED)
        assert ind.status == STATUS.PENDING
        assert updated.status == STATUS.CONVERGED

    def test_with_energy(self):
        ind = Individual.from_init(generation=0)
        updated = ind.with_energy(raw=-100.0, grand_canonical=-0.5)
        assert updated.raw_energy == -100.0
        assert ind.raw_energy is None

    def test_mark_duplicate(self):
        ind = Individual.from_init(generation=0)
        dup = ind.mark_duplicate()
        assert dup.status == STATUS.DUPLICATE
        assert dup.weight == 0.0
        assert not dup.is_selectable

    def test_is_selectable(self):
        ind = Individual.from_init(generation=0)
        assert not ind.is_selectable
        assert ind.with_status(STATUS.CONVERGED).is_selectable
        assert not ind.with_status(STATUS.CONVERGED).mark_duplicate().is_selectable

    def test_extra_data_roundtrip(self):
        data = {"adsorbate_counts": {"O": 2}, "note": "test"}
        ind = Individual.from_init(generation=0, extra_data=data)
        assert ind.extra_data["adsorbate_counts"]["O"] == 2


class TestDatabase:

    def test_setup(self, temp_db):
        assert temp_db.path.exists()

    def test_insert_and_get(self, temp_db):
        ind = Individual.from_init(generation=0, geometry_path="/tmp/t.vasp",
                                    extra_data={"adsorbate_counts": {"O": 1}})
        temp_db.insert(ind)
        got = temp_db.get(ind.id)
        assert got is not None
        assert got.id == ind.id
        assert got.extra_data["adsorbate_counts"]["O"] == 1

    def test_get_nonexistent(self, temp_db):
        assert temp_db.get("nope") is None

    def test_update(self, temp_db):
        ind = Individual.from_init(generation=0)
        temp_db.insert(ind)
        updated = ind.with_status(STATUS.CONVERGED).with_energy(-100.0, -0.5)
        temp_db.update(updated)
        got = temp_db.get(ind.id)
        assert got.status == STATUS.CONVERGED
        assert got.grand_canonical_energy == pytest.approx(-0.5)

    def test_get_by_status(self, temp_db):
        for _ in range(3):
            temp_db.insert(Individual.from_init(generation=0))
        for _ in range(2):
            temp_db.insert(Individual.from_init(generation=0).with_status(STATUS.CONVERGED))
        assert len(temp_db.get_by_status(STATUS.PENDING)) == 3
        assert len(temp_db.get_by_status(STATUS.CONVERGED)) == 2

    def test_selectable_pool_excludes_duplicates(self, temp_db):
        for _ in range(3):
            temp_db.insert(Individual.from_init(generation=0).with_status(STATUS.CONVERGED))
        temp_db.insert(Individual.from_init(generation=0).mark_duplicate())
        assert len(temp_db.selectable_pool()) == 3

    def test_best_ordering(self, temp_db):
        for e in [-0.5, -1.0, -0.3, -2.0, -0.1]:
            temp_db.insert(
                Individual.from_init(generation=0)
                .with_status(STATUS.CONVERGED)
                .with_energy(raw=-100.0, grand_canonical=e)
            )
        best = temp_db.best(n=3)
        assert len(best) == 3
        assert best[0].grand_canonical_energy == pytest.approx(-2.0)

    def test_count_by_status(self, temp_db):
        temp_db.insert(Individual.from_init(generation=0))
        temp_db.insert(Individual.from_init(generation=0))
        temp_db.insert(Individual.from_init(generation=0).with_status(STATUS.CONVERGED))
        counts = temp_db.count_by_status()
        assert counts[STATUS.PENDING] == 2
        assert counts[STATUS.CONVERGED] == 1

    def test_find_by_geometry_path_substring(self, temp_db):
        ind = Individual.from_init(generation=0, geometry_path="/run/gen_000/struct_0001/POSCAR")
        temp_db.insert(ind)
        found = temp_db.find_by_geometry_path_substring("struct_0001")
        assert found is not None
        assert found.id == ind.id

    def test_to_dataframe(self, temp_db):
        for i in range(5):
            temp_db.insert(Individual.from_init(generation=i % 2))
        df = temp_db.to_dataframe()
        assert len(df) == 5
        assert "id" in df.columns
