"""Test Individual data model."""

from galoop.individual import Individual, STATUS, OPERATOR

def test_individual_creation():
    ind = Individual(
        id="test_001",
        generation=0,
        operator=OPERATOR.INIT,
        status=STATUS.PENDING,
    )
    assert ind.id == "test_001"
    assert ind.generation == 0
    assert ind.status == STATUS.PENDING
    assert ind.weight == 1.0

def test_individual_from_init():
    ind = Individual.from_init(
        generation=0,
        geometry_path="/tmp/test.vasp",
        extra_data={"adsorbate_counts": {"O": 1}},
    )
    assert ind.operator == OPERATOR.INIT
    assert ind.status == STATUS.PENDING
    assert ind.extra_data["adsorbate_counts"]["O"] == 1

def test_individual_from_parents():
    parent1 = Individual.from_init(generation=0)
    parent2 = Individual.from_init(generation=0)
    
    child = Individual.from_parents(
        generation=1,
        parents=[parent1, parent2],
        operator=OPERATOR.SPLICE,
    )
    assert child.generation == 1
    assert child.operator == OPERATOR.SPLICE
    assert len(child.parent_ids) == 2

def test_individual_with_status():
    ind = Individual.from_init(generation=0)
    ind_converged = ind.with_status(STATUS.CONVERGED)
    
    assert ind.status == STATUS.PENDING  # Original unchanged
    assert ind_converged.status == STATUS.CONVERGED  # New copy has new status

def test_individual_with_energy():
    ind = Individual.from_init(generation=0)
    ind_with_e = ind.with_energy(raw=-100.0, grand_canonical=-0.5)
    
    assert ind_with_e.raw_energy == -100.0
    assert ind_with_e.grand_canonical_energy == -0.5

def test_individual_mark_duplicate():
    ind = Individual.from_init(generation=0)
    ind_dup = ind.mark_duplicate()
    
    assert ind_dup.status == STATUS.DUPLICATE
    assert ind_dup.weight == 0.0

def test_individual_is_selectable():
    ind_pending = Individual.from_init(generation=0, 
                                       extra_data={"adsorbate_counts": {}})
    assert not ind_pending.is_selectable
    
    ind_converged = ind_pending.with_status(STATUS.CONVERGED)
    assert ind_converged.is_selectable
    
    ind_dup = ind_converged.mark_duplicate()
    assert not ind_dup.is_selectable  # weight=0, not selectable
