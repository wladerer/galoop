"""Test that all modules import correctly."""

def test_import_individual():
    from galoop.individual import Individual, STATUS, OPERATOR
    assert STATUS.PENDING == "pending"
    assert STATUS.CONVERGED == "converged"
    assert OPERATOR.INIT == "init"

def test_import_database():
    from galoop.database import GaloopDB
    assert GaloopDB is not None

def test_import_fingerprint():
    from galoop.fingerprint import compute_soap, tanimoto_similarity
    assert callable(compute_soap)
    assert callable(tanimoto_similarity)

def test_import_config():
    from galoop.config import load_config, GaloopConfig
    assert GaloopConfig is not None
    assert callable(load_config)

def test_import_cli():
    from galoop.cli import cli
    assert cli is not None

def test_import_calculator():
    from galoop.engine.calculator import Pipeline, CalculatorStage, build_pipeline
    assert Pipeline is not None
    assert callable(build_pipeline)

def test_import_scheduler():
    from galoop.engine.scheduler import Scheduler, LocalScheduler, SlurmScheduler, PbsScheduler
    assert Scheduler is not None
    assert LocalScheduler is not None

def test_import_surface():
    from galoop.science.surface import load_slab, load_adsorbate, place_adsorbate
    assert callable(load_slab)
    assert callable(load_adsorbate)
    assert callable(place_adsorbate)

def test_import_energy():
    from galoop.science.energy import grand_canonical_energy, is_desorbed
    assert callable(grand_canonical_energy)
    assert callable(is_desorbed)

def test_import_reproduce():
    from galoop.science.reproduce import splice, merge, mutate_add, mutate_remove
    assert callable(splice)
    assert callable(merge)
    assert callable(mutate_add)
    assert callable(mutate_remove)
