"""Test database operations."""

import tempfile
from pathlib import Path
from galoop.database import GaloopDB
from galoop.individual import Individual, STATUS

def test_database_setup():
    """Test database creation and schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with GaloopDB(db_path) as db:
            db.setup()
            assert db_path.exists()

def test_database_insert_and_retrieve():
    """Test insert and retrieval."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with GaloopDB(db_path) as db:
            db.setup()
            
            ind = Individual.from_init(
                generation=0,
                geometry_path="/tmp/test.vasp",
                extra_data={"adsorbate_counts": {"O": 1}},
            )
            
            db.insert(ind)
            retrieved = db.get(ind.id)
            
            assert retrieved.id == ind.id
            assert retrieved.generation == 0
            assert retrieved.extra_data["adsorbate_counts"]["O"] == 1

def test_database_update():
    """Test updating a structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with GaloopDB(db_path) as db:
            db.setup()
            
            ind = Individual.from_init(generation=0)
            db.insert(ind)
            
            ind_updated = ind.with_status(STATUS.CONVERGED).with_energy(-100.0, -0.5)
            db.update(ind_updated)
            
            retrieved = db.get(ind.id)
            assert retrieved.status == STATUS.CONVERGED
            assert retrieved.grand_canonical_energy == -0.5

def test_database_get_by_status():
    """Test filtering by status."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with GaloopDB(db_path) as db:
            db.setup()
            
            # Insert multiple structures with different statuses
            for i in range(3):
                ind = Individual.from_init(generation=0)
                db.insert(ind)
            
            for i in range(2):
                ind = Individual.from_init(generation=0).with_status(STATUS.CONVERGED)
                db.insert(ind)
            
            pending = db.get_by_status(STATUS.PENDING)
            converged = db.get_by_status(STATUS.CONVERGED)
            
            assert len(pending) == 3
            assert len(converged) == 2

def test_database_selectable_pool():
    """Test getting selectable structures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        with GaloopDB(db_path) as db:
            db.setup()
            
            # Add converged unique structures
            for i in range(3):
                ind = Individual.from_init(generation=0).with_status(STATUS.CONVERGED)
                db.insert(ind)
            
            # Add converged duplicate (weight=0)
            ind_dup = Individual.from_init(generation=0).mark_duplicate()
            db.insert(ind_dup)
            
            pool = db.selectable_pool()
            assert len(pool) == 3  # Only the non-duplicates
