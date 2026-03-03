"""Shared pytest fixtures."""

import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Provide a temporary, initialised GaloopDB."""
    from galoop.database import GaloopDB

    db_path = tmp_path / "test.db"
    db = GaloopDB(db_path)
    db.connect()
    db.setup()
    yield db
    db.close()
