"""
galoop/database.py

SQLite interface.  Single source of truth for structure state.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from galoop.individual import Individual, STATUS


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _enc(value) -> str | None:
    return None if value is None else json.dumps(value)


def _dec_list(value) -> list:
    return [] if not value else json.loads(value)


def _dec_dict(value) -> dict:
    return {} if not value else json.loads(value)


def row_to_individual(row: sqlite3.Row) -> Individual:
    """Convert a DB row to an Individual instance."""
    return Individual(
        id=row["id"],
        generation=row["generation"],
        parent_ids=_dec_list(row["parent_ids"]),
        operator=row["operator"],
        status=row["status"],
        raw_energy=row["raw_energy"],
        grand_canonical_energy=row["grand_canonical_energy"],
        weight=row["weight"],
        geometry_path=row["geometry_path"],
        extra_data=_dec_dict(row["extra_data"]),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS structures (
    id                      TEXT PRIMARY KEY,
    generation              INTEGER NOT NULL,
    parent_ids              TEXT    NOT NULL DEFAULT '[]',
    operator                TEXT    NOT NULL DEFAULT 'init',
    status                  TEXT    NOT NULL DEFAULT 'pending',
    raw_energy              REAL,
    grand_canonical_energy  REAL,
    weight                  REAL    NOT NULL DEFAULT 1.0,
    geometry_path           TEXT,
    extra_data              TEXT    NOT NULL DEFAULT '{}',
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_status ON structures (status);
CREATE INDEX IF NOT EXISTS idx_generation ON structures (generation);
CREATE INDEX IF NOT EXISTS idx_gce ON structures (grand_canonical_energy);

CREATE TRIGGER IF NOT EXISTS trg_updated_at
AFTER UPDATE ON structures
BEGIN
    UPDATE structures SET updated_at = datetime('now') WHERE id = NEW.id;
END;
"""


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------

class GaloopDB:
    """Thin wrapper around SQLite for structure bookkeeping."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # -- connection lifecycle ----------------------------------------------

    def connect(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> GaloopDB:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Cursor]:
        """Transaction context manager with auto-commit / rollback."""
        assert self._conn is not None, "Database not connected"
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- setup -------------------------------------------------------------

    def setup(self) -> None:
        """Create tables and indices (idempotent)."""
        assert self._conn is not None, "Database not connected"
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- CRUD --------------------------------------------------------------

    def insert(self, ind: Individual) -> str:
        with self._tx() as cur:
            cur.execute(
                """INSERT INTO structures
                   (id, generation, parent_ids, operator, status,
                    raw_energy, grand_canonical_energy, weight,
                    geometry_path, extra_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ind.id,
                    ind.generation,
                    _enc(ind.parent_ids),
                    ind.operator,
                    ind.status,
                    ind.raw_energy,
                    ind.grand_canonical_energy,
                    ind.weight,
                    ind.geometry_path,
                    _enc(ind.extra_data),
                ),
            )
        return ind.id

    def update(self, ind: Individual) -> None:
        with self._tx() as cur:
            cur.execute(
                """UPDATE structures SET
                       status=?, raw_energy=?, grand_canonical_energy=?,
                       weight=?, geometry_path=?, extra_data=?
                   WHERE id=?""",
                (
                    ind.status,
                    ind.raw_energy,
                    ind.grand_canonical_energy,
                    ind.weight,
                    ind.geometry_path,
                    _enc(ind.extra_data),
                    ind.id,
                ),
            )

    def get(self, ind_id: str) -> Individual | None:
        row = self._conn.execute(
            "SELECT * FROM structures WHERE id=?", (ind_id,)
        ).fetchone()
        return row_to_individual(row) if row else None

    def get_by_status(self, status: str) -> list[Individual]:
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status=?", (status,)
        ).fetchall()
        return [row_to_individual(r) for r in rows]

    def selectable_pool(self) -> list[Individual]:
        """Converged structures with weight > 0 (eligible for parent selection)."""
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status=? AND weight>0",
            (STATUS.CONVERGED,),
        ).fetchall()
        return [row_to_individual(r) for r in rows]

    def best(self, n: int = 10) -> list[Individual]:
        rows = self._conn.execute(
            """SELECT * FROM structures
               WHERE status=? AND grand_canonical_energy IS NOT NULL
               ORDER BY grand_canonical_energy ASC LIMIT ?""",
            (STATUS.CONVERGED, n),
        ).fetchall()
        return [row_to_individual(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM structures GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def find_by_geometry_path_substring(self, substr: str) -> Individual | None:
        """Find the first Individual whose geometry_path contains *substr*."""
        row = self._conn.execute(
            "SELECT * FROM structures WHERE geometry_path LIKE ? LIMIT 1",
            (f"%{substr}%",),
        ).fetchone()
        return row_to_individual(row) if row else None

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT * FROM structures ORDER BY generation, rowid",
            self._conn,
        )
        for col in ("parent_ids", "extra_data"):
            df[col] = df[col].apply(
                lambda v: json.loads(v) if v else ([] if col == "parent_ids" else {})
            )
        return df
