"""
gocia/database.py

Lean SQLite interface. Single source of truth for structure state.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd

from galoop.individual import Individual, STATUS


def _enc(value) -> str | None:
    """Encode to JSON."""
    return None if value is None else json.dumps(value)


def _dec_list(value) -> list:
    """Decode from JSON."""
    return [] if not value else json.loads(value)


def _dec_dict(value) -> dict:
    """Decode from JSON."""
    return {} if not value else json.loads(value)


def _row_to_individual(row: sqlite3.Row) -> Individual:
    """Convert DB row to Individual."""
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


_SCHEMA = """
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


class GaloopDB:
    """SQLite database interface."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open connection."""
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")

    def close(self) -> None:
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> GaloopDB:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """Context manager for transactions."""
        conn = self._conn
        if conn is None:
            raise RuntimeError("Not connected.")
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def setup(self) -> None:
        """Create tables."""
        conn = self._conn
        if conn is None:
            raise RuntimeError("Not connected.")
        for statement in _SCHEMA.split(";"):
            if statement.strip():
                conn.execute(statement)
        conn.commit()

    def insert(self, ind: Individual) -> str:
        """Insert a new Individual."""
        with self.transaction() as cur:
            cur.execute(
                """
                INSERT INTO structures (
                    id, generation, parent_ids, operator, status,
                    raw_energy, grand_canonical_energy, weight,
                    geometry_path, extra_data
                ) VALUES (
                    :id, :gen, :parents, :op, :status,
                    :raw_e, :gce, :weight,
                    :geom, :extra
                )
                """,
                {
                    "id": ind.id,
                    "gen": ind.generation,
                    "parents": _enc(ind.parent_ids),
                    "op": ind.operator,
                    "status": ind.status,
                    "raw_e": ind.raw_energy,
                    "gce": ind.grand_canonical_energy,
                    "weight": ind.weight,
                    "geom": ind.geometry_path,
                    "extra": _enc(ind.extra_data),
                },
            )
        return ind.id

    def update(self, ind: Individual) -> None:
        """Update an existing Individual."""
        with self.transaction() as cur:
            cur.execute(
                """
                UPDATE structures SET
                    status                 = :status,
                    raw_energy             = :raw_e,
                    grand_canonical_energy = :gce,
                    weight                 = :weight,
                    geometry_path          = :geom,
                    extra_data             = :extra
                WHERE id = :id
                """,
                {
                    "id": ind.id,
                    "status": ind.status,
                    "raw_e": ind.raw_energy,
                    "gce": ind.grand_canonical_energy,
                    "weight": ind.weight,
                    "geom": ind.geometry_path,
                    "extra": _enc(ind.extra_data),
                },
            )

    def get(self, ind_id: str) -> Individual | None:
        """Fetch a single Individual by id."""
        row = self._conn.execute(
            "SELECT * FROM structures WHERE id = ?", (ind_id,)
        ).fetchone()
        return _row_to_individual(row) if row else None

    def get_by_status(self, status: str) -> list[Individual]:
        """Get all Individuals with a given status."""
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status = ?", (status,)
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def selectable_pool(self) -> list[Individual]:
        """Get all Individuals eligible for selection (converged, weight > 0)."""
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status = ? AND weight > 0",
            (STATUS.CONVERGED,),
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def best(self, n: int = 10) -> list[Individual]:
        """Get top-n Individuals by grand canonical energy."""
        rows = self._conn.execute(
            """
            SELECT * FROM structures
            WHERE status = ? AND grand_canonical_energy IS NOT NULL
            ORDER BY grand_canonical_energy ASC
            LIMIT ?
            """,
            (STATUS.CONVERGED, n),
        ).fetchall()
        return [_row_to_individual(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        """Count structures by status."""
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as n FROM structures GROUP BY status"
        ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    def to_dataframe(self) -> pd.DataFrame:
        """Export to pandas DataFrame."""
        df = pd.read_sql_query(
            "SELECT * FROM structures ORDER BY generation, rowid",
            self._conn,
        )
        for col in ("parent_ids", "extra_data"):
            df[col] = df[col].apply(
                lambda v: json.loads(v) if v else ([] if col == "parent_ids" else {})
            )
        return df
