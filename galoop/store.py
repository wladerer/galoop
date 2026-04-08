"""
galoop/store.py

SQLite-backed population store for the GA.

Uses WAL mode for safe concurrent reads (e.g. report while GA loop is
running).  Each structure is a row in the ``structures`` table; geometry
files live on disk under ``structures/{id}/``.

Replaces the signac GaloopProject with a simpler, dependency-free backend.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from galoop.individual import STATUS, Individual

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS structures (
    id                     TEXT PRIMARY KEY,
    parent_ids             TEXT NOT NULL DEFAULT '[]',
    operator               TEXT NOT NULL DEFAULT 'init',
    status                 TEXT NOT NULL DEFAULT 'pending',
    raw_energy             REAL,
    grand_canonical_energy REAL,
    weight                 REAL NOT NULL DEFAULT 1.0,
    geometry_path          TEXT,
    extra_data             TEXT NOT NULL DEFAULT '{}',
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_status ON structures(status);
CREATE INDEX IF NOT EXISTS idx_gce    ON structures(grand_canonical_energy);

CREATE TABLE IF NOT EXISTS run_params (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TRIGGER IF NOT EXISTS trg_updated_at
    AFTER UPDATE ON structures
    FOR EACH ROW
    BEGIN
        UPDATE structures SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
    END;
"""


# ---------------------------------------------------------------------------
# Config diff helpers
# ---------------------------------------------------------------------------

_ENERGY_CRITICAL: frozenset[str] = frozenset({
    "slab.energy",
    "conditions.potential",
    "conditions.pH",
    "conditions.temperature",
    "conditions.pressure",
})


def _flatten_config(cfg: dict) -> dict[str, object]:
    flat: dict[str, object] = {}
    for k, v in cfg.items():
        if k == "adsorbates" and isinstance(v, list):
            for ads in v:
                sym = ads.get("symbol", "?")
                for ak, av in ads.items():
                    flat[f"adsorbates.{sym}.{ak}"] = av
        elif k == "calculator_stages" and isinstance(v, list):
            for stage in v:
                name = stage.get("name", "?")
                for sk, sv in stage.items():
                    flat[f"calculator_stages.{name}.{sk}"] = sv
        elif isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}.{sk}"] = sv
        else:
            flat[k] = v
    return flat


def diff_configs(stored: dict, current: dict) -> list[dict]:
    """Return a list of changed fields between two serialised configs."""
    s = _flatten_config(stored)
    c = _flatten_config(current)
    diffs = []
    for key in sorted(set(s) | set(c)):
        old = s.get(key, "<absent>")
        new = c.get(key, "<absent>")
        if old != new:
            critical = key in _ENERGY_CRITICAL or "chemical_potential" in key
            diffs.append({"field": key, "old": old, "new": new,
                          "energy_critical": critical})
    return diffs


# ---------------------------------------------------------------------------
# GaloopStore
# ---------------------------------------------------------------------------

class GaloopStore:
    """SQLite population store.  Drop-in replacement for GaloopProject."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.run_dir / "galoop.db"
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=30.0,  # wait up to 30s if DB is locked
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def workspace(self) -> Path:
        """Directory containing per-structure subdirectories."""
        d = self.run_dir / "structures"
        d.mkdir(exist_ok=True)
        return d

    def close(self) -> None:
        """Close the database connection, checkpointing WAL first.

        Both calls are best-effort; we don't want a teardown failure to mask
        a more interesting upstream exception.
        """
        import contextlib
        with contextlib.suppress(sqlite3.Error):
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    # -- individual directory -------------------------------------------------

    def individual_dir(self, ind_id: str) -> Path:
        """Return (and create) the filesystem directory for a structure."""
        d = self.workspace / ind_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # -- row ↔ Individual conversion -----------------------------------------

    @staticmethod
    def _row_to_individual(row: sqlite3.Row) -> Individual:
        return Individual(
            id=row["id"],
            parent_ids=json.loads(row["parent_ids"]),
            operator=row["operator"],
            status=row["status"],
            raw_energy=row["raw_energy"],
            grand_canonical_energy=row["grand_canonical_energy"],
            weight=row["weight"],
            geometry_path=row["geometry_path"],
            extra_data=json.loads(row["extra_data"]),
        )

    # -- insert / update ------------------------------------------------------

    def insert(self, ind: Individual) -> Path:
        """Insert an Individual and return its structure directory."""
        self._conn.execute(
            """INSERT OR IGNORE INTO structures
               (id, parent_ids, operator, status, raw_energy,
                grand_canonical_energy, weight, geometry_path, extra_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ind.id,
                json.dumps(ind.parent_ids),
                ind.operator,
                ind.status or STATUS.PENDING,
                ind.raw_energy,
                ind.grand_canonical_energy,
                ind.weight,
                ind.geometry_path,
                json.dumps(ind.extra_data),
            ),
        )
        self._conn.commit()
        return self.individual_dir(ind.id)

    def update(self, ind: Individual) -> None:
        """Persist mutable fields of an Individual."""
        self._conn.execute(
            """UPDATE structures SET
                   status = ?,
                   raw_energy = ?,
                   grand_canonical_energy = ?,
                   weight = ?,
                   geometry_path = ?,
                   extra_data = ?
               WHERE id = ?""",
            (
                ind.status,
                ind.raw_energy,
                ind.grand_canonical_energy,
                ind.weight,
                ind.geometry_path,
                json.dumps(ind.extra_data),
                ind.id,
            ),
        )
        self._conn.commit()

    # -- lookup ---------------------------------------------------------------

    def get(self, ind_id: str) -> Individual | None:
        row = self._conn.execute(
            "SELECT * FROM structures WHERE id = ?", (ind_id,)
        ).fetchone()
        return self._row_to_individual(row) if row else None

    def get_by_status(self, status: str) -> list[Individual]:
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status = ?", (status,)
        ).fetchall()
        return [self._row_to_individual(r) for r in rows]

    def selectable_pool(self) -> list[Individual]:
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status = ? AND weight > 0",
            (STATUS.CONVERGED,),
        ).fetchall()
        return [self._row_to_individual(r) for r in rows]

    def best(self, n: int = 10) -> list[Individual]:
        rows = self._conn.execute(
            """SELECT * FROM structures
               WHERE status = ? AND grand_canonical_energy IS NOT NULL
               ORDER BY grand_canonical_energy
               LIMIT ?""",
            (STATUS.CONVERGED, n),
        ).fetchall()
        return [self._row_to_individual(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) as cnt FROM structures GROUP BY status"
        ).fetchall()
        return {r["status"]: r["cnt"] for r in rows}

    def all_converged_unique(self) -> list[Individual]:
        """Return converged structures with positive weight."""
        rows = self._conn.execute(
            "SELECT * FROM structures WHERE status = ? AND weight > 0",
            (STATUS.CONVERGED,),
        ).fetchall()
        return [self._row_to_individual(r) for r in rows]

    def is_empty(self) -> bool:
        row = self._conn.execute("SELECT COUNT(*) as cnt FROM structures").fetchone()
        return row["cnt"] == 0

    # -- dataframe export -----------------------------------------------------

    def to_dataframe(self):
        import pandas as pd
        rows = self._conn.execute("SELECT * FROM structures").fetchall()
        data = []
        for r in rows:
            data.append({
                "id": r["id"],
                "parent_ids": json.loads(r["parent_ids"]),
                "operator": r["operator"],
                "status": r["status"],
                "raw_energy": r["raw_energy"],
                "grand_canonical_energy": r["grand_canonical_energy"],
                "weight": r["weight"],
                "geometry_path": r["geometry_path"],
                "extra_data": json.loads(r["extra_data"]),
            })
        return pd.DataFrame(data)

    # -- config tracking ------------------------------------------------------

    def save_config_snapshot(self, config: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_params (key, value) VALUES (?, ?)",
            ("config", json.dumps(config)),
        )
        self._conn.commit()

    def diff_config(self, current: dict) -> list[dict]:
        row = self._conn.execute(
            "SELECT value FROM run_params WHERE key = ?", ("config",)
        ).fetchone()
        if row is None:
            return []
        stored = json.loads(row["value"])
        return diff_configs(stored, current)
