"""
galoop/project.py

Signac-based workspace management.  Replaces GaloopDB + sentinel files.

Each structure is a signac Job:
  statepoint : {id, operator, parent_ids, extra_data}   — immutable identity
  doc        : {status, raw_energy, grand_canonical_energy, weight,
                geometry_path, dup_of, tanimoto}         — mutable runtime state

Status lifecycle
----------------
pending   → submitted (row picks up, submits to cluster)
          → relaxed   (pipeline finished, _relax command wrote outcome)
          → converged (unique; GCE computed by main loop)
          → duplicate (main loop classified as dup)
          → failed
          → desorbed
"""

from __future__ import annotations

import logging
from pathlib import Path

import signac

from galoop.individual import Individual, STATUS, OPERATOR

log = logging.getLogger(__name__)

# Extra status written by _relax; signals "pipeline done, not yet classified"
STATUS_RELAXED = "relaxed"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _deep_plain(obj):
    """Recursively convert signac SyncedCollections to plain Python types.

    Signac wraps nested dicts/lists in JSONAttrDict/JSONAttrList which are
    MutableMapping / MutableSequence subclasses but NOT dict/list subclasses.
    """
    from collections.abc import Mapping, Sequence
    if isinstance(obj, Mapping):
        return {k: _deep_plain(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_deep_plain(v) for v in obj]
    return obj


def _job_to_individual(job) -> Individual:
    sp = _deep_plain(dict(job.statepoint))
    doc = _deep_plain(dict(job.doc))
    extra = dict(sp.get("extra_data", {}))
    # Merge doc-level fields (dup_of, tanimoto) into extra_data for compat
    for k in ("dup_of", "tanimoto"):
        if k in doc:
            extra[k] = doc[k]
    return Individual(
        id=sp["id"],
        parent_ids=sp.get("parent_ids", []),
        operator=sp.get("operator", OPERATOR.INIT),
        status=doc.get("status", STATUS.PENDING),
        raw_energy=doc.get("raw_energy"),
        grand_canonical_energy=doc.get("grand_canonical_energy"),
        weight=doc.get("weight", 1.0),
        geometry_path=doc.get("geometry_path"),
        extra_data=extra,
    )


# ---------------------------------------------------------------------------
# Config diff helpers  (moved from database.py)
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
# GaloopProject
# ---------------------------------------------------------------------------

class GaloopProject:
    """Thin wrapper around a signac Project for structure bookkeeping."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self._project = signac.init_project(path=str(self.run_dir))

    @property
    def workspace(self) -> Path:
        return self.run_dir / "workspace"

    # -- job creation -------------------------------------------------------

    def create_job(self, ind: Individual):
        """Create (or open) a signac job for *ind* and return it."""
        statepoint = {
            "id": ind.id,
            "operator": ind.operator,
            "parent_ids": ind.parent_ids,
            "extra_data": ind.extra_data,
        }
        job = self._project.open_job(statepoint)
        job.init()
        # setdefault so that resuming a run doesn't clobber existing state
        job.doc.setdefault("status", ind.status or STATUS.PENDING)
        job.doc.setdefault("weight", ind.weight)
        if ind.raw_energy is not None:
            job.doc["raw_energy"] = ind.raw_energy
        if ind.grand_canonical_energy is not None:
            job.doc["grand_canonical_energy"] = ind.grand_canonical_energy
        if ind.geometry_path:
            job.doc["geometry_path"] = ind.geometry_path
        return job

    # -- lookup -------------------------------------------------------------

    def get_job_by_id(self, ind_id: str):
        jobs = list(self._project.find_jobs({"id": ind_id}))
        return jobs[0] if jobs else None

    # -- update -------------------------------------------------------------

    def update(self, ind: Individual) -> None:
        job = self.get_job_by_id(ind.id)
        if job is None:
            log.warning("No job found for id %s", ind.id)
            return
        job.doc["status"] = ind.status
        if ind.raw_energy is not None:
            job.doc["raw_energy"] = ind.raw_energy
        if ind.grand_canonical_energy is not None:
            job.doc["grand_canonical_energy"] = ind.grand_canonical_energy
        job.doc["weight"] = ind.weight
        if ind.geometry_path:
            job.doc["geometry_path"] = ind.geometry_path
        # Persist doc-level display fields (dup_of, tanimoto)
        for k in ("dup_of", "tanimoto"):
            if k in ind.extra_data:
                job.doc[k] = ind.extra_data[k]

    # -- queries ------------------------------------------------------------

    def get_by_status(self, status: str) -> list[Individual]:
        return [
            _job_to_individual(j)
            for j in self._project
            if j.doc.get("status") == status
        ]

    def selectable_pool(self) -> list[Individual]:
        return [
            _job_to_individual(j)
            for j in self._project
            if j.doc.get("status") == STATUS.CONVERGED
            and j.doc.get("weight", 1.0) > 0
        ]

    def best(self, n: int = 10) -> list[Individual]:
        candidates = [
            j for j in self._project
            if j.doc.get("status") == STATUS.CONVERGED
            and j.doc.get("grand_canonical_energy") is not None
        ]
        candidates.sort(key=lambda j: j.doc["grand_canonical_energy"])
        return [_job_to_individual(j) for j in candidates[:n]]

    def count_by_status(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for job in self._project:
            st = job.doc.get("status", "unknown")
            counts[st] = counts.get(st, 0) + 1
        return counts

    # -- compatibility aliases (mirrors old GaloopDB API) -------------------

    def insert(self, ind: Individual):
        """Create a job for *ind* and return it (mirrors GaloopDB.insert)."""
        return self.create_job(ind)

    def get(self, ind_id: str) -> Individual | None:
        """Return the Individual with *ind_id*, or None (mirrors GaloopDB.get)."""
        job = self.get_job_by_id(ind_id)
        return _job_to_individual(job) if job is not None else None

    def find_by_geometry_path_substring(self, substr: str) -> Individual | None:
        for job in self._project:
            if substr in job.doc.get("geometry_path", ""):
                return _job_to_individual(job)
        return None

    def all_converged_unique_jobs(self):
        """Yield signac jobs that are converged and not duplicates."""
        for job in self._project:
            if (job.doc.get("status") == STATUS.CONVERGED
                    and job.doc.get("weight", 1.0) > 0):
                yield job

    # -- config tracking ----------------------------------------------------

    def to_dataframe(self):
        """Return all structures as a pandas DataFrame (compatible with old GaloopDB.to_dataframe)."""
        import pandas as pd

        rows = []
        for job in self._project:
            sp = dict(job.statepoint)
            doc = dict(job.doc)
            rows.append({
                "id": sp.get("id"),
                "parent_ids": sp.get("parent_ids", []),
                "operator": sp.get("operator", "init"),
                "status": doc.get("status", "pending"),
                "raw_energy": doc.get("raw_energy"),
                "grand_canonical_energy": doc.get("grand_canonical_energy"),
                "weight": doc.get("weight", 1.0),
                "geometry_path": doc.get("geometry_path"),
                "extra_data": sp.get("extra_data", {}),
            })
        return pd.DataFrame(rows)

    def save_config_snapshot(self, config: dict) -> None:
        self._project.doc["config"] = config

    def diff_config(self, current: dict) -> list[dict]:
        stored = self._project.doc.get("config")
        if stored is None:
            return []
        return diff_configs(stored, current)
