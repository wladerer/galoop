"""
galoop/individual.py

Minimal Individual data model for the GA.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class STATUS:
    """Structure lifecycle states."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    CONVERGED = "converged"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    DESORBED = "desorbed"

    _TERMINAL = frozenset({"converged", "failed", "duplicate", "desorbed"})
    _ACTIVE = frozenset({"pending", "submitted"})

    @staticmethod
    def is_terminal(status: str) -> bool:
        return status in STATUS._TERMINAL

    @staticmethod
    def is_active(status: str) -> bool:
        return status in STATUS._ACTIVE


class OPERATOR:
    """GA operator labels."""

    INIT = "init"
    SPLICE = "splice"
    MERGE = "merge"
    MUTATE_ADD = "mutate_add"
    MUTATE_REMOVE = "mutate_remove"
    MUTATE_DISPLACE = "mutate_displace"
    MUTATE_RATTLE_SLAB = "mutate_rattle_slab"
    MUTATE_TRANSLATE = "mutate_translate"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

def _short_uuid() -> str:
    return str(uuid.uuid4())[:8]


class Individual(BaseModel):
    """
    A single candidate structure in the GA population.

    Fields
    ------
    id : 8-char UUID fragment
    parent_ids : IDs of parent structures (empty for initial population)
    operator : which GA operator produced this structure
    status : current lifecycle state
    raw_energy : DFT total energy (eV), set after relaxation
    grand_canonical_energy : CHE-corrected fitness (eV)
    weight : selection weight (0.0 for duplicates, 1.0 otherwise)
    geometry_path : filesystem path to POSCAR/CONTCAR
    extra_data : adsorbate counts, stage energies, notes
    """

    model_config = {"frozen": False}

    id: str = Field(default_factory=_short_uuid)
    parent_ids: list[str] = Field(default_factory=list)
    operator: str = OPERATOR.INIT
    status: str = STATUS.PENDING

    raw_energy: float | None = None
    grand_canonical_energy: float | None = None
    weight: float = 1.0

    geometry_path: str | None = None
    extra_data: dict[str, Any] = Field(default_factory=dict)

    # -- factory methods ---------------------------------------------------

    @classmethod
    def from_init(
        cls,
        geometry_path: str | None = None,
        extra_data: dict | None = None,
    ) -> Individual:
        """Create a structure for the initial random population."""
        return cls(
            operator=OPERATOR.INIT,
            status=STATUS.PENDING,
            geometry_path=geometry_path,
            extra_data=extra_data or {},
        )

    @classmethod
    def from_parents(
        cls,
        parents: list[Individual],
        operator: str,
        geometry_path: str | None = None,
        extra_data: dict | None = None,
    ) -> Individual:
        """Create an offspring from one or more parents."""
        return cls(
            parent_ids=[p.id for p in parents],
            operator=operator,
            status=STATUS.PENDING,
            geometry_path=geometry_path,
            extra_data=extra_data or {},
        )

    # -- immutable update helpers ------------------------------------------

    def with_status(self, status: str) -> Individual:
        """Return a copy with a new status."""
        copy = self.model_copy()
        copy.status = status
        return copy

    def with_energy(self, raw: float, grand_canonical: float) -> Individual:
        """Return a copy with energies set."""
        copy = self.model_copy()
        copy.raw_energy = raw
        copy.grand_canonical_energy = grand_canonical
        copy.status = STATUS.CONVERGED
        return copy

    def with_weight(self, weight: float) -> Individual:
        copy = self.model_copy()
        copy.weight = weight
        return copy

    def mark_duplicate(self) -> Individual:
        """Return a copy marked as duplicate with zero selection weight."""
        copy = self.model_copy()
        copy.status = STATUS.DUPLICATE
        copy.weight = 0.0
        return copy

    # -- predicates --------------------------------------------------------

    @property
    def is_selectable(self) -> bool:
        return self.status == STATUS.CONVERGED and self.weight > 0

    @property
    def is_converged(self) -> bool:
        return self.status == STATUS.CONVERGED
