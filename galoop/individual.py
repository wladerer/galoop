"""
gocia/individual.py

Simplified Individual model — minimal fields, clear semantics.
"""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class STATUS:
    """Status constants for the structure lifecycle."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    CONVERGED = "converged"
    FAILED = "failed"
    DUPLICATE = "duplicate"
    DESORBED = "desorbed"

    @staticmethod
    def is_terminal(status: str) -> bool:
        return status in {STATUS.CONVERGED, STATUS.FAILED, STATUS.DUPLICATE, STATUS.DESORBED}

    @staticmethod
    def is_running_or_submitted(status: str) -> bool:
        return status in {STATUS.SUBMITTED, STATUS.PENDING}


class OPERATOR:
    """Operator constants."""
    INIT = "init"
    SPLICE = "splice"
    MERGE = "merge"
    MUTATE_ADD = "mutate_add"
    MUTATE_REMOVE = "mutate_remove"
    MUTATE_DISPLACE = "mutate_displace"


class Individual(BaseModel):
    """
    A single structure in the GA.

    Minimal fields:
    - id, generation, parents, operator: genealogy
    - status: pipeline state
    - raw_energy, grand_canonical_energy: fitness
    - weight: selection weight (1.0 or 0.0 for duplicates)
    - geometry_path: where the CONTCAR lives
    - extra_data: adsorbate counts, stage energies, etc.
    """

    model_config = {"frozen": False}

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    generation: int = 0
    parent_ids: list[str] = Field(default_factory=list)
    operator: str = OPERATOR.INIT
    status: str = STATUS.PENDING

    raw_energy: float | None = None
    grand_canonical_energy: float | None = None
    weight: float = 1.0

    geometry_path: str | None = None
    extra_data: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_init(
        cls,
        generation: int,
        geometry_path: str | None = None,
        extra_data: dict | None = None,
    ) -> Individual:
        """Create a fresh structure for the initial population."""
        return cls(
            generation=generation,
            operator=OPERATOR.INIT,
            status=STATUS.PENDING,
            geometry_path=geometry_path,
            extra_data=extra_data or {},
        )

    @classmethod
    def from_parents(
        cls,
        generation: int,
        parents: list[Individual],
        operator: str,
        geometry_path: str | None = None,
        extra_data: dict | None = None,
    ) -> Individual:
        """Create a structure from parent(s)."""
        return cls(
            generation=generation,
            parent_ids=[p.id for p in parents],
            operator=operator,
            status=STATUS.PENDING,
            geometry_path=geometry_path,
            extra_data=extra_data or {},
        )

    def with_status(self, status: str) -> Individual:
        """Return a copy with new status."""
        copy = self.model_copy()
        copy.status = status
        return copy

    def with_energy(self, raw: float, grand_canonical: float) -> Individual:
        """Return a copy with new energies."""
        copy = self.model_copy()
        copy.raw_energy = raw
        copy.grand_canonical_energy = grand_canonical
        return copy

    def with_weight(self, weight: float) -> Individual:
        """Return a copy with new weight."""
        copy = self.model_copy()
        copy.weight = weight
        return copy

    def mark_duplicate(self) -> Individual:
        """Mark as duplicate, zero weight."""
        copy = self.model_copy()
        copy.status = STATUS.DUPLICATE
        copy.weight = 0.0
        return copy

    @property
    def is_selectable(self) -> bool:
        """Can this structure be selected as a parent?"""
        return self.status == STATUS.CONVERGED and self.weight > 0

    @property
    def is_converged(self) -> bool:
        return self.status == STATUS.CONVERGED
