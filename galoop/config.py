"""
galoop/config.py

Configuration loading and Pydantic validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

class SlabConfig(BaseModel):
    geometry: str = Field(..., description="Path to POSCAR/CONTCAR")
    energy: float = Field(..., description="DFT energy of bare slab (eV)")
    sampling_zmin: float = Field(..., description="Min z for adsorbate placement (Å)")
    sampling_zmax: float = Field(..., description="Max z for adsorbate placement (Å)")

    @model_validator(mode="after")
    def _zmin_lt_zmax(self) -> SlabConfig:
        if self.sampling_zmin >= self.sampling_zmax:
            raise ValueError("sampling_zmin must be < sampling_zmax")
        return self


class AdsorbateConfig(BaseModel):
    symbol: str = Field(..., description="Chemical symbol or formula")
    chemical_potential: float = Field(..., description="Standard chemical potential (eV)")
    n_orientations: int = Field(default=1, ge=1)
    min_count: int = Field(default=0, ge=0)
    max_count: int = Field(default=5, ge=0)
    geometry: str | None = Field(default=None, description="Path to geometry file")
    coordinates: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _max_ge_min(self) -> AdsorbateConfig:
        if self.max_count < self.min_count:
            raise ValueError("max_count must be >= min_count")
        return self


class StageConfig(BaseModel):
    name: str
    type: str = Field(..., description="Calculator type: mace | vasp")
    fmax: float = Field(default=0.05, gt=0.0)
    max_steps: int = Field(default=300, ge=1)
    energy_per_atom_tol: float = Field(default=10.0, gt=0.0)
    max_force_tol: float = Field(default=50.0, gt=0.0)
    incar: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _normalise_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ("mace", "vasp"):
            raise ValueError(f"type must be 'mace' or 'vasp', got '{v}'")
        return v


class SchedulerConfig(BaseModel):
    type: str = Field(default="local")
    nworkers: int = Field(default=4, ge=1)
    walltime: str = Field(default="01:00:00")
    resources: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _normalise_type(cls, v: str) -> str:
        v = v.lower()
        if v not in ("local", "slurm", "pbs"):
            raise ValueError(f"type must be 'local', 'slurm', or 'pbs', got '{v}'")
        return v


class FingerprintConfig(BaseModel):
    r_cut: float = Field(default=6.0, gt=0.0)
    n_max: int = Field(default=8, ge=1)
    l_max: int = Field(default=6, ge=0)
    duplicate_threshold: float = Field(default=0.90, gt=0.0, le=1.0)


class GAConfig(BaseModel):
    population_size: int = Field(default=20, ge=2)
    max_generations: int = Field(default=50, ge=1)
    min_generations: int = Field(default=5, ge=1)
    max_stall_generations: int = Field(default=10, ge=1)
    min_adsorbates: int = Field(default=1, ge=0)
    max_adsorbates: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def _gen_bounds(self) -> GAConfig:
        if self.min_generations > self.max_generations:
            raise ValueError("min_generations must be <= max_generations")
        return self

    @model_validator(mode="after")
    def _ads_bounds(self) -> GAConfig:
        if self.min_adsorbates > self.max_adsorbates:
            raise ValueError("min_adsorbates must be <= max_adsorbates")
        return self


class ConditionsConfig(BaseModel):
    temperature: float = Field(default=298.15, gt=0.0)
    pressure: float = Field(default=1.0, gt=0.0)
    potential: float = Field(default=0.0)
    pH: float = Field(default=0.0, ge=0.0, le=14.0)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class GaloopConfig(BaseModel):
    slab: SlabConfig
    adsorbates: list[AdsorbateConfig] = Field(..., min_length=1)
    calculator_stages: list[StageConfig] = Field(..., min_length=1)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ga: GAConfig = Field(default_factory=GAConfig)
    conditions: ConditionsConfig = Field(default_factory=ConditionsConfig)
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)
    mace_model: str = Field(default="medium")
    mace_device: str = Field(default="cpu")

    @field_validator("mace_device")
    @classmethod
    def _valid_device(cls, v: str) -> str:
        v = v.lower()
        if v not in ("cpu", "cuda", "auto"):
            raise ValueError(f"mace_device must be cpu/cuda/auto, got '{v}'")
        return v

    @model_validator(mode="after")
    def _unique_stage_names(self) -> GaloopConfig:
        names = [s.name for s in self.calculator_stages]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate calculator stage names")
        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> GaloopConfig:
    """Load and validate a YAML config file."""
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install pyyaml") from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open() as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raise ValueError("Config file is empty")

    return GaloopConfig.model_validate(raw)
