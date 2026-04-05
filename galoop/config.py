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
    binding_index: int = Field(default=0, ge=0, description="Index of the surface-binding atom")
    geometry: str | None = Field(default=None, description="Path to geometry file")
    coordinates: list[list[float]] | None = Field(default=None)

    @model_validator(mode="after")
    def _max_ge_min(self) -> AdsorbateConfig:
        if self.max_count < self.min_count:
            raise ValueError("max_count must be >= min_count")
        return self

    @model_validator(mode="after")
    def _require_geometry_for_polyatomic(self) -> AdsorbateConfig:
        from galoop.science.surface import parse_formula
        if len(parse_formula(self.symbol)) > 1:
            if self.geometry is None and self.coordinates is None:
                raise ValueError(
                    f"Adsorbate '{self.symbol}' has multiple atoms; "
                    "specify 'geometry' (file path) or 'coordinates' (inline positions) in the config"
                )
        return self


class StageConfig(BaseModel):
    name: str
    type: str = Field(..., description="Calculator type: mace | vasp")
    fmax: float = Field(default=0.05, gt=0.0)
    max_steps: int = Field(default=300, ge=1)
    energy_per_atom_tol: float = Field(default=10.0, gt=0.0)
    max_force_tol: float = Field(default=50.0, gt=0.0)
    incar: dict[str, Any] = Field(default_factory=dict)
    fix_slab_first: bool = Field(default=False,
        description="Pre-relax adsorbates with slab fully fixed before the main relax")
    prescan_fmax: float | None = Field(default=None, gt=0.0,
        description="Force cutoff for the prescan (defaults to fmax if unset)")

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
    energy_tol_pct: float = Field(default=5.0, ge=0.0,
        description="Energy gate: max % difference relative to existing structure")
    dist_hist_bins: int = Field(default=50, ge=10,
        description="Number of bins for pairwise distance histogram")
    dist_hist_threshold: float = Field(default=0.95, ge=0.0, le=1.0,
        description="Min cosine similarity for distance histogram gate")


class OperatorWeightsConfig(BaseModel):
    splice: float = Field(default=0.30, ge=0.0)
    merge: float = Field(default=0.20, ge=0.0)
    mutate_add: float = Field(default=0.15, ge=0.0)
    mutate_remove: float = Field(default=0.10, ge=0.0)
    mutate_displace: float = Field(default=0.10, ge=0.0)
    mutate_rattle_slab: float = Field(default=0.05, ge=0.0)
    mutate_translate: float = Field(default=0.10, ge=0.0)

    @model_validator(mode="after")
    def _nonzero_sum(self) -> "OperatorWeightsConfig":
        total = (self.splice + self.merge + self.mutate_add
                 + self.mutate_remove + self.mutate_displace
                 + self.mutate_rattle_slab + self.mutate_translate)
        if total <= 0:
            raise ValueError("Operator weights must sum to a positive value")
        return self


class GAConfig(BaseModel):
    population_size: int = Field(default=20, ge=2)
    max_structures: int = Field(default=1000, ge=1)
    min_structures: int = Field(default=100, ge=1)
    max_stall: int = Field(default=10, ge=1)
    min_adsorbates: int = Field(default=1, ge=0)
    max_adsorbates: int = Field(default=8, ge=1)
    boltzmann_temperature: float = Field(
        default=0.1, gt=0.0,
        description="kT-like scale for Boltzmann parent selection (eV)",
    )
    rattle_amplitude: float = Field(
        default=0.1, gt=0.0,
        description="Gaussian sigma for slab surface rattling (Å)",
    )
    operator_weights: OperatorWeightsConfig = Field(
        default_factory=OperatorWeightsConfig,
        description="Relative probabilities for each GA operator",
    )

    @model_validator(mode="after")
    def _struct_bounds(self) -> GAConfig:
        if self.min_structures > self.max_structures:
            raise ValueError("min_structures must be <= max_structures")
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
    mace_model: str = Field(
        default="medium",
        description=(
            "MACE model identifier ('small', 'medium', 'large', 'medium-0b3', …) "
            "or an absolute/relative path to a custom .pt model file."
        ),
    )
    mace_device: str = Field(default="cpu")
    mace_dtype: str = Field(
        default="float32",
        description="Floating-point dtype for MACE ('float32' or 'float64').",
    )

    @field_validator("mace_device")
    @classmethod
    def _valid_device(cls, v: str) -> str:
        v = v.lower()
        if v not in ("cpu", "cuda", "auto"):
            raise ValueError(f"mace_device must be cpu/cuda/auto, got '{v}'")
        return v

    @field_validator("mace_dtype")
    @classmethod
    def _valid_dtype(cls, v: str) -> str:
        if v not in ("float32", "float64"):
            raise ValueError(f"mace_dtype must be float32 or float64, got '{v}'")
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
