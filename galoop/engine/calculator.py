"""
galoop/engine/calculator.py

Multi-stage calculator pipeline: MACE-MP and/or VASP.

Each stage relaxes the geometry and writes a CONTCAR for the next stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.optimize import BFGS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class StageResult(NamedTuple):
    """Outcome of a single calculator stage."""
    converged: bool
    energy: float
    n_steps: int
    trajectory_file: str | None


# ---------------------------------------------------------------------------
# Single stage
# ---------------------------------------------------------------------------

class CalculatorStage:
    """One stage in the multi-stage relaxation pipeline."""

    def __init__(
        self,
        name: str,
        type: str,                       # noqa: A002 — shadows builtin; kept for config compat
        fmax: float = 0.05,
        max_steps: int = 300,
        energy_per_atom_tol: float = 10.0,
        max_force_tol: float = 50.0,
        incar: dict | None = None,
        fix_slab_first: bool = False,
        prescan_fmax: float | None = None,
        **extra,                         # absorb unknown keys from config
    ):
        self.name = name
        self.calc_type = type.lower()
        self.fmax = fmax
        self.max_steps = max_steps
        self.energy_per_atom_tol = energy_per_atom_tol
        self.max_force_tol = max_force_tol
        self.incar = incar or {}
        self.fix_slab_first = fix_slab_first
        self.prescan_fmax = prescan_fmax if prescan_fmax is not None else fmax

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
        mace_model: str = "medium",
        mace_device: str = "cpu",
        mace_dtype: str = "float32",
        n_slab_atoms: int = 0,
    ) -> StageResult:
        """
        Relax *atoms* with this stage's calculator.

        Writes ``CONTCAR`` inside ``struct_dir/stage_{name}/``.
        """
        from ase.constraints import FixAtoms

        stage_dir = Path(struct_dir) / f"stage_{self.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log.debug("Running %s in %s", self.name, stage_dir)

        if self.fix_slab_first and n_slab_atoms > 0 and n_slab_atoms < len(atoms):
            prescan_dir = Path(struct_dir) / f"stage_{self.name}_prescan"
            prescan_dir.mkdir(parents=True, exist_ok=True)

            prescan_atoms = atoms.copy()
            prescan_atoms.set_constraint(FixAtoms(indices=list(range(n_slab_atoms))))

            log.debug("Prescan (adsorbates-only) for %s in %s (fmax=%.4f)", self.name, prescan_dir, self.prescan_fmax)
            try:
                if self.calc_type == "mace":
                    self._run_mace(prescan_atoms, prescan_dir, mace_model, mace_device, mace_dtype, fmax=self.prescan_fmax)
                elif self.calc_type == "vasp":
                    self._run_vasp(prescan_atoms, prescan_dir, fmax=self.prescan_fmax)
            except Exception as exc:
                log.warning("%s prescan failed (%s) — proceeding with original geometry", self.name, exc)
            else:
                prescan_contcar = prescan_dir / "CONTCAR"
                if prescan_contcar.exists():
                    try:
                        loaded = read(str(prescan_contcar), format="vasp")
                        loaded.set_constraint(atoms.constraints)
                        atoms = loaded
                    except Exception as exc:
                        log.warning("Failed to load prescan CONTCAR: %s", exc)

        try:
            if self.calc_type == "mace":
                return self._run_mace(atoms, stage_dir, mace_model, mace_device, mace_dtype)
            elif self.calc_type == "vasp":
                return self._run_vasp(atoms, stage_dir)
            else:
                raise ValueError(f"Unknown calculator type: {self.calc_type}")
        except Exception as exc:
            log.error("%s failed: %s", self.name, exc)
            return StageResult(False, float("nan"), 0, None)

    # -- MACE --------------------------------------------------------------

    def _run_mace(
        self,
        atoms: Atoms,
        stage_dir: Path,
        mace_model: str,
        mace_device: str,
        mace_dtype: str = "float32",
        fmax: float | None = None,
    ) -> StageResult:
        """Relax with a MACE calculator (foundation model or custom .pt file)."""
        from pathlib import Path as _Path
        try:
            model_path = _Path(mace_model)
            if model_path.exists():
                # Custom / fine-tuned model file → use MACECalculator directly
                from mace.calculators import MACECalculator
                calc = MACECalculator(
                    model_paths=str(model_path),
                    device=mace_device,
                    default_dtype=mace_dtype,
                )
            else:
                # Named foundation model ("small", "medium", "large", "medium-0b3", …)
                from mace.calculators import mace_mp
                calc = mace_mp(model=mace_model, device=mace_device, default_dtype=mace_dtype)
        except ImportError as exc:
            raise ImportError(
                "MACE requires mace-torch.  Install with: pip install mace-torch"
            ) from exc
        atoms = atoms.copy()
        atoms.calc = calc

        traj_path = stage_dir / "trajectory.traj"
        log_path = stage_dir / "relax.log"
        dyn = BFGS(
            atoms,
            trajectory=str(traj_path),
            logfile=str(log_path),
        )

        try:
            converged = dyn.run(fmax=fmax if fmax is not None else self.fmax, steps=self.max_steps)
        except Exception as exc:
            log.warning("MACE relax failed: %s", exc)
            converged = False

        energy = float(atoms.get_potential_energy())
        n_steps = getattr(dyn, "nsteps", 0)

        # Sanity check: unreasonable energy per atom → treat as failed
        epa = abs(energy) / max(len(atoms), 1)
        if epa > self.energy_per_atom_tol:
            log.warning(
                "%s energy/atom = %.2f eV — exceeds tolerance %.1f",
                self.name, epa, self.energy_per_atom_tol,
            )
            converged = False

        self._write_contcar(atoms, stage_dir)

        return StageResult(
            converged=converged,
            energy=energy,
            n_steps=n_steps,
            trajectory_file=str(traj_path),
        )

    # -- VASP --------------------------------------------------------------

    def _run_vasp(self, atoms: Atoms, stage_dir: Path, fmax: float | None = None) -> StageResult:
        """Relax with VASP via ASE's Vasp calculator."""
        from ase.calculators.vasp import Vasp

        # ASE's Vasp calculator expects **lowercase** keyword arguments.
        # The user-facing config may use uppercase INCAR keys, so we lower-case them.
        vasp_kwargs: dict[str, object] = {
            "ismear": 0,
            "sigma": 0.05,
            "algo": "Fast",
            "lreal": "Auto",
            "lwave": False,
            "lcharg": False,
            "ibrion": 2,
            "nsw": self.max_steps,
            "ediffg": -(fmax if fmax is not None else self.fmax),
            "ediff": 1e-5,
        }
        for key, val in self.incar.items():
            vasp_kwargs[key.lower()] = val

        try:
            calc = Vasp(directory=str(stage_dir), **vasp_kwargs)
        except Exception as exc:
            log.warning("VASP init failed: %s", exc)
            return StageResult(False, float("nan"), 0, None)

        atoms = atoms.copy()
        atoms.calc = calc

        try:
            energy = float(atoms.get_potential_energy())
            converged = True
        except Exception as exc:
            log.warning("VASP run failed: %s", exc)
            energy = float("nan")
            converged = False

        # Read VASP-optimised geometry
        contcar = stage_dir / "CONTCAR"
        if contcar.exists():
            try:
                atoms = read(str(contcar), format="vasp")
            except Exception as exc:
                log.warning("Failed to read VASP CONTCAR: %s", exc)

        return StageResult(
            converged=converged,
            energy=energy,
            n_steps=0,   # would need OUTCAR parsing for exact count
            trajectory_file=None,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _write_contcar(atoms: Atoms, stage_dir: Path) -> None:
        try:
            write(str(stage_dir / "CONTCAR"), atoms, format="vasp")
        except Exception as exc:
            log.warning("Failed to write CONTCAR: %s", exc)


# ---------------------------------------------------------------------------
# Multi-stage pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Run a sequence of :class:`CalculatorStage` objects on one structure."""

    def __init__(self, stages: list[CalculatorStage]):
        if not stages:
            raise ValueError("Pipeline must have at least one stage")
        self.stages = stages

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
        mace_model: str = "medium",
        mace_device: str = "cpu",
        mace_dtype: str = "float32",
        n_slab_atoms: int = 0,
    ) -> dict:
        """
        Run all stages sequentially.

        Returns
        -------
        dict with keys:
            converged  — True only if **all** stages converged
            final_energy — energy from the last stage
            stage_results — {name: StageResult}
            final_atoms — geometry after the last stage
        """
        struct_dir = Path(struct_dir)
        struct_dir.mkdir(parents=True, exist_ok=True)

        current_atoms = atoms.copy()
        stage_results: dict[str, StageResult] = {}
        all_converged = True

        for stage in self.stages:
            result = stage.run(
                current_atoms, struct_dir,
                mace_model=mace_model, mace_device=mace_device, mace_dtype=mace_dtype,
                n_slab_atoms=n_slab_atoms,
            )
            stage_results[stage.name] = result

            if not result.converged:
                all_converged = False
                log.warning("%s did not converge", stage.name)

            # Load output geometry for next stage
            contcar = struct_dir / f"stage_{stage.name}" / "CONTCAR"
            if contcar.exists():
                try:
                    current_atoms = read(str(contcar), format="vasp")
                except Exception as exc:
                    log.warning("Failed to load %s output: %s", stage.name, exc)
                    all_converged = False
                    break
            else:
                log.warning("No CONTCAR from %s", stage.name)
                all_converged = False
                break

        # Write final outputs
        final_contcar = struct_dir / "CONTCAR"
        try:
            write(str(final_contcar), current_atoms, format="vasp")
        except Exception as exc:
            log.warning("Failed to write final CONTCAR: %s", exc)

        last_stage_name = self.stages[-1].name
        final_energy = stage_results[last_stage_name].energy
        energy_file = struct_dir / "FINAL_ENERGY"
        try:
            energy_file.write_text(f"{final_energy:.10f}\n")
        except Exception as exc:
            log.warning("Failed to write energy file: %s", exc)

        return {
            "converged": all_converged,
            "final_energy": final_energy,
            "stage_results": stage_results,
            "final_atoms": current_atoms,
        }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_pipeline(stage_configs: list) -> Pipeline:
    """
    Build a :class:`Pipeline` from a list of stage config dicts or Pydantic models.
    """
    stages: list[CalculatorStage] = []
    for cfg in stage_configs:
        d = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
        stages.append(CalculatorStage(**d))
    return Pipeline(stages)
