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
        **extra,                         # absorb unknown keys from config
    ):
        self.name = name
        self.calc_type = type.lower()
        self.fmax = fmax
        self.max_steps = max_steps
        self.energy_per_atom_tol = energy_per_atom_tol
        self.max_force_tol = max_force_tol
        self.incar = incar or {}

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
        mace_model: str = "medium",
        mace_device: str = "cpu",
    ) -> StageResult:
        """
        Relax *atoms* with this stage's calculator.

        Writes ``CONTCAR`` inside ``struct_dir/stage_{name}/``.
        """
        stage_dir = Path(struct_dir) / f"stage_{self.name}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        log.debug("Running %s in %s", self.name, stage_dir)

        try:
            if self.calc_type == "mace":
                return self._run_mace(atoms, stage_dir, mace_model, mace_device)
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
    ) -> StageResult:
        """Relax with a MACE-MP foundation model."""
        try:
            from mace.calculators import mace_mp
        except ImportError as exc:
            raise ImportError(
                "MACE-MP requires mace-torch.  Install with: pip install mace-torch"
            ) from exc

        calc = mace_mp(
            model=mace_model,          # "small", "medium", or "large"
            device=mace_device,
            default_dtype="float32",
        )
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
            converged = dyn.run(fmax=self.fmax, steps=self.max_steps)
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

    def _run_vasp(self, atoms: Atoms, stage_dir: Path) -> StageResult:
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
            "ediffg": -self.fmax,
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
                mace_model=mace_model, mace_device=mace_device,
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
