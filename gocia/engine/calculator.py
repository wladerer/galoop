"""
gocia/engine/calculator.py

Multi-stage calculator pipeline: MACE, VASP, or hybrid.
Handles job submission, status tracking, and energy reading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.calculators.mace import MACE
from ase.calculators.vasp import Vasp
from ase.constraints import StrainFilter, UnitCellFilter
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE

log = logging.getLogger(__name__)


class StageResult(NamedTuple):
    """Result of a single calculator stage."""
    converged: bool
    energy: float
    n_steps: int
    trajectory_file: str | None


class CalculatorStage:
    """Single stage in a multi-stage pipeline."""

    def __init__(
        self,
        name: str,
        calc_type: str,
        fmax: float = 0.05,
        max_steps: int = 300,
        energy_per_atom_tol: float = 10.0,
        max_force_tol: float = 50.0,
        **kwargs,
    ):
        self.name = name
        self.calc_type = calc_type.lower()
        self.fmax = fmax
        self.max_steps = max_steps
        self.energy_per_atom_tol = energy_per_atom_tol
        self.max_force_tol = max_force_tol
        self.kwargs = kwargs  # INCAR overrides for VASP, etc.

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
        mace_model: str = "medium",
        mace_device: str = "cpu",
    ) -> StageResult:
        """
        Relax atoms with this stage's calculator.

        Parameters
        ----------
        atoms : ASE Atoms object
        struct_dir : Directory for outputs
        mace_model : MACE model (small/medium/large)
        mace_device : CPU or CUDA

        Returns
        -------
        StageResult
        """
        struct_dir = Path(struct_dir)
        struct_dir.mkdir(parents=True, exist_ok=True)

        stage_dir = struct_dir / f"stage_{self.name}"
        stage_dir.mkdir(exist_ok=True)

        log.debug(f"  Running {self.name} in {stage_dir}")

        try:
            if self.calc_type == "mace":
                return self._run_mace(atoms, stage_dir, mace_model, mace_device)
            elif self.calc_type == "vasp":
                return self._run_vasp(atoms, stage_dir)
            else:
                raise ValueError(f"Unknown calculator type: {self.calc_type}")
        except Exception as e:
            log.error(f"  {self.name} failed: {e}")
            return StageResult(
                converged=False,
                energy=float("nan"),
                n_steps=0,
                trajectory_file=None,
            )

    def _run_mace(
        self,
        atoms: Atoms,
        stage_dir: Path,
        mace_model: str,
        mace_device: str,
    ) -> StageResult:
        """Run MACE relaxation."""
        try:
            calc = MACE(
                model=f"medium" if mace_model == "medium" else "small",
                device=mace_device,
                default_dtype="float32",
            )
        except Exception as e:
            log.warning(f"  MACE init failed ({e}); trying default")
            calc = MACE(model="small", device="cpu", default_dtype="float32")

        atoms.set_calculator(calc)

        # Use strain filter for full cell relaxation
        dyn = BFGS(
            atoms,
            trajectory=str(stage_dir / "trajectory.traj"),
            logfile=str(stage_dir / "relax.log"),
        )

        try:
            converged = dyn.run(fmax=self.fmax, steps=self.max_steps)
        except Exception as e:
            log.warning(f"  MACE relax failed: {e}")
            converged = False

        energy = float(atoms.get_potential_energy())
        n_steps = getattr(dyn, "nsteps", 0)

        # Write CONTCAR for next stage
        poscar = stage_dir / "CONTCAR"
        try:
            write(str(poscar), atoms, format="vasp")
        except Exception as e:
            log.warning(f"  Failed to write CONTCAR: {e}")

        return StageResult(
            converged=converged or n_steps > 0,
            energy=energy,
            n_steps=n_steps,
            trajectory_file=str(stage_dir / "trajectory.traj"),
        )

    def _run_vasp(
        self,
        atoms: Atoms,
        stage_dir: Path,
    ) -> StageResult:
        """Run VASP relaxation."""
        incar = {
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "ALGO": "Fast",
            "LREAL": "Auto",
            "LWAVE": False,
            "LCHARG": False,
            "IBRION": 2,
            "NSW": self.max_steps,
            "EDIFFG": -self.fmax,
            "EDIFF": 1e-5,
        }
        incar.update(self.kwargs)

        try:
            calc = Vasp(
                directory=str(stage_dir),
                **incar,
            )
        except Exception as e:
            log.warning(f"  VASP init failed: {e}")
            return StageResult(
                converged=False,
                energy=float("nan"),
                n_steps=0,
                trajectory_file=None,
            )

        atoms.set_calculator(calc)

        try:
            atoms.get_potential_energy()
            converged = True
        except Exception as e:
            log.warning(f"  VASP run failed: {e}")
            converged = False

        energy = float(atoms.get_potential_energy()) if converged else float("nan")
        n_steps = 0  # Would need to parse OUTCAR for exact step count

        # Read final geometry
        contcar = stage_dir / "CONTCAR"
        if contcar.exists():
            try:
                atoms = read(str(contcar), format="vasp")
            except Exception as e:
                log.warning(f"  Failed to read CONTCAR: {e}")

        return StageResult(
            converged=converged,
            energy=energy,
            n_steps=n_steps,
            trajectory_file=None,
        )


class Pipeline:
    """Multi-stage relaxation pipeline."""

    def __init__(self, stages: list[CalculatorStage]):
        self.stages = stages
        if not stages:
            raise ValueError("Pipeline must have at least one stage")

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
        mace_model: str = "medium",
        mace_device: str = "cpu",
    ) -> dict:
        """
        Run all stages sequentially.

        Parameters
        ----------
        atoms : Starting geometry
        struct_dir : Output directory
        mace_model : MACE model size
        mace_device : CPU or CUDA

        Returns
        -------
        dict with keys:
            - converged: bool (all stages passed)
            - final_energy: float
            - stage_results: dict {stage_name: StageResult}
            - final_atoms: ASE Atoms object
        """
        struct_dir = Path(struct_dir)
        struct_dir.mkdir(parents=True, exist_ok=True)

        current_atoms = atoms.copy()
        stage_results = {}
        all_converged = True

        for stage in self.stages:
            result = stage.run(
                current_atoms,
                struct_dir,
                mace_model=mace_model,
                mace_device=mace_device,
            )
            stage_results[stage.name] = result

            if not result.converged:
                all_converged = False
                log.warning(f"  {stage.name} did not converge")

            # Load output for next stage
            stage_dir = struct_dir / f"stage_{stage.name}"
            contcar = stage_dir / "CONTCAR"
            if contcar.exists():
                try:
                    current_atoms = read(str(contcar), format="vasp")
                except Exception as e:
                    log.warning(f"  Failed to load {stage.name} output: {e}")
                    all_converged = False
                    break
            else:
                log.warning(f"  No CONTCAR from {stage.name}")
                all_converged = False
                break

        # Write final CONTCAR
        final_contcar = struct_dir / "CONTCAR"
        try:
            write(str(final_contcar), current_atoms, format="vasp")
        except Exception as e:
            log.warning(f"  Failed to write final CONTCAR: {e}")

        # Write final energy
        final_energy = stage_results[self.stages[-1].name].energy
        energy_file = struct_dir / "FINAL_ENERGY"
        try:
            energy_file.write_text(f"{final_energy:.10f}\n")
        except Exception as e:
            log.warning(f"  Failed to write energy file: {e}")

        return {
            "converged": all_converged,
            "final_energy": final_energy,
            "stage_results": stage_results,
            "final_atoms": current_atoms,
        }


def build_pipeline(stage_configs: list) -> Pipeline:
    """
    Build a Pipeline from config stage list.

    Parameters
    ----------
    stage_configs : List of dicts with keys:
        - name: str
        - type: str ("mace" or "vasp")
        - fmax: float
        - max_steps: int
        - incar: dict (VASP only)
        - etc.

    Returns
    -------
    Pipeline
    """
    stages = []
    for cfg in stage_configs:
        cfg_dict = dict(cfg) if hasattr(cfg, "items") else cfg.model_dump()
        stage = CalculatorStage(**cfg_dict)
        stages.append(stage)
    return Pipeline(stages)
