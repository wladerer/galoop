"""
galoop/engine/calculator.py

Multi-stage calculator pipeline. Backends are pluggable via
:mod:`galoop.engine.backends`: any ASE-compatible calculator can be used
by naming a built-in (``mace``, ``vasp``) or an import path
(``pkg.module:factory``) in the stage config.

Each stage relaxes the geometry and writes a CONTCAR for the next stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
from ase import Atoms
from ase.io import write
from ase.optimize import BFGS

from galoop.engine import backends
from galoop.science.surface import read_atoms

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
    """One stage in the multi-stage relaxation pipeline.

    Dispatch is handled by :mod:`galoop.engine.backends`: ``type`` is
    resolved to a ``(factory, drives_own_relaxation)`` pair at stage-build
    time. Factories whose calculator drives its own relaxation (e.g. VASP
    via ``ibrion``/``nsw``) are called once per relax; the others use a
    Python-side BFGS loop.
    """

    def __init__(
        self,
        name: str,
        type: str,
        fmax: float = 0.05,
        max_steps: int = 300,
        energy_per_atom_tol: float = 10.0,
        max_force_tol: float = 50.0,
        params: dict | None = None,
        fix_slab_first: bool = False,
        prescan_fmax: float | None = None,
        **extra,                         # absorb unknown keys from config
    ):
        self.name = name
        self.type = type
        self.fmax = fmax
        self.max_steps = max_steps
        self.energy_per_atom_tol = energy_per_atom_tol
        self.max_force_tol = max_force_tol
        self.params = dict(params or {})
        self.fix_slab_first = fix_slab_first
        self.prescan_fmax = prescan_fmax if prescan_fmax is not None else fmax

        # Resolve the backend now so misconfigured yamls fail early, not on
        # the first worker that tries to run this stage.
        self._factory, self._drives_own_relaxation = backends.resolve(type)

    def make_calculator(self, stage_dir: Path | None = None):
        """Build an ASE calculator for this stage.

        ``stage_dir`` is injected into ``params["directory"]`` for backends
        that need a writable work dir (VASP). MACE and other pure-Python
        calculators ignore it.
        """
        params = dict(self.params)
        if stage_dir is not None:
            params.setdefault("directory", str(stage_dir))
        return self._factory(params)

    def run(
        self,
        atoms: Atoms,
        struct_dir: Path,
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

            log.debug("Prescan (adsorbates-only) for %s in %s (fmax=%.4f)",
                      self.name, prescan_dir, self.prescan_fmax)
            try:
                self._relax(prescan_atoms, prescan_dir, fmax=self.prescan_fmax)
            except Exception as exc:
                log.warning("%s prescan failed (%s) — proceeding with original geometry",
                            self.name, exc)
            else:
                prescan_contcar = prescan_dir / "CONTCAR"
                if prescan_contcar.exists():
                    try:
                        loaded = read_atoms(prescan_contcar, format="vasp")
                        loaded.set_constraint(atoms.constraints)
                        atoms = loaded
                    except Exception as exc:
                        log.warning("Failed to load prescan CONTCAR: %s", exc)

        try:
            return self._relax(atoms, stage_dir)
        except Exception as exc:
            log.error("%s failed: %s", self.name, exc)
            return StageResult(False, float("nan"), 0, None)

    # -- backend-agnostic relax driver -------------------------------------

    def _relax(
        self,
        atoms: Atoms,
        stage_dir: Path,
        fmax: float | None = None,
    ) -> StageResult:
        """One unified relax path.

        Branches on ``self._drives_own_relaxation``:

        - **False (MLIP-style)**: attach the calculator and run a Python BFGS
          loop, writing a trajectory file. Works for MACE, fairchem, Orb,
          SevenNet, anything with an ASE Calculator interface.
        - **True (VASP-style)**: the calculator relaxes internally; we just
          call ``get_potential_energy()`` and read the post-run CONTCAR.
        """
        try:
            calc = self.make_calculator(stage_dir=stage_dir)
        except ImportError as exc:
            raise ImportError(
                f"Backend {self.type!r} could not be loaded: {exc}"
            ) from exc

        if self._drives_own_relaxation:
            return self._run_with_internal_relax(atoms, stage_dir, calc)
        return self._run_with_bfgs(atoms, stage_dir, calc, fmax=fmax)

    def _run_with_bfgs(
        self,
        atoms: Atoms,
        stage_dir: Path,
        calc,
        fmax: float | None,
    ) -> StageResult:
        """BFGS loop for calculators that only provide energy+forces."""
        atoms = atoms.copy()
        # Cached calculators retain stale results from prior atoms of a
        # different size — clear before reuse to avoid shape-mismatch errors.
        import contextlib
        with contextlib.suppress(AttributeError):
            calc.results.clear()
        atoms.calc = calc

        traj_path = stage_dir / "trajectory.traj"
        log_path = stage_dir / "relax.log"
        dyn = BFGS(atoms, trajectory=str(traj_path), logfile=str(log_path))

        try:
            converged = dyn.run(
                fmax=fmax if fmax is not None else self.fmax,
                steps=self.max_steps,
            )
        except Exception as exc:
            log.warning("%s BFGS relax failed: %s", self.name, exc)
            converged = False

        n_steps = getattr(dyn, "nsteps", 0)
        try:
            energy = float(atoms.get_potential_energy())
        except Exception:
            energy = float("nan")
            converged = False

        if not self._energy_sane(energy, atoms):
            converged = False

        self._write_contcar(atoms, stage_dir)
        return StageResult(
            converged=converged,
            energy=energy,
            n_steps=n_steps,
            trajectory_file=str(traj_path),
        )

    def _run_with_internal_relax(
        self,
        atoms: Atoms,
        stage_dir: Path,
        calc,
    ) -> StageResult:
        """For calculators that relax inside get_potential_energy() (VASP)."""
        # CalculatorStage-level fmax/max_steps override whatever the factory
        # baked in, so users get consistent knobs across backends. Only
        # calculators exposing ASE's ``.set()`` API (Vasp does) are updated.
        if hasattr(calc, "set"):
            try:
                calc.set(nsw=self.max_steps, ediffg=-self.fmax)
            except Exception as exc:
                log.debug("Could not override nsw/ediffg on %s: %s",
                          type(calc).__name__, exc)

        atoms = atoms.copy()
        atoms.calc = calc

        try:
            energy = float(atoms.get_potential_energy())
            converged = True
        except Exception as exc:
            log.warning("%s internal relax failed: %s", self.name, exc)
            energy = float("nan")
            converged = False

        # Prefer the backend's own output geometry if present.
        contcar = stage_dir / "CONTCAR"
        if contcar.exists():
            try:
                atoms = read_atoms(contcar, format="vasp")
            except Exception as exc:
                log.warning("Failed to read stage CONTCAR: %s", exc)

        if converged and not self._energy_sane(energy, atoms):
            converged = False

        self._write_contcar(atoms, stage_dir)
        return StageResult(
            converged=converged,
            energy=energy,
            n_steps=0,
            trajectory_file=None,
        )

    # -- helpers -----------------------------------------------------------

    def _energy_sane(self, energy: float, atoms: Atoms) -> bool:
        """Sanity check: unreasonable energy per atom → treat as failed."""
        epa = abs(energy) / max(len(atoms), 1)
        if epa > self.energy_per_atom_tol:
            log.warning(
                "%s energy/atom = %.2f eV — exceeds tolerance %.1f",
                self.name, epa, self.energy_per_atom_tol,
            )
            return False
        return True

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
            result = stage.run(current_atoms, struct_dir, n_slab_atoms=n_slab_atoms)
            stage_results[stage.name] = result

            if not result.converged:
                all_converged = False
                log.warning("%s did not converge", stage.name)

            # Load output geometry for next stage
            contcar = struct_dir / f"stage_{stage.name}" / "CONTCAR"
            if contcar.exists():
                try:
                    current_atoms = read_atoms(contcar, format="vasp")
                except Exception as exc:
                    log.warning("Failed to load %s output: %s", stage.name, exc)
                    all_converged = False
                    break
            else:
                log.warning("No CONTCAR from %s", stage.name)
                all_converged = False
                break

            # Inter-stage connectivity check: verify adsorbates are surface-bound.
            # Use generous cutoff (1.5×) since geometry may not be fully relaxed.
            # The stricter post-relax check in the harvest loop catches the rest.
            if (
                n_slab_atoms > 0
                and n_slab_atoms < len(current_atoms)
                and len(self.stages) > 1
                and not _check_surface_binding(current_atoms, n_slab_atoms, bond_mult=1.5)
            ):
                log.warning(
                    "%s produced unbound adsorbate molecule(s) — aborting pipeline",
                    stage.name,
                )
                all_converged = False
                break

        # Write final outputs
        final_contcar = struct_dir / "CONTCAR"
        try:
            write(str(final_contcar), current_atoms, format="vasp")
        except Exception as exc:
            log.warning("Failed to write final CONTCAR: %s", exc)

        if not stage_results:
            return {
                "converged": False,
                "final_energy": None,
                "stage_results": stage_results,
                "final_atoms": current_atoms,
            }
        last_stage_name = list(stage_results.keys())[-1]
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

def _check_surface_binding(
    atoms: Atoms,
    n_slab: int,
    bond_mult: float = 1.3,
) -> bool:
    """True if every adsorbate molecule has at least one atom bonded to the slab.

    Uses only ASE primitives to keep engine/ isolated from galoop package.
    """
    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList, natural_cutoffs

    ads_indices = list(range(n_slab, len(atoms)))
    if not ads_indices:
        return True

    # Group adsorbate atoms into molecules by covalent bonding
    pos = atoms.get_positions()
    nums = atoms.get_atomic_numbers()
    n = len(ads_indices)
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cutoff = 1.25 * (covalent_radii[nums[ads_indices[i]]]
                             + covalent_radii[nums[ads_indices[j]]])
            d = np.linalg.norm(pos[ads_indices[i]] - pos[ads_indices[j]])
            if d < cutoff:
                adj[i].append(j)
                adj[j].append(i)

    visited = [False] * n
    molecules: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(ads_indices[node])
            stack.extend(adj[node])
        molecules.append(component)

    # Check each molecule has at least one atom bonded to slab
    cutoffs = natural_cutoffs(atoms, mult=bond_mult)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True, skin=0.0)
    nl.update(atoms)

    slab_set = set(range(n_slab))
    for mol in molecules:
        has_contact = False
        for atom_idx in mol:
            neighbors, _ = nl.get_neighbors(atom_idx)
            if slab_set & set(int(n) for n in neighbors):
                has_contact = True
                break
        if not has_contact:
            return False

    return True


def build_pipeline(stage_configs: list) -> Pipeline:
    """
    Build a :class:`Pipeline` from a list of stage config dicts or Pydantic models.
    """
    stages: list[CalculatorStage] = []
    for cfg in stage_configs:
        d = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
        stages.append(CalculatorStage(**d))
    return Pipeline(stages)
