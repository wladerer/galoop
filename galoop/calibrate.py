"""
galoop/calibrate.py

Auto-compute slab energy and CHE chemical potentials using the same
calculator pipeline configured for the GA run.

Called before the GA loop when slab.energy or any adsorbate
chemical_potential is None.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from ase import Atoms
from ase.build import molecule
from ase.io import read, write

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Element reference molecules
# ---------------------------------------------------------------------------
# Each element needs a reference molecule and stoichiometry.
# mu_element = E(molecule) / n_atoms_of_element_in_molecule
#
# Example: H2 has 2 H atoms → mu_H = E(H2) / 2
#          N2 has 2 N atoms → mu_N = E(N2) / 2
#          CH4 has 1 C and 4 H → mu_C = E(CH4) - 4 * mu_H

# (molecule_formula, {element: count_in_molecule})
# Order matters: molecules with fewer element types should come first
# so their elements are resolved before compounds that depend on them.
_DEFAULT_REFERENCES: list[tuple[str, dict[str, int]]] = [
    ("H2",  {"H": 2}),
    ("N2",  {"N": 2}),
    ("H2O", {"H": 2, "O": 1}),
    ("CH4", {"H": 4, "C": 1}),
]


def _decompose_formula(symbol: str) -> dict[str, int]:
    """Parse a simple chemical formula into element counts.

    Handles formulas like O, OH, H2O, OOH, HCOO, NO, NH3, CH3OH, etc.
    """
    from galoop.science.surface import parse_formula
    from collections import Counter
    return dict(Counter(parse_formula(symbol)))


# ---------------------------------------------------------------------------
# Pipeline runner for calibration
# ---------------------------------------------------------------------------

def _run_pipeline(atoms: Atoms, struct_dir: Path, config,
                  n_slab_atoms: int = 0) -> float:
    """Run the full calculator pipeline on *atoms* and return the final energy.

    Uses the exact same pipeline stages as the GA run for consistency.
    """
    from galoop.engine.calculator import build_pipeline

    struct_dir = Path(struct_dir)
    struct_dir.mkdir(parents=True, exist_ok=True)
    write(str(struct_dir / "POSCAR"), atoms, format="vasp")

    pipeline = build_pipeline(config.calculator_stages)
    result = pipeline.run(
        atoms,
        struct_dir,
        mace_model=config.mace_model,
        mace_device=config.mace_device,
        mace_dtype=config.mace_dtype,
        n_slab_atoms=n_slab_atoms,
    )

    if not result["converged"] or math.isnan(float(result["final_energy"])):
        raise RuntimeError(
            f"Calibration pipeline did not converge in {struct_dir}"
        )

    return float(result["final_energy"])


def _prepare_molecule(formula: str) -> Atoms:
    """Build a molecule in a periodic box for pipeline relaxation."""
    mol = molecule(formula)
    mol.cell = [10.0, 10.0, 10.0]
    mol.pbc = True
    mol.center()
    return mol


# ---------------------------------------------------------------------------
# Resolve elemental chemical potentials from reference molecules
# ---------------------------------------------------------------------------

def _resolve_elemental_potentials(
    needed_elements: set[str],
    config,
    cal_dir: Path,
) -> dict[str, float]:
    """Compute per-element chemical potentials from reference molecules.

    For each reference molecule, relax it through the pipeline and solve
    for the unknown element's potential.  For example:

        E(H2)  → mu_H = E(H2) / 2
        E(H2O) → mu_O = E(H2O) - 2 * mu_H
        E(CH4) → mu_C = E(CH4) - 4 * mu_H
        E(N2)  → mu_N = E(N2) / 2
    """
    mu: dict[str, float] = {}
    mol_energies: dict[str, float] = {}

    for mol_formula, composition in _DEFAULT_REFERENCES:
        # Check if this molecule provides any element we still need
        elements_in_mol = set(composition.keys())
        unsolved = elements_in_mol - set(mu.keys())
        if not (unsolved & needed_elements):
            continue
        # All elements except one must already be solved
        if len(unsolved) > 1:
            log.warning(
                "Cannot resolve %s from %s — need %s solved first",
                unsolved, mol_formula, unsolved - {next(iter(unsolved))},
            )
            continue

        log.info("  Relaxing %s …", mol_formula)
        e_mol = _run_pipeline(
            _prepare_molecule(mol_formula),
            cal_dir / mol_formula,
            config,
        )
        mol_energies[mol_formula] = e_mol
        log.info("  E(%s) = %.6f eV", mol_formula, e_mol)

        # Solve for the unknown element
        target_elem = next(iter(unsolved))
        known_contribution = sum(
            count * mu[elem]
            for elem, count in composition.items()
            if elem in mu
        )
        target_count = composition[target_elem]
        mu[target_elem] = (e_mol - known_contribution) / target_count
        log.info("  mu(%s) = %.6f eV", target_elem, mu[target_elem])

    # Check for unresolved elements
    missing = needed_elements - set(mu.keys())
    if missing:
        raise ValueError(
            f"Cannot auto-derive chemical potentials for elements {missing}. "
            f"Add reference molecules or set chemical_potential explicitly. "
            f"Resolved so far: {mu}"
        )

    return mu, mol_energies


# ---------------------------------------------------------------------------
# Main calibration entry point
# ---------------------------------------------------------------------------

def calibrate(config, run_dir: Path | None = None) -> dict:
    """
    Auto-compute missing slab energy and chemical potentials.

    Uses the same calculator pipeline (MACE, VASP, or multi-stage) that
    the GA run will use, so reference energies are consistent with
    production relaxations.

    Modifies *config* in place and returns a summary dict of computed values.

    Parameters
    ----------
    config : GaloopConfig (mutable)
    run_dir : optional directory to cache calibration results

    Returns
    -------
    dict with keys like 'slab_energy', 'E_H2', 'E_H2O', 'mu_O', etc.
    """
    results = {}
    needs_molecules = any(
        a.chemical_potential is None for a in config.adsorbates
    )
    needs_slab = config.slab.energy is None

    if not needs_slab and not needs_molecules:
        return results

    cal_dir = (run_dir / "calibration") if run_dir else Path("calibration")
    cal_dir.mkdir(parents=True, exist_ok=True)

    log.info("Auto-calibration: computing reference energies via pipeline …")

    # --- Slab energy ---
    if needs_slab:
        log.info("  Relaxing bare slab …")
        slab = read(str(config.slab.geometry), format="vasp")
        e_slab = _run_pipeline(
            slab, cal_dir / "slab",
            config, n_slab_atoms=len(slab),
        )
        config.slab.energy = e_slab
        results["slab_energy"] = e_slab
        log.info("  Slab energy: %.4f eV", e_slab)

    # --- Reference molecules → elemental potentials → adsorbate potentials ---
    if needs_molecules:
        # Figure out which elements we need
        needed_elements: set[str] = set()
        for ads in config.adsorbates:
            if ads.chemical_potential is None:
                needed_elements.update(_decompose_formula(ads.symbol).keys())

        mu_elem, mol_energies = _resolve_elemental_potentials(
            needed_elements, config, cal_dir,
        )

        for formula, energy in mol_energies.items():
            results[f"E_{formula}"] = energy
        for elem, mu in mu_elem.items():
            results[f"mu_{elem}"] = mu

        # Derive adsorbate chemical potentials from elemental potentials
        for ads in config.adsorbates:
            if ads.chemical_potential is not None:
                continue
            composition = _decompose_formula(ads.symbol)
            mu_ads = sum(count * mu_elem[elem] for elem, count in composition.items())
            ads.chemical_potential = mu_ads
            results[f"mu_{ads.symbol}"] = mu_ads
            decomp = " + ".join(f"{c}{e}" for e, c in composition.items())
            log.info("  mu(%s) = %.6f eV  (%s)", ads.symbol, mu_ads, decomp)

    # Write calibration results for reference
    with open(cal_dir / "reference_energies.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    log.info("Auto-calibration complete.")
    return results
