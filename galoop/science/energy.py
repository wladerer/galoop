"""
gocia/science/energy.py

Electrochemical energy calculations: CHE grand canonical energy.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Physical constants
FARADAY = 96485.33  # C/mol
KB = 8.314e-3  # kJ/(mol·K) or eV/(K)
KB_EV = 8.617333e-5  # eV/K


def grand_canonical_energy(
    raw_energy: float,
    adsorbate_counts: dict[str, int],
    chemical_potentials: dict[str, float],
    potential: float = 0.0,
    pH: float = 0.0,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float:
    """
    Calculate grand canonical (CHE) energy.

    Parameters
    ----------
    raw_energy : DFT total energy (eV)
    adsorbate_counts : {symbol: count} of adsorbates
    chemical_potentials : {symbol: mu_0} standard chemical potential (eV)
    potential : Electrode potential (V vs RHE)
    pH : pH of solution
    temperature : Temperature (K)
    pressure : Pressure (atm, for ideal gas correction if needed)

    Returns
    -------
    float : Grand canonical energy (eV)

    Notes
    -----
    CHE framework (Nørskov et al., 2005):
    G = E_DFT + sum_i (n_i * mu_i) - n_e * U - n_H * kT * ln(10) * pH

    where:
    - E_DFT is the DFT energy
    - mu_i is the electrochemical potential of species i
    - n_i is the count of species i
    - n_e is the number of electrons transferred
    - U is the applied potential (V vs RHE)
    - pH is the solution pH
    - kT * ln(10) ≈ 0.0592 V at 298 K
    """
    G = raw_energy

    # Add adsorbate contributions
    for symbol, count in adsorbate_counts.items():
        if symbol not in chemical_potentials:
            log.warning(f"  No chemical potential for {symbol}; skipping")
            continue
        mu = chemical_potentials[symbol]
        G += count * mu

    # Electrochemical correction
    # In the CHE model, the electrode potential shifts the energy
    # For simplicity, we assume the adsorbate does not accept/donate electrons
    # (or that electron transfer is implicit in chemical potentials)

    # pH correction (for aqueous species)
    # Only applies if there are proton-containing adsorbates (e.g., OH, OOH)
    n_H = 0
    for symbol, count in adsorbate_counts.items():
        # Count protons in the formula (heuristic)
        if "H" in symbol and symbol != "H":
            # Assume OH has 1 H, OOH has 1 H, H2O has 2 H, etc.
            h_per_species = symbol.count("H")
            n_H += count * h_per_species

    if n_H > 0 and pH != 0.0:
        # pH correction: proton energy shifts with pH
        # ΔG_pH = -n_H * kT * ln(10) * pH ≈ -0.0592 eV * n_H * pH at 298 K
        kT_ln10_298K = 0.05916  # eV (standard at 298.15 K)
        # Temperature scaling
        kT_ln10 = KB_EV * temperature * np.log(10)
        pH_correction = -n_H * kT_ln10 * pH
        G += pH_correction

    return float(G)


def is_desorbed(
    atoms,
    slab_info,
    z_threshold: float = 15.0,
) -> bool:
    """
    Check if adsorbates have desorbed.

    Parameters
    ----------
    atoms : Final relaxed structure
    slab_info : SlabInfo with slab metadata
    z_threshold : Z-coordinate above which is desorbed (Å)

    Returns
    -------
    bool : True if any adsorbate atom is above z_threshold
    """
    if len(atoms) <= slab_info.n_slab_atoms:
        return False

    adsorbate_positions = atoms.get_positions()[slab_info.n_slab_atoms :]
    z_coords = adsorbate_positions[:, 2]

    return np.any(z_coords > z_threshold)


def zero_point_energy_correction(
    atoms,
    n_slab_atoms: int,
    T: float = 298.15,
) -> float:
    """
    Approximate zero-point energy (ZPE) correction.

    Parameters
    ----------
    atoms : Structure
    n_slab_atoms : Number of slab atoms (frozen)
    T : Temperature (K)

    Returns
    -------
    float : Approximate ZPE or entropic correction (eV)

    Notes
    -----
    This is a placeholder. Real ZPE requires vibrational analysis (phononopy, etc.).
    For now, returns a rough estimate based on the number of adsorbate atoms.
    """
    n_adsorbate = len(atoms) - n_slab_atoms
    if n_adsorbate <= 0:
        return 0.0

    # Rough estimate: ~0.04 eV per adsorbate atom at 298 K
    zpe = n_adsorbate * 0.04

    # Temperature entropy contribution (rough)
    # TS ≈ n * kT * (some constant factor)
    # For now, skip temperature dependence
    entropy_contrib = 0.0

    return zpe + entropy_contrib


def compare_energies(
    E1: float,
    E2: float,
    tolerance: float = 0.01,
) -> str:
    """
    Compare two energies with tolerance.

    Returns
    -------
    str : "lower", "equal", "higher"
    """
    diff = E1 - E2
    if abs(diff) < tolerance:
        return "equal"
    return "lower" if diff < 0 else "higher"
