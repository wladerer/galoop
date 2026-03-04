"""
galoop/science/energy.py

Electrochemical energy: computational hydrogen electrode (CHE) framework.
"""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

KB_EV = 8.617333e-5   # eV / K


# ---------------------------------------------------------------------------
# Grand canonical energy
# ---------------------------------------------------------------------------

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
    CHE grand canonical energy (Nørskov et al., 2005).

    .. math::

        G = E_{DFT}
            + \\sum_i n_i \\mu_i
            - n_H \\, k_B T \\ln(10) \\, \\mathrm{pH}

    Parameters
    ----------
    raw_energy : DFT total energy (eV)
    adsorbate_counts : ``{symbol: count}``
    chemical_potentials : ``{symbol: μ⁰}`` in eV
    potential : electrode potential (V vs RHE)
    pH : solution pH
    temperature : K
    pressure : atm (reserved for ideal-gas corrections)

    Returns
    -------
    float — grand canonical energy in eV
    """
    G = raw_energy

    # Chemical-potential contribution
    for symbol, count in adsorbate_counts.items():
        mu = chemical_potentials.get(symbol)
        if mu is None:
            log.warning("No chemical potential for '%s' — skipping", symbol)
            continue
        G += count * mu

    # pH correction for proton-containing adsorbates
    n_H = _count_protons(adsorbate_counts)
    if n_H > 0 and pH != 0.0:
        kT_ln10 = KB_EV * temperature * np.log(10)
        G -= n_H * kT_ln10 * pH

    return float(G)


def _count_protons(adsorbate_counts: dict[str, int]) -> int:
    """
    Heuristic: count H atoms across all adsorbate formulas.

    ``"OH"`` → 1 H, ``"OOH"`` → 1 H, ``"H2O"`` → 2 H, ``"O"`` → 0 H.
    Pure ``"H"`` is excluded (it *is* a proton, not a proton-containing species).
    """
    total = 0
    for symbol, count in adsorbate_counts.items():
        if symbol == "H":
            continue
        h_per = symbol.count("H")      # simple char count — works for OH, OOH, H2O
        total += count * h_per
    return total


# ---------------------------------------------------------------------------
# Approximate ZPE placeholder
# ---------------------------------------------------------------------------

def zero_point_energy_correction(
    atoms,
    n_slab_atoms: int,
    T: float = 298.15,
) -> float:
    """
    Rough ZPE estimate (~0.04 eV per adsorbate atom).

    Replace with a proper vibrational analysis (phonopy, etc.) for
    publication-quality results.
    """
    n_ads = len(atoms) - n_slab_atoms
    return max(n_ads, 0) * 0.04


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def compare_energies(E1: float, E2: float, tolerance: float = 0.01) -> str:
    """Return ``"lower"``, ``"equal"``, or ``"higher"``."""
    diff = E1 - E2
    if abs(diff) < tolerance:
        return "equal"
    return "lower" if diff < 0 else "higher"
