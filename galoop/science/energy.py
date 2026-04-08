"""
galoop/science/energy.py

Electrochemical energy: computational hydrogen electrode (CHE) framework.

Full CHE (Nørskov et al., 2004/2005):
  Ω = E_DFT + Σ_i n_i μ'_i

where μ'_i is the effective chemical potential of adsorbate species i,
incorporating reaction conditions:

  For species formed from water (O, OH, OOH, H2O):
    μ'_i = μ°_i + n_e(eU) + n_H(k_BT ln10 pH)

  For gas-phase species (CO, N2, etc.):
    μ'_i = μ°_i + 0.5 k_BT ln(P/P°)

  For H adatoms (from H+ + e-):
    μ'_H = μ°_H - eU - k_BT ln10 pH
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

log = logging.getLogger(__name__)

KB_EV = 8.617333e-5   # eV / K


# ---------------------------------------------------------------------------
# Per-species electrochemical metadata
# ---------------------------------------------------------------------------
# For each adsorbate species, we need:
#   n_electron : number of electrons transferred in the adsorption reaction
#                (positive = electrons released, e.g. H2O → O* + 2H+ + 2e-)
#   n_proton   : number of protons released (usually = n_electron for CHE)
#   phase      : "aqueous" (from solution, pH/U dependent) or
#                "gas" (from gas phase, pressure dependent)
#
# These are the standard CHE half-reactions:
#   H*   ← H+(aq) + e-                    n_e = -1, n_H+ = -1  (consuming)
#   O*   ← H2O(l) - 2H+(aq) - 2e-        n_e = +2, n_H+ = +2
#   OH*  ← H2O(l) - H+(aq) - e-          n_e = +1, n_H+ = +1
#   OOH* ← 2H2O(l) - 3H+(aq) - 3e-      n_e = +3, n_H+ = +3
#   H2O* ← H2O(l)                         n_e = 0,  n_H+ = 0
#   CO*  ← CO(g)                           n_e = 0,  phase = gas
#   N*   ← ½N2(g)                          n_e = 0,  phase = gas
#   NH3* ← NH3(g)                          n_e = 0,  phase = gas

_CHE_METADATA: dict[str, dict] = {
    "H":   {"n_electron": -1, "n_proton": -1, "phase": "aqueous"},
    "O":   {"n_electron":  2, "n_proton":  2, "phase": "aqueous"},
    "OH":  {"n_electron":  1, "n_proton":  1, "phase": "aqueous"},
    "OOH": {"n_electron":  3, "n_proton":  3, "phase": "aqueous"},
    "H2O": {"n_electron":  0, "n_proton":  0, "phase": "aqueous"},
    "CO":  {"n_electron":  0, "n_proton":  0, "phase": "gas"},
    "N":   {"n_electron":  0, "n_proton":  0, "phase": "gas"},
    "N2":  {"n_electron":  0, "n_proton":  0, "phase": "gas"},
    "NH3": {"n_electron":  0, "n_proton":  0, "phase": "gas"},
    "CH4": {"n_electron":  0, "n_proton":  0, "phase": "gas"},
}


def _get_species_metadata(symbol: str) -> dict:
    """Look up or infer CHE metadata for an adsorbate species.

    If the species isn't in the table, infer from element composition:
    - Species containing only O and H: assume aqueous (water-derived)
    - Everything else: assume gas phase, no electron transfer
    """
    if symbol in _CHE_METADATA:
        return _CHE_METADATA[symbol]

    # Infer: count O and H
    n_O = symbol.count("O")
    n_H = symbol.count("H")
    has_other = any(c.isalpha() and c not in "OH" for c in symbol)

    if has_other or (n_O == 0 and n_H == 0):
        # Contains C, N, etc. — treat as gas phase
        return {"n_electron": 0, "n_proton": 0, "phase": "gas"}
    else:
        # Pure O/H species — aqueous, CHE convention
        # n_e = 2*n_O - n_H (from water decomposition stoichiometry)
        n_e = 2 * n_O - n_H
        return {"n_electron": n_e, "n_proton": n_e, "phase": "aqueous"}


# ---------------------------------------------------------------------------
# Grand canonical energy
# ---------------------------------------------------------------------------

def grand_canonical_energy(
    raw_energy: float,
    adsorbate_counts: Mapping[str, int],
    chemical_potentials: Mapping[str, float],
    potential: float = 0.0,
    pH: float = 0.0,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float:
    """
    Full CHE grand canonical energy.

    .. math::

        \\Omega = E_{DFT} + \\sum_i n_i \\mu'_i

    where the effective chemical potential μ'_i depends on reaction conditions:

    For aqueous species (O, OH, OOH — derived from H₂O):
        μ'_i = μ°_i + n_e × eU + n_{H+} × k_BT ln(10) × pH

    For gas-phase species (CO, N₂, NH₃):
        μ'_i = μ°_i + ½ k_BT ln(P/P°)   [per molecule]

    For H adatoms (from H⁺ + e⁻ → H*):
        μ'_H = μ°_H - eU - k_BT ln(10) × pH

    Parameters
    ----------
    raw_energy : DFT total energy (eV)
    adsorbate_counts : ``{symbol: count}``
    chemical_potentials : ``{symbol: μ°}`` in eV (at U=0, pH=0, P=1 atm)
    potential : electrode potential (V vs SHE). Use 0.0 for RHE reference.
    pH : solution pH
    temperature : K
    pressure : total gas pressure (atm). Applies to gas-phase adsorbates.

    Returns
    -------
    float — grand canonical energy in eV
    """
    kT = KB_EV * temperature
    kT_ln10 = kT * np.log(10)

    G = raw_energy

    for symbol, count in adsorbate_counts.items():
        if count == 0:
            continue

        mu_0 = chemical_potentials.get(symbol)
        if mu_0 is None:
            log.warning("No chemical potential for '%s' — skipping", symbol)
            continue

        meta = _get_species_metadata(symbol)
        n_e = meta["n_electron"]
        n_Hp = meta["n_proton"]
        phase = meta["phase"]

        # Effective chemical potential at given conditions
        mu_eff = mu_0

        # Electrochemical correction: each electron transfer shifts by eU
        if n_e != 0 and potential != 0.0:
            mu_eff += n_e * potential  # positive n_e = electrons released

        # pH correction: each proton released/consumed shifts by kT ln(10) pH
        if n_Hp != 0 and pH != 0.0:
            mu_eff += n_Hp * kT_ln10 * pH

        # Gas-phase pressure correction
        if phase == "gas" and pressure != 1.0 and pressure > 0:
            mu_eff += 0.5 * kT * np.log(pressure)

        G += count * mu_eff

    return float(G)


# ---------------------------------------------------------------------------
# Convenience: decompose Ω at a grid of conditions
# ---------------------------------------------------------------------------

def grand_canonical_energy_grid(
    raw_energy: float,
    adsorbate_counts: dict[str, int],
    chemical_potentials: dict[str, float],
    pH: float | np.ndarray = 0.0,
    potential: float | np.ndarray = 0.0,
    temperature: float = 298.15,
    pressure: float = 1.0,
) -> float | np.ndarray:
    """Vectorized GCE computation over arrays of pH and/or potential.

    Accepts scalar or array pH/potential. Returns matching shape.
    Useful for Pourbaix diagram construction.
    """
    pH = np.atleast_1d(np.asarray(pH, dtype=float))
    potential = np.atleast_1d(np.asarray(potential, dtype=float))

    # Broadcast to common shape
    pH_grid, U_grid = np.broadcast_arrays(pH, potential)
    result = np.full(pH_grid.shape, raw_energy, dtype=float)

    kT = KB_EV * temperature
    kT_ln10 = kT * np.log(10)

    for symbol, count in adsorbate_counts.items():
        if count == 0:
            continue
        mu_0 = chemical_potentials.get(symbol)
        if mu_0 is None:
            continue

        meta = _get_species_metadata(symbol)
        n_e = meta["n_electron"]
        n_Hp = meta["n_proton"]
        phase = meta["phase"]

        mu_eff = mu_0
        correction = np.zeros_like(pH_grid)

        if n_e != 0:
            correction += n_e * U_grid
        if n_Hp != 0:
            correction += n_Hp * kT_ln10 * pH_grid
        if phase == "gas" and pressure != 1.0 and pressure > 0:
            correction += 0.5 * kT * np.log(pressure)

        result += count * (mu_eff + correction)

    return float(result) if result.ndim == 0 else result


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
