"""Generate the 7 follow-up galoop.yaml configs with precomputed refs.

Uses CHE-style references:
  mu(H)   = 0.5 * E(H2)
  mu(O)   = E(H2O) - E(H2)        (H2O + H2 balance → O + 2H2 → H2O, rearranged)
  mu(N)   = 0.5 * E(N2)
  mu(NH)  = 0.5 * E(N2) + 0.5 * E(H2)
  mu(NH2) = 0.5 * E(N2) + E(H2)
  mu(NH3) = E(NH3)
  mu(OH)  = E(H2O) - 0.5 * E(H2)
  mu(OOH) = 2*E(H2O) - 1.5*E(H2)   (2 H2O - 1.5 H2 = OOH + electrons; standard CHE form)
  mu(CO)  = E(CO)
  mu(H2O) = E(H2O)
  mu(C)   — tricky; we use E(CO) - (E(H2O) - E(H2)) = E(CO) - mu(O)
  mu(CH)  = mu(C) + 0.5*E(H2)
  mu(CH2) = mu(C) + E(H2)
  mu(CH3) = mu(C) + 1.5*E(H2)
"""
from __future__ import annotations

import textwrap
from pathlib import Path

MOL_E = {
    'CO': -14.417733837642619,
    'H2': -6.967667595804677,
    'H2O': -14.154034733892173,
    'N2': -16.18002508678477,
    'NH3': -19.380593626315676,
    'O2': -8.469472117744427,
    'OH': -7.4362105155008305,
    'OOH': -12.329537389950417,
}
SLAB_E = {
    'pt111_orr': -244.64749406138407,
    'pt211_nrr': -181.57526243336218,
    'ag111_co': -96.63050391962514,
    'pd111_hsat': -206.12734269899704,
    'ni111_chx': -215.83818177800833,
    'au111_co': -118.40990457747725,
    'ru0001_oh': -382.7588259349825,
}

mu_H   = 0.5 * MOL_E['H2']
mu_O   = MOL_E['H2O'] - MOL_E['H2']
mu_N   = 0.5 * MOL_E['N2']
mu_NH  = mu_N + mu_H
mu_NH2 = mu_N + 2*mu_H
mu_NH3 = MOL_E['NH3']
mu_OH  = MOL_E['H2O'] - 0.5*MOL_E['H2']
mu_OOH = 2*MOL_E['H2O'] - 1.5*MOL_E['H2']
mu_CO  = MOL_E['CO']
mu_H2O = MOL_E['H2O']
mu_C   = MOL_E['CO'] - mu_O
mu_CH  = mu_C + mu_H
mu_CH2 = mu_C + 2*mu_H
mu_CH3 = mu_C + 3*mu_H

MU = dict(H=mu_H, O=mu_O, N=mu_N, NH=mu_NH, NH2=mu_NH2, NH3=mu_NH3,
          OH=mu_OH, OOH=mu_OOH, CO=mu_CO, H2O=mu_H2O,
          C=mu_C, CH=mu_CH, CH2=mu_CH2, CH3=mu_CH3)

print("Chemical potentials (eV):")
for k, v in MU.items():
    print(f"  mu({k:3s}) = {v:.6f}")

# Determine slab top z by reading slab.vasp
from ase.io import read
def ztop(tag):
    a = read(f"runs/{tag}_camp/slab.vasp", format="vasp")
    return float(a.get_positions()[:,2].max())


def yaml_for(tag, slab_e_key, adsorbate_block, conditions, ga_block=None,
             max_adsorbates=10, min_adsorbates=1):
    zt = ztop(tag)
    zmin = zt + 1.5
    zmax = zt + 4.0
    ga_block = ga_block or textwrap.dedent("""\
      population_size: 20
      min_structures: 30
      max_structures: 300
      max_stall: 15
      min_adsorbates: {min_ads}
      max_adsorbates: {max_ads}
      boltzmann_temperature: 0.1
      rattle_amplitude: 0.15
      displace_amplitude: 1.0
      translate_amplitude: 1.5
      gpr_guided: true
      gpr_fraction: 0.4
      gpr_min_samples: 20
      gpr_kappa: 1.5
      operator_weights:
        splice: 0.22
        merge: 0.13
        mutate_add: 0.20
        mutate_remove: 0.12
        mutate_displace: 0.10
        mutate_translate: 0.15
        mutate_rattle_slab: 0.08""").format(min_ads=min_adsorbates, max_ads=max_adsorbates)

    yaml = f"""slab:
  geometry: /home/wladerer/github/galoop/runs/{tag}_camp/slab.vasp
  energy: {SLAB_E[slab_e_key]:.6f}
  sampling_zmin: {zmin:.2f}
  sampling_zmax: {zmax:.2f}

adsorbates:
{adsorbate_block}

calculator_stages:
  - name: uma_oc20
    type: calc:make_calculator
    fmax: 0.05
    max_steps: 200
    fix_slab_first: true
    prescan_fmax: 0.1
    params:
      model: uma-s-1p1
      task: oc20
      device: cuda

scheduler:
  type: local
  nworkers: 1

ga:
{textwrap.indent(ga_block, '  ')}

conditions:
{textwrap.indent(conditions, '  ')}

fingerprint:
  r_cut: 5.0
  n_max: 6
  l_max: 4
  duplicate_threshold: 0.88
  energy_tol_pct: 3.0
"""
    Path(f"runs/{tag}_camp/galoop.yaml").write_text(yaml)
    print(f"wrote runs/{tag}_camp/galoop.yaml  zmin={zmin:.2f} zmax={zmax:.2f}")


# ---- Pt(111) ORR: O, OH, OOH, H at U=0.8 V ----
yaml_for(
    "pt111_orr", "pt111_orr",
    adsorbate_block=f"""  - symbol: O
    chemical_potential: {MU['O']:.6f}
    min_count: 0
    max_count: 6
  - symbol: OH
    chemical_potential: {MU['OH']:.6f}
    min_count: 0
    max_count: 6
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 0.97]
  - symbol: OOH
    chemical_potential: {MU['OOH']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - O: [1.45, 0.0, 0.0]
      - H: [1.70, 0.95, 0.0]
  - symbol: H
    chemical_potential: {MU['H']:.6f}
    min_count: 0
    max_count: 4""",
    conditions="""potential: 0.8
pH: 0.0
temperature: 298.15""",
    max_adsorbates=10, min_adsorbates=1,
)

# ---- Pt(211) NRR: N, NH, NH2, NH3, H at U=-0.5 V ----
yaml_for(
    "pt211_nrr", "pt211_nrr",
    adsorbate_block=f"""  - symbol: N
    chemical_potential: {MU['N']:.6f}
    min_count: 0
    max_count: 4
  - symbol: NH
    chemical_potential: {MU['NH']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - N: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 1.03]
  - symbol: NH2
    chemical_potential: {MU['NH2']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - N: [0.0, 0.0, 0.0]
      - H: [0.0,  0.82, 0.55]
      - H: [0.0, -0.82, 0.55]
  - symbol: NH3
    chemical_potential: {MU['NH3']:.6f}
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - N: [0.0, 0.0, 0.0]
      - H: [0.0,   0.94, 0.39]
      - H: [0.81, -0.47, 0.39]
      - H: [-0.81, -0.47, 0.39]
  - symbol: H
    chemical_potential: {MU['H']:.6f}
    min_count: 0
    max_count: 4""",
    conditions="""potential: -0.5
pH: 0.0
temperature: 298.15""",
    max_adsorbates=10, min_adsorbates=1,
)

# ---- Ag(111) CO/H/H2O at U=0 ----
yaml_for(
    "ag111_co", "ag111_co",
    adsorbate_block=f"""  - symbol: CO
    chemical_potential: {MU['CO']:.6f}
    min_count: 0
    max_count: 8
    binding_index: 0
    coordinates:
      - C: [0.0, 0.0, 0.0]
      - O: [0.0, 0.0, 1.1503]
  - symbol: H
    chemical_potential: {MU['H']:.6f}
    min_count: 0
    max_count: 4
  - symbol: H2O
    chemical_potential: {MU['H2O']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - H: [0.0, 0.7572, 0.5858]
      - H: [0.0, -0.7572, 0.5858]""",
    conditions="""potential: 0.0
pH: 0.0
temperature: 298.15""",
    max_adsorbates=12, min_adsorbates=1,
)

# ---- Pd(111) H saturation: ONLY H, max_count=16 ----
yaml_for(
    "pd111_hsat", "pd111_hsat",
    adsorbate_block=f"""  - symbol: H
    chemical_potential: {MU['H']:.6f}
    min_count: 1
    max_count: 16""",
    conditions="""potential: 0.0
pH: 0.0
temperature: 298.15""",
    max_adsorbates=16, min_adsorbates=1,
)

# ---- Ni(111) methanation CHx: C, CH, CH2, CH3, H ----
yaml_for(
    "ni111_chx", "ni111_chx",
    adsorbate_block=f"""  - symbol: C
    chemical_potential: {MU['C']:.6f}
    min_count: 0
    max_count: 4
  - symbol: CH
    chemical_potential: {MU['CH']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - C: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 1.09]
  - symbol: CH2
    chemical_potential: {MU['CH2']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - C: [0.0, 0.0, 0.0]
      - H: [0.0,  0.88, 0.60]
      - H: [0.0, -0.88, 0.60]
  - symbol: CH3
    chemical_potential: {MU['CH3']:.6f}
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - C: [0.0, 0.0, 0.0]
      - H: [0.0,   0.98, 0.37]
      - H: [0.85, -0.49, 0.37]
      - H: [-0.85, -0.49, 0.37]
  - symbol: H
    chemical_potential: {MU['H']:.6f}
    min_count: 0
    max_count: 4""",
    conditions="""potential: 0.0
pH: 0.0
temperature: 298.15""",
    max_adsorbates=10, min_adsorbates=1,
)

# ---- Au(111) CO/OH: "inert" metal sanity ----
yaml_for(
    "au111_co", "au111_co",
    adsorbate_block=f"""  - symbol: CO
    chemical_potential: {MU['CO']:.6f}
    min_count: 0
    max_count: 6
    binding_index: 0
    coordinates:
      - C: [0.0, 0.0, 0.0]
      - O: [0.0, 0.0, 1.1503]
  - symbol: OH
    chemical_potential: {MU['OH']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 0.97]""",
    conditions="""potential: 0.0
pH: 0.0
temperature: 298.15""",
    max_adsorbates=8, min_adsorbates=1,
)

# ---- Ru(0001) H2O/OH/O ----
yaml_for(
    "ru0001_oh", "ru0001_oh",
    adsorbate_block=f"""  - symbol: O
    chemical_potential: {MU['O']:.6f}
    min_count: 0
    max_count: 6
  - symbol: OH
    chemical_potential: {MU['OH']:.6f}
    min_count: 0
    max_count: 6
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 0.97]
  - symbol: H2O
    chemical_potential: {MU['H2O']:.6f}
    min_count: 0
    max_count: 4
    binding_index: 0
    coordinates:
      - O: [0.0, 0.0, 0.0]
      - H: [0.0, 0.7572, 0.5858]
      - H: [0.0, -0.7572, 0.5858]""",
    conditions="""potential: 0.0
pH: 0.0
temperature: 298.15""",
    max_adsorbates=10, min_adsorbates=1,
)

print("\nAll 7 yamls written.")
