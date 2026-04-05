# galoop

Grand canonical genetic algorithm for electrochemical surface adsorbate structure search.

## Literature notes

### GOCIA — Grand canonical global optimizer (Zhang et al., PCCP 2025, 27, 696-706)
- Grand canonical GA (GCGA) that evolves composition and geometry simultaneously
- Three placement methods: growth sampling (bond-length from covalent radii), box sampling (random), graph sampling (identify atop/bridge/hollow sites via NetworkX)
- Iterative local optimization with connectivity checks between stages: no desorbed species, no broken fragments, force surface bonds, remove non-bonded atoms
- Over-mating penalty: fitness multiplied by `1 + (N_mate)^(-3/4)`
- Recommends running at multiple chemical potentials and merging ensembles for better compositional coverage
- Multi-stage relaxation: low precision (rationalize) -> mid precision (near LM) -> high precision (final)
- Similarity check via sorted interatomic distance lists (Vilhelmsen & Hammer, JCP 2014, 141, 044711)
- DOI: 10.1039/d4cp03801k

### Gradient-based grand canonical optimization with fractional atomic existence (Christiansen & Hammer, arXiv:2507.19438, 2025)
- Extends GNN message-passing to include continuous existence variable q_i in [0,1] per atom
- Gibbs free energy differentiable w.r.t. both Cartesian coordinates and existence: dG/dx_i and dG/dq_i
- Composition optimized via gradient descent (FIRE), not discrete enumeration
- Sigmoid transform q(chi) = 1/(1+exp(-chi)) keeps free variable unconstrained
- Snap to 0/1 at threshold (q < 0.1 -> 0, q > 0.9 -> 1), then continue Cartesian-only relaxation
- Optional harmonic bias to target species count: l(q) = k * (sum(q_i * I(Z_i=Z)) - N_Z)^2
- Applied to Cu(110) surface oxide: identifies Cu2O2 and Cu5O4 as stable phases, matching DFT
- Trained on Orb uMLIP labels (orb-v3-conservative-inf-omat), 58534 configurations, ~2h on A100
- DOI: 10.48550/arXiv.2507.19438

### Computational Hydrogen Electrode (Norskov et al., J. Phys. Chem. B 2004, 108, 17886-17892)
- Reference framework for electrochemical free energies at solid-liquid interfaces
- At U=0 V vs SHE and pH=0, the chemical potential of (H+ + e-) equals 1/2 E(H2_gas) by definition
- Free energy of adsorption: dG = dE + dZPE - TdS, where dE is DFT binding energy
- Potential dependence: dG(U) = dG(0) + neU for reactions involving n electron transfers
- pH dependence: dG(pH) = dG(0) + kT ln(10) * pH per proton transfer
- Applied to ORR on Pt: identifies OOH and OH binding as potential-determining steps
- Volcano plot: optimal catalyst binds O neither too strongly nor too weakly
- Key assumption: activation barriers scale with reaction free energies (BEP relations)
- DOI: 10.1021/jp047349j
