# galoop

Galoop is my own pet project to extend GCGA algorithms. I am using Gaussian Process Regression to guide GCGA towards a global minimum within composition space. The genetic algorithm acts to promote structural diversity while GPR steers the evolution. This method seems to help accelerate the compositional search and find when a surface is truly saturated under particular experimental conditions. A side effect of this is skipping over shallow basins in which local minima exist, perhaps skewing ensemble averaged results one might hope to glean from a GCGA campaign.  

## CLI 

Galoop comes with a set of cli commands since it is intended for use on terminal centered computing environments like HPCs. Some examples are

`galoop status` - shows best candidates and progress of the campaign

`galoop run` - starts the campaign 

`galoop stop` - gracefully ends the campaign 

The most commonly used command would be `galoop status` which gives the user a good idea of the success rate of relaxations, the frequency of various mutation operators, and the paths to the structures with the best (most negative) grand potential. 

## Input File

`galoop` input files are validated as pydantic models to ensure that the program is not handed innapropriate input settings. 

### User Notes

galoop requires a yaml file with, at the minimum, the pristine surface and adsorbate geometries. The remaining input parameters have been tuned over countless GPR-GCGA campaigns. Furthermore, there is an optional calibration step that calculates the reference slab energy and chemical potentials for the adsorbates at the highest level of theory the user has chosen. 

There are many parameters the user can change. But thankfully, there is no need for setting environment variables are corralling endless helper scripts or geometry reference files. I've included a sample input file to share the possible levers one has on the campaigns. I've annotated some of the more interesting features.


```yaml
slab:
  geometry: /home/wladerer/github/galoop/runs/pt111_nrr/slab.vasp
  sampling_zmin: 13.0 
  sampling_zmax: 17.0

adsorbates:
  - symbol: N
    min_count: 0
    max_count: 4

  - symbol: NH
    min_count: 0
    max_count: 3
    binding_index: 0        # defines the atom that binds to the surface
    coordinates:
      - [0.0, 0.0, 0.0]     # N
      - [0.0, 0.0, 1.03]    # H

  - symbol: NH2
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - [0.0, 0.0, 0.0]     # N
      - [0.0, 0.82, 0.55]   # H
      - [0.0, -0.82, 0.55]  # H

  - symbol: NH3
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - [0.0, 0.0, 0.0]      # N
      - [0.0, 0.94, 0.39]     # H
      - [0.81, -0.47, 0.39]   # H
      - [-0.81, -0.47, 0.39]  # H

  - symbol: H
    min_count: 0
    max_count: 6

calculator_stages: #the user may define as many steps in the geometry optimization of each structure - all handled and dispatched by Parsl
  - name: preopt
    type: mace
    fmax: 0.05
    max_steps: 500
    fix_slab_first: true
    prescan_fmax: 0.1

scheduler:
  type: local # Includes slurm, PBS, and SGE
  nworkers: 3

ga:
  population_size: 20
  min_structures: 20
  max_structures: 500
  max_stall: 25
  min_adsorbates: 1
  max_adsorbates: 10
  boltzmann_temperature: 0.1
  rattle_amplitude: 0.3
  displace_amplitude: 2.0
  translate_amplitude: 2.0
  gpr_guided: true
  gpr_fraction: 0.4
  gpr_min_samples: 15
  gpr_kappa: 1.5
  operator_weights: # these have been tuned over dozens of campaigns according to their ability to produce novel and reasonable structures
    splice: 0.25
    merge: 0.15
    mutate_add: 0.15
    mutate_remove: 0.12
    mutate_displace: 0.05
    mutate_translate: 0.23
    mutate_rattle_slab: 0.05

conditions: #you can define the global working conditions and the CHE framework applies the correction for you (no vibrations though)
  potential: -0.5
  pH: 0.0
  temperature: 298.15

fingerprint: #this is for tuning duplicate detection 
  r_cut: 5.0
  n_max: 6
  l_max: 4
  duplicate_threshold: 0.95
  energy_tol_pct: 3.0

mace_model: medium #you are able to put the path to a mace model (.pt or .model file) to use your own pre-trained MLIP
mace_device: cuda #change according to your hardware
mace_dtype: float32

``` 


## My own Literature notes

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
