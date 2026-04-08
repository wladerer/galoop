# galoop

Galoop is my own pet project to extend GCGA algorithm implemented in GOCIA by Z. Zhang. I am using Gaussian Process Regression to guide GCGA towards a global minimum within composition space. The genetic algorithm acts to promote structural diversity while GPR steers the evolution. This method seems to help accelerate the compositional search and find when a surface is truly saturated under particular experimental conditions. A side effect of this is skipping over shallow basins in which local minima exist, perhaps skewing ensemble averaged results one might hope to glean from a GCGA campaign.  

## CLI 

Galoop comes with a set of cli commands since it is intended for use on terminal centered computing environments like HPCs. Some examples are

`galoop init` - scaffold a new campaign directory with a minimal `galoop.yaml`

`galoop status` - shows best candidates and progress of the campaign

`galoop run` - starts the campaign 

`galoop stop` - gracefully ends the campaign 

The most commonly used command would be `galoop status` which gives the user a good idea of the success rate of relaxations, the frequency of various mutation operators, and the paths to the structures with the best (most negative) grand potential.

### `galoop init` — scaffold a campaign

`galoop init` is the easiest way to start a new campaign. It creates `galoop.yaml` (and optionally a `calc.py` backend template) in a target directory with placeholder values you fill in before launching.

```bash
# Barebones — writes galoop.yaml with a <PATH_TO_YOUR_SLAB_FILE> placeholder
galoop init runs/my_campaign

# Link an existing slab file (copied into the campaign dir as slab.vasp)
galoop init runs/my_campaign -s path/to/my_slab.vasp

# VASP backend instead of the default MACE
galoop init runs/my_campaign -s slab.vasp -b vasp

# Custom MLIP backend: scaffolds both galoop.yaml AND a calc.py template
galoop init runs/my_campaign -s slab.vasp -b custom

# Same, but explicitly request the calc.py template alongside the default
# MACE config
galoop init runs/my_campaign -s slab.vasp --calc-template
```

After scaffolding, edit the placeholders (the `<LIKE_THIS>` tokens) in `galoop.yaml` — notably `sampling_zmin`/`sampling_zmax` and your adsorbate list — and then:

```bash
galoop run -c runs/my_campaign/galoop.yaml -d runs/my_campaign -v
```

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
    binding_index: 0        # row index of the atom that binds to the surface
    coordinates:            # each row is {Element: [x, y, z]}
      - N: [0.0, 0.0, 0.0]
      - H: [0.0, 0.0, 1.03]

  - symbol: NH2
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - N: [0.0, 0.0, 0.0]
      - H: [0.0,  0.82, 0.55]
      - H: [0.0, -0.82, 0.55]

  - symbol: NH3
    min_count: 0
    max_count: 3
    binding_index: 0
    coordinates:
      - N: [0.0, 0.0, 0.0]
      - H: [ 0.0,   0.94, 0.39]
      - H: [ 0.81, -0.47, 0.39]
      - H: [-0.81, -0.47, 0.39]

  - symbol: H
    min_count: 0
    max_count: 6

calculator_stages: #the user may define as many steps in the geometry optimization of each structure - all handled and dispatched by Parsl
  - name: preopt
    type: mace                 # built-in name OR "pkg.module:factory" for a custom MLIP
    fmax: 0.05
    max_steps: 500
    fix_slab_first: true
    prescan_fmax: 0.1
    params:                    # passed straight to the backend factory
      model: medium            # small | medium | large | path/to/custom.pt
      device: cuda             # cpu | cuda | auto
      dtype: float32

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

``` 

## Custom MLIP backends

Galoop's `calculator_stages[].type` field accepts two kinds of values:

- **Built-in names**: `mace` and `vasp`. These ship with galoop.
- **Import paths**: `pkg.module:factory_callable` — a dotted path to any Python callable that returns an ASE `Calculator`. This lets you bring your own MLIP without modifying galoop source. The module just has to be importable on the `PYTHONPATH` of whoever is running the stage (you, plus Parsl workers if you're on SLURM/PBS).

### When to use this

MACE-MP-0 is general-purpose, but catalysis-specific models (trained on adsorbate-on-slab DFT data) are often much more accurate for the kind of campaigns galoop is designed for. Examples of openly available models that plug in cleanly:

- **fairchem UMA / eSEN** (Meta FAIR) — `pip install fairchem-core`. Multi-task models trained on OC20 + OMat24 + others. Strong on both catalysis and bulk materials.
- **Orb-v2 / Orb-v3** (Orbital Materials) — `pip install orb-models`. Includes OMat24-trained checkpoints.
- **MatterSim** (Microsoft) — `pip install mattersim`. General materials MLIP with T/P awareness.
- **SevenNet** — efficient equivariant, has an OC20-finetuned variant.

### Walkthrough: using fairchem UMA (OMat24 among others)

**1. Scaffold the campaign with a custom backend:**

```bash
galoop init runs/uma_nrr -s my_slab.vasp -b custom
```

This writes `runs/uma_nrr/galoop.yaml` with `type: calc:make_calculator` and `runs/uma_nrr/calc.py` as a placeholder factory.

**2. Install fairchem:**

```bash
uv pip install fairchem-core
```

**3. Fill in `calc.py`:**

```python
# runs/uma_nrr/calc.py
from __future__ import annotations
import threading
from typing import Any

_CACHE: dict[tuple, Any] = {}
_LOCK = threading.Lock()


def make_calculator(params: dict):
    """Return a fairchem UMA calculator, memoized per process."""
    key = (
        params.get("model", "uma-s-1p1"),
        params.get("task", "oc20"),
        params.get("device", "cuda"),
    )
    if key in _CACHE:
        return _CACHE[key]

    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]

        from fairchem.core import pretrained_mlip
        from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

        model_name, task, device = key
        predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
        calc = FAIRChemCalculator(predictor, task_name=task)

        _CACHE[key] = calc
        return calc
```

The caching is important for Parsl workers: without it you'd reload the model on every structure, which dominates wall-clock on fast campaigns.

**4. Edit `galoop.yaml` to reference the factory and tune params:**

```yaml
calculator_stages:
  # Cheap screening with small UMA
  - name: screen
    type: calc:make_calculator
    fmax: 0.08
    max_steps: 150
    fix_slab_first: true
    prescan_fmax: 0.15
    params:
      model: uma-s-1p1        # small → fast
      task: oc20              # adsorbate-on-slab head
      device: cuda

  # Refine survivors with the medium model
  - name: refine
    type: calc:make_calculator
    fmax: 0.03
    max_steps: 400
    params:
      model: uma-m-1p1
      task: oc20
      device: cuda

# Optional: a separate (usually cheaper) backend for snap_to_surface during
# initial-population building. If omitted, snap reuses calculator_stages[0]
# with fmax/max_steps overridden to 0.2 / 30.
snap_stage:
  name: snap
  type: calc:make_calculator
  fmax: 0.2
  max_steps: 30
  params:
    model: uma-s-1p1
    task: oc20
    device: cuda
```

**5. Make `calc.py` importable and launch:**

```bash
cd runs/uma_nrr
PYTHONPATH=. galoop run -c galoop.yaml -d . -v
```

For SLURM/PBS runs, add `PYTHONPATH` to `scheduler.resources.env` in the yaml — Parsl's `worker_init` picks it up automatically:

```yaml
scheduler:
  type: slurm
  nworkers: 4
  resources:
    partition: gpu
    env:
      PYTHONPATH: /path/to/runs/uma_nrr
```

### How dispatch works under the hood

When `CalculatorStage` is instantiated, galoop calls `backends.resolve(type_str)`:

1. If `type_str` contains a colon (`pkg.mod:func`), `importlib.import_module("pkg.mod")` + `getattr(..., "func")` gives the factory.
2. Otherwise, `type_str` is looked up in the built-in registry (`mace`, `vasp`).

The factory is called once per stage-build with the `params` dict, returns an ASE Calculator, and galoop's `CalculatorStage` runs BFGS against it (or, for VASP-style backends that relax internally, calls `get_potential_energy()` once). See `galoop/engine/backends.py` for the registry and `galoop/engine/calculator.py` for the stage runner.

### DFT-style "drives its own relaxation" backends

If your calculator performs the relaxation internally (like ASE's `Vasp`), expose the factory as a module-level tuple instead of a bare callable:

```python
def _make(params):
    from ase.calculators.vasp import Vasp
    return Vasp(**params.get("incar", {}))

MY_BACKEND = (_make, True)   # second element = drives_own_relaxation
```

and reference it as `type: my_module:MY_BACKEND`. Galoop will call `get_potential_energy()` once and read the resulting CONTCAR instead of running a Python-side BFGS loop.


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
