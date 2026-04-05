# galoop quickstart

Galoop is a steady-state genetic algorithm for adsorbate structure search on surfaces. It uses [signac](https://signac.io) for workspace management and [row](https://row.readthedocs.io) for cluster job submission.

## Installation

```bash
# Clone and install (editable)
git clone https://github.com/wladerer/galoop
cd galoop
uv venv .venv/galoop
uv pip install -e . --python .venv/galoop/bin/python

# Activate
source .venv/galoop/bin/activate
```

Core dependencies: `ase`, `signac`, `row`, `mace-torch`, `dscribe`, `pydantic`, `click`.

---

## Minimal run layout

```
my_run/
  galoop.yaml    # config (you write this)
  slab.vasp      # bare slab POSCAR
```

After `galoop run`, galoop creates:

```
my_run/
  galoop.yaml
  slab.vasp
  workflow.toml           # copied automatically for row
  .signac/config          # signac project metadata
  workspace/
    <job_id>/
      POSCAR              # starting geometry
      CONTCAR             # relaxed geometry
      FINAL_ENERGY        # energy in eV
      stage_relax/        # calculator stage outputs
```

---

## Step 1 — Prepare your slab

Build a slab POSCAR with ASE or VESTA. Fix the bulk layers using selective dynamics (or let galoop fix them by `sampling_zmin`).

```python
from ase.build import fcc100
from ase.constraints import FixAtoms
from ase.io import write
import numpy as np

slab = fcc100("Cu", size=(3, 3, 4), vacuum=8.0, periodic=True)

# Fix bottom two layers
z = slab.get_positions()[:, 2]
threshold = sorted(set(z.round(2)))[2]   # top of second layer
slab.set_constraint(FixAtoms(indices=[i for i, zi in enumerate(z) if zi < threshold]))

write("slab.vasp", slab, format="vasp")

# Note the z-range of the top layer for the config
print("Top layer z:", z.max().round(2))   # e.g. 13.41
# Place adsorbates just above the top layer:
# sampling_zmin ≈ top_z + 1.5,  sampling_zmax ≈ top_z + 2.5
```

Get the bare slab energy from your calculator:

```python
from mace.calculators import mace_mp
slab.calc = mace_mp(model="small", device="cpu")
print(slab.get_potential_energy())   # use this as slab.energy in the config
```

---

## Step 2 — Write `galoop.yaml`

```yaml
slab:
  geometry: slab.vasp
  energy: -135.87          # bare slab energy from your calculator (eV)
  sampling_zmin: 14.9      # bottom of adsorbate placement window (Å)
  sampling_zmax: 15.9      # top of adsorbate placement window (Å)

adsorbates:
  - symbol: OH
    chemical_potential: -7.2     # μ_O + 0.5*μ_H2 (eV, matched to your calculator)
    binding_index: 0             # index of the surface-binding atom (O here)
    coordinates:
      - [0.0, 0.0, 0.0]         # O at binding site
      - [0.0, 0.0, 0.97]        # H pointing up
    min_count: 0
    max_count: 3

  - symbol: CO
    chemical_potential: -14.1
    binding_index: 0             # C binds to surface
    coordinates:
      - [0.0, 0.0, 0.0]         # C
      - [0.0, 0.0, 1.15]        # O pointing up
    min_count: 0
    max_count: 3

calculator_stages:
  - name: relax
    type: mace                   # or "vasp" for DFT
    fmax: 0.05
    max_steps: 300
    fix_slab_first: true         # pre-relax adsorbates with slab frozen

# For a two-stage MACE→VASP workflow:
#   - name: preopt
#     type: mace
#     fmax: 0.10
#     max_steps: 200
#   - name: dft
#     type: vasp
#     fmax: 0.05
#     max_steps: 100
#     incar: {ENCUT: 520, EDIFF: 1.0e-6}

mace_model: small        # small | medium | large | path/to/custom.pt
mace_device: cpu         # cpu | cuda | auto

scheduler:
  type: local            # local | slurm | pbs
  nworkers: 4            # max pending jobs spawned at once

ga:
  population_size: 20
  max_structures: 500
  min_structures: 100
  max_stall: 15          # poll cycles without GCE improvement before stopping
  min_adsorbates: 1
  max_adsorbates: 4
  boltzmann_temperature: 0.15   # eV — parent selection sharpness
  rattle_amplitude: 0.10        # Å — slab surface rattling sigma

conditions:
  temperature: 298.15
  potential: 0.0         # V vs RHE
  pH: 0.0
```

### Chemical potential calibration

Chemical potentials must be consistent with your calculator's reference energies.
For MACE-small, compute molecule energies and use thermochemical corrections:

```python
# Example: μ_OH = E(H2O) - 0.5*E(H2) + corrections
# Rough values for MACE-small:  OH ≈ -7.2 eV,  CO ≈ -14.1 eV
# For VASP, use your standard DFT references (O2, H2, CO gas).
```

---

## Step 3 — Run

### On a cluster (recommended)

Two processes run concurrently from your run directory:

```bash
cd my_run

# Terminal 1 — GA controller (spawns offspring, evaluates relaxed structures)
galoop run --config galoop.yaml

# Terminal 2 — row submits pending jobs to the cluster scheduler
row run
```

`galoop run` copies `workflow.toml` to the run directory on first launch.
`row run` picks up every `pending` job and calls `galoop _relax <id> --run-dir .`.

For Slurm, set resources in `workflow.toml`:

```toml
[group.action.resources]
walltime = "02:00:00"
processes = {per_submission = 1, cores_per_process = 4}
```

### Local / development (no cluster)

Use the provided local runner, which relaxes each structure inline:

```bash
python runs/cu001_test/run_local.py
```

Or copy `run_local.py` to your run directory and adjust the `RUN_DIR` path at the top.

---

## Monitoring

```bash
# Live status
galoop status --run-dir my_run

# Example output:
#   converged            23
#   duplicate            14
#   desorbed              2
#   failed                1
#   pending               4
#
#   total evaluated:     40
#   duplicate rate:      38%
#
#   Top 5 by grand canonical energy:
#   1. G=-225.26 eV  2b83a5a4  op=init         ads={'OH': 2}
#   2. G=-225.13 eV  b387951f  op=init         ads={'OH': 2}
#   3. G=-213.63 eV  558ddce3  op=mutate_add   ads={'OH': 2, 'CO': 1}
#   ...

# HTML report with energy landscape, structure table, duplicate clusters
galoop report --run-dir my_run --config galoop.yaml

# Chemical-environment graphs (opens browser)
galoop graph --run-dir my_run --config galoop.yaml

# Stop gracefully after the current batch finishes
galoop stop my_run
```

---

## Status lifecycle

```
pending → submitted → relaxed → converged   (unique; GCE computed)
                             → duplicate   (same topology as existing converged)
                             → failed      (calculator error or bad geometry)
                             → desorbed    (adsorbate drifted off surface)
```

---

## GA operators

| Operator | Default weight | Description |
|---|---|---|
| `splice` | 0.30 | Molecule-aware Z-cut crossover: whole adsorbate molecules from A (below cut) + B (above) |
| `merge` | 0.20 | Combine adsorbates from both parents onto A's slab (clash-filtered) |
| `mutate_add` | 0.15 | Add one adsorbate at a random site |
| `mutate_remove` | 0.10 | Remove one adsorbate |
| `mutate_displace` | 0.10 | Displace one adsorbate atom (0.5 Å sigma) |
| `mutate_translate` | 0.10 | Translate one whole adsorbate molecule as a rigid body (0.8 Å sigma in x/y) |
| `mutate_rattle_slab` | 0.05 | Perturb unfixed slab surface atoms (`rattle_amplitude`) |

Override weights in `galoop.yaml`:

```yaml
ga:
  operator_weights:
    splice: 0.30
    merge: 0.20
    mutate_add: 0.15
    mutate_remove: 0.05
    mutate_displace: 0.10
    mutate_translate: 0.15
    mutate_rattle_slab: 0.05
```

---

## Convergence

The run stops when any of these is true:
- `total converged >= max_structures`
- `total converged >= min_structures` **and** `stall_count >= max_stall`

`stall_count` increments once per poll cycle (not per structure) when no new best grand canonical energy is found. Reset to 0 whenever a new best is set.

---

## Long-running campaigns on HPC

For multi-day runs (hundreds of VASP evaluations), the approach changes slightly: avoid relying on interactive login node sessions and submit both the GA controller and the row submitter as low-resource Slurm jobs.

### Submit both processes as Slurm jobs

`submit_controller.sh` — the GA controller:

```bash
#!/bin/bash
#SBATCH --job-name=galoop_ctrl
#SBATCH --partition=long          # partition with sufficient walltime (e.g. 7 days)
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=galoop_controller.log

source /path/to/.venv/bin/activate
cd /path/to/my_run
galoop run --config galoop.yaml
```

`submit_row.sh` — the job submitter:

```bash
#!/bin/bash
#SBATCH --job-name=row_submitter
#SBATCH --partition=long
#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=row_submitter.log

source /path/to/.venv/bin/activate
cd /path/to/my_run
while true; do row run; sleep 60; done
```

```bash
sbatch submit_controller.sh
sbatch submit_row.sh
```

The controller uses negligible resources — 1 CPU, a few hundred MB of RAM. The actual VASP work runs in separate jobs submitted by `row run`.

### Restarting after interruption

`galoop run` is fully restartable. On startup it rebuilds the structure cache from the workspace and finds the current best GCE — no manual intervention needed. Just resubmit `submit_controller.sh`.

One caveat: `stall_count` is in-memory and resets to 0 on restart. A run that was near convergence will require up to `max_stall` additional cycles without improvement before stopping again. For long campaigns, set `max_stall` generously (15–20) to account for this.

Structures that were `submitted` (Slurm job running) when the controller died will finish and write `relaxed` to the workspace. The restarted controller picks them up on the first poll — nothing is lost.

### Recommended settings for long campaigns

```yaml
ga:
  population_size: 20
  max_structures: 500        # hard cap — guarantees termination regardless of stall
  min_structures: 200        # must reach this before stall check applies
  max_stall: 20              # generous — accounts for one restart resetting counter
  boltzmann_temperature: 0.2 # slightly higher to maintain diversity over many evals

scheduler:
  type: slurm
  nworkers: 30               # structures kept in-flight simultaneously
```

### High-security clusters (no internet on compute nodes)

MACE foundation models are downloaded at runtime by default. On air-gapped systems, pre-download the model on a node with internet access and point the config to the cached file:

```python
# Run once on a node with internet access
from mace.calculators import mace_mp
mace_mp(model="small", device="cpu")
# Model is cached at ~/.cache/mace/ — note the path
```

```yaml
# galoop.yaml — use the absolute path instead of a model name
mace_model: /home/you/.cache/mace/mace_mp_small.model
```

Similarly, ensure all Python packages are installed into a self-contained venv before the run starts — compute nodes will not be able to reach PyPI.

### Monitoring without an active session

```bash
# Check current status from the login node at any time
galoop status --run-dir my_run

# Generate an HTML report snapshot
galoop report --run-dir my_run --config galoop.yaml

# Append status to a log periodically (add to crontab)
# */30 * * * * galoop status --run-dir /path/to/my_run >> /path/to/my_run/status_history.log
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Very high duplicate rate (>60%) | Boltzmann temperature too low — always selecting the same parent | Increase `boltzmann_temperature` (try 0.3–0.5) |
| `energy/atom = 1e6 eV` warnings | Splice/merge produced clashing geometry | Normal; caught by `energy_per_atom_tol`, structure marked failed |
| Desorption on every structure | `sampling_zmax` too high | Lower `sampling_zmax` closer to the surface |
| Clash placement warnings on every spawn | `max_adsorbates` too high for cell size | Reduce `max_adsorbates` or use a larger supercell |
| `workflow.toml not found` | Running `row run` from wrong directory | Run from the directory containing `galoop.yaml`; `galoop run` copies `workflow.toml` there automatically |
