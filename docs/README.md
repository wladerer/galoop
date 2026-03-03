# Lean galoop

A simplified genetic algorithm for exploring surface adsorbate structures under electrochemical conditions.

## Key Design Decisions

### 1. **SOAP-Only Post-Relaxation Duplicate Detection**
- No pre-submission fingerprinting → faster spawning, cleaner logic
- After full relaxation (all calculator stages), compute SOAP and check against cache
- Simple: Tanimoto similarity ≥ threshold → duplicate (weight = 0)
- SOAP vectors cached in-memory during run, rebuilt from CONTCAR files on restart

### 2. **Asynchronous Steady-State Loop**
- Job finishes → process it → spawn one replacement
- Workers stay busy even while long VASP jobs are running
- No blocking on generation boundaries

### 3. **Flexible Multi-Stage Calculator Pipeline**
Configure in `galoop.yaml` to run:
- **MACE only** (fast, low accuracy)
- **MACE → VASP coarse → VASP fine** (hybrid, production)
- **VASP only** (pure DFT)
- Any other combination

### 4. **Minimal Data Model**
`Individual` has only:
- `id`, `generation`, `parents`, `operator`: genealogy
- `status`: pipeline state (pending/submitted/converged/failed/duplicate/desorbed)
- `raw_energy`, `grand_canonical_energy`: fitness
- `weight`: 1.0 (unique) or 0.0 (duplicate)
- `geometry_path`, `extra_data`: structure location and adsorbate counts

### 5. **DB as Single Source of Truth**
- Sentinel files are verification only, not authoritative
- On restart, walk DB and rebuild SOAP cache from CONTCAR files
- Cleaner than the original sentinel-based restart system

---

## Quickstart

### Installation
```bash
pip install click pydantic pyyaml numpy pandas ase dscribe
# For VASP stages also:
pip install vasp  # (or get it from your HPC)
```

### Setup

1. **Prepare your slab geometry:**
   ```bash
   # Create slab.vasp with selective dynamics (frozen bottom layers)
   ```

2. **Create galoop.yaml:**
   ```bash
   cp example_galoop.yaml galoop.yaml
   # Edit to match your system
   ```

3. **Initialize the run:**
   ```bash
   python cli.py run --config galoop.yaml --run-dir ./my_run
   ```
   This:
   - Validates config
   - Creates `my_run/galoop.db` with schema
   - Generates `gen_000/` with random initial population
   - Starts the main GA loop

### Monitor

```bash
# Check status
python cli.py status --run-dir ./my_run

# Request graceful stop (finishes current generation)
python cli.py stop --run-dir ./my_run
```

### Post-Processing

After the run:
- Best structures are in `gen_NNN/` directories
- `galoop.db` has complete genealogy
- Analyze with pandas/SQLite:
  ```python
  import pandas as pd
  df = pd.read_sql("SELECT * FROM structures WHERE status='converged' ORDER BY grand_canonical_energy", 
                   'sqlite:///my_run/galoop.db')
  ```

---

## Architecture Overview

### File Structure
```
gocia/
  individual.py       # Simplified Individual model + STATUS/OPERATOR constants
  database.py         # SQLite CRUD
  fingerprint.py      # SOAP computation + Tanimoto similarity
  config.py           # Pydantic config validation
  galoop.py           # Main async GA loop
  cli.py              # Command-line interface
  
  (external, not reimplemented)
  engine/
    calculator.py     # Pipeline runner (stages: MACE, VASP)
    scheduler.py      # Job submission (local, SLURM, PBS)
  science/
    surface.py        # Slab loading, adsorbate placement
    energy.py         # CHE grand canonical energy
    reproduce.py      # GA operators (splice, merge, mutate)
```

### Main Loop Flow

```
while not done:
  poll scheduler for finished jobs
  
  for each finished job:
    read CONTCAR
    compute SOAP
    check against soap_cache
    
    if duplicate:
      mark duplicate, weight = 0
    else:
      compute fitness (CHE energy)
      cache SOAP vector
      update DB
    
  spawn one replacement (splice/merge/mutate or random)
  submit to scheduler
  
  check stop criteria (generations, stall)
```

### Selection

Parents are drawn **weighted by fitness** from all converged (weight > 0) structures:
```python
weights = exp(-gce / kT)  # Boltzmann weighting
indices = rng.choice(n_parents, p=weights/sum(weights))
parents = [pool[i] for i in indices]
```

No age bias, no generation boundaries — the entire converged pool is fair game.

---

## Configuration

### `calculator_stages`
List of relaxation steps in order. Each has:
- `name`: unique label
- `type`: "mace" or "vasp"
- `fmax`: force convergence (eV/Å)
- `max_steps`: step limit
- `incar` (VASP only): INCAR overrides

Example hybrid:
```yaml
calculator_stages:
  - name: mace_quick
    type: mace
    fmax: 0.10
    max_steps: 300
  
  - name: vasp_polish
    type: vasp
    fmax: 0.02
    max_steps: 100
    incar:
      ENCUT: 520
      EDIFF: 1.0e-6
```

### `ga`
- `population_size`: structures per checkpoint bucket
- `max_generations`: hard stop after N buckets
- `min_generations`: run at least N buckets
- `max_stall_generations`: stop if best hasn't improved for N buckets
- `min_adsorbates`, `max_adsorbates`: per-structure coverage bounds

### `fingerprint`
- `r_cut`, `n_max`, `l_max`: SOAP parameters
- `duplicate_threshold`: Tanimoto ≥ this → duplicate (0.90 is typical)

### `conditions`
CHE grand canonical energy is evaluated at fixed T, P, U, pH.

---

## Restart Safety

If the job is interrupted:
1. All submitted/running jobs are tracked in `active_jobs` dict
2. On restart, `_fill_workers()` reads DB and resubmits any pending structures
3. Finished structures (sentinel = CONVERGED) are post-processed
4. SOAP cache is rebuilt from all converged structures' CONTCAR files

No data loss, no manual intervention needed.

---

## Customization

### Add a New Operator

Implement in `gocia/science/reproduce.py`, then add to `_sample_operator()`:
```python
def _sample_operator(config, rng):
    ops = [SPLICE, MERGE, MUTATE_ADD, MUTATE_REMOVE, MY_NEW_OP]
    probs = [0.3, 0.2, 0.2, 0.2, 0.1]
    return rng.choice(ops, p=probs)
```

### Change Duplicate Threshold

Edit `galoop.yaml`:
```yaml
fingerprint:
  duplicate_threshold: 0.85  # Stricter (fewer duplicates flagged)
```

### Add a New Adsorbate

Just add to `adsorbates` in galoop.yaml:
```yaml
adsorbates:
  - symbol: N
    chemical_potential: -4.5
    min_count: 0
    max_count: 2
```

The GA handles arbitrary species automatically.

---

## Notes

- **SOAP computation is expensive** — runs ~once per converged structure. For 100+ structures, consider caching or switching to lighter descriptors post-run.
- **MACE pre-opt is optional** — if you only want VASP, just omit the MACE stage.
- **Selective dynamics** (FixAtoms) in slab geometry are respected during all relaxations.
- **CHE correction** automatically applies electrochemical potential + pH to adsorbate chemical potentials.

---

## References

- CHE framework: Nørskov, J. K., et al. J. Electrochem. Soc. (2005)
- SOAP descriptor: Bartók, A. P., et al. Phys. Rev. B (2013)
- dscribe: Himanen, L., et al. Comput. Phys. Commun. (2020)
