# Lean GOCIA — Quick Reference Card

## Files You Have

### Core (11 Python modules, ~3500 lines)
```
individual.py      (120 lines)  ← Data model
database.py        (250 lines)  ← SQLite CRUD
fingerprint.py     (180 lines)  ← SOAP duplicate detection
config.py          (250 lines)  ← Configuration validation
galoop.py          (700 lines)  ← Main GA loop
cli.py             (100 lines)  ← CLI interface
calculator.py      (350 lines)  ← MACE/VASP pipeline
scheduler.py       (400 lines)  ← SLURM/PBS/local scheduling
surface.py         (300 lines)  ← Slab + adsorbate placement
energy.py          (150 lines)  ← CHE energy + desorption
reproduce.py       (300 lines)  ← GA operators
```

### Configuration & Docs (5 files)
```
example_gocia.yaml              ← Template config
README.md                       ← User guide
IMPLEMENTATION_SUMMARY.md       ← Architecture details
INTEGRATION_GUIDE.md            ← Integration instructions
COMPLETE_DELIVERABLE.md         ← This summary
```

---

## Setup in 10 Steps

### 1. Create project structure
```bash
mkdir my_gocia && cd my_gocia
mkdir -p gocia/engine gocia/science
```

### 2. Copy all Python files
```bash
cp individual.py database.py fingerprint.py config.py galoop.py cli.py gocia/
cp calculator.py gocia/engine/
cp scheduler.py gocia/engine/
cp surface.py energy.py reproduce.py gocia/science/
touch gocia/__init__.py gocia/engine/__init__.py gocia/science/__init__.py
```

### 3. Copy config template
```bash
cp example_gocia.yaml gocia.yaml
```

### 4. Edit gocia.yaml
```bash
vim gocia.yaml
# Update at minimum:
# - slab.geometry (path to POSCAR)
# - adsorbates (your species)
# - calculator_stages (MACE and/or VASP)
```

### 5. Install dependencies
```bash
pip install numpy pandas ase pydantic pyyaml click dscribe
pip install mace-torch  # if using MACE
```

### 6. Verify imports
```python
python -c "from gocia.individual import Individual; print('✓ OK')"
```

### 7. Initialize database
```python
python -c "
from gocia.database import GociaDB
from pathlib import Path
Path('runs').mkdir(exist_ok=True)
with GociaDB('runs/test.db') as db:
    db.setup()
    print('✓ DB created')
"
```

### 8. Test config loading
```python
python -c "
from gocia.config import load_config
cfg = load_config('gocia.yaml')
print(f'✓ Config loaded: {len(cfg.adsorbates)} adsorbates, {len(cfg.calculator_stages)} stages')
"
```

### 9. Create minimal run script
```bash
cat > run_ga.py << 'EOF'
#!/usr/bin/env python
if __name__ == "__main__":
    from gocia.cli import cli
    cli()
EOF
chmod +x run_ga.py
```

### 10. Launch!
```bash
python run_ga.py run --config gocia.yaml --run-dir ./my_run --seed 42
```

---

## One-Liner Commands

### Monitor run
```bash
watch -n 5 "python run_ga.py status --run-dir ./my_run"
```

### Stop gracefully
```bash
python run_ga.py stop --run-dir ./my_run
```

### Export best structures
```bash
python -c "
import pandas as pd
df = pd.read_sql(
    'SELECT id, generation, grand_canonical_energy, extra_data FROM structures WHERE status=\"converged\" ORDER BY grand_canonical_energy LIMIT 20',
    'sqlite:///my_run/gocia.db'
)
df.to_csv('best_structures.csv', index=False)
print(f'Exported {len(df)} structures')
"
```

### Check run status
```bash
ls -lh my_run/gen_*/struct_*/CONTCAR | wc -l
```

---

## Configuration Cheat Sheet

### Minimal Config
```yaml
slab:
  geometry: slab.vasp
  energy: -100.0
  sampling_zmin: 10.0
  sampling_zmax: 15.0

adsorbates:
  - symbol: O
    chemical_potential: -4.92

calculator_stages:
  - name: mace
    type: mace
    fmax: 0.10

scheduler:
  type: local
  nworkers: 4
```

### Production Config (MACE + VASP)
```yaml
calculator_stages:
  - name: mace_preopt
    type: mace
    fmax: 0.10
    max_steps: 300

  - name: vasp_coarse
    type: vasp
    fmax: 0.05
    incar:
      ENCUT: 400
      EDIFF: 1.0e-5

  - name: vasp_fine
    type: vasp
    fmax: 0.02
    incar:
      ENCUT: 520
      EDIFF: 1.0e-6
```

### High-Throughput (MACE only, many adsorbates)
```yaml
ga:
  population_size: 50
  max_generations: 100
  max_stall_generations: 15

adsorbates:
  - symbol: O
    chemical_potential: -4.92
    max_count: 8
  - symbol: OH
    chemical_potential: -3.75
    max_count: 6
  - symbol: OOH
    chemical_potential: -3.50
    max_count: 3
```

---

## Database Queries

### Count by status
```sql
SELECT status, COUNT(*) FROM structures GROUP BY status;
```

### Top 10 structures
```sql
SELECT id, generation, grand_canonical_energy FROM structures 
WHERE status='converged' 
ORDER BY grand_canonical_energy 
LIMIT 10;
```

### Structures by generation
```sql
SELECT generation, COUNT(*) FROM structures 
GROUP BY generation 
ORDER BY generation;
```

### Export as CSV (pandas)
```python
import pandas as pd
df = pd.read_sql("SELECT * FROM structures", "sqlite:///gocia.db")
df.to_csv("export.csv", index=False)
```

---

## Common Issues & Fixes

| Error | Fix |
|-------|-----|
| `No module named 'dscribe'` | `pip install dscribe` |
| `ValidationError: duplicate_threshold` | Check fingerprint config in YAML |
| `FileNotFoundError: slab.vasp` | Check slab.geometry path in YAML |
| `sbatch: command not found` | Change scheduler.type to `local` or install SLURM |
| Database locked | No other process accessing DB; wait a moment |
| MACE download fails | Check internet; or set MACE_HOME and pre-download |
| VASP calculation hangs | Check VASP_PP_PATH environment variable |

---

## Performance Tuning

### Faster runs (sacrificing accuracy)
```yaml
calculator_stages:
  - name: mace_quick
    type: mace
    fmax: 0.15        # ← Loosen convergence
    max_steps: 100    # ← Fewer steps
```

### More thorough (slower)
```yaml
calculator_stages:
  - name: mace_tight
    type: mace
    fmax: 0.05        # ← Tighter convergence
    max_steps: 500    # ← More steps
```

### Parallel workers
```yaml
scheduler:
  nworkers: 8         # ← Increase parallelism
```

### Duplicate threshold
```yaml
fingerprint:
  duplicate_threshold: 0.85  # ← Stricter (fewer duplicates allowed)
```

---

## Typical Run Timeline (20-structure population)

| Stage | MACE-only | MACE+VASP |
|-------|-----------|-----------|
| Initial setup | 1 min | 1 min |
| Gen 0 placement | 5 min | 5 min |
| Gen 0 evaluation | 20 min | 4 hrs |
| Gen 1-5 (each) | 2-3 hrs | 1-2 days |
| **Total (5 gens)** | **6-8 hrs** | **1-2 weeks** |

---

## Restart After Interrupt

If the job is killed/crashes:
```bash
# All state is in gocia.db, no data loss
python run_ga.py run --config gocia.yaml --run-dir ./my_run --seed 42
# Loop resumes from where it left off
```

---

## Post-Processing

### Identify best structure
```python
import pandas as pd
from pathlib import Path
df = pd.read_sql("SELECT * FROM structures WHERE status='converged'", 
                 "sqlite:///my_run/gocia.db")
best = df.loc[df['grand_canonical_energy'].idxmin()]
struct_path = Path(best['geometry_path']).parent / 'CONTCAR'
print(f"Best: {struct_path}")
```

### Gather top 5 structures
```bash
python -c "
import pandas as pd
from pathlib import Path
import shutil

df = pd.read_sql(
    'SELECT geometry_path FROM structures WHERE status=\"converged\" ORDER BY grand_canonical_energy LIMIT 5',
    'sqlite:///my_run/gocia.db'
)

for i, path in enumerate(df['geometry_path']):
    src = Path(path).parent / 'CONTCAR'
    shutil.copy(src, f'best_{i:02d}.vasp')
    print(f'Copied {i}: {src}')
"
```

---

## Files to Read (in order)

1. **This file** — Quick reference
2. **example_gocia.yaml** — Understand config structure
3. **README.md** — Architecture overview
4. **galoop.py** (read `run()` function) — Main loop flow
5. **individual.py** — Data model
6. **fingerprint.py** — SOAP duplicate detection

---

## Key Concepts

| Term | Meaning |
|------|---------|
| **SOAP** | Smooth Overlap of Atomic Positions; fingerprint for duplicate detection |
| **CHE** | Computational Hydrogen Electrode; electrochemical energy correction |
| **Splice** | GA operator: cut both parents, recombine |
| **Merge** | GA operator: combine both parents' adsorbates |
| **GCE** | Grand Canonical Energy (fitness metric) |
| **Sentinel** | Verification file (status on disk) |
| **Stage** | One step in multi-stage calculator pipeline |

---

## Defaults

| Parameter | Default | Good Range |
|-----------|---------|------------|
| duplicate_threshold | 0.90 | 0.85-0.95 |
| temperature | 298.15 K | 298-573 K |
| MACE fmax | varies | 0.05-0.20 |
| VASP fmax | varies | 0.02-0.10 |
| population_size | 20 | 10-50 |
| max_generations | 50 | 5-200 |
| max_stall_generations | 10 | 5-20 |

---

## Success Indicators

✅ **Run is working:**
- `gen_000/` created with random structures
- `gocia.db` exists and grows
- MACE/VASP calculations run without error
- New generations appear (`gen_001/`, `gen_002/`, etc.)
- Best energy improves over generations

❌ **Something's wrong:**
- CONTCAR files are empty
- All structures marked duplicate
- Calculations timeout or OOM
- DB locked or corrupted
- SOAP vectors have NaN

---

## Contact & Feedback

If you find bugs or have ideas:
- Check the logs: `cat my_run/slurm.out` (SLURM) or `stdout.txt` (local)
- Review the code: all files are well-commented
- Test in isolation: try MACE-only first, then add VASP
- Read the traceback: Python's error messages are usually clear

---

**Ready to run! Type:**
```bash
python run_ga.py run --config gocia.yaml --run-dir ./my_run --seed 42
```

**Happy searching!** 🧬
