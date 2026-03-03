# Integration Guide: Lean galoop with Existing Modules

This guide explains how to integrate the new lean galoop core with the existing (proven) modules from the original codebase.

---

## What to Keep from Original

### `gocia/engine/calculator.py`
**Keep entirely.** It handles:
- Multi-stage pipeline orchestration
- MACE relaxation via mace_mp()
- VASP relaxation via ASE Vasp calculator
- Sentinel file writing
- Trajectory.h5 output
- Energy sanity checks

**Import in galoop.py:**
```python
from gocia.engine.calculator import build_pipeline, run_pipeline
```

### `gocia/engine/scheduler.py`
**Keep entirely.** It provides:
- Scheduler abstraction (local/SLURM/PBS)
- Job script rendering
- Status polling (sacct/squeue/qstat)
- Job cancellation

**Import in galoop.py:**
```python
from gocia.engine.scheduler import build_scheduler
```

### `gocia/science/surface.py`
**Keep entirely.** It handles:
- Slab loading with FixAtoms constraint parsing
- Adsorbate geometry loading (files or inline)
- Random XY/Z placement
- Steric clash detection
- Formula parsing (OH, OOH, etc.)

**Import in galoop.py:**
```python
from gocia.science.surface import load_slab, load_adsorbate, place_adsorbate
```

### `gocia/science/energy.py`
**Keep entirely.** It provides:
- CHE grand canonical energy calculation
- Desorption detection
- Electrochemical potential + pH correction

**Import in galoop.py:**
```python
from gocia.science.energy import grand_canonical_energy, is_desorbed
```

### `gocia/science/reproduce.py`
**Keep entirely.** It provides all GA operators:
- `splice(a, b, n_slab, rng)` → (child1, child2)
- `merge(a, b, n_slab, rng)` → child
- `mutate_add(atoms, n_slab, symbol, position)` → atoms
- `mutate_remove(atoms, n_slab, symbol, rng)` → atoms
- `mutate_displace(atoms, n_slab, symbol, new_pos, rng)` → atoms

**Import in galoop.py:**
```python
from gocia.science.reproduce import splice, merge, mutate_add, mutate_remove, mutate_displace
```

---

## What's New (Replacement)

### `gocia/individual.py`
**Replaces:** Original `gocia/individual.py`
- Much simpler (no isomer tracking, no stored fingerprints)
- Cleaner data model

### `gocia/database.py`
**Replaces:** `gocia/engine/database.py` (partially)
- Keeps the same DB schema (mostly)
- Simplifies CRUD operations
- Removes run tracking table
- Removes fingerprint storage

### `gocia/fingerprint.py`
**Replaces:** `gocia/science/fingerprint.py`
- SOAP-only (removes histograms, Coulomb matrix, etc.)
- Simpler `classify_postrelax()`
- No pre-submission checking

### `gocia/config.py`
**Replaces:** Original `gocia/config.py`
- Simplified config schema
- Removes some obscure parameters
- Keeps essential ones

### `gocia/galoop.py`
**Replaces:** Original `gocia/galoop.py`
- Cleaner async steady-state loop
- No pre-submission gating
- Simpler spawn pipeline

### `gocia/cli.py`
**Replaces:** Original `gocia/cli.py`
- Minimal but functional
- Can be extended with `gocia progress`, `gocia pourbaix`, `gocia coverage` etc. later

---

## Integration Checklist

### 1. **Project Structure**
```
gocia/
├── __init__.py                    (new)
├── individual.py                  (NEW)
├── database.py                    (NEW)
├── fingerprint.py                 (NEW, replaces old)
├── config.py                      (NEW, replaces old)
├── galoop.py                      (NEW, replaces old)
├── cli.py                         (NEW, replaces old)
│
├── engine/
│   ├── __init__.py
│   ├── calculator.py              (KEEP, no changes)
│   └── scheduler.py               (KEEP, no changes)
│
└── science/
    ├── __init__.py
    ├── surface.py                 (KEEP, no changes)
    ├── energy.py                  (KEEP, no changes)
    ├── reproduce.py               (KEEP, no changes)
    └── (remove fingerprint.py — replaced by new one)
```

### 2. **Imports in galoop.py**
```python
# From engine (KEEP EXISTING)
from gocia.engine.calculator import build_pipeline, run_pipeline
from gocia.engine.scheduler import build_scheduler

# From science (KEEP EXISTING)
from gocia.science.surface import load_slab, load_adsorbate, place_adsorbate
from gocia.science.energy import grand_canonical_energy, is_desorbed
from gocia.science.reproduce import splice, merge, mutate_add, mutate_remove, mutate_displace

# New (from this codebase)
from gocia.individual import Individual, STATUS, OPERATOR
from gocia.database import GaloopDB
from gocia.fingerprint import classify_postrelax, compute_soap
from gocia.config import GaloopConfig
```

### 3. **Fix Circular Import**
In the new `galoop.py`, I have:
```python
from gocia.engine.calculator import build_pipeline
from gocia.engine.scheduler import build_scheduler
```

But these imports are **inside function signatures**, not at module level, to avoid circular imports. Keep it that way.

### 4. **Compatibility Notes**

**Database Schema:**
The new `database.py` uses the same schema as the original, minus:
- No `fingerprint` column (SOAP not stored)
- No `is_isomer` / `isomer_of` fields (simplified to weight = 0)
- No `desorption_flag` (desorbed → status = "desorbed")

If migrating from an existing run, you may need to:
```sql
-- Backup old DB
cp galoop.db galoop.db.bak

-- Drop old columns if they exist
ALTER TABLE structures DROP COLUMN fingerprint;
ALTER TABLE structures DROP COLUMN is_isomer;
ALTER TABLE structures DROP COLUMN isomer_of;
ALTER TABLE structures DROP COLUMN desorption_flag;

-- Reclassify old structures
UPDATE structures SET status = 'duplicate', weight = 0.0 WHERE status = 'isomer';
UPDATE structures SET status = 'desorbed' WHERE status = 'desorbed';
```

**Config Schema:**
Old `galoop.yaml` files need minimal updates:
- **Remove:** `fingerprint.soap_gate`, `weight_soap`, `weight_hist`, `energy_skip_tol`
- **Keep:** `r_cut`, `n_max`, `l_max`, `duplicate_threshold`
- All other sections (slab, adsorbates, calculator_stages, ga, conditions) are identical

**Individual Model:**
Old code using `ind.fingerprint`, `ind.is_isomer`, `ind.isomer_of` will break. Grep for these and remove references.

### 5. **Testing Integration**

```python
# test_integration.py
from pathlib import Path
from gocia.config import load_config
from gocia.database import GaloopDB
from gocia.individual import Individual, STATUS
from gocia.science.surface import load_slab
from gocia.engine.calculator import build_pipeline
from gocia.engine.scheduler import build_scheduler

# 1. Load config
cfg = load_config("galoop.yaml")

# 2. Create DB
with GaloopDB("galoop.db") as db:
    db.setup()
    
    # 3. Insert a test structure
    ind = Individual.from_init(
        generation=0,
        geometry_path="/tmp/struct_0000/POSCAR",
        extra_data={"adsorbate_counts": {"O": 1}},
    )
    db.insert(ind)
    
    # 4. Retrieve it
    retrieved = db.get(ind.id)
    assert retrieved.operator == "init"
    assert retrieved.status == "pending"
    
# 5. Test calculator pipeline
stages = build_pipeline(cfg.calculator_stages)
assert len(stages) > 0
assert stages[0].name == "mace_preopt"  # or whatever your first stage is

# 6. Test scheduler
scheduler = build_scheduler(cfg.scheduler)
assert scheduler is not None

# 7. Test slab loading
slab_info = load_slab(
    cfg.slab.geometry,
    zmin=cfg.slab.sampling_zmin,
    zmax=cfg.slab.sampling_zmax,
)
assert slab_info.n_slab_atoms > 0

print("✓ All integration tests passed!")
```

---

## Step-by-Step Migration

### For a Completely New Project

```bash
# 1. Create project structure
mkdir my_gocia_project && cd my_gocia_project
mkdir gocia

# 2. Copy files from this implementation
cp individual.py config.py database.py fingerprint.py galoop.py cli.py gocia/

# 3. Copy (don't modify) from original galoop
cp -r /path/to/original/gocia/engine gocia/
cp -r /path/to/original/gocia/science gocia/
rm gocia/science/fingerprint.py  # Use our new one instead

# 4. Copy example config and README
cp example_galoop.yaml ./galoop.yaml
cp README.md ./

# 5. Edit galoop.yaml to match your system
vim galoop.yaml

# 6. Run
python cli.py run --config galoop.yaml --run-dir . --seed 42
```

### For an Existing galoop Run

```bash
cd /path/to/existing/run

# 1. Backup everything
cp -r . ../backup_before_lean_upgrade/

# 2. Replace core files (keeping engine/ and science/)
cp /path/to/new/individual.py gocia/
cp /path/to/new/database.py gocia/
cp /path/to/new/fingerprint.py gocia/science/  # REPLACE OLD ONE
cp /path/to/new/config.py gocia/
cp /path/to/new/galoop.py gocia/
cp /path/to/new/cli.py gocia/

# 3. If you have an old galoop.db, migrate it (see above)
# 4. Update galoop.yaml (remove old fingerprint params)
# 5. Resume
python -m gocia.cli run --config galoop.yaml --run-dir .
```

---

## Troubleshooting

### Import Errors
If you see `ImportError: cannot import name 'X' from 'gocia'`:
- Check that you've copied all 6 files (individual.py, database.py, etc.)
- Check that engine/ and science/ subdirectories exist and have __init__.py files
- Use absolute imports: `from gocia.individual import Individual`

### Database Errors
If you see `sqlite3.OperationalError: no such table`:
- Call `db.setup()` to create the schema
- Or migrate your old DB (see above)

### Config Errors
If you see `ValidationError: fingerprint` or similar:
- Update galoop.yaml to remove old fingerprint parameters
- Use example_galoop.yaml as a template

### Missing Modules
If you see `ModuleNotFoundError: No module named 'dscribe'`:
```bash
pip install dscribe
```

---

## Performance Expectations

On a typical workstation:
- **Initial population** (20 structures): 1-2 minutes
- **MACE-only relaxation**: 30 seconds per structure
- **MACE + VASP coarse + VASP fine**: 10-30 minutes per structure
- **SOAP duplicate check**: 1-5 seconds per structure (post-relax)
- **Spawn + submission**: <1 second per structure

With 4 workers, expect:
- MACE-only: 1 full GA generation (20 structures) ≈ 1-2 hours
- Hybrid MACE+VASP: 1 generation ≈ 1-3 days (VASP limited)

---

## Questions?

Refer back to:
- **Code organization**: IMPLEMENTATION_SUMMARY.md
- **Design philosophy**: README.md
- **Usage**: example_galoop.yaml + README.md
- **Data model**: individual.py (well-commented)

Good luck!
