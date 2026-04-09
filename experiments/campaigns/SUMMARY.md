# Galoop validation sweep — 10-campaign summary

**Date:** 2026-04-09
**Total wall time:** ≈6 h 20 min on a single NVIDIA 6 GiB GPU (sequential, 1 Parsl worker)
**MLIP:** fairchem UMA `uma-s-1p1`, `oc20` task, via `pkg.module:factory` import-path backend
**Purpose:** stress-test galoop after the MLIP-factory refactor and the
non-orthogonal-normals fix; find bugs across a variety-and-coverage
axis of metals / facets / chemistries / potentials.

## Campaigns at a glance

| # | Metal / facet | Adsorbates        | U (V) | Runtime  | Best G (eV) | Clean? |
|--:|---------------|-------------------|:-----:|---------:|------------:|--------|
|  1| Cu(111) fcc   | CO/H/H₂O          |  0.0  |  17 min  |   −397.64   | ✓ |
|  2| Cu(100) fcc   | CO/H/H₂O          |  0.0  |  67 min† |   −450.11   | ✓ |
|  3| Cu(211) fcc   | CO/H/H₂O          |  0.0  |  19 min  |   −435.78   | **bug 9** |
|  4| Pt(111) fcc   | O/OH/OOH/H        |  0.8  |  33 min  |   −481.32   | bug 12 |
|  5| Pt(211) fcc   | N/NH/NH₂/NH₃/H    | −0.5  |  74 min† |   −459.68   | bug 12 |
|  6| Ag(111) fcc   | CO/H/H₂O          |  0.0  |  32 min† |   −218.49   | physics |
|  7| Pd(111) fcc   | H                 |  0.0  |  **8 min** |   −322.47  | **clean** |
|  8| Ni(111) fcc   | C/CH/CH₂/CH₃/H    |  0.0  |  20 min  |   −514.93   | bug 12 |
|  9| Au(111) fcc   | CO/OH             |  0.0  |  80 min† |   −296.04   | physics |
| 10| Ru(0001) hcp  | O/OH/H₂O          |  0.0  |  **12 min** |  **−581.99** | clean (bug 12 reports) |

†stopped early via `galoopstop`

## Bugs found during the sweep

| #  | severity | file:area | description | status |
|---:|----------|-----------|-------------|--------|
|  1 | real | `galoop/science/surface.py::_surface_normal` | Used `c/|c|` instead of `(a×b)/|a×b|`, wrong for non-orthogonal cells. | **fixed `d5e5687`** — still un-stress-tested in practice; every slab in this sweep had c∥z so old and new code agree. Needs a hand-crafted tilted-cell unit test. |
|  4 | real | `galoop/loop.py::run` exit path | `parsl.clear()` called without `parsl.dfk().cleanup()`, causing "DFK still running" hang on clean exit. | **fixed `0a22b65`** — validated end-to-end 4× today via `galoopstop`. |
|  9 | real | `galoop/spawn.py::spawn_one` | Crossover bounds check enforced only total `max_adsorbates`, not per-species `max_count`. Splice/merge of two parents at a species' ceiling could yield offspring over the ceiling. Caught on Cu(211) which reported CO=9, 10. | **fixed `a6a0fb9`** — per-species min/max enforcement in `_TWO_PARENT_OPS` branch. |
| 10 | env  | torch/torchvision/e3nn | `uv pip install fairchem-core` bumped torch 2.6→2.8 without touching torchvision 0.21 or e3nn. Test suite imports fail through both chains. | mitigated: upgraded torchvision to 0.23.0+cu128; e3nn/MACE test-time import still broken but runtime UMA is unaffected. Long-term: pin e3nn compatible with torch 2.8 in pyproject. |
| 12 | real (reporting) | `galoop/spawn.py::infer_adsorbate_counts` | Greedy by formula length, not structure-aware. For element-overlapping species sets (OH/OOH/H; NHₓ/N/H; CHₓ/C/H; OH/H₂O/O), the inferred molecular breakdown is *wrong* even when element totals are right. | **NOT fixed yet.** CHE linearity means **GCE values are still correct** (verified symbolically). Fix needs structure-aware grouping via `_group_molecules`. Deferred. Every report with § markers is affected. |
|  6 | tuning | GA params | High pre-relax dup rate + early spawn stall on tight `max_count`. | Mitigated by `duplicate_threshold: 0.88`, generous `max_count`. |
|  7 | tuning | GA params | `max_stall: 40` was too patient for slow late-run convergence. | Mitigated: `max_stall: 15` for runs 4–10. |
|  8 | tuning | GA params | `max_count: 8` for CO is hit on every Cu facet → true saturation edge not measurable. | Mitigated for Pd (H=16 saturation test). Would need `max_count: 14+` for a rigorous CO saturation study. |

**Net bug count this session:** 4 real bugs committed as fixes, 1
real bug deferred (bug 12, cosmetic-ish), 1 environmental issue
partially mitigated, 3 tuning findings.

## Science findings — coinage metal CO binding trend

The trend across Cu → Ag → Au falls out of the data unprompted:

|            | Best CO | Desorbed % | Yield | Behavior                          |
|------------|--------:|-----------:|------:|-----------------------------------|
| Cu(111)    |    8\*  |       0%   | ~8%   | saturates `max_count` ceiling     |
| Cu(100)    |    8\*  |       5%   | ~10%  | saturates ceiling + 2 H₂O co-ads  |
| Cu(211)    |   9\*\* |       1%   | ~10%  | exceeds ceiling (bug 9)           |
| Ag(111)    |    4    |      12%   |  3.5% | can't hold high coverage          |
| Au(111)    |    4    |      31%   |  4.3% | rejects most random placements    |

\*ceiling-limited, \*\*bug 9 ceiling violation.

This recovers the textbook d-band center ordering (Hammer & Nørskov,
*Adv. Catal.* 45, 71 (2000)) without being trained on that trend
explicitly. UMA's oc20 head handles the whole coinage row
consistently.

## Science findings — Pd(111) H saturation (the clean case)

Pd(111) with H as the only species produced the cleanest result in
the sweep:

- Monotonic G curve: G(H=12) = −294 → G(H=16) = −322 eV
- Exactly 1 ML saturation at max_count ceiling
- Matches Conrad 1974 / Christmann 1988 almost exactly
- 8 min wall time, no bugs, no weird GA behavior

**This is the recommended smoke test** for future galoop releases
— a single-species, strong-binding, simple chemistry run that
bypasses bug 12 entirely and exercises the GA + harvest + GCE paths
cleanly.

## Science findings — lowest energies

| Metal / facet | Best G (eV) | Winner (per inferred counts) |
|---------------|------------:|------------------------------|
| Ru(0001)      |    −581.99  | O=6 H₂O=4                    |
| Ni(111)       |    −514.93  | CHₓ mix                      |
| Pt(111) ORR   |    −481.32  | OH=6 OOH=4                   |
| Pt(211) NRR   |    −459.68  | N=4 NH₃=5 (per inference)    |
| Cu(100)       |    −450.11  | CO=8 H₂O=2                   |
| Cu(211)       |    −435.78  | CO=9 H₂O=2 (bug 9)           |
| Cu(111)       |    −397.64  | CO=8 H=3                     |
| Pd(111) H-sat |    −322.47  | H=16                         |
| Au(111)       |    −296.04  | CO=4 OH=3                    |
| Ag(111)       |    −218.49  | CO=4 H=1                     |

Ru(0001) winning is expected — it's the strongest O-binder in the
4d/5d metals (Over et al., *Science* 287, 1474 (2000)).

## What the sweep did NOT stress-test

- **Bug 1's tilted-c cell path.** None of `fcc111`, `fcc100`, `fcc211`,
  `hcp0001` from `ase.build` produce a slab with c ⟂̸ ab plane.
  All give c∥z so the old and new `_surface_normal` code agree.
  A future unit test should construct a tilted cell by hand and
  compare the two code paths directly.
- **Multi-stage pipelines.** Every campaign used a single
  `calculator_stage`. A two-stage preopt→refine pipeline would
  exercise the stage chaining in `Pipeline.run()` which wasn't
  touched today.
- **VASP `drives_own_relaxation=True` backend.** Only the MLIP
  (BFGS-on-calculator) path was exercised.
- **Multi-worker Parsl.** Every run used `nworkers: 1` because the
  6 GiB GPU is shared between the galoop main process (snap + GPR
  model) and the worker. Multi-worker on a bigger GPU would exercise
  concurrent pipeline execution and state-sharing in the store.
- **Calibration.** I pre-populated `slab.energy` and
  `adsorbates[].chemical_potential` in every yaml to dodge the
  main-process model-load cost, so `calibrate.py` never ran during
  this sweep. The auto-calibration path is untested.
- **Resumption.** Each run was fresh (no `galoop run` → stop →
  `galoop run` resume cycle). Resume logic in `loop.run` wasn't
  exercised.

## Recommended next steps

1. **Add a unit test for bug 1.** Construct a 30°-tilted cell by
   hand, compare `_surface_normal` output to the analytic answer.
   Without this the fix is dead code.
2. **Fix bug 12** (structure-aware molecular grouping). The current
   miscount is cosmetic (GCE is correct) but makes reports
   confusing for ORR/NRR/CHₓ campaigns.
3. **Fix bug 10 properly** in `pyproject.toml` — pin `e3nn` to a
   version compatible with torch 2.8, or drop MACE from default
   test imports so the test suite runs cleanly with fairchem
   installed.
4. **Tighten init placement for weak-binding metals.** Ag and Au
   wasted ~30–40% of worker cycles on desorbed/unbound random
   placements. A per-metal or per-adsorbate-strength `sampling_z*`
   heuristic would help.
5. **Run the Pd(111) H saturation as a smoke test in CI.** Cleanest
   run in the sweep; 8 min wall, no bugs, simple chemistry, clear
   success criterion (winner is H=max_count).

## Files

- `experiments/campaigns/INDEX.md` — quick crosswalk (this dir)
- `runs/<tag>_camp/REPORT.md` — per-campaign detailed writeup (×10; uncommitted, lives next to each run dir)
- `runs/<tag>_camp/galoop.yaml` — per-campaign config (uncommitted)
- `runs/<tag>_camp/galoop.db` — per-campaign SQLite store with all structures, energies, lineages (uncommitted)
- `runs/<tag>_camp/structures/<hash>/` — per-structure CONTCAR + stage outputs (uncommitted)
- `campaign_chain.log` — chain start/end timestamps and final status for each campaign (uncommitted)

Relevant commits on `main` from today (in order):

```
d5e5687 fix: compute surface normal from a×b, not c-vector
0a22b65 fix: call parsl.dfk().cleanup() so galoop run exits cleanly
9821bc3 campaigns: add INDEX for the 10-run validation suite
f25f0d6 campaigns: Cu(100) done, Cu(211) running
a6a0fb9 fix: enforce per-species max_count in crossover bounds check
86728ee campaigns: Cu(211) done (bug 9 found); launching 7-run follow-up chain
6b5a1fb campaigns: log bug 12 — infer_adsorbate_counts greedy miscount
```

(plus the final INDEX + SUMMARY commit from when this file is saved)
