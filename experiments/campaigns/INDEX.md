# Campaign index — galoop validation runs (2026-04-09 onward)

10-campaign GPU validation run for galoop after the MLIP-factory and
non-orthogonal-normals refactors. Goal: variety + coverage; find bugs
as they happen.

**Backend (all runs):** fairchem UMA `uma-s-1p1` / `oc20` task on CUDA.
**Common GA params:** `population_size: 20`, `max_structures: 450`,
`max_stall: 40`, `duplicate_threshold: 0.88`, `gpr_guided: true`,
single Parsl worker. Per-run yamls under `runs/<tag>/galoop.yaml`;
per-run REPORT.md is uncommitted (lives next to the run dir, which is
`.gitignore`d).

## Campaigns

| #  | tag                 | metal/facet | adsorbates              | conditions          | status      | best G (eV) | top ads counts             | wall   |
|---:|---------------------|-------------|-------------------------|---------------------|-------------|------------:|----------------------------|--------|
|  1 | cu111_co_camp       | Cu(111)     | CO/H/H₂O                | U=0,  pH=0          | done        |  −397.640   | CO:8 H:3 H₂O:0 (ceiling)   | 17 min |
|  2 | cu100_co_camp       | Cu(100)     | CO/H/H₂O                | U=0,  pH=0          | done†       |  −450.106   | CO:8 H:1 H₂O:2 (ceiling)   | 67 min |
|  3 | cu211_co_camp       | Cu(211)     | CO/H/H₂O                | U=0,  pH=0          | done‡       |  −435.784   | CO:9\* H:1 H₂O:2 (bug 9)   | 19 min |
|  4 | pt111_orr_camp      | Pt(111)     | O/OH/OOH/H              | U=0.8, pH=0         | done        |  −481.319   | OH:6 OOH:4 (ceiling§)      | 33 min |
|  5 | pt211_nrr_camp      | Pt(211)     | N/NH/NH₂/NH₃/H          | U=−0.5, pH=0        | done†       |  −459.675   | N:4 NH₃:5§                 | 74 min |
|  6 | ag111_co_camp       | Ag(111)     | CO/H/H₂O                | U=0,  pH=0          | done†       |  −218.485   | CO:4 H:1 (weak binding)    | 32 min |
|  7 | pd111_hsat_camp     | Pd(111)     | H (max=16)              | U=0,  pH=0          | done ✨     |  −322.469   | **H:16 = 1 ML (clean)**    |  8 min |
|  8 | ni111_chx_camp      | Ni(111)     | C/CH/CH₂/CH₃/H          | U=0,  pH=0          | done        |  −514.931   | CHₓ mix§ (GPR-dominated)   | 20 min |
|  9 | au111_co_camp       | Au(111)     | CO/OH                   | U=0,  pH=0          | done†       |  −296.041   | CO:4 OH:3 (most inert)     | 80 min |
| 10 | ru0001_oh_camp      | Ru(0001)    | H₂O/OH/O                | U=0,  pH=0          | done ✨     |  **−581.995** | O:6 H₂O:4 (lowest G in sweep) | 12 min |

**§ breakdown subject to bug 12** (element-overlapping species → greedy
`infer_adsorbate_counts` miscount). GCE values are still correct.
**✨ clean runs** with no constraint violations, low reject rate, fast
turnaround — the GA pipeline's happy path.
**Total sweep wall time:** ~6 h 20 min across 10 campaigns on a single
NVIDIA 6 GiB GPU.

## Bugs found / fixed during the run

(running list — added as observed, with commit SHA if a fix landed)

| # | severity | bug                                                                                          | status                                        |
|--:|----------|-----------------------------------------------------------------------------------------------|-----------------------------------------------|
| 1 | medium   | `_surface_normal` used `c/|c|`; wrong for tilted-c cells (the very case the commit named)    | fixed `d5e5687` (a×b normal); not yet stress-tested |
| 4 | minor    | `parsl.dfk().cleanup()` not called on exit → "DFK still running" hang                         | fixed `0a22b65`                               |
| 6 | tuning   | high pre-relax dup rate + early spawn stall on tight `max_count`                              | mitigated by raising `max_count`, lowering `duplicate_threshold` to 0.88 |
| 7 | tuning   | `max_stall: 40` is too patient when convergence rate drops to ~1/12 min late in a run        | will use `max_stall: 15` for runs 4–10 to fail fast |
| 8 | tuning   | `max_count: 8` for CO is hit on every Cu facet → saturation point not measurable             | will raise to `max_count: 14` for runs 4+ where it makes sense |
| 9 | **real** | `spawn_one` crossover bounds check enforces only `max_adsorbates` (total), NOT per-species `max_count`. Splice/merge of two parents at the CO ceiling can emit children exceeding it.  | fixed `a6a0fb9` (per-species min/max enforcement in `_TWO_PARENT_OPS` branch) |
|10 | env      | `fairchem-core` install bumped torch 2.6→2.8 but left `torchvision==0.21` (torch 2.6 era). `torchvision::nms` import fails; anything importing through `galoop.spawn`→MACE→e3nn also fails with `e3nn` codegen pickling mismatch. | workaround: upgraded torchvision to 0.23.0+cu128 (fixes torchvision). `e3nn`/MACE chain still broken in test env — NOT blocking UMA runtime. Long-term fix: pin e3nn ≥ torch-2.8-compat or strip MACE tests from default ignore list. |
|12 | reporting | `infer_adsorbate_counts` greedily groups atoms by formula length, not structure-aware. For species that share elements (OOH/OH/O/H, or C/CH/CH₂/CH₃/H), actual {OH:6,OOH:3,H:1} gets reported as {OOH:6,H:4} — same element count, wrong molecular breakdown. | **GCE is unaffected** — CHE chem pots are linear in element content, so different groupings with the same atoms give identical Σ µᵢ·nᵢ (verified symbolically). Reported "top 5 with species counts X,Y,Z" for ORR/NRR/CHₓ runs are therefore *unreliable molecular breakdowns* but *correct GCE*. Fix requires structure-aware counting via `_group_molecules`. Deferred — will not mis-fix mid-run. |
| - | note     | bug 4 (`parsl.dfk().cleanup()`) validated end-to-end on Cu(100) — `galoopstop` triggered a clean exit with `DFK cleanup complete` and no hang |  |

† Cu(100) was stopped early via `galoopstop` after 67 min (24 converged,
85% dup rate, best unchanged for ~45 min). The run produced a useful
result; further evaluation would have been a poor use of GPU time.

‡ Cu(211) winner violated configured `max_count=6` for CO (hit 9 and
10). The relaxations are physically valid but the run doesn't answer
the "what's CO coverage at max_count=6" question. Flagged in
`runs/cu211_co_camp/REPORT.md`; fix landed as commit `a6a0fb9`; not
re-run as the result is still a useful data point showing UMA
accommodates higher CO coverage on stepped surfaces.

(snap_to_surface ablation findings from 2026-04-08 are intentionally
not committed to this index — the snap removal was rolled back, the
ablation lives only in session history.)
