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
|  2 | cu100_co_camp       | Cu(100)     | CO/H/H₂O                | U=0,  pH=0          | running     |     —       | —                          |  —     |
|  3 | cu211_co_camp       | Cu(211)     | CO/H/H₂O                | U=0,  pH=0          | queued      |     —       | —                          |  —     |
|  4 | (Pt(111) ORR)       | Pt(111)     | O/OH/OOH/H              | U=0.8, pH=0         | planned     |     —       | —                          |  —     |
|  5 | (Pt(211) NRR)       | Pt(211)     | N/NH/NH₂/NH₃/H          | U=−0.5, pH=0        | planned     |     —       | —                          |  —     |
|  6 | (Ag(111) CO)        | Ag(111)     | CO/H/H₂O                | U=0,  pH=0          | planned     |     —       | —                          |  —     |
|  7 | (Pd(111) H sat)     | Pd(111)     | H (max=16)              | U=0,  pH=0          | planned     |     —       | —                          |  —     |
|  8 | (Ni(111) CHₓ)       | Ni(111)     | C/CH/CH₂/CH₃/H          | U=0,  pH=0          | planned     |     —       | —                          |  —     |
|  9 | (Au(111) inert)     | Au(111)     | CO/OH                   | U=0,  pH=0          | planned     |     —       | —                          |  —     |
| 10 | (Ru(0001) hcp)      | Ru(0001)    | H₂O/OH/O                | U=0,  pH=0          | planned     |     —       | —                          |  —     |

## Bugs found / fixed during the run

(running list — added as observed, with commit SHA if a fix landed)

| # | severity | bug                                                                                          | status                                        |
|--:|----------|-----------------------------------------------------------------------------------------------|-----------------------------------------------|
| 1 | medium   | `_surface_normal` used `c/|c|`; wrong for tilted-c cells (the very case the commit named)    | fixed `d5e5687` (a×b normal); not yet stress-tested |
| 4 | minor    | `parsl.dfk().cleanup()` not called on exit → "DFK still running" hang                         | fixed `0a22b65`                               |
| 6 | tuning   | high pre-relax dup rate + early spawn stall on tight `max_count`                              | mitigated by raising `max_count`, lowering `duplicate_threshold` to 0.88 |

(snap_to_surface ablation findings from 2026-04-08 are intentionally
not committed to this index — the snap removal was rolled back, the
ablation lives only in session history.)
