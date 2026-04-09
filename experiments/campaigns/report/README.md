# Validation-sweep report

LaTeX report + reproducibility scripts for the 10-campaign galoop
validation sweep run on 2026-04-09.

## Layout

```
report/
├── report.tex             ← source
├── Makefile
├── README.md              ← you are here
├── stats.json             ← aggregated per-campaign counts (generated)
├── scripts/
│   ├── 01_structure_figs.py   — top-down + side renders per campaign
│   ├── 02_heatmap_rdf.py      — adsorbate heatmaps + RDFs
│   ├── 03_soap_sweep.py       — SOAP cutoff sensitivity curves
│   └── 04_gpr_analysis.py     — GPR vs GA operator analysis
└── figures/
    ├── struct_*.png        — per-campaign best-structure views
    ├── heatmap_grid.png
    ├── rdf_grid.png
    ├── soap_dup_curves.png
    ├── soap_rcut_unique.png
    ├── operator_counts.png
    ├── g_evolution.png
    └── gpr_vs_nongpr_box.png
```

## Building the PDF

```bash
# 1. Install LaTeX once:
sudo apt install -y texlive-latex-recommended texlive-latex-extra \
                    texlive-science texlive-fonts-recommended latexmk

# 2. Build:
cd experiments/campaigns/report
make            # -> report.pdf
```

## Regenerating figures from source data

All figures come from the per-run SQLite stores under
`runs/*_camp/galoop.db` and the relaxed CONTCARs in
`runs/*_camp/structures/`. If you re-run a campaign or want to
refresh the figures:

```bash
cd experiments/campaigns/report
make figures
make                # rebuild PDF with new figures
```

The scripts expect to be run from the repository root OR from
`experiments/campaigns/report/` — the path resolution uses
`__file__` to find the repo root automatically.

## Dependencies (Python)

- `ase` (slab I/O, atom rendering)
- `matplotlib` (everything)
- `numpy`, `pandas`
- `dscribe` (SOAP descriptor for Section 1)
- `pyyaml`
- `galoop` itself (the scripts import `galoop.store.GaloopStore` and
  `galoop.individual.STATUS`)

All of these are already in the project's `pyproject.toml`.
