# Galoop manuscript

Full scientific writeup of what the validation sweep demonstrated —
theory, architecture, results, and forward-looking work. Sibling of
the more technical `../report/` directory, which is the raw
validation-sweep artefact trail; this one is the document you'd hand
to a collaborator.

## Layout

```
manuscript/
├── manuscript.tex          ← main LaTeX source
├── scientific_report.sty   ← style pkg (from K-Dense scientific-writing skill)
├── Makefile                ← latexmk build
├── README.md               ← you are here
├── figures/                ← manuscript-specific figures (empty; reuses ../report/figures/)
└── diagrams/
    ├── flowchart.mmd       ← canonical Mermaid source for Fig. 1
    └── uml.mmd             ← canonical Mermaid source for Fig. 2
```

The `diagrams/*.mmd` files are the **canonical text-of-truth**
versions of the flowchart and UML class diagram, per the
`markdown-mermaid-writing` skill. Inside `manuscript.tex` the same
diagrams are re-rendered in TikZ so the PDF build has no external
render dependency (mermaid-cli, headless Chrome, etc.). When editing,
update the Mermaid source FIRST, then sync the TikZ by hand.

## Figures

The manuscript reuses the figures from the validation-sweep report
next door via `\graphicspath{{figures/}{../report/figures/}}`. That
avoids copying the 1.8 MB PNG set twice; the `report/` directory
stays authoritative.

If you need to regenerate any figure, run the scripts in
`../report/scripts/` — they query the per-run SQLite stores under
`runs/*_camp/` directly.

## Building

### One-time install

```bash
sudo apt install -y texlive-latex-recommended texlive-latex-extra \
                    texlive-science texlive-fonts-recommended \
                    texlive-pictures latexmk
```

`texlive-pictures` is required for the TikZ shape libraries
(`shapes.geometric`, `arrows.meta`, `positioning`, `fit`, `calc`,
`backgrounds`) used by the flowchart and UML diagrams.

### Build

```bash
make            # pdflatex — default, works everywhere
make xe         # xelatex — proper Helvetica rendering
```

The default pdflatex build falls back to Computer Modern Sans where
`scientific_report.sty` asks for Helvetica; the layout is the same
but the body font is slightly different.

### Clean

```bash
make clean      # removes build artefacts, keeps the PDF
make distclean  # also removes manuscript.pdf
```

## Editing

- **Introduction is intentionally bullet points.** The author is
  expected to expand them into prose during a revision pass. See the
  `\begin{itemize}` block in section~\ref{sec:intro}.
- **Math sections** (`\section{Theoretical background}`) are fully
  worked — CHE, SOAP+Tanimoto, GPR+UCB, and the new kappa-annealing
  schedule all have their equations typeset.
- **Diagrams** live in two forms:
  - `diagrams/*.mmd` — Mermaid text, the canonical version. Edit
    this first.
  - TikZ blocks inside `manuscript.tex` — what the PDF actually
    shows. Sync by hand to the Mermaid source.
- **Style package commands** (from `scientific_report.sty`) used in
  the document: `\makereporttitle`, `keyfindings`, `methodology`,
  `limitations` box environments. See the style pkg for the full set.
- **Bibliography** is a minimal inline `thebibliography` block with
  three entries (Nørskov 2004, Hammer-Nørskov 2000, Bartók 2013). If
  the writeup grows, move to a proper `.bib` file.

## Relation to the `report/` directory

| Aspect | `report/` | `manuscript/` |
|---|---|---|
| Purpose | Technical artefact log of the validation sweep | Scientific writeup of what was demonstrated |
| Audience | Me six months from now, debugging | A collaborator or reviewer |
| Introduction | None — jumps into results | Bullet-point outline for prose expansion |
| Math background | None | Full CHE / SOAP / GPR / kappa derivations |
| Diagrams | None | TikZ flowchart + UML class diagram |
| Figures | The 17 PNGs generated from scripts/ | Same, reused via `\graphicspath` |
| Style | Plain `article` | `scientific_report.sty` (K-Dense skill) |

Both compile with the same texlive install. If you only want one,
the `report/` version is the more complete technical record; the
`manuscript/` version is the one to send someone.
