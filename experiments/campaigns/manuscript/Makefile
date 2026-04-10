# Build the galoop manuscript PDF.
#
# Requires the same LaTeX packages as the validation-sweep report:
#     sudo apt install -y texlive-latex-recommended texlive-latex-extra \
#                         texlive-science texlive-fonts-recommended \
#                         texlive-pictures latexmk
#
# texlive-pictures carries TikZ's shape libraries (the flowchart and
# the UML diagram need `shapes.geometric`, `arrows.meta`, etc.).
#
# Compile with pdflatex by default (works for everything except the
# Helvetica font substitution — scientific_report.sty requests
# helvet, which pdflatex handles via the cm-super family). For proper
# Helvetica rendering, use `make xe`.

TEX       = manuscript.tex
PDF       = manuscript.pdf
LATEXMK   = latexmk -pdf -interaction=nonstopmode -halt-on-error
XELATEXMK = latexmk -xelatex -interaction=nonstopmode -halt-on-error

FIGURES_LOCAL  := $(wildcard figures/*.png)
FIGURES_REPORT := $(wildcard ../report/figures/*.png)
DIAGRAMS       := $(wildcard diagrams/*.mmd)

all: $(PDF)

$(PDF): $(TEX) $(FIGURES_LOCAL) $(FIGURES_REPORT) scientific_report.sty
	$(LATEXMK) $(TEX)

xe: $(TEX) $(FIGURES_LOCAL) $(FIGURES_REPORT) scientific_report.sty
	$(XELATEXMK) $(TEX)

clean:
	latexmk -c $(TEX) 2>/dev/null || true
	rm -f manuscript.aux manuscript.log manuscript.out manuscript.toc \
	      manuscript.fls manuscript.fdb_latexmk manuscript.synctex.gz \
	      manuscript.bbl manuscript.blg manuscript.run.xml

distclean: clean
	rm -f $(PDF)

.PHONY: all xe clean distclean
