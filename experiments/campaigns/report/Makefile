# Build the validation-sweep report.
#
# Requires:
#   sudo apt install -y texlive-latex-recommended texlive-latex-extra \
#                       texlive-science texlive-fonts-recommended latexmk
#
# Usage:
#   make            # full build -> report.pdf
#   make figures    # regenerate all figures from the campaign DBs
#   make clean      # remove LaTeX build artefacts (keeps PDF)
#   make distclean  # also remove the PDF

PYTHON ?= python
SCRIPTS := scripts
FIG_DIR := figures

FIGURES := \
	$(FIG_DIR)/struct_cu111_co_camp.png \
	$(FIG_DIR)/heatmap_grid.png \
	$(FIG_DIR)/rdf_grid.png \
	$(FIG_DIR)/soap_dup_curves.png \
	$(FIG_DIR)/soap_rcut_unique.png \
	$(FIG_DIR)/operator_counts.png \
	$(FIG_DIR)/g_evolution.png \
	$(FIG_DIR)/gpr_vs_nongpr_box.png

all: report.pdf

report.pdf: report.tex $(FIGURES)
	latexmk -pdf -interaction=nonstopmode report.tex

figures:
	$(PYTHON) $(SCRIPTS)/01_structure_figs.py
	$(PYTHON) $(SCRIPTS)/02_heatmap_rdf.py
	$(PYTHON) $(SCRIPTS)/03_soap_sweep.py
	$(PYTHON) $(SCRIPTS)/04_gpr_analysis.py

clean:
	latexmk -c report.tex
	rm -f report.aux report.log report.out report.toc report.fls \
	      report.fdb_latexmk report.synctex.gz

distclean: clean
	rm -f report.pdf

.PHONY: all figures clean distclean
