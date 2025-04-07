# Makefile for compiling the gene-disease paper

# Set the main LaTeX file name without extension
MAIN = paper_draft

# Commands
LATEX = pdflatex
BIBTEX = bibtex

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references/references.bib
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot

distclean: clean
	rm -f $(MAIN).pdf

.PHONY: all clean distclean 