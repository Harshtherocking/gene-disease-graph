#!/bin/bash

# Compile the paper with latex and bibtex

echo "Compiling gene-disease association prediction paper..."

# Change to the paper directory
cd "$(dirname "$0")"

# Run the compilation process
pdflatex paper_draft.tex
bibtex paper_draft
pdflatex paper_draft.tex
pdflatex paper_draft.tex

# Check if the PDF was created successfully
if [ -f "paper_draft.pdf" ]; then
    echo "PDF compiled successfully: paper_draft.pdf"
else
    echo "Error: PDF compilation failed"
    exit 1
fi

echo "Done!" 