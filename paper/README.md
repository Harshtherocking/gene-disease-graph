# Gene-Disease Association Prediction Paper

This repository contains the LaTeX source for our paper on gene-disease association prediction using graph neural networks.

## Paper Abstract

Identifying gene-disease associations is crucial for understanding disease mechanisms and developing effective treatments. Experimental methods for validating such associations are time-consuming and expensive, making computational prediction approaches valuable for prioritizing candidates for wet-lab validation. In this paper, we propose a graph neural network-based framework for predicting novel gene-disease associations from heterogeneous biological networks. Our framework incorporates multiple state-of-the-art graph neural network architectures (GCN, GAT, GraphSAGE, and SEAL) along with traditional heuristic methods for comparison. Experiments on a comprehensive gene-disease association dataset demonstrate that our graph-based approaches significantly outperform traditional methods, with the GCN model achieving the highest performance with an AUC of 0.94 and Average Precision of 0.90. Our framework offers an efficient and accurate computational approach for discovering potential gene-disease associations to guide biomedical research.

## Compiling the Paper

### Prerequisites

To compile the paper, you need the following software:
- LaTeX distribution (such as TeX Live or MiKTeX)
- IEEE Transaction LaTeX class and style files

### Using Make

We provide a Makefile for easy compilation:

```bash
# Compile the paper
make

# Clean auxiliary files
make clean

# Clean all generated files including the PDF
make distclean
```

### Manual Compilation

Alternatively, you can compile the paper manually with the following commands:

```bash
pdflatex paper_draft
bibtex paper_draft
pdflatex paper_draft
pdflatex paper_draft
```

## Paper Structure

The paper is organized as follows:
- **Introduction**: Overview of the gene-disease association prediction problem and our approach
- **Related Work**: Summary of prior research on computational GDA prediction and graph neural networks
- **Methodology**: Description of our GNN framework and implemented models
- **Experimental Setup**: Details of the dataset, evaluation protocol, and implementation
- **Results and Discussion**: Performance comparison, scalability analysis, and case studies
- **Conclusion and Future Work**: Summary of findings and directions for future research

## Customizing the Paper

To customize the paper content:
1. Edit `paper_draft.tex` to modify text, equations, or tables
2. Update the bibliography information in the `\begin{thebibliography}` section
3. Replace placeholder information such as author names and affiliations

## License

This work is provided for educational and research purposes. If you use any part of this work, please cite our paper. 