# Gene-Disease Link Prediction Framework

A comprehensive framework for predicting gene-disease associations using graph neural networks and embedding-based methods.

## Overview

This project implements a suite of graph-based models for predicting potential associations between genes and diseases. The framework includes state-of-the-art graph neural networks (GCN, GAT, GraphSAGE, SEAL), embedding-based methods (DeepWalk, Node2Vec), and traditional heuristic approaches for comprehensive comparison.

The best model (GraphSAGE) achieves 0.994 AUC and 0.992 AP on our validation dataset, demonstrating the effectiveness of graph neural networks for this task.

## Repository Structure

```
.
├── link_prediction/             # Main code for link prediction models
│   ├── algorithms/              # Implementation of various algorithms
│   │   ├── gcn.py               # Graph Convolutional Network
│   │   ├── gat.py               # Graph Attention Network
│   │   ├── graphsage.py         # GraphSAGE implementation
│   │   ├── seal.py              # SEAL implementation
│   │   ├── embedding_methods.py # DeepWalk and Node2Vec implementations
│   │   └── heuristic_methods.py # Traditional heuristic methods
│   ├── main.py                  # Main script for training and evaluation
│   ├── metrics/                 # Code for computing evaluation metrics
│   ├── utils/                   # Utility functions
│   ├── generate_metrics_report.py  # Generate detailed metrics report
│   ├── generate_latex_table.py     # Generate LaTeX table for paper
│   ├── generate_sample_metrics.py  # Generate sample metrics for testing
│   ├── generate_simple_chart.py    # Create bar chart comparison
│   └── generate_multi_metrics.py   # Generate multiple visualization charts
├── paper/                       # Research paper and related materials
│   ├── paper_draft.tex          # LaTeX source for the research paper
│   ├── figures/                 # Figures and visualizations
│   ├── references/              # Bibliography and references
│   └── compile_paper.sh         # Script to compile the paper
└── data/                        # Data directory (not included in repo)
    └── GRAPH                    # Graph data file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gene-disease-link-prediction.git
cd gene-disease-link-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv graphresearch
source graphresearch/bin/activate  # On Windows: graphresearch\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train a specific model:

```bash
cd link_prediction
python main.py --model [MODEL_NAME] --device [cuda/cpu] --epochs [NUM_EPOCHS]
```

Available models:
- GCN
- GAT 
- GraphSAGE
- SEAL
- DeepWalk
- Node2Vec
- Heuristic

To train all models:

```bash
python main.py --all --max-nodes 500  # Limit to 500 nodes for faster training
```

### Running Embedding Methods

A dedicated script is available for testing embedding methods:

```bash
./run_embedding_test.sh
```

### Generating Reports and Visualizations

The framework includes several tools for generating reports:

```bash
# Generate comprehensive metrics report
./generate_metrics_report.py

# Generate LaTeX table for publication
./generate_latex_table.py

# Generate comparative charts
./generate_simple_chart.py
./generate_multi_metrics.py
```

## Results

Our extensive evaluation shows that GNN-based models consistently outperform traditional approaches:

| Model        | AUC    | AP     | F1     | Training Time (s) |
|--------------|--------|--------|--------|-------------------|
| GraphSAGE    | 0.994  | 0.992  | 0.975  | 0.00              |
| GAT          | 0.994  | 0.991  | 0.974  | 0.00              |
| SEAL         | 0.972  | 0.839  | 0.899  | 6.60              |
| GCN          | 0.952  | 0.908  | 0.943  | 2.18              |
| DeepWalk     | 0.877  | 0.952  | 0.890  | 5.51              |
| Node2Vec     | 0.862  | 0.869  | 0.920  | 5.05              |
| Heuristic    | 0.831  | 0.790  | 0.764  | 2.58              |

## Research Paper

A research paper detailing our framework and findings is available in the `paper/` directory. To compile the paper:

```bash
cd paper
./compile_paper.sh
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our paper:

```
@article{gene_disease_prediction,
  title={A Graph Neural Network Framework for Gene-Disease Association Prediction},
  author={Author Names},
  journal={Journal Name},
  year={2023}
}
```

## Acknowledgments

- The DGL team for their graph neural network library
- Contributors to the DG-Miner dataset 