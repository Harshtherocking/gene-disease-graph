# Gene-Disease Link Prediction Framework

This framework provides a comprehensive collection of link prediction algorithms for bipartite gene-disease networks.

[*Project Report*](../REPORT.md)

## Overview

The framework includes the following components:

1. Data preparation: Loads the disease-gene graph and splits edges into train/validation/test sets
2. Multiple link prediction algorithms (both traditional and deep learning-based)
3. Comprehensive evaluation and comparison of model performance
4. Visualization of results

## Link Prediction Algorithms

The framework includes the following algorithms:

1. **Matrix Factorization (MF)**: A simple embedding-based approach that learns node representations through matrix factorization.
2. **Graph Convolutional Network (GCN)**: Uses GCN layers to learn node representations that capture the graph structure.
3. **Graph Attention Network (GAT)**: Extends GCNs with attention mechanisms to weight the importance of neighbors.
4. **GraphSAGE**: A neighborhood aggregation approach for inductive learning on graphs.
5. **SEAL (Subgraph Extraction And Learning)**: Extracts enclosing subgraphs around node pairs and uses them for link prediction.
6. **Heuristic Methods**:
   - Random Forest on topological features
   - Gradient Boosting Decision Trees on topological features

## Directory Structure

```
link_prediction/
├── data_preparation.py         # Data loading and splitting
├── base_model.py               # Base class for all link prediction models
├── algorithms/                 # Implementation of algorithms
│   ├── matrix_factorization.py
│   ├── gcn.py
│   ├── gat.py
│   ├── graphsage.py
│   ├── seal.py
│   └── heuristic_methods.py
├── metrics.py                  # Evaluation metrics and comparison
├── models/                     # Saved models and metrics
└── main.py                     # Main script to run the pipeline
```

## Usage

### Prerequisites

Make sure you have the required dependencies installed:

```bash
pip install torch dgl networkx pandas scikit-learn matplotlib seaborn tqdm
```

### Running the Pipeline

To run the complete pipeline (data preparation, model training, and evaluation):

```bash
python main.py --all
```

You can also run individual steps:

```bash
# Just prepare the data
python main.py --prepare-data

# Train specific models
python main.py --train-models --models MF GCN GraphSAGE

# Train models in parallel
python main.py --train-models --parallel

# Evaluate models and generate reports
python main.py --evaluate
```

### Using a Smaller Dataset

For faster experimentation, you can use a smaller subset of the graph by specifying the `--max-nodes` parameter:

```bash
# Use a small subset with only 1000 nodes
python main.py --all --max-nodes 1000 --epochs 10
```

This will:
1. Create a subgraph with only 1000 randomly selected nodes
2. Train the models with fewer epochs (10 instead of the default 20)
3. For SEAL, use an even smaller subset of edges to speed up computation

Running with these settings can reduce the training time from potentially hours to minutes, which is useful for testing and debugging purposes.

### GPU Acceleration

The framework supports GPU acceleration via CUDA, which can significantly speed up training for the neural network models (MF, GCN, GAT, GraphSAGE, and SEAL). To use GPU:

```bash
# Run with GPU acceleration
python main.py --all --device cuda

# For quick testing on GPU
./run_quick_test.sh
```

If you have limited GPU memory, you can combine the smaller dataset approach with GPU acceleration:

```bash
# Use a small dataset on GPU
python main.py --all --max-nodes 1000 --epochs 10 --device cuda
```

The SEAL model will automatically adjust its batch size based on available GPU memory to optimize performance and avoid out-of-memory errors.

### Training Options

```
usage: main.py [-h] [--prepare-data] [--train-models] [--evaluate] [--all]
               [--models {MF,GCN,GAT,GraphSAGE,SEAL,RF,GBDT,all} [{MF,GCN,GAT,GraphSAGE,SEAL,RF,GBDT,all} ...]]
               [--epochs EPOCHS] [--device DEVICE] [--parallel] [--max-nodes MAX_NODES]

Link Prediction Pipeline

optional arguments:
  -h, --help            show this help message and exit
  --prepare-data        Prepare data for link prediction
  --train-models        Train all models
  --evaluate            Evaluate all models and generate reports
  --all                 Run the complete pipeline
  --models {MF,GCN,GAT,GraphSAGE,SEAL,RF,GBDT,all} [{MF,GCN,GAT,GraphSAGE,SEAL,RF,GBDT,all} ...]
                        Models to train
  --epochs EPOCHS       Number of epochs to train
  --device DEVICE       Device to use for training
  --parallel            Train models in parallel
  --max-nodes MAX_NODES Maximum number of nodes to include in the subgraph
```

## Output

The framework generates the following outputs:

1. Trained model files in the `models/` directory
2. Test metrics for each model
3. A CSV file with a comparison of all models
4. Visualization plots:
   - Model comparison bar charts
   - Radar chart comparing models across multiple metrics
   - Feature importance plots for heuristic models

## Extending the Framework

To add a new link prediction algorithm:

1. Create a new file in the `algorithms/` directory
2. Implement a new class that inherits from `LinkPredictionModel`
3. Implement the required methods: `__init__`, `forward`, etc.
4. Add the model to the `train_model` function in `main.py`

## Improvement Ideas

1. **Hyperparameter Optimization**: Implement grid search or Bayesian optimization for hyperparameters.
2. **Ensemble Methods**: Combine predictions from multiple models for improved accuracy.
3. **More Advanced GNN Architectures**: Implement newer architectures like GraphTransformers.
4. **Node Feature Incorporation**: Add support for node features beyond graph structure.
5. **Explainability**: Add methods to explain link predictions.
6. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation.
7. **Scalability Improvements**: Optimize for larger graphs using minibatch training.
8. **Interactive Visualization**: Create interactive dashboards for exploring results.

## Performance Considerations

- **Memory Usage**: The full graph can be memory-intensive. Use the `--max-nodes` parameter to reduce memory requirements.
- **SEAL Performance**: The SEAL algorithm is computationally expensive due to subgraph extraction. By default, it uses a smaller subset of the data for training.
- **Parallel Training**: Enable `--parallel` to train models concurrently, but be aware that this will increase memory usage.

## License

This project is licensed under the MIT License.

## Acknowledgments

This framework is built using DGL (Deep Graph Library) and PyTorch for efficient graph neural network implementations. 