#!/bin/bash

# Run a fast link prediction test with a single model on GPU
echo "Running link prediction quick test using GPU"
echo "This will train just a single model (GCN) on a tiny subset of data"

# Enable GPU usage
echo "Using GPU with CUDA for training"

# Get the script directory and project root
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Check if GRAPH file exists
if [ -f "$PROJECT_ROOT/GRAPH" ]; then
    echo "Found graph file at $PROJECT_ROOT/GRAPH"
else
    echo "Warning: Graph file not found at $PROJECT_ROOT/GRAPH"
    echo "Make sure the GRAPH file is in the project root directory."
fi

# Activate the virtualenv
echo "Activating graphresearch virtualenv"
source "$PROJECT_ROOT/graphresearch/bin/activate"

# Run the pipeline with a tiny subset of data and only the GCN model
cd "$SCRIPT_DIR"
python main.py --all --max-nodes 500 --epochs 5 --models GCN --device cuda

echo "Done! The test model has been saved to 'link_prediction/models/'"
echo "Check 'link_prediction/metrics.py' to evaluate its performance" 