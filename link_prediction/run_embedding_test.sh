#!/bin/bash

# Run a test of DeepWalk and Node2Vec methods
echo "Running link prediction with embedding methods"
echo "This will train DeepWalk and Node2Vec models on a small subset of data"

# Enable GPU usage if available
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    DEVICE="cuda"
    echo "Using GPU with CUDA for training"
else
    DEVICE="cpu"
    echo "Using CPU for training"
fi

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

# Run the pipeline with a small subset of data and only the embedding models
cd "$SCRIPT_DIR"
python main.py --all --max-nodes 500 --epochs 1 --models DeepWalk Node2Vec --device $DEVICE

echo "Done! The embedding models have been saved to 'link_prediction/models/'"
echo "Check 'link_prediction/metrics.py' to evaluate their performance" 