#!/bin/bash

# Run the link prediction pipeline with a small subset of the data
echo "Running link prediction with a small subset of the data (1000 nodes)"
echo "This will complete in a reasonable amount of time for testing purposes"

# Enable GPU usage
# Remove the line that disables CUDA devices
# export CUDA_VISIBLE_DEVICES=""
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

# Run the pipeline with a small subset of data
cd "$SCRIPT_DIR"
python main.py --all --max-nodes 1000 --epochs 10 --device cuda

echo "Done! Check the 'link_prediction' directory for results."
echo "The model comparison will be available in 'link_prediction/model_comparison.csv'"
echo "Visualizations will be in 'link_prediction/model_comparison.png' and 'link_prediction/radar_comparison.png'" 