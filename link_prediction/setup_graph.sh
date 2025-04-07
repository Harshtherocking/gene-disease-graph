#!/bin/bash

# Get the script directory and project root
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Setting up the GRAPH file for link prediction..."

# Check if GRAPH file exists in the project root
if [ -f "$PROJECT_ROOT/GRAPH" ]; then
    echo "Found GRAPH file at $PROJECT_ROOT/GRAPH"
    
    # Create a symbolic link in the link_prediction directory
    echo "Creating a symbolic link in the link_prediction directory..."
    ln -sf "$PROJECT_ROOT/GRAPH" "$SCRIPT_DIR/GRAPH"
    
    if [ -f "$SCRIPT_DIR/GRAPH" ]; then
        echo "Setup complete! The GRAPH file is now accessible to the link prediction scripts."
    else
        echo "Failed to create symbolic link. Please check file permissions."
    fi
else
    echo "ERROR: GRAPH file not found at $PROJECT_ROOT/GRAPH"
    echo "Please make sure the GRAPH file is in the project root directory before running this script."
    echo "The file should be at: $PROJECT_ROOT/GRAPH"
    exit 1
fi

echo "You can now run the link prediction scripts:"
echo "  ./run_quick_test.sh    - For a quick test with just the GCN model"
echo "  ./run_small_example.sh - For a more comprehensive evaluation with all models" 