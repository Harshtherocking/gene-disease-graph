#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_metric_files(base_dir='models'):
    """Find all metric files in the models directory"""
    metric_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_test_metrics.pkl'):
                metric_files.append(os.path.join(root, file))
    return metric_files

def load_metrics(file_path):
    """Load metrics from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            metrics = pickle.load(f)
        return metrics
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def extract_model_name(file_path):
    """Extract model name from file path"""
    base_name = os.path.basename(file_path)
    model_name = base_name.replace('_test_metrics.pkl', '')
    return model_name

def clean_model_name(model_name):
    """Convert model name to a nicer format for the paper"""
    if model_name == 'GCNLinkPrediction':
        return 'GCN'
    elif model_name == 'GATLinkPrediction':
        return 'GAT'
    elif model_name == 'GraphSAGELinkPrediction':
        return 'GraphSAGE'
    elif model_name == 'SEALLinkPrediction':
        return 'SEAL'
    elif model_name == 'HeuristicLinkPrediction':
        return 'Heuristic (RF)'
    elif model_name == 'RF':
        return 'Random Forest'
    elif model_name == 'GBDT':
        return 'GBDT'
    elif model_name == 'MatrixFactorization':
        return 'MF'
    else:
        return model_name

def generate_report(metric_files):
    """Generate a comprehensive report of all metrics"""
    all_metrics = []
    
    for file_path in metric_files:
        metrics = load_metrics(file_path)
        if metrics is None:
            continue
        
        model_name = extract_model_name(file_path)
        model_name = clean_model_name(model_name)
        
        # Extract key metrics
        metric_row = {
            'Model': model_name,
            'AUC': metrics.get('auc', 0),
            'AP': metrics.get('ap', 0),
            'F1': metrics.get('f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'Training Time (s)': metrics.get('training_time', 0)
        }
        
        all_metrics.append(metric_row)
    
    # Create a DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Remove duplicates keeping the highest AUC value for each model
    df = df.sort_values('AUC', ascending=False).drop_duplicates('Model')
    
    # Sort by AUC
    df = df.sort_values('AUC', ascending=False)
    
    return df

def create_bar_chart(df, metric='AUC', output_file='model_comparison.png'):
    """Create a horizontal bar chart comparing models on a specific metric"""
    plt.figure(figsize=(10, 6))
    
    # Sort by the metric
    df_sorted = df.sort_values(metric, ascending=True)
    
    # Create color map based on model types
    colors = []
    for model in df_sorted['Model']:
        if 'GCN' in model:
            colors.append('#1f77b4')  # blue for GCN
        elif 'GAT' in model:
            colors.append('#ff7f0e')  # orange for GAT
        elif 'GraphSAGE' in model:
            colors.append('#2ca02c')  # green for GraphSAGE
        elif 'SEAL' in model:
            colors.append('#d62728')  # red for SEAL
        elif 'DeepWalk' in model or 'Node2Vec' in model:
            colors.append('#9467bd')  # purple for embedding methods
        else:
            colors.append('#8c564b')  # brown for others
    
    # Create the bar chart
    bars = plt.barh(df_sorted['Model'], df_sorted[metric], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.xlabel(metric)
    plt.ylabel('Model')
    plt.title(f'Model Comparison by {metric}')
    plt.xlim(0, 1.05)  # For probabilities (AUC, AP)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_file}")
    
    return plt

def main():
    # Find all metric files
    metric_files = find_metric_files()
    
    if not metric_files:
        print("No metric files found. Please train models first.")
        return
    
    print(f"Found {len(metric_files)} metric files.")
    
    # Generate report
    df = generate_report(metric_files)
    
    # Create charts for different metrics
    create_bar_chart(df, 'AUC', 'model_comparison.png')
    
    # Print the DataFrame for verification
    print("\nMetrics DataFrame (sorted by AUC):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 