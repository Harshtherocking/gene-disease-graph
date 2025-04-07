#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

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

def generate_report(metric_files):
    """Generate a comprehensive report of all metrics"""
    all_metrics = []
    
    for file_path in metric_files:
        metrics = load_metrics(file_path)
        if metrics is None:
            continue
        
        model_name = extract_model_name(file_path)
        
        # Extract key metrics
        metric_row = {
            'Model': model_name,
            'AUC': metrics.get('auc', 0),
            'AP': metrics.get('ap', 0),
            'F1': metrics.get('f1', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0)
        }
        
        all_metrics.append(metric_row)
    
    # Create a DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by AUC
    df = df.sort_values('AUC', ascending=False)
    
    return df

def print_table(df):
    """Print a formatted table of metrics"""
    # Round numeric columns to 4 decimal places
    numeric_cols = ['AUC', 'AP', 'F1', 'Precision', 'Recall']
    df[numeric_cols] = df[numeric_cols].round(4)
    
    # Convert to tabulate format
    table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
    
    print("\n=== Link Prediction Models Performance Metrics ===\n")
    print(table)
    
    return table

def plot_comparison(df, metric='AUC', output_file=None):
    """Generate a bar plot comparing models on a specific metric"""
    plt.figure(figsize=(10, 6))
    
    # Sort by the specific metric
    df_sorted = df.sort_values(metric)
    
    # Create color map - different colors for different model types
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
    
    # Create bar plot
    bars = plt.barh(df_sorted['Model'], df_sorted[metric], color=colors)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center')
    
    plt.xlabel(metric)
    plt.ylabel('Model')
    plt.title(f'Model Comparison by {metric}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")
    
    plt.show()

def save_report_to_markdown(df, output_file='link_prediction/metrics_report.md'):
    """Save the report to a markdown file"""
    with open(output_file, 'w') as f:
        f.write("# Link Prediction Models Performance Report\n\n")
        f.write("## Performance Metrics\n\n")
        
        # Format the table for markdown
        md_table = df.to_markdown(index=False)
        f.write(md_table)
        f.write("\n\n")
        
        # Add interpretation
        f.write("## Interpretation\n\n")
        
        best_model = df.iloc[0]['Model']
        best_auc = df.iloc[0]['AUC']
        best_ap = df.iloc[0]['AP']
        
        f.write(f"The best performing model is **{best_model}** with an AUC of {best_auc:.4f} and AP of {best_ap:.4f}.\n\n")
        
        # Compare GNN vs traditional methods
        gnn_models = df[df['Model'].str.contains('GCN|GAT|GraphSAGE|SEAL')]
        trad_models = df[~df['Model'].str.contains('GCN|GAT|GraphSAGE|SEAL')]
        
        if not gnn_models.empty and not trad_models.empty:
            avg_gnn_auc = gnn_models['AUC'].mean()
            avg_trad_auc = trad_models['AUC'].mean()
            
            f.write(f"On average, GNN-based models achieve an AUC of {avg_gnn_auc:.4f}, ")
            
            if avg_gnn_auc > avg_trad_auc:
                f.write(f"which is {(avg_gnn_auc - avg_trad_auc):.4f} higher than traditional methods.\n\n")
            else:
                f.write(f"which is {(avg_trad_auc - avg_gnn_auc):.4f} lower than traditional methods.\n\n")
        
        # Check if embedding models exist
        emb_models = df[df['Model'].str.contains('DeepWalk|Node2Vec')]
        if not emb_models.empty:
            best_emb = emb_models.iloc[0]['Model']
            best_emb_auc = emb_models.iloc[0]['AUC']
            f.write(f"Among embedding-based models, **{best_emb}** performs best with an AUC of {best_emb_auc:.4f}.\n\n")
        
        f.write("## Recommendations\n\n")
        f.write("Based on the performance metrics:\n\n")
        f.write(f"1. **{best_model}** should be the primary model for gene-disease link prediction tasks.\n")
        f.write("2. Consider ensemble approaches combining the top 2-3 models for potentially improved performance.\n")
        f.write("3. For large-scale applications where computational efficiency is important, evaluate the trade-off between model performance and training/inference time.\n")
    
    print(f"Report saved to {output_file}")

def main():
    # Create output directory if it doesn't exist
    os.makedirs('link_prediction/reports', exist_ok=True)
    
    # Find all metric files
    metric_files = find_metric_files()
    
    if not metric_files:
        print("No metric files found. Please train models first.")
        return
    
    print(f"Found {len(metric_files)} metric files.")
    
    # Generate report
    df = generate_report(metric_files)
    
    # Print table
    print_table(df)
    
    # Save report to markdown
    save_report_to_markdown(df, 'link_prediction/reports/metrics_report.md')
    
    # Generate plots for different metrics
    for metric in ['AUC', 'AP', 'F1']:
        output_file = f'link_prediction/reports/{metric.lower()}_comparison.png'
        plot_comparison(df, metric, output_file)

if __name__ == "__main__":
    main() 