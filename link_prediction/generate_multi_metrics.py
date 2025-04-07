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

def create_bar_chart(df, metric='AUC', output_file=None):
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
    
    # Save the figure if output_file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to {output_file}")
    
    return plt

def create_multi_metric_chart(df, metrics=['AUC', 'AP', 'F1'], output_file='multi_metric_comparison.png'):
    """Create a grouped bar chart comparing models across multiple metrics"""
    # Get top N models by average of the selected metrics
    N = min(6, len(df))
    df['avg_metric'] = df[metrics].mean(axis=1)
    top_models = df.sort_values('avg_metric', ascending=False).head(N)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Number of metrics and models
    n_metrics = len(metrics)
    n_models = len(top_models)
    
    # Width of a bar 
    width = 0.8 / n_metrics
    
    # Set position of bar on X axis
    positions = np.arange(n_models)
    
    # Colors for different metrics
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create bars for each metric
    for i, metric in enumerate(metrics):
        values = top_models[metric].values
        offset = (i - n_metrics/2 + 0.5) * width
        bars = ax.bar(positions + offset, values, width, label=metric, color=colors[i % len(colors)])
        
        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)
    
    # Set the x-axis tick labels to be the model names
    ax.set_xticks(positions)
    ax.set_xticklabels(top_models['Model'], rotation=45, ha='right')
    
    # Set plot labels and title
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Across Multiple Metrics')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=n_metrics)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Multi-metric chart saved to {output_file}")
    
    return fig

def create_training_time_vs_auc(df, output_file='time_vs_auc.png'):
    """Create a scatter plot of Training Time vs AUC"""
    plt.figure(figsize=(10, 6))
    
    # Create color map based on model types
    colors = []
    for model in df['Model']:
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
    
    # Create the scatter plot
    plt.scatter(df['Training Time (s)'], df['AUC'], c=colors, s=100, alpha=0.7)
    
    # Add model names as labels
    for i, txt in enumerate(df['Model']):
        plt.annotate(txt, (df['Training Time (s)'].iloc[i], df['AUC'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('AUC')
    plt.title('Model Performance vs Training Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Time vs AUC chart saved to {output_file}")
    
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
    
    # Create output directory
    os.makedirs('charts', exist_ok=True)
    
    # Create individual charts for different metrics
    for metric in ['AUC', 'AP', 'F1']:
        output_file = f'charts/{metric.lower()}_comparison.png'
        create_bar_chart(df, metric, output_file)
    
    # Create multi-metric chart
    create_multi_metric_chart(df, ['AUC', 'AP', 'F1'], 'charts/multi_metric_comparison.png')
    
    # Create training time vs AUC chart
    create_training_time_vs_auc(df, 'charts/time_vs_auc.png')
    
    # Print the DataFrame for verification
    print("\nMetrics DataFrame (sorted by AUC):")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 