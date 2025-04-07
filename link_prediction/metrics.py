import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import torch
import sys
import glob

# Import models
from algorithms.matrix_factorization import MatrixFactorization
from algorithms.gcn import GCNLinkPrediction
from algorithms.gat import GATLinkPrediction
from algorithms.graphsage import GraphSAGELinkPrediction
from algorithms.seal import SEALLinkPrediction
from algorithms.heuristic_methods import HeuristicLinkPrediction

def load_data():
    """
    Load the processed data for link prediction
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "link_prediction_data.pkl")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    with open(data_path, 'rb') as f:
        data_split = pickle.load(f)
    
    return data_split

def load_model(model_class, model_path, data_split, **kwargs):
    """
    Load a trained model
    """
    model = model_class(data_split['graph'], **kwargs)
    if model.load_model():
        return model
    else:
        return None

def collect_all_metrics():
    """
    Collect all metrics from trained models
    """
    # Load data
    data_split = load_data()
    
    # Dictionary to store model names and their metrics
    metrics_dict = {}
    
    # Models to evaluate
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Find all test metrics files
    metric_files = glob.glob(os.path.join(models_dir, "*_test_metrics.pkl"))
    
    # Load metrics from each file
    for metric_file in metric_files:
        model_name = os.path.basename(metric_file).replace("_test_metrics.pkl", "")
        with open(metric_file, 'rb') as f:
            metrics = pickle.load(f)
        metrics_dict[model_name] = metrics
    
    return metrics_dict

def collect_training_metrics():
    """
    Collect training metrics from all models
    """
    # Dictionary to store model names and their training metrics
    training_metrics = {}
    
    # Models to evaluate
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, "models")
    
    # Find all training metrics files
    metric_files = glob.glob(os.path.join(models_dir, "*_training_metrics.pkl"))
    
    # Load metrics from each file
    for metric_file in metric_files:
        model_name = os.path.basename(metric_file).replace("_training_metrics.pkl", "")
        with open(metric_file, 'rb') as f:
            metrics = pickle.load(f)
        training_metrics[model_name] = metrics
    
    return training_metrics

def generate_comparison_report():
    """
    Generate a comparison report of all trained models
    """
    metrics_dict = collect_all_metrics()
    training_metrics = collect_training_metrics()
    
    if not metrics_dict:
        print("No metrics found. Train models first.")
        return
    
    # Create a dataframe for comparison
    model_names = list(metrics_dict.keys())
    metrics_df = pd.DataFrame(index=model_names)
    
    # Add test metrics
    for model_name, metrics in metrics_dict.items():
        for metric_name, value in metrics.items():
            metrics_df.loc[model_name, f"test_{metric_name}"] = value
    
    # Add training time if available
    for model_name, metrics in training_metrics.items():
        if model_name in metrics_df.index:  # Only process models that are in the metrics_df
            if 'training_time' in metrics:
                metrics_df.loc[model_name, 'training_time'] = metrics['training_time']
            
            # Add validation metrics if available
            if 'val_metrics' in metrics and metrics['val_metrics']:
                # In the training process, val_metrics is stored as lists for each metric
                # {'auc': [0.8, 0.85, 0.9], 'ap': [0.7, 0.75, 0.8]}
                # Take the last value from each list as the final validation metric
                if isinstance(metrics['val_metrics'], dict):
                    for metric_name, values in metrics['val_metrics'].items():
                        if isinstance(values, list) and values:
                            # Use the last value in the list
                            metrics_df.loc[model_name, f"val_{metric_name}"] = values[-1]
    
    # Sort by test AUC (descending)
    if 'test_auc' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('test_auc', ascending=False)
    
    # Save comparison to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_df.to_csv(os.path.join(output_dir, "model_comparison.csv"))
    
    # Print comparison table
    print("Link Prediction Model Comparison:")
    print("="*80)
    print(metrics_df)
    
    # Generate visualizations
    plot_model_comparison(metrics_df)
    
    return metrics_df

def plot_model_comparison(metrics_df):
    """
    Generate comparison plots
    """
    if metrics_df is None or metrics_df.empty:
        print("No metrics available for plotting.")
        return
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Plot AUC comparison
    if 'test_auc' in metrics_df.columns:
        ax = plt.subplot(2, 2, 1)
        sns.barplot(x=metrics_df.index, y='test_auc', data=metrics_df, palette="viridis")
        plt.title("AUC Comparison", fontsize=14)
        plt.ylabel("AUC", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylim(0.5, 1.0)
    
    # Plot AP comparison
    if 'test_ap' in metrics_df.columns:
        ax = plt.subplot(2, 2, 2)
        sns.barplot(x=metrics_df.index, y='test_ap', data=metrics_df, palette="viridis")
        plt.title("Average Precision Comparison", fontsize=14)
        plt.ylabel("Average Precision", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylim(0.5, 1.0)
    
    # Plot F1 comparison
    if 'test_f1' in metrics_df.columns:
        ax = plt.subplot(2, 2, 3)
        sns.barplot(x=metrics_df.index, y='test_f1', data=metrics_df, palette="viridis")
        plt.title("F1 Score Comparison", fontsize=14)
        plt.ylabel("F1 Score", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.ylim(0.0, 1.0)
    
    # Plot training time comparison if available
    if 'training_time' in metrics_df.columns:
        ax = plt.subplot(2, 2, 4)
        sns.barplot(x=metrics_df.index, y='training_time', data=metrics_df, palette="viridis")
        plt.title("Training Time Comparison (seconds)", fontsize=14)
        plt.ylabel("Training Time (s)", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a radar plot for model comparison if all necessary metrics are available
    metrics_to_plot = ['test_auc', 'test_ap', 'test_f1', 'test_precision', 'test_recall']
    
    # Check if all required metrics exist
    all_metrics_available = all(metric in metrics_df.columns for metric in metrics_to_plot)
    
    if all_metrics_available:
        plot_radar_chart(metrics_df, metrics_to_plot, output_dir)
    else:
        print("Skipping radar chart - some required metrics are missing.")
    
    print(f"Visualizations saved to {output_dir}")

def plot_radar_chart(metrics_df, metrics_to_plot, output_dir):
    """
    Create a radar chart for comparing models across multiple metrics
    """
    # Handle missing values in the dataframe
    radar_df = metrics_df.copy()
    for metric in metrics_to_plot:
        if metric not in radar_df.columns:
            radar_df[metric] = 0.0
        else:
            # Fill NaN values with 0
            radar_df[metric] = radar_df[metric].fillna(0.0)
    
    # Number of metrics
    N = len(metrics_to_plot)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create the plot
    ax = plt.subplot(111, polar=True)
    
    # Add labels for each metric
    plt.xticks(angles[:-1], [m.replace('test_', '') for m in metrics_to_plot], fontsize=12)
    
    # Set y-limits
    ax.set_ylim(0, 1)
    
    # Add metrics for each model
    for i, model in enumerate(radar_df.index):
        values = radar_df.loc[model, metrics_to_plot].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Plot the values
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    
    plt.title("Model Comparison Across Metrics", fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_feature_importance_plot():
    """
    Generate a plot of feature importance for the heuristic models
    """
    training_metrics = collect_training_metrics()
    
    # Filter out models with feature importance
    feature_importance_models = {}
    for model_name, metrics in training_metrics.items():
        if 'feature_importance' in metrics and metrics['feature_importance'] is not None:
            feature_importance_models[model_name] = metrics['feature_importance']
    
    if not feature_importance_models:
        print("No feature importance data found.")
        return
    
    output_dir = os.path.join(os.getcwd(), "link_prediction")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dataframe for feature importance
    feature_names = list(next(iter(feature_importance_models.values())).keys())
    importance_df = pd.DataFrame(index=feature_names)
    
    for model_name, importance in feature_importance_models.items():
        importance_df[model_name] = pd.Series(importance)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    importance_df.plot(kind='bar', figsize=(12, 8))
    plt.title("Feature Importance Comparison", fontsize=15)
    plt.ylabel("Importance", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title="Models")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_dir}")

def run_all_evaluations():
    """
    Run all evaluations and generate reports
    """
    print("Generating model comparison report...")
    metrics_df = generate_comparison_report()
    
    print("\nGenerating feature importance plot...")
    generate_feature_importance_plot()
    
    print("\nEvaluation complete. Results saved to the link_prediction directory.")
    
    return metrics_df

if __name__ == "__main__":
    run_all_evaluations() 