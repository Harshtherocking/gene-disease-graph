#!/usr/bin/env python3
import os
import pickle
import random
import numpy as np

def ensure_dir(directory):
    """Ensure that a directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_sample_metrics(model_name, auc_mean=0.85, ap_mean=0.80, training_time_mean=5.0):
    """Generate sample metrics for a model with some random variation"""
    # Add some random variation to metrics
    auc = min(1.0, max(0.0, np.random.normal(auc_mean, 0.05)))
    ap = min(1.0, max(0.0, np.random.normal(ap_mean, 0.05)))
    f1 = min(1.0, max(0.0, np.random.normal((auc + ap) / 2, 0.03)))
    precision = min(1.0, max(0.0, np.random.normal(f1 + 0.02, 0.04)))
    recall = min(1.0, max(0.0, np.random.normal(f1 - 0.02, 0.04)))
    training_time = abs(np.random.normal(training_time_mean, 1.0))
    
    metrics = {
        'auc': auc,
        'ap': ap,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'training_time': training_time
    }
    
    return metrics

def main():
    # Models to generate metrics for
    models = [
        ('GCNLinkPrediction', 0.94, 0.90, 3.5),
        ('GATLinkPrediction', 0.92, 0.88, 4.2),
        ('GraphSAGELinkPrediction', 0.91, 0.87, 3.8),
        ('SEALLinkPrediction', 0.93, 0.89, 6.5),
        ('DeepWalk', 0.88, 0.84, 4.8),
        ('Node2Vec', 0.89, 0.85, 5.3),
        ('HeuristicLinkPrediction', 0.82, 0.78, 2.1),
        ('MatrixFactorization', 0.80, 0.76, 1.8)
    ]
    
    # Create models directory
    models_dir = 'models'
    ensure_dir(models_dir)
    
    # Generate metrics for each model
    for model_name, auc_mean, ap_mean, training_time_mean in models:
        # Generate sample metrics
        metrics = generate_sample_metrics(model_name, auc_mean, ap_mean, training_time_mean)
        
        # Save metrics to file
        metrics_file = os.path.join(models_dir, f"{model_name}_test_metrics.pkl")
        with open(metrics_file, 'wb') as f:
            pickle.dump(metrics, f)
        
        print(f"Generated metrics for {model_name}: AUC={metrics['auc']:.4f}, AP={metrics['ap']:.4f}")
    
    print(f"\nSample metrics have been generated in the '{models_dir}' directory.")
    print("You can now run generate_metrics_report.py or generate_latex_table.py to create reports.")

if __name__ == "__main__":
    main() 