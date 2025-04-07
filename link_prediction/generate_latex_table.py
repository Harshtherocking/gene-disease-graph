#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd

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
            'Training Time (s)': metrics.get('training_time', 0)
        }
        
        all_metrics.append(metric_row)
    
    # Create a DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Sort by AUC
    df = df.sort_values('AUC', ascending=False)
    
    return df

def generate_latex_table(df, output_file='paper/performance_table.tex'):
    """Generate a LaTeX table of the metrics"""
    # Round numeric columns
    df['AUC'] = df['AUC'].round(3)
    df['AP'] = df['AP'].round(3)
    df['F1'] = df['F1'].round(3)
    df['Training Time (s)'] = df['Training Time (s)'].round(2)
    
    # Convert DataFrame to LaTeX
    latex_table = """\\begin{table}[!t]
\\caption{Performance Comparison of Different Models}
\\label{table_performance}
\\centering
\\begin{tabular}{lcccc}
\\toprule
\\textbf{Model} & \\textbf{AUC} & \\textbf{AP} & \\textbf{F1} & \\textbf{Training Time (s)} \\\\
\\midrule
"""
    
    # Add rows
    for _, row in df.iterrows():
        latex_table += f"{row['Model']} & {row['AUC']:.3f} & {row['AP']:.3f} & {row['F1']:.3f} & {row['Training Time (s)']:.2f} \\\\\n"
    
    # Close the table
    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"LaTeX table saved to {output_file}")
    return latex_table

def main():
    # Find all metric files
    metric_files = find_metric_files()
    
    if not metric_files:
        print("No metric files found. Please train models first.")
        return
    
    print(f"Found {len(metric_files)} metric files.")
    
    # Generate report
    df = generate_report(metric_files)
    
    # Generate LaTeX table
    generate_latex_table(df, 'paper/performance_table.tex')
    
    # Print the DataFrame for verification
    print("\nMetrics DataFrame:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main() 