import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from utils.data_preparation import load_graph, split_edges_into_parts
from utils.heuristic_methods import run_traditional_methods_evaluation
from utils.node_embeddings import run_embedding_methods_evaluation
from utils.gnn_methods import run_gnn_methods_evaluation

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_results(results, output_dir):
    """
    Plot and save results.
    
    Args:
        results: List of dictionaries with results
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Plot AUC-ROC scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='auc', hue='split', data=df)
    plt.title('AUC-ROC Scores by Method and Split')
    plt.xlabel('Method')
    plt.ylabel('AUC-ROC')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_roc_scores.png'))
    
    # Plot AP scores
    plt.figure(figsize=(12, 8))
    sns.barplot(x='method', y='ap', hue='split', data=df)
    plt.title('Average Precision Scores by Method and Split')
    plt.xlabel('Method')
    plt.ylabel('Average Precision')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_scores.png'))
    
    # Plot accuracy scores if available
    if 'accuracy' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='method', y='accuracy', hue='split', data=df)
        plt.title('Accuracy Scores by Method and Split')
        plt.xlabel('Method')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_scores.png'))
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Print summary
    print("\nResults Summary:")
    print(df.groupby(['method', 'split'])[['auc', 'ap']].mean().round(4))

def run_pipeline(args):
    """
    Run the complete link prediction pipeline.
    
    Args:
        args: Command line arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load graph
    print("Loading graph...")
    graph, encoder = load_graph()
    print(f"Graph loaded with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Split edges into parts
    print("Splitting edges into parts...")
    all_splits = split_edges_into_parts(
        graph, 
        num_parts=args.num_parts,
        train_ratio=args.train_ratio, 
        val_ratio=args.val_ratio, 
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Initialize results list
    all_results = []
    
    # Process each part
    for part_idx, split_data in enumerate(all_splits):
        print(f"\n{'='*50}")
        print(f"Processing Part {part_idx+1}/{args.num_parts}")
        print(f"{'='*50}")
        
        print(f"Training edges: {len(split_data['train_pos'][0])}")
        print(f"Validation edges: {len(split_data['val_pos'][0])}")
        print(f"Test edges: {len(split_data['test_pos'][0])}")
        
        # Run traditional heuristic methods
        if args.run_heuristics:
            print("\n" + "="*50)
            print(f"Running traditional heuristic methods for Part {part_idx+1}...")
            print("="*50)
            heuristic_results = run_traditional_methods_evaluation(split_data)
            # Add part information to results
            for result in heuristic_results:
                result['part'] = part_idx
            all_results.extend(heuristic_results)
        
        # Run node embedding methods
        if args.run_embeddings:
            print("\n" + "="*50)
            print(f"Running node embedding methods for Part {part_idx+1}...")
            print("="*50)
            embedding_results = run_embedding_methods_evaluation(
                split_data,
                embedding_dim=args.embedding_dim,
                methods=args.embedding_methods
            )
            # Add part information to results
            for result in embedding_results:
                result['part'] = part_idx
            all_results.extend(embedding_results)
        
        # Run GNN methods
        if args.run_gnn:
            print("\n" + "="*50)
            print(f"Running GNN methods for Part {part_idx+1}...")
            print("="*50)
            gnn_results = run_gnn_methods_evaluation(
                split_data,
                in_dim=args.in_dim,
                hidden_dim=args.hidden_dim,
                out_dim=args.out_dim,
                num_layers=args.num_layers,
                gnn_types=args.gnn_types
            )
            # Add part information to results
            for result in gnn_results:
                result['part'] = part_idx
            all_results.extend(gnn_results)
    
    # Plot and save results
    print("\n" + "="*50)
    print("Plotting and saving results...")
    print("="*50)
    plot_results(all_results, args.output_dir)
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Link Prediction Pipeline for Disease-Gene Association Graph")
    
    # Data parameters
    parser.add_argument("--num_parts", type=int, default=5, help="Number of parts to split the data into")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Ratio of training edges within each part")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Ratio of validation edges within each part")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Ratio of test edges within each part")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Method selection
    parser.add_argument("--run_heuristics", action="store_true", help="Run traditional heuristic methods")
    parser.add_argument("--run_embeddings", action="store_true", help="Run node embedding methods")
    parser.add_argument("--run_gnn", action="store_true", help="Run GNN methods")
    
    # Node embedding parameters
    parser.add_argument("--embedding_dim", type=int, default=128, help="Dimension of node embeddings")
    parser.add_argument("--embedding_methods", nargs="+", default=["node2vec", "deepwalk"], 
                        help="Node embedding methods to use")
    
    # GNN parameters
    parser.add_argument("--in_dim", type=int, default=None, help="Input dimension for GNN")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension for GNN")
    parser.add_argument("--out_dim", type=int, default=16, help="Output dimension for GNN")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--gnn_types", nargs="+", default=["gcn", "gat", "sage"], 
                        help="GNN types to use")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no methods are specified, run all
    if not (args.run_heuristics or args.run_embeddings or args.run_gnn):
        args.run_heuristics = True
        args.run_embeddings = True
        args.run_gnn = True
    
    # Run pipeline
    run_pipeline(args) 