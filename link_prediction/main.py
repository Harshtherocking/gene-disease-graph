import os
import sys
import argparse
import torch
import dgl
import pickle
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

# Import the data preparation module
from data_preparation import prepare_data_for_link_prediction

# Import all models
from algorithms.matrix_factorization import MatrixFactorization
from algorithms.gcn import GCNLinkPrediction
from algorithms.gat import GATLinkPrediction
from algorithms.graphsage import GraphSAGELinkPrediction
from algorithms.seal import SEALLinkPrediction
from algorithms.heuristic_methods import HeuristicLinkPrediction
from algorithms.embedding_methods import DeepWalk, Node2Vec

# Import metrics evaluation
from metrics import run_all_evaluations

def parse_args():
    parser = argparse.ArgumentParser(description='Link Prediction Pipeline')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare data for link prediction')
    parser.add_argument('--train-models', action='store_true', help='Train all models')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate all models and generate reports')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--models', nargs='+', type=str, 
                        choices=['MF', 'GCN', 'GAT', 'GraphSAGE', 'SEAL', 'RF', 'GBDT', 'DeepWalk', 'Node2Vec', 'all'],
                        default=['all'], help='Models to train')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument('--parallel', action='store_true', help='Train models in parallel')
    parser.add_argument('--max-nodes', type=int, default=2000, 
                        help='Maximum number of nodes to include in the subgraph')
    return parser.parse_args()

def load_data(max_nodes=2000):
    """
    Load the processed data for link prediction
    """
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define data path relative to the script directory
    data_path = os.path.join(script_dir, "link_prediction_data.pkl")
    
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Preparing data...")
        data_split = prepare_data_for_link_prediction(max_nodes=max_nodes)
    else:
        print(f"Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            data_split = pickle.load(f)
        
        # Check if we need to recreate the data with fewer nodes
        current_nodes = data_split['graph'].num_nodes()
        if current_nodes > max_nodes:
            print(f"Current data uses {current_nodes} nodes, recreating with max {max_nodes} nodes")
            data_split = prepare_data_for_link_prediction(max_nodes=max_nodes)
    
    return data_split

def train_model(model_name, data_split, epochs=20, device='cpu'):
    """
    Train a specific model
    """
    print(f"Training {model_name} model...")
    start_time = time.time()
    
    # Create and train the model with optimized parameters for faster training
    if model_name == 'MF':
        model = MatrixFactorization(data_split['graph'], hidden_dim=64)
        model.train_model(data_split, epochs=epochs, batch_size=256, device=device)
    elif model_name == 'GCN':
        model = GCNLinkPrediction(data_split['graph'], hidden_dim=64, n_layers=2)
        model.train_model(data_split, epochs=epochs, batch_size=256, device=device)
    elif model_name == 'GAT':
        model = GATLinkPrediction(data_split['graph'], hidden_dim=64, n_layers=2, num_heads=2)
        model.train_model(data_split, epochs=epochs, batch_size=256, device=device)
    elif model_name == 'GraphSAGE':
        model = GraphSAGELinkPrediction(data_split['graph'], hidden_dim=64, n_layers=2)
        model.train_model(data_split, epochs=epochs, batch_size=256, device=device)
    elif model_name == 'SEAL':
        # Use even smaller hop size and fewer training samples for SEAL as it's computationally intensive
        model = SEALLinkPrediction(data_split['graph'], hidden_dim=32, hop=1)
        
        # For SEAL, we'll use a smaller subset of the data to speed up training
        seal_data_split = {k: v for k, v in data_split.items()}
        
        # Limit training samples for SEAL
        max_samples = min(1000, len(data_split['train_pos_u']))
        indices = torch.randperm(len(data_split['train_pos_u']))[:max_samples]
        seal_data_split['train_pos_u'] = data_split['train_pos_u'][indices]
        seal_data_split['train_pos_v'] = data_split['train_pos_v'][indices]
        indices = torch.randperm(len(data_split['train_neg_u']))[:max_samples]
        seal_data_split['train_neg_u'] = data_split['train_neg_u'][indices]
        seal_data_split['train_neg_v'] = data_split['train_neg_v'][indices]
        
        # Limit validation samples
        if len(data_split['val_pos_u']) > 0:
            max_val_samples = min(500, len(data_split['val_pos_u']))
            indices = torch.randperm(len(data_split['val_pos_u']))[:max_val_samples]
            seal_data_split['val_pos_u'] = data_split['val_pos_u'][indices]
            seal_data_split['val_pos_v'] = data_split['val_pos_v'][indices]
            indices = torch.randperm(len(data_split['val_neg_u']))[:max_val_samples]
            seal_data_split['val_neg_u'] = data_split['val_neg_u'][indices]
            seal_data_split['val_neg_v'] = data_split['val_neg_v'][indices]
        
        # Limit test samples for evaluation
        max_test_samples = min(500, len(data_split['test_pos_u']))
        indices = torch.randperm(len(data_split['test_pos_u']))[:max_test_samples]
        seal_data_split['test_pos_u'] = data_split['test_pos_u'][indices]
        seal_data_split['test_pos_v'] = data_split['test_pos_v'][indices]
        indices = torch.randperm(len(data_split['test_neg_u']))[:max_test_samples]
        seal_data_split['test_neg_u'] = data_split['test_neg_u'][indices]
        seal_data_split['test_neg_v'] = data_split['test_neg_v'][indices]
        
        # When using CUDA, adjust batch size based on available GPU memory
        batch_size = 64
        if device == 'cuda':
            # Check available GPU memory and adjust batch size
            if torch.cuda.is_available():
                try:
                    free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                    free_memory_gb = free_memory / (1024 ** 3)  # Convert to GB
                    
                    # Adjust batch size based on available memory
                    if free_memory_gb > 8:  # More than 8GB free
                        batch_size = 128
                    elif free_memory_gb > 4:  # More than 4GB free
                        batch_size = 96
                    else:  # Less memory available
                        batch_size = 32
                    
                    print(f"Adjusted SEAL batch size to {batch_size} based on {free_memory_gb:.2f}GB available GPU memory")
                except:
                    # Fall back to default if there's any issue
                    batch_size = 64
                    print("Could not determine GPU memory, using default SEAL batch size")
        
        print(f"Using a smaller subset for SEAL: {len(seal_data_split['train_pos_u'])} train, {len(seal_data_split['test_pos_u'])} test samples")
        model.train_model(seal_data_split, epochs=min(10, epochs), batch_size=batch_size, device=device)
    elif model_name == 'RF':
        # Simplify feature extraction for RF to make it faster
        model = HeuristicLinkPrediction(
            data_split['graph'], 
            model_type='randomforest',
            include_features=['common_neighbors', 'jaccard_coefficient', 'preferential_attachment']
        )
        model.train_model(data_split)
    elif model_name == 'GBDT':
        # Simplify feature extraction for GBDT to make it faster
        model = HeuristicLinkPrediction(
            data_split['graph'], 
            model_type='gbdt',
            include_features=['common_neighbors', 'jaccard_coefficient', 'preferential_attachment']
        )
        model.train_model(data_split)
    elif model_name == 'DeepWalk':
        # Use optimized parameters for DeepWalk
        model = DeepWalk(
            data_split['graph'], 
            embedding_dim=64, 
            walk_length=40,  # Shorter walks for efficiency
            num_walks=5,     # Fewer walks
            window_size=5
        )
        model.train_model(data_split, device=device)
    elif model_name == 'Node2Vec':
        # Use optimized parameters for Node2Vec
        model = Node2Vec(
            data_split['graph'], 
            embedding_dim=64, 
            walk_length=40,  # Shorter walks for efficiency
            num_walks=5,     # Fewer walks
            window_size=5,
            p=1.0,           # Return parameter
            q=2.0            # In-out parameter (favor outward exploration)
        )
        model.train_model(data_split, device=device)
    else:
        print(f"Unknown model: {model_name}")
        return
    
    # Evaluate on test set
    test_auc, test_ap = model.evaluate(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v'],
        device
    )
    
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save detailed metrics
    test_metrics = model.get_metrics(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v'],
        device
    )
    
    # Save test metrics
    metrics_path = os.path.join(model.save_dir, f"{model.name}_test_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    
    elapsed_time = time.time() - start_time
    print(f"Model {model_name} trained and evaluated in {elapsed_time:.2f} seconds")
    
    return test_metrics

def train_all_models(args):
    """
    Train all selected models
    """
    print("Training models...")
    data_split = load_data(args.max_nodes)
    
    # Determine which models to train
    models_to_train = []
    if 'all' in args.models:
        models_to_train = ['MF', 'GCN', 'GAT', 'GraphSAGE', 'SEAL', 'RF', 'GBDT', 'DeepWalk', 'Node2Vec']
    else:
        models_to_train = args.models
    
    print(f"Training the following models: {', '.join(models_to_train)}")
    
    # Train models
    if args.parallel:
        # Train models in parallel using process pool
        print("Training models in parallel...")
        # Get the number of available cores
        num_cores = mp.cpu_count()
        print(f"Using {num_cores} cores for parallel training")
        
        # Create a process pool
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all training jobs
            futures = [executor.submit(train_model, model_name, data_split, args.epochs, args.device) 
                      for model_name in models_to_train]
            
            # Wait for all jobs to complete and get results
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error during model training: {e}")
    else:
        # Train models sequentially
        for model_name in models_to_train:
            try:
                train_model(model_name, data_split, args.epochs, args.device)
            except Exception as e:
                print(f"Error training {model_name}: {e}")
    
    print("All models trained successfully!")

def main():
    """
    Main function to run the link prediction pipeline
    """
    args = parse_args()
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directories
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Prepare data if requested
    if args.prepare_data or args.all:
        print(f"Preparing data for link prediction (max_nodes={args.max_nodes})...")
        prepare_data_for_link_prediction(max_nodes=args.max_nodes)
    
    # Train models if requested
    if args.train_models or args.all:
        print("Training models...")
        train_all_models(args)
    
    # Evaluate models if requested
    if args.evaluate or args.all:
        print("Evaluating models and generating reports...")
        run_all_evaluations()
    
    if not args.prepare_data and not args.train_models and not args.evaluate and not args.all:
        print("No action specified. Use --help to see available options.")

if __name__ == "__main__":
    main() 