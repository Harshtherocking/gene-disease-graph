import os
import dgl
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split
from .graphloader import GraphLoader

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)

def load_graph():
    """Load the disease-gene association graph."""
    # Get the project directory
    home = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define input and output paths
    input_file = os.path.join(home, "DG-Miner_miner-disease-gene.tsv")
    output_file = os.path.join(home, "GRAPH")
    
    # Check if graph already exists
    if os.path.exists(output_file):
        print(f"Loading existing graph from {output_file}...")
        graph_list, _ = dgl.load_graphs(output_file)
        graph = graph_list[0]
        # Load the encoder from graphloader to get mappings
        graph_loader = GraphLoader(input_file, output_file)
        graph_loader.load_data()  # This will populate the encoder
        encoder = graph_loader.encoder
    else:
        print(f"Creating new graph from {input_file}...")
        graph_loader = GraphLoader(input_file, output_file)
        graph_loader.process()
        graph_list, _ = dgl.load_graphs(output_file)
        graph = graph_list[0]
        encoder = graph_loader.encoder
    
    return graph, encoder

def split_edges_into_parts(graph, num_parts=5, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split the graph edges into 5 parts, then split each part into training, validation and test sets.
    Also generates negative edges for training.
    
    Args:
        graph: DGL graph
        num_parts: Number of parts to split the data into
        train_ratio: Ratio of training edges within each part
        val_ratio: Ratio of validation edges within each part
        test_ratio: Ratio of test edges within each part
        seed: Random seed
        
    Returns:
        List of dictionaries, each containing train/val/test splits for one part
    """
    set_seed(seed)
    
    # Get all edges
    u, v = graph.edges()
    eids = np.arange(graph.number_of_edges())
    
    # Shuffle edge IDs
    np.random.shuffle(eids)
    
    # Split into num_parts
    part_size = len(eids) // num_parts
    parts = []
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = start_idx + part_size if i < num_parts - 1 else len(eids)
        parts.append(eids[start_idx:end_idx])
    
    # For each part, split into train/val/test
    all_splits = []
    for part_idx, part_eids in enumerate(parts):
        print(f"Processing part {part_idx+1}/{num_parts} with {len(part_eids)} edges")
        
        # Split edge IDs within this part
        train_size = int(len(part_eids) * train_ratio)
        val_size = int(len(part_eids) * val_ratio)
        
        # Shuffle and split
        np.random.shuffle(part_eids)
        train_eids = part_eids[:train_size]
        val_eids = part_eids[train_size:train_size+val_size]
        test_eids = part_eids[train_size+val_size:]
        
        # Create train/val/test masks
        train_mask = torch.zeros(graph.number_of_edges(), dtype=torch.bool)
        val_mask = torch.zeros(graph.number_of_edges(), dtype=torch.bool)
        test_mask = torch.zeros(graph.number_of_edges(), dtype=torch.bool)
        
        train_mask[train_eids] = True
        val_mask[val_eids] = True
        test_mask[test_eids] = True
        
        # Create training subgraph
        train_graph = dgl.edge_subgraph(graph, train_eids)
        
        # Get positive edges for each split
        train_pos_u, train_pos_v = graph.find_edges(train_eids)
        val_pos_u, val_pos_v = graph.find_edges(val_eids)
        test_pos_u, test_pos_v = graph.find_edges(test_eids)
        
        # Function to sample negative edges
        def sample_negative_edges(num_samples, existing_edges):
            existing_edges_set = set(zip(existing_edges[0].tolist(), existing_edges[1].tolist()))
            neg_u, neg_v = [], []
            while len(neg_u) < num_samples:
                # Sample random source and destination nodes
                u = np.random.randint(0, graph.number_of_nodes())
                v = np.random.randint(0, graph.number_of_nodes())
                
                # Check if edge exists
                if (u, v) not in existing_edges_set and u != v:
                    neg_u.append(u)
                    neg_v.append(v)
            
            return torch.tensor(neg_u), torch.tensor(neg_v)
        
        # Sample negative edges for training, validation, and testing
        train_neg_u, train_neg_v = sample_negative_edges(len(train_pos_u), (u, v))
        val_neg_u, val_neg_v = sample_negative_edges(len(val_pos_u), (u, v))
        test_neg_u, test_neg_v = sample_negative_edges(len(test_pos_u), (u, v))
        
        # Package results for this part
        part_result = {
            'part_idx': part_idx,
            'full_graph': graph,
            'train_graph': train_graph,
            'train_pos': (train_pos_u, train_pos_v),
            'train_neg': (train_neg_u, train_neg_v),
            'val_pos': (val_pos_u, val_pos_v),
            'val_neg': (val_neg_u, val_neg_v),
            'test_pos': (test_pos_u, test_pos_v),
            'test_neg': (test_neg_u, test_neg_v),
        }
        
        all_splits.append(part_result)
    
    return all_splits

def get_node_types(graph, encoder):
    """
    Determine the type (disease or gene) for each node.
    Returns a dictionary mapping node ID to node type.
    """
    node_types = {}
    for node_id, name in encoder.id_to_name.items():
        # Assuming disease names and gene names have different patterns
        # You may need to adjust this logic based on your actual data
        if name in encoder.name_to_id:
            if "MESH" in name:  # Adjust this condition based on your disease naming convention
                node_types[node_id] = "disease"
            else:
                node_types[node_id] = "gene"
    
    return node_types

if __name__ == "__main__":
    # Example usage
    graph, encoder = load_graph()
    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    # Split into 5 parts
    all_splits = split_edges_into_parts(graph, num_parts=5)
    
    # Print summary for each part
    for part_idx, split_data in enumerate(all_splits):
        print(f"\nPart {part_idx+1}:")
        print(f"  Training edges: {len(split_data['train_pos'][0])}")
        print(f"  Validation edges: {len(split_data['val_pos'][0])}")
        print(f"  Test edges: {len(split_data['test_pos'][0])}") 