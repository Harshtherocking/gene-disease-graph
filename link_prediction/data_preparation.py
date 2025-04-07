import os
import dgl
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random
import pickle

def load_graph(max_nodes=5000):
    """
    Load the DGL graph from the saved file and extract a small subgraph
    
    Args:
        max_nodes: Maximum number of nodes to include in the subgraph
    """
    # Try multiple possible locations for the graph file
    possible_paths = [
        os.path.join(os.getcwd(), "GRAPH"),  # Root directory
        os.path.join(os.getcwd(), "link_prediction", "GRAPH"),  # link_prediction directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GRAPH")  # Parent directory
    ]
    
    graph_path = None
    for path in possible_paths:
        if os.path.exists(path):
            graph_path = path
            break
    
    if graph_path is None:
        raise FileNotFoundError(f"Graph file not found in any of the expected locations: {possible_paths}")
    
    print(f"Loading graph from {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    full_graph = graphs[0]
    print(f"Original graph has {full_graph.num_nodes()} nodes and {full_graph.num_edges()} edges")
    
    # Generate a subgraph with a limited number of nodes for faster processing
    if max_nodes < full_graph.num_nodes():
        # Randomly sample nodes
        all_nodes = torch.arange(full_graph.num_nodes())
        selected_nodes = torch.tensor(random.sample(range(full_graph.num_nodes()), max_nodes))
        
        # Create a subgraph
        subgraph = dgl.node_subgraph(full_graph, selected_nodes)
        print(f"Created subgraph with {subgraph.num_nodes()} nodes and {subgraph.num_edges()} edges")
        return subgraph
    else:
        print(f"Using full graph with {full_graph.num_nodes()} nodes")
        return full_graph

def split_edge_data(graph, val_ratio=0.1, test_ratio=0.2, random_state=42):
    """
    Split the edges in the graph into training, validation and test sets
    Returns a dictionary with the original graph and masks for training, validation and test
    """
    # Get all edges from the graph
    u, v = graph.edges()
    edge_ids = torch.arange(graph.num_edges())
    
    # Create edge list
    edge_list = list(zip(u.tolist(), v.tolist(), edge_ids.tolist()))
    
    # Split edges for validation and test
    train_edges, test_edges = train_test_split(
        edge_list, test_size=test_ratio, random_state=random_state
    )
    
    if val_ratio > 0:
        train_edges, val_edges = train_test_split(
            train_edges, test_size=val_ratio/(1-test_ratio), random_state=random_state
        )
    else:
        val_edges = []
        
    # Create masks for the edges
    train_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    val_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    test_mask = torch.zeros(graph.num_edges(), dtype=torch.bool)
    
    # Set masks based on edge IDs
    train_mask[[edge[2] for edge in train_edges]] = True
    val_mask[[edge[2] for edge in val_edges]] = True
    test_mask[[edge[2] for edge in test_edges]] = True
    
    # Generate negative edges for training, validation and test
    train_neg_edges = generate_negative_edges(graph, len(train_edges))
    val_neg_edges = generate_negative_edges(graph, len(val_edges))
    test_neg_edges = generate_negative_edges(graph, len(test_edges))
    
    data_split = {
        'graph': graph,
        'train_pos_u': torch.tensor([edge[0] for edge in train_edges]),
        'train_pos_v': torch.tensor([edge[1] for edge in train_edges]),
        'train_neg_u': torch.tensor([edge[0] for edge in train_neg_edges]),
        'train_neg_v': torch.tensor([edge[1] for edge in train_neg_edges]),
        'val_pos_u': torch.tensor([edge[0] for edge in val_edges]),
        'val_pos_v': torch.tensor([edge[1] for edge in val_edges]),
        'val_neg_u': torch.tensor([edge[0] for edge in val_neg_edges]),
        'val_neg_v': torch.tensor([edge[1] for edge in val_neg_edges]),
        'test_pos_u': torch.tensor([edge[0] for edge in test_edges]),
        'test_pos_v': torch.tensor([edge[1] for edge in test_edges]),
        'test_neg_u': torch.tensor([edge[0] for edge in test_neg_edges]),
        'test_neg_v': torch.tensor([edge[1] for edge in test_neg_edges]),
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask
    }
    
    return data_split

def generate_negative_edges(graph, num_samples):
    """
    Generate negative edges (not in the original graph)
    """
    num_nodes = graph.num_nodes()
    existing_edges = set(zip(graph.edges()[0].tolist(), graph.edges()[1].tolist()))
    
    neg_edges = []
    while len(neg_edges) < num_samples:
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        
        # Skip self-loops and existing edges
        if src != dst and (src, dst) not in existing_edges and (dst, src) not in existing_edges:
            neg_edges.append((src, dst))
            # Add to existing edges to avoid duplicates
            existing_edges.add((src, dst))
    
    return neg_edges

def prepare_data_for_link_prediction(max_nodes=2000):
    """
    Main function to prepare data for link prediction
    
    Args:
        max_nodes: Maximum number of nodes to include in the subgraph
    """
    graph = load_graph(max_nodes=max_nodes)
    data_split = split_edge_data(graph)
    
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save the split data
    output_dir = script_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "link_prediction_data.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(data_split, f)
    
    print(f"Data prepared and saved to {output_path}")
    print(f"Train: {len(data_split['train_pos_u'])} positive edges, {len(data_split['train_neg_u'])} negative edges")
    print(f"Val: {len(data_split['val_pos_u'])} positive edges, {len(data_split['val_neg_u'])} negative edges")
    print(f"Test: {len(data_split['test_pos_u'])} positive edges, {len(data_split['test_neg_u'])} negative edges")
    
    return data_split

if __name__ == "__main__":
    prepare_data_for_link_prediction(max_nodes=2000) 