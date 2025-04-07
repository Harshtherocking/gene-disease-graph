import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import sys
import os
from tqdm import tqdm
from scipy.sparse import csr_matrix
import networkx as nx

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class GCNConv(nn.Module):
    """
    Basic GCN convolution layer for SEAL
    """
    def __init__(self, in_dim, out_dim):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, g, x):
        with g.local_scope():
            g.ndata['h'] = x
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
            h = g.ndata['h']
            return self.linear(h)

class DGCNN(nn.Module):
    """
    Deep Graph Convolutional Neural Network for SEAL
    Adapted from the paper "An End-to-End Deep Learning Architecture for Graph Classification"
    """
    def __init__(self, in_dim, hidden_dims=[32, 32, 32, 1], k=0.6):
        super(DGCNN, self).__init__()
        self.k = k  # Ratio of nodes to keep in SortPooling
        
        # GCN layers
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GCNConv(in_dim, hidden_dims[0]))
        
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(GCNConv(hidden_dims[i-1], hidden_dims[i]))
        
        # MLP layers
        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.Linear(hidden_dims[-1] * 2, 128))
        self.mlp_layers.append(nn.Linear(128, 64))
        self.mlp_layers.append(nn.Linear(64, 1))
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(dim) for dim in hidden_dims
        ])
    
    def forward(self, g, h):
        # List to store all node representations
        h_list = [h]
        
        # Apply GCN layers with batch normalization and ReLU
        for i, conv in enumerate(self.conv_layers):
            h = F.relu(self.batch_norms[i](conv(g, h)))
            h_list.append(h)
        
        # Concatenate all GCN outputs (similar to DenseNet connections)
        h_final = torch.cat(h_list, dim=1)
        
        # Global mean pooling
        with g.local_scope():
            g.ndata['h'] = h_final
            h_graph = dgl.mean_nodes(g, 'h')
        
        # Apply MLP layers
        for i, mlp in enumerate(self.mlp_layers):
            h_graph = mlp(h_graph)
            if i < len(self.mlp_layers) - 1:
                h_graph = F.relu(h_graph)
        
        return h_graph.view(-1)

class SEALLinkPrediction(LinkPredictionModel):
    """
    SEAL (Subgraph Extraction And Learning) model for link prediction
    """
    def __init__(self, graph, hidden_dim=32, hop=1, save_dir=None):
        super(SEALLinkPrediction, self).__init__('SEAL', graph, hidden_dim, save_dir)
        self.num_nodes = graph.num_nodes()
        self.hop = hop  # Number of hops to include in enclosing subgraph
        
        # Convert DGL graph to NetworkX for subgraph extraction
        self.nx_graph = dgl.to_networkx(graph, node_attrs=None, edge_attrs=None)
        
        # DGCNN model for subgraph classification
        # Input dimension = node label dimension (hop + 1 for structrual identity + 1 for edge existence)
        self.model = DGCNN(in_dim=self.hop + 2, hidden_dims=[32, 32, 32, 32])
        
        # Store subgraphs for faster access
        self.subgraph_cache = {}
    
    def extract_enclosing_subgraphs(self, u_list, v_list, batch_size=512):
        """
        Extract enclosing subgraphs for a batch of node pairs
        """
        subgraphs = []
        node_labels = []
        
        for idx in range(0, len(u_list), batch_size):
            batch_u = u_list[idx:idx+batch_size]
            batch_v = v_list[idx:idx+batch_size]
            
            for u, v in zip(batch_u, batch_v):
                # Check if subgraph is in cache
                cache_key = (min(u.item(), v.item()), max(u.item(), v.item()))
                if cache_key in self.subgraph_cache:
                    subgraph, label = self.subgraph_cache[cache_key]
                else:
                    # Extract enclosing subgraph
                    subgraph, label = self._extract_subgraph(u.item(), v.item())
                    # Cache subgraph
                    self.subgraph_cache[cache_key] = (subgraph, label)
                
                subgraphs.append(subgraph)
                node_labels.append(label)
        
        # Batch subgraphs
        batched_graph = dgl.batch(subgraphs)
        node_labels = torch.cat(node_labels, dim=0)
        
        return batched_graph, node_labels
    
    def _extract_subgraph(self, u, v):
        """
        Extract the enclosing subgraph around nodes u and v
        """
        # Get the enclosing subgraph
        nodes = set([u, v])
        for _ in range(self.hop):
            neighbors = set()
            for node in nodes:
                neighbors.update(self.nx_graph.neighbors(node))
            nodes.update(neighbors)
        
        # Create the subgraph
        subgraph = self.nx_graph.subgraph(list(nodes))
        
        # Convert to DGL graph
        dgl_subgraph = dgl.from_networkx(subgraph)
        
        # Compute Double-Radius Node Labeling (DRNL)
        node_labels = self._double_radius_node_labeling(subgraph, u, v)
        
        # Add an additional channel for structural identities (1 for u and v, 0 for others)
        if u in subgraph and v in subgraph:
            # Map original node ids to subgraph node ids
            node_map = {orig_id: i for i, orig_id in enumerate(subgraph.nodes())}
            u_idx, v_idx = node_map[u], node_map[v]
            structural_labels = torch.zeros(dgl_subgraph.num_nodes())
            structural_labels[u_idx] = 1
            structural_labels[v_idx] = 1
            
            # Combine node labels and structural labels
            dgl_subgraph.ndata['feat'] = torch.cat([
                node_labels.unsqueeze(1),
                structural_labels.unsqueeze(1)
            ], dim=1)
        else:
            # If u or v is not in the subgraph (should not happen with enclosing subgraphs)
            dgl_subgraph.ndata['feat'] = torch.cat([
                node_labels.unsqueeze(1),
                torch.zeros(dgl_subgraph.num_nodes(), 1)
            ], dim=1)
        
        return dgl_subgraph, dgl_subgraph.ndata['feat']
    
    def _double_radius_node_labeling(self, subgraph, src, dst):
        """
        Double Radius Node Labeling (DRNL) as described in the SEAL paper
        """
        # Map original node ids to subgraph node ids
        node_map = {orig_id: i for i, orig_id in enumerate(subgraph.nodes())}
        
        # If src or dst is not in the subgraph, return all zeros
        if src not in node_map or dst not in node_map:
            return torch.zeros(len(node_map))
        
        src_idx, dst_idx = node_map[src], node_map[dst]
        
        # Initialize distances
        nx_subgraph = nx.Graph(subgraph)
        src_distances = nx.single_source_shortest_path_length(nx_subgraph, src)
        dst_distances = nx.single_source_shortest_path_length(nx_subgraph, dst)
        
        labels = []
        for node in subgraph.nodes():
            node_idx = node_map[node]
            d_src = src_distances.get(node, self.hop + 1)
            d_dst = dst_distances.get(node, self.hop + 1)
            
            if d_src <= self.hop and d_dst <= self.hop:
                # DRNL formula
                label = 1 + min(d_src, d_dst) + (d_src + d_dst) * (self.hop + 1) // 2
            else:
                # Node is outside the hop limit
                label = 0
            
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.float)
    
    def forward(self, u, v):
        """
        Forward pass
        """
        # Handle batching
        if len(u.shape) == 0:
            u = u.unsqueeze(0)
            v = v.unsqueeze(0)
        
        # Extract enclosing subgraphs
        subgraphs, node_labels = self.extract_enclosing_subgraphs(u, v)
        
        # Pass through the model
        scores = self.model(subgraphs, node_labels)
        
        return scores

if __name__ == "__main__":
    import pickle
    
    # Load data
    with open('../link_prediction_data.pkl', 'rb') as f:
        data_split = pickle.load(f)
    
    # Create model
    model = SEALLinkPrediction(data_split['graph'], hop=1)
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    model.train_model(data_split, epochs=50, batch_size=64, device=device)
    
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
    
    print("Test Metrics:", test_metrics)
    
    # Save test metrics
    metrics_path = os.path.join(model.save_dir, f"{model.name}_test_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f) 