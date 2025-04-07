import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    """
    def __init__(self, in_dim, out_dim):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        
    def forward(self, g, h):
        # Normalize features by degree
        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).unsqueeze(1)
        h = h * norm
        
        # Message passing
        g.ndata['h'] = h
        g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h = g.ndata.pop('h')
        
        # Apply normalization on the output
        h = h * norm
        
        # Linear transformation
        h = self.linear(h)
        return h

class GCNLinkPrediction(LinkPredictionModel):
    """
    Graph Convolutional Network for link prediction
    """
    def __init__(self, graph, hidden_dim=128, n_layers=2, dropout=0.1, save_dir=None):
        super(GCNLinkPrediction, self).__init__('GCN', graph, hidden_dim, save_dir)
        self.num_nodes = graph.num_nodes()
        
        # Node initial features (random embedding if no features)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # GCN layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        # Output layer
        self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Linear(hidden_dim * 2, 1)  # Prediction layer
    
    def encode(self):
        """
        Encode nodes using GCN
        """
        # Ensure node embeddings are on the same device as the model
        h = self.node_embeddings.weight.to(self.graph.device)
        
        # Apply GCN layers
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            # Make sure the graph is on the same device as the model
            if self.graph.device != h.device:
                local_graph = self.graph.to(h.device)
            else:
                local_graph = self.graph
            h = layer(local_graph, h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
        
        return h
    
    def forward(self, u, v):
        # Encode all nodes
        h = self.encode()
        
        # Get node embeddings
        u_embeddings = h[u]
        v_embeddings = h[v]
        
        # Concatenate embeddings and predict
        edge_features = torch.cat([u_embeddings, v_embeddings], dim=1)
        scores = self.prediction(edge_features).squeeze(1)
        
        return scores

if __name__ == "__main__":
    import pickle
    
    # Load data
    with open('../link_prediction_data.pkl', 'rb') as f:
        data_split = pickle.load(f)
    
    # Create model
    model = GCNLinkPrediction(data_split['graph'])
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    model.train_model(data_split, epochs=100, device=device)
    
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