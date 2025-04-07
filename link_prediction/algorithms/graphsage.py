import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE Layer
    """
    def __init__(self, in_dim, out_dim, aggregator_type='mean', dropout=0.0, batch_norm=True):
        super(GraphSAGELayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator_type = aggregator_type
        self.batch_norm = batch_norm
        
        # Message functions for different aggregators
        if aggregator_type == 'mean':
            self.fc_self = nn.Linear(in_dim, out_dim, bias=True)
            self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        elif aggregator_type == 'gcn':
            self.fc = nn.Linear(in_dim, out_dim, bias=True)
        elif aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_dim, in_dim, bias=True)
            self.fc_self = nn.Linear(in_dim, out_dim, bias=True)
            self.fc_neigh = nn.Linear(in_dim, out_dim, bias=False)
        else:
            raise ValueError("Aggregator type not supported: {}".format(aggregator_type))
        
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.reset_parameters()
    
    def reset_parameters(self):
        """
        Initialize parameters
        """
        if self.aggregator_type == 'mean':
            nn.init.xavier_uniform_(self.fc_self.weight)
            nn.init.xavier_uniform_(self.fc_neigh.weight)
        elif self.aggregator_type == 'gcn':
            nn.init.xavier_uniform_(self.fc.weight)
        elif self.aggregator_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight)
            nn.init.xavier_uniform_(self.fc_self.weight)
            nn.init.xavier_uniform_(self.fc_neigh.weight)
    
    def forward(self, g, h):
        """
        Forward computation
        """
        h_self = h
        
        # Aggregate messages from neighbors
        if self.aggregator_type == 'mean':
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = g.ndata['neigh']
            h = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        
        elif self.aggregator_type == 'gcn':
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            h_neigh = g.ndata['neigh']
            # GCN-like aggregation
            h = self.fc(h_self + h_neigh)
        
        elif self.aggregator_type == 'pool':
            g.ndata['h'] = self.activation(self.fc_pool(h))
            g.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = g.ndata['neigh']
            h = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        
        # Apply batch normalization and activation
        if self.batch_norm:
            h = self.bn(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        return h

class GraphSAGELinkPrediction(LinkPredictionModel):
    """
    GraphSAGE for link prediction
    """
    def __init__(self, graph, hidden_dim=128, n_layers=2, dropout=0.1, aggregator_type='mean', save_dir=None):
        super(GraphSAGELinkPrediction, self).__init__('GraphSAGE', graph, hidden_dim, save_dir)
        self.num_nodes = graph.num_nodes()
        
        # Node initial features (random embedding if no features)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # GraphSAGE layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator_type, dropout))
        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator_type, dropout))
        # Output layer
        self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator_type, dropout))
        
        self.prediction = nn.Linear(hidden_dim * 2, 1)  # Prediction layer
    
    def encode(self):
        """
        Encode nodes using GraphSAGE
        """
        h = self.node_embeddings.weight.to(self.graph.device)
        
        # Apply GraphSAGE layers
        for layer in self.layers:
            # Make sure the graph is on the same device as the model
            if self.graph.device != h.device:
                local_graph = self.graph.to(h.device)
            else:
                local_graph = self.graph
                
            h = layer(local_graph, h)
        
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
    model = GraphSAGELinkPrediction(data_split['graph'])
    
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