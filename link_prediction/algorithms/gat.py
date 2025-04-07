import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class GATLayer(nn.Module):
    """
    Graph Attention Network Layer
    """
    def __init__(self, in_dim, out_dim, num_heads=4, feat_drop=0.0, attn_drop=0.0, residual=True):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # Residual connection
        self.residual = residual
        if residual:
            self.res_fc = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        if residual:
            nn.init.xavier_uniform_(self.res_fc.weight)
    
    def forward(self, g, h):
        h_in = h  # For residual connection
        h = self.feat_drop(h)
        
        # Linear transformation
        feat = self.fc(h).view(-1, self.num_heads, self._out_feats)
        
        # Calculate attention scores
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        
        # Add attention scores to graph
        g.ndata.update({'ft': feat, 'el': el, 'er': er})
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(g.edata.pop('e'))
        
        # Attention weights
        g.edata['a'] = self.attn_drop(torch.softmax(e, dim=1))
        
        # Message passing
        g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = g.ndata['ft']
        
        # Residual connection
        if self.residual:
            resval = self.res_fc(h_in).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        
        return rst

class GATLinkPrediction(LinkPredictionModel):
    """
    Graph Attention Network for link prediction
    """
    def __init__(self, graph, hidden_dim=128, n_layers=2, num_heads=4, dropout=0.1, save_dir=None):
        super(GATLinkPrediction, self).__init__('GAT', graph, hidden_dim, save_dir)
        self.num_nodes = graph.num_nodes()
        
        # Node initial features (random embedding if no features)
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_embeddings.weight)
        
        # GAT layers
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(GATLayer(hidden_dim, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        # Hidden layers
        for _ in range(1, n_layers - 1):
            self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, feat_drop=dropout, attn_drop=dropout))
        # Output layer
        self.layers.append(GATLayer(hidden_dim * num_heads, hidden_dim, 1, feat_drop=dropout, attn_drop=dropout))
        
        self.dropout = nn.Dropout(dropout)
        self.prediction = nn.Linear(hidden_dim * 2, 1)  # Prediction layer
        
        # Fix the _out_feats attribute in GAT layers
        for i, layer in enumerate(self.layers):
            layer._out_feats = hidden_dim
    
    def encode(self):
        """
        Encode nodes using GAT
        """
        h = self.node_embeddings.weight.to(self.graph.device)
        
        # Apply GAT layers
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
                # Reshape multi-head outputs
                h = h.reshape(h.shape[0], -1)
                h = F.elu(h)
            else:
                # For the last layer, take the average of heads
                h = h.mean(1)
        
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
    model = GATLinkPrediction(data_split['graph'])
    
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