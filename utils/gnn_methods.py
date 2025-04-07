import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm
from .data_preparation import load_graph, split_edges_into_parts

class GCNLayer(nn.Module):
    """Graph Convolutional Network layer."""
    
    def __init__(self, in_dim, out_dim, bias=True):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, graph, feat):
        """
        Forward pass.
        
        Args:
            graph: DGL graph
            feat: Node features
            
        Returns:
            Updated node features
        """
        with graph.local_scope():
            # Normalize node features
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            
            # Apply normalization to node features
            feat = feat * norm.unsqueeze(-1)
            
            # Graph convolution
            graph.ndata['h'] = feat
            graph.update_all(
                fn=lambda edges: {'m': edges.src['h']},
                reduce_fn=lambda nodes: {'h': torch.sum(nodes.mailbox['m'], dim=1)}
            )
            h = graph.ndata['h']
            
            # Apply normalization to aggregated features
            h = h * norm.unsqueeze(-1)
            
            # Linear transformation
            out = torch.mm(h, self.weight)
            
            if self.bias is not None:
                out += self.bias
            
            return out

class GATLayer(nn.Module):
    """Graph Attention Network layer."""
    
    def __init__(self, in_dim, out_dim, num_heads=1, feat_drop=0.6, attn_drop=0.6, alpha=0.2):
        super(GATLayer, self).__init__()
        self.gat_layer = dglnn.GATConv(in_dim, out_dim, num_heads, feat_drop, attn_drop, alpha)
    
    def forward(self, graph, feat):
        """
        Forward pass.
        
        Args:
            graph: DGL graph
            feat: Node features
            
        Returns:
            Updated node features
        """
        return self.gat_layer(graph, feat)

class SAGELayer(nn.Module):
    """GraphSAGE layer."""
    
    def __init__(self, in_dim, out_dim, aggregator_type='mean'):
        super(SAGELayer, self).__init__()
        self.sage_layer = dglnn.SAGEConv(in_dim, out_dim, aggregator_type)
    
    def forward(self, graph, feat):
        """
        Forward pass.
        
        Args:
            graph: DGL graph
            feat: Node features
            
        Returns:
            Updated node features
        """
        return self.sage_layer(graph, feat)

class GNNModel(nn.Module):
    """Graph Neural Network model for link prediction."""
    
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, gnn_type='gcn', dropout=0.5):
        super(GNNModel, self).__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers
        
        # Input layer
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        
        # GNN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gcn':
                self.layers.append(GCNLayer(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.layers.append(GATLayer(hidden_dim, hidden_dim))
            elif gnn_type == 'sage':
                self.layers.append(SAGELayer(hidden_dim, hidden_dim))
            else:
                raise ValueError(f"GNN type {gnn_type} not supported. Choose from ['gcn', 'gat', 'sage']")
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, graph, features):
        """
        Forward pass.
        
        Args:
            graph: DGL graph
            features: Node features
            
        Returns:
            Node embeddings
        """
        h = self.input_layer(features)
        
        for i, layer in enumerate(self.layers):
            h_new = layer(graph, h)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Skip connection
            if i < self.num_layers - 1:
                h = h + h_new
            else:
                h = h_new
        
        return self.output_layer(h)

class LinkPredictor(nn.Module):
    """Link prediction model using node embeddings."""
    
    def __init__(self, in_dim, hidden_dim=64, out_dim=1):
        super(LinkPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, u_embeds, v_embeds):
        """
        Forward pass.
        
        Args:
            u_embeds: Source node embeddings
            v_embeds: Destination node embeddings
            
        Returns:
            Link prediction scores
        """
        features = torch.cat([u_embeds, v_embeds], dim=1)
        return self.layers(features).squeeze()

class GNNLinkPredictor:
    """GNN-based link prediction model."""
    
    def __init__(self, graph, in_dim, hidden_dim=64, out_dim=16, num_layers=2, 
                 gnn_type='gcn', dropout=0.5, lr=0.01, weight_decay=5e-4):
        """
        Initialize the model.
        
        Args:
            graph: DGL graph
            in_dim: Input dimension (node feature dimension)
            hidden_dim: Hidden dimension
            out_dim: Output dimension (node embedding dimension)
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn', 'gat', 'sage')
            dropout: Dropout rate
            lr: Learning rate
            weight_decay: Weight decay for optimizer
        """
        self.graph = graph
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize node features if not present
        if 'feat' not in graph.ndata:
            # Use one-hot encoding for node IDs as features
            num_nodes = graph.number_of_nodes()
            graph.ndata['feat'] = torch.eye(num_nodes, dtype=torch.float32)
        
        # Move graph to device
        self.graph = graph.to(self.device)
        
        # Initialize models
        self.gnn = GNNModel(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        ).to(self.device)
        
        self.predictor = LinkPredictor(
            in_dim=out_dim,
            hidden_dim=hidden_dim,
            out_dim=1
        ).to(self.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.Adam(
            list(self.gnn.parameters()) + list(self.predictor.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def train(self, train_pos, train_neg, val_pos=None, val_neg=None, 
              epochs=100, batch_size=1024, patience=10):
        """
        Train the model.
        
        Args:
            train_pos: Training positive edges
            train_neg: Training negative edges
            val_pos: Validation positive edges
            val_neg: Validation negative edges
            epochs: Number of epochs
            batch_size: Batch size
            patience: Early stopping patience
            
        Returns:
            Trained model
        """
        # Get node features
        features = self.graph.ndata['feat'].to(self.device)
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            self.gnn.train()
            self.predictor.train()
            
            # Shuffle training data
            train_pos_u, train_pos_v = train_pos
            train_neg_u, train_neg_v = train_neg
            
            # Combine positive and negative edges
            train_u = torch.cat([train_pos_u, train_neg_u])
            train_v = torch.cat([train_pos_v, train_neg_v])
            train_labels = torch.cat([torch.ones(len(train_pos_u)), torch.zeros(len(train_neg_u))])
            
            # Create batches
            indices = torch.randperm(len(train_u))
            num_batches = (len(indices) + batch_size - 1) // batch_size
            
            total_loss = 0
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]
                
                batch_u = train_u[batch_indices].to(self.device)
                batch_v = train_v[batch_indices].to(self.device)
                batch_labels = train_labels[batch_indices].to(self.device)
                
                # Forward pass
                node_embeddings = self.gnn(self.graph, features)
                u_embeds = node_embeddings[batch_u]
                v_embeds = node_embeddings[batch_v]
                preds = self.predictor(u_embeds, v_embeds)
                
                # Compute loss
                loss = self.loss_fn(preds, batch_labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_indices)
            
            avg_loss = total_loss / len(train_u)
            
            # Evaluate on validation set if provided
            if val_pos is not None and val_neg is not None:
                val_auc, val_ap = self.evaluate(val_pos, val_neg)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
                
                # Early stopping
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def evaluate(self, pos_edges, neg_edges):
        """
        Evaluate the model on positive and negative edges.
        
        Args:
            pos_edges: Positive edges
            neg_edges: Negative edges
            
        Returns:
            AUC-ROC and AP scores
        """
        self.gnn.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            # Get node features
            features = self.graph.ndata['feat'].to(self.device)
            
            # Get node embeddings
            node_embeddings = self.gnn(self.graph, features)
            
            # Get positive and negative edges
            pos_u, pos_v = pos_edges
            neg_u, neg_v = neg_edges
            
            # Move to device
            pos_u = pos_u.to(self.device)
            pos_v = pos_v.to(self.device)
            neg_u = neg_u.to(self.device)
            neg_v = neg_v.to(self.device)
            
            # Get embeddings
            pos_u_embeds = node_embeddings[pos_u]
            pos_v_embeds = node_embeddings[pos_v]
            neg_u_embeds = node_embeddings[neg_u]
            neg_v_embeds = node_embeddings[neg_v]
            
            # Get predictions
            pos_preds = self.predictor(pos_u_embeds, pos_v_embeds)
            neg_preds = self.predictor(neg_u_embeds, neg_v_embeds)
            
            # Combine predictions and labels
            preds = torch.cat([pos_preds, neg_preds]).cpu().numpy()
            labels = torch.cat([torch.ones(len(pos_preds)), torch.zeros(len(neg_preds))]).numpy()
            
            # Calculate metrics
            auc = roc_auc_score(labels, preds)
            ap = average_precision_score(labels, preds)
            
            return auc, ap
    
    def predict(self, u, v):
        """
        Predict link probability between nodes u and v.
        
        Args:
            u: Source node ID
            v: Destination node ID
            
        Returns:
            Link probability
        """
        self.gnn.eval()
        self.predictor.eval()
        
        with torch.no_grad():
            # Get node features
            features = self.graph.ndata['feat'].to(self.device)
            
            # Get node embeddings
            node_embeddings = self.gnn(self.graph, features)
            
            # Get embeddings for u and v
            u_embed = node_embeddings[u].unsqueeze(0)
            v_embed = node_embeddings[v].unsqueeze(0)
            
            # Get prediction
            pred = self.predictor(u_embed, v_embed)
            
            # Apply sigmoid to get probability
            prob = torch.sigmoid(pred).item()
            
            return prob

def run_gnn_methods_evaluation(split_data, in_dim=None, hidden_dim=64, out_dim=16, 
                              num_layers=2, gnn_types=['gcn', 'gat', 'sage']):
    """
    Run evaluation for GNN methods.
    
    Args:
        split_data: Dictionary containing train/val/test splits
        in_dim: Input dimension (node feature dimension)
        hidden_dim: Hidden dimension
        out_dim: Output dimension (node embedding dimension)
        num_layers: Number of GNN layers
        gnn_types: List of GNN types to evaluate
        
    Returns:
        Dictionary with results for each method
    """
    print("Evaluating GNN methods...")
    
    # Get graph
    graph = split_data['train_graph']
    
    # Set input dimension if not provided
    if in_dim is None:
        if 'feat' in graph.ndata:
            in_dim = graph.ndata['feat'].size(1)
        else:
            in_dim = graph.number_of_nodes()  # One-hot encoding
    
    results = []
    
    for gnn_type in gnn_types:
        print(f"\nTraining {gnn_type.upper()} model...")
        
        # Initialize model
        model = GNNLinkPredictor(
            graph=graph,
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            gnn_type=gnn_type
        )
        
        # Train model
        model.train(
            train_pos=split_data['train_pos'],
            train_neg=split_data['train_neg'],
            val_pos=split_data['val_pos'],
            val_neg=split_data['val_neg'],
            epochs=100,
            patience=10
        )
        
        # Evaluate on validation set
        print(f"Validation set results for {gnn_type.upper()}:")
        val_auc, val_ap = model.evaluate(split_data['val_pos'], split_data['val_neg'])
        print(f"  AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
        results.append({
            'method': gnn_type,
            'split': 'validation',
            'auc': val_auc,
            'ap': val_ap
        })
        
        # Evaluate on test set
        print(f"Test set results for {gnn_type.upper()}:")
        test_auc, test_ap = model.evaluate(split_data['test_pos'], split_data['test_neg'])
        print(f"  AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        results.append({
            'method': gnn_type,
            'split': 'test',
            'auc': test_auc,
            'ap': test_ap
        })
    
    return results

if __name__ == "__main__":
    # Load data
    graph, encoder = load_graph()
    all_splits = split_edges_into_parts(graph, num_parts=5)
    
    # Run evaluation on first part
    results = run_gnn_methods_evaluation(all_splits[0]) 