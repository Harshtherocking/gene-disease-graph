import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class MatrixFactorization(LinkPredictionModel):
    """
    Simple Matrix Factorization model for link prediction
    Each node is represented by a learned embedding. The score for an edge (u, v)
    is computed as the dot product of their embeddings.
    """
    def __init__(self, graph, hidden_dim=128, save_dir=None):
        super(MatrixFactorization, self).__init__('MatrixFactorization', graph, hidden_dim, save_dir)
        self.num_nodes = graph.num_nodes()
        
        # Node embeddings
        self.node_embeddings = nn.Embedding(self.num_nodes, hidden_dim)
        # Initialize embeddings
        nn.init.xavier_uniform_(self.node_embeddings.weight)
    
    def forward(self, u, v):
        # Get node embeddings
        u_embeddings = self.node_embeddings(u)
        v_embeddings = self.node_embeddings(v)
        
        # Compute dot product
        scores = torch.sum(u_embeddings * v_embeddings, dim=1)
        
        return scores

if __name__ == "__main__":
    import pickle
    
    # Load data
    with open('../link_prediction_data.pkl', 'rb') as f:
        data_split = pickle.load(f)
    
    # Create model
    model = MatrixFactorization(data_split['graph'])
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train model
    model.train_model(data_split, epochs=50, device=device)
    
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
    import os
    metrics_path = os.path.join(model.save_dir, f"{model.name}_test_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f) 