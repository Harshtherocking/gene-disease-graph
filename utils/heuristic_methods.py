import dgl
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import networkx as nx
from .data_preparation import load_graph, split_edges_into_parts

class HeuristicLinkPredictor:
    def __init__(self, graph):
        """Initialize with a DGL graph."""
        self.graph = graph
        # Convert to NetworkX for easier heuristic computation
        self.nx_graph = dgl.to_networkx(graph, edge_attrs=None).to_undirected()
    
    def _get_common_neighbors_score(self, u, v):
        """Calculate common neighbors score."""
        u_neighbors = set(self.nx_graph.neighbors(u.item()))
        v_neighbors = set(self.nx_graph.neighbors(v.item()))
        return len(u_neighbors.intersection(v_neighbors))
    
    def _get_jaccard_score(self, u, v):
        """Calculate Jaccard coefficient."""
        u_neighbors = set(self.nx_graph.neighbors(u.item()))
        v_neighbors = set(self.nx_graph.neighbors(v.item()))
        intersection = len(u_neighbors.intersection(v_neighbors))
        union = len(u_neighbors.union(v_neighbors))
        return intersection / union if union > 0 else 0
    
    def _get_adamic_adar_score(self, u, v):
        """Calculate Adamic-Adar index."""
        u_neighbors = set(self.nx_graph.neighbors(u.item()))
        v_neighbors = set(self.nx_graph.neighbors(v.item()))
        common_neighbors = u_neighbors.intersection(v_neighbors)
        
        score = 0
        for w in common_neighbors:
            # Calculate 1/log(degree) for each common neighbor
            degree = self.nx_graph.degree(w)
            if degree > 1:  # Avoid log(1) which is 0
                score += 1 / np.log(degree)
        
        return score
    
    def _get_preferential_attachment_score(self, u, v):
        """Calculate preferential attachment score."""
        u_degree = self.nx_graph.degree(u.item())
        v_degree = self.nx_graph.degree(v.item())
        return u_degree * v_degree
    
    def calculate_scores(self, u_list, v_list, method="common_neighbors"):
        """
        Calculate scores for given node pairs using the specified method.
        
        Args:
            u_list, v_list: Lists of source and destination nodes
            method: One of "common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"
            
        Returns:
            List of scores
        """
        score_func = {
            "common_neighbors": self._get_common_neighbors_score,
            "jaccard": self._get_jaccard_score,
            "adamic_adar": self._get_adamic_adar_score,
            "preferential_attachment": self._get_preferential_attachment_score
        }
        
        if method not in score_func:
            raise ValueError(f"Method {method} not supported. Choose from {list(score_func.keys())}")
        
        scores = []
        for u, v in zip(u_list, v_list):
            scores.append(score_func[method](u, v))
        
        return torch.tensor(scores)
    
    def evaluate(self, pos_edges, neg_edges, method="common_neighbors"):
        """
        Evaluate the link prediction performance on positive and negative edges.
        
        Args:
            pos_edges: Tuple of (u, v) tensors for positive edges
            neg_edges: Tuple of (u, v) tensors for negative edges
            method: Heuristic method to use
            
        Returns:
            Dictionary with AUC-ROC, AP, and accuracy metrics
        """
        pos_u, pos_v = pos_edges
        neg_u, neg_v = neg_edges
        
        # Calculate scores
        pos_scores = self.calculate_scores(pos_u, pos_v, method)
        neg_scores = self.calculate_scores(neg_u, neg_v, method)
        
        # Combine scores and create labels
        scores = torch.cat([pos_scores, neg_scores]).numpy()
        labels = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))]).numpy()
        
        # Calculate metrics
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        
        # For accuracy, we need to binarize the scores
        median_score = np.median(scores)
        binary_preds = (scores >= median_score).astype(int)
        acc = accuracy_score(labels, binary_preds)
        
        return {
            "auc": auc,
            "ap": ap,
            "accuracy": acc,
            "method": method
        }

def run_traditional_methods_evaluation(split_data):
    """
    Run evaluation for all traditional heuristic methods.
    
    Args:
        split_data: Dictionary containing train/val/test splits
        
    Returns:
        Dictionary with results for each method
    """
    print("Evaluating traditional heuristic methods...")
    
    # Initialize predictor
    predictor = HeuristicLinkPredictor(split_data['full_graph'])
    
    # Methods to evaluate
    methods = ["common_neighbors", "jaccard", "adamic_adar", "preferential_attachment"]
    
    # Store results
    results = []
    
    # Evaluate on validation set
    print("Validation set results:")
    for method in methods:
        result = predictor.evaluate(
            split_data['val_pos'], 
            split_data['val_neg'],
            method
        )
        print(f"  {method}: AUC = {result['auc']:.4f}, AP = {result['ap']:.4f}, Accuracy = {result['accuracy']:.4f}")
        result['split'] = 'validation'
        results.append(result)
    
    # Evaluate on test set
    print("Test set results:")
    for method in methods:
        result = predictor.evaluate(
            split_data['test_pos'], 
            split_data['test_neg'],
            method
        )
        print(f"  {method}: AUC = {result['auc']:.4f}, AP = {result['ap']:.4f}, Accuracy = {result['accuracy']:.4f}")
        result['split'] = 'test'
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Load data
    graph, encoder = load_graph()
    all_splits = split_edges_into_parts(graph, num_parts=5)
    
    # Run evaluation on first part
    results = run_traditional_methods_evaluation(all_splits[0]) 