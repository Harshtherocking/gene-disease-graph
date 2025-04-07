import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import sys
import os
import networkx as nx
import dgl
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import pickle
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class HeuristicFeatures:
    """
    Extract heuristic features for link prediction
    """
    def __init__(self, graph, include_features=None):
        self.graph = graph
        self.nx_graph = dgl.to_networkx(graph, node_attrs=None, edge_attrs=None)
        self.adj = nx.to_scipy_sparse_array(self.nx_graph, dtype=np.float32)
        
        # List of available features
        self.available_features = [
            'common_neighbors', 'jaccard_coefficient', 'adamic_adar', 
            'preferential_attachment', 'resource_allocation',
            'total_neighbors', 'shortest_path', 'katz'
        ]
        
        # Set features to extract
        if include_features is None:
            self.include_features = self.available_features
        else:
            self.include_features = include_features
            
        # Precompute some measures
        self._precompute()
    
    def _precompute(self):
        """
        Precompute some measures for efficiency
        """
        # Get degree of each node
        self.degree = np.array(self.nx_graph.degree())[:, 1].astype(np.float32)
        
        # Compute shortest path lengths for small graphs
        if self.nx_graph.number_of_nodes() < 10000 and 'shortest_path' in self.include_features:
            self.shortest_paths = dict(nx.all_pairs_shortest_path_length(self.nx_graph))
        else:
            self.shortest_paths = None
        
        # Compute Katz scores if needed
        if 'katz' in self.include_features:
            try:
                # Use a relatively small beta to avoid convergence issues
                beta = 0.01
                self.katz_matrix = np.linalg.inv(np.eye(self.nx_graph.number_of_nodes()) - beta * nx.to_numpy_array(self.nx_graph)) - np.eye(self.nx_graph.number_of_nodes())
            except:
                print("Warning: Katz centrality computation failed. Removing it from features.")
                self.include_features.remove('katz')
                self.katz_matrix = None
        else:
            self.katz_matrix = None
    
    def get_features(self, src_nodes, dst_nodes):
        """
        Extract features for a batch of node pairs
        """
        num_pairs = len(src_nodes)
        features = np.zeros((num_pairs, len(self.include_features)), dtype=np.float32)
        
        for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
            features[i] = self._get_node_pair_features(src, dst)
        
        return features
    
    def _get_node_pair_features(self, src, dst):
        """
        Extract features for a single node pair
        """
        features = []
        
        # Common neighbors
        if 'common_neighbors' in self.include_features:
            common_neighbors = len(sorted(nx.common_neighbors(self.nx_graph, src, dst)))
            features.append(common_neighbors)
        
        # Jaccard coefficient
        if 'jaccard_coefficient' in self.include_features:
            try:
                j_coeff = next(nx.jaccard_coefficient(self.nx_graph, [(src, dst)]))[2]
                features.append(j_coeff)
            except:
                features.append(0)
        
        # Adamic-Adar index
        if 'adamic_adar' in self.include_features:
            try:
                aa_index = next(nx.adamic_adar_index(self.nx_graph, [(src, dst)]))[2]
                features.append(aa_index)
            except:
                features.append(0)
        
        # Preferential attachment
        if 'preferential_attachment' in self.include_features:
            try:
                pa_score = next(nx.preferential_attachment(self.nx_graph, [(src, dst)]))[2]
                features.append(pa_score)
            except:
                # Fallback to manual calculation
                pa_score = self.degree[src] * self.degree[dst]
                features.append(pa_score)
        
        # Resource allocation index
        if 'resource_allocation' in self.include_features:
            try:
                ra_index = next(nx.resource_allocation_index(self.nx_graph, [(src, dst)]))[2]
                features.append(ra_index)
            except:
                features.append(0)
        
        # Total neighbors
        if 'total_neighbors' in self.include_features:
            total = len(list(self.nx_graph.neighbors(src))) + len(list(self.nx_graph.neighbors(dst)))
            features.append(total)
        
        # Shortest path
        if 'shortest_path' in self.include_features:
            if self.shortest_paths is not None:
                try:
                    path_length = self.shortest_paths[src][dst]
                    features.append(path_length)
                except:
                    features.append(10)  # Some large value
            else:
                # Compute on the fly for large graphs
                try:
                    path_length = nx.shortest_path_length(self.nx_graph, source=src, target=dst)
                    features.append(path_length)
                except:
                    features.append(10)  # Some large value
        
        # Katz
        if 'katz' in self.include_features and self.katz_matrix is not None:
            try:
                katz_score = self.katz_matrix[src, dst]
                features.append(katz_score)
            except:
                features.append(0)
        
        return np.array(features, dtype=np.float32)

class HeuristicLinkPrediction(LinkPredictionModel):
    """
    Link prediction model using heuristic topological features
    """
    def __init__(self, graph, model_type='randomforest', save_dir=None, include_features=None):
        super(HeuristicLinkPrediction, self).__init__(f'Heuristic_{model_type}', graph, None, save_dir)
        self.graph = graph
        self.model_type = model_type
        
        # Initialize the classifier
        if model_type == 'randomforest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'gbdt':
            self.classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_type == 'logistic':
            self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize the feature extractor with specified features
        self.feature_extractor = HeuristicFeatures(graph, include_features=include_features)
        
        # Model is trained or not
        self.is_trained = False
    
    def forward(self, u, v):
        """
        Forward pass to predict if an edge exists between nodes u and v
        """
        # Convert to numpy arrays
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        
        # Extract features
        features = self.feature_extractor.get_features(u, v)
        
        # Predict
        if self.is_trained:
            scores = self.classifier.predict_proba(features)[:, 1]
            return torch.tensor(scores, dtype=torch.float32)
        else:
            raise RuntimeError("Model not trained yet.")
    
    def train_model(self, data_split, device='cpu', **kwargs):
        """
        Train the model using the provided data split
        """
        print(f"Training {self.name}")
        start_time = time.time()
        
        # Get train data
        train_pos_u = data_split['train_pos_u'].cpu().numpy()
        train_pos_v = data_split['train_pos_v'].cpu().numpy()
        train_neg_u = data_split['train_neg_u'].cpu().numpy()
        train_neg_v = data_split['train_neg_v'].cpu().numpy()
        
        # Extract features
        print("Extracting features for positive pairs...")
        pos_features = self.feature_extractor.get_features(train_pos_u, train_pos_v)
        print("Extracting features for negative pairs...")
        neg_features = self.feature_extractor.get_features(train_neg_u, train_neg_v)
        
        # Combine features and labels
        X = np.vstack([pos_features, neg_features])
        y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
        
        # Train the classifier
        print("Training classifier...")
        self.classifier.fit(X, y)
        self.is_trained = True
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set if available
        val_metrics = {}
        if len(data_split['val_pos_u']) > 0:
            val_pos_u = data_split['val_pos_u'].cpu().numpy()
            val_pos_v = data_split['val_pos_v'].cpu().numpy()
            val_neg_u = data_split['val_neg_u'].cpu().numpy()
            val_neg_v = data_split['val_neg_v'].cpu().numpy()
            
            # Extract features
            print("Extracting features for validation set...")
            val_pos_features = self.feature_extractor.get_features(val_pos_u, val_pos_v)
            val_neg_features = self.feature_extractor.get_features(val_neg_u, val_neg_v)
            
            # Combine features and labels
            X_val = np.vstack([val_pos_features, val_neg_features])
            y_val = np.hstack([np.ones(len(val_pos_features)), np.zeros(len(val_neg_features))])
            
            # Predict and evaluate
            val_scores = self.classifier.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_scores)
            val_ap = average_precision_score(y_val, val_scores)
            
            val_metrics = {'auc': val_auc, 'ap': val_ap}
            print(f"Validation AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
        
        # Save the model
        self.save_model()
        
        # Save training info
        training_info = {
            'val_metrics': val_metrics,
            'training_time': training_time,
            'feature_importance': None
        }
        
        # Feature importance for tree-based models
        if hasattr(self.classifier, 'feature_importances_'):
            feature_importance = dict(zip(
                self.feature_extractor.include_features,
                self.classifier.feature_importances_
            ))
            training_info['feature_importance'] = feature_importance
            print("Feature importance:", feature_importance)
        
        metrics_path = os.path.join(self.save_dir, f"{self.name}_training_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(training_info, f)
        
        return training_info
    
    def evaluate(self, pos_u, pos_v, neg_u, neg_v, device='cpu'):
        """
        Evaluate the model on positive and negative edges
        """
        # Convert to numpy arrays
        pos_u = pos_u.cpu().numpy()
        pos_v = pos_v.cpu().numpy()
        neg_u = neg_u.cpu().numpy()
        neg_v = neg_v.cpu().numpy()
        
        # Extract features
        pos_features = self.feature_extractor.get_features(pos_u, pos_v)
        neg_features = self.feature_extractor.get_features(neg_u, neg_v)
        
        # Combine features and labels
        X = np.vstack([pos_features, neg_features])
        y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
        
        # Predict and evaluate
        scores = self.classifier.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        
        return auc, ap
    
    def get_metrics(self, pos_u, pos_v, neg_u, neg_v, device='cpu', threshold=0.5):
        """
        Get comprehensive metrics for the model
        """
        # Convert to numpy arrays
        pos_u = pos_u.cpu().numpy()
        pos_v = pos_v.cpu().numpy()
        neg_u = neg_u.cpu().numpy()
        neg_v = neg_v.cpu().numpy()
        
        # Extract features
        pos_features = self.feature_extractor.get_features(pos_u, pos_v)
        neg_features = self.feature_extractor.get_features(neg_u, neg_v)
        
        # Combine features and labels
        X = np.vstack([pos_features, neg_features])
        y = np.hstack([np.ones(len(pos_features)), np.zeros(len(neg_features))])
        
        # Predict and evaluate
        scores = self.classifier.predict_proba(X)[:, 1]
        predictions = (scores > threshold).astype(int)
        
        metrics = {
            'auc': roc_auc_score(y, scores),
            'ap': average_precision_score(y, scores),
            'f1': f1_score(y, predictions),
            'precision': precision_score(y, predictions),
            'recall': recall_score(y, predictions)
        }
        
        return metrics
    
    def save_model(self):
        """
        Save the classifier model
        """
        os.makedirs(self.save_dir, exist_ok=True)
        model_path = os.path.join(self.save_dir, f"{self.name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """
        Load the classifier model
        """
        model_path = os.path.join(self.save_dir, f"{self.name}.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            self.is_trained = True
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} not found")
            return False

if __name__ == "__main__":
    import pickle
    
    # Load data
    with open('../link_prediction_data.pkl', 'rb') as f:
        data_split = pickle.load(f)
    
    # Create model with Random Forest
    rf_model = HeuristicLinkPrediction(data_split['graph'], model_type='randomforest')
    
    # Train model
    rf_model.train_model(data_split)
    
    # Evaluate on test set
    test_auc, test_ap = rf_model.evaluate(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save detailed metrics
    test_metrics = rf_model.get_metrics(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    print("Test Metrics:", test_metrics)
    
    # Save test metrics
    metrics_path = os.path.join(rf_model.save_dir, f"{rf_model.name}_test_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f)
    
    # Create model with Gradient Boosting
    gbdt_model = HeuristicLinkPrediction(data_split['graph'], model_type='gbdt')
    
    # Train model
    gbdt_model.train_model(data_split)
    
    # Evaluate on test set
    test_auc, test_ap = gbdt_model.evaluate(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    print(f"Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Save detailed metrics
    test_metrics = gbdt_model.get_metrics(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    # Save test metrics
    metrics_path = os.path.join(gbdt_model.save_dir, f"{gbdt_model.name}_test_metrics.pkl")
    with open(metrics_path, 'wb') as f:
        pickle.dump(test_metrics, f) 