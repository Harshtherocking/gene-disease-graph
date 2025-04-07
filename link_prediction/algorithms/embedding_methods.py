import torch
import numpy as np
import networkx as nx
import random
import sys
import os
import pickle
from tqdm import tqdm
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_model import LinkPredictionModel

class RandomWalkMethod(LinkPredictionModel):
    """
    Base class for random walk-based embedding methods (DeepWalk, Node2Vec)
    """
    def __init__(self, graph, embedding_dim=128, walk_length=80, num_walks=10, window_size=10, 
                 p=1.0, q=1.0, method_name="RandomWalk", save_dir=None):
        super(RandomWalkMethod, self).__init__(method_name, graph, embedding_dim, save_dir)
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p  # Return parameter (for Node2Vec)
        self.q = q  # In-out parameter (for Node2Vec)
        
        # Convert DGL graph to NetworkX graph for random walks
        self.nx_graph = dgl.to_networkx(graph, node_attrs=None, edge_attrs=None)
        self.embeddings = None
        self.node_list = list(self.nx_graph.nodes())
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    def simulate_walks(self):
        """
        Simulate random walks from each node (basic random walk)
        """
        walks = []
        nodes = list(self.nx_graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in tqdm(nodes, desc=f"Generating {self.name} walks"):
                walks.append(self._random_walk(node))
        
        return walks
    
    def _random_walk(self, start_node):
        """
        Perform a single random walk from start_node
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            curr = walk[-1]
            neighbors = list(self.nx_graph.neighbors(curr))
            
            if len(neighbors) > 0:
                walk.append(random.choice(neighbors))
            else:
                break
        
        return [str(node) for node in walk]
    
    def learn_embeddings(self, walks):
        """
        Learn node embeddings using Word2Vec from the walks
        """
        model = Word2Vec(walks, vector_size=self.hidden_dim, window=self.window_size,
                        min_count=0, sg=1, workers=4, epochs=5)
        
        # Create embedding matrix
        embeddings = {}
        for node in self.node_list:
            node_str = str(node)
            if node_str in model.wv:
                embeddings[node] = model.wv[node_str]
            else:
                # For nodes not in any walk, initialize randomly
                embeddings[node] = np.random.normal(0, 1, self.hidden_dim)
        
        return embeddings
    
    def train_model(self, data_split, epochs=1, batch_size=64, device='cpu'):
        """
        Train the random walk model and classifier for link prediction
        """
        print(f"Training {self.name} on {device}")
        
        # Generate walks and learn embeddings
        walks = self.simulate_walks()
        self.embeddings = self.learn_embeddings(walks)
        
        # Extract node embeddings for edges
        X_train, y_train = self._get_edge_features(
            data_split['train_pos_u'], data_split['train_pos_v'],
            data_split['train_neg_u'], data_split['train_neg_v']
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        
        # Save the model
        self.save_model()
        
        # Evaluate on validation set if available
        if len(data_split.get('val_pos_u', [])) > 0:
            val_auc, val_ap = self.evaluate(
                data_split['val_pos_u'], 
                data_split['val_pos_v'],
                data_split['val_neg_u'], 
                data_split['val_neg_v']
            )
            print(f"Validation AUC: {val_auc:.4f}, AP: {val_ap:.4f}")
        
        return {"val_metrics": {"auc": [val_auc], "ap": [val_ap]}, "training_time": 0}
    
    def _get_edge_features(self, pos_u, pos_v, neg_u, neg_v):
        """
        Generate edge features by combining node embeddings
        """
        pos_u_list = pos_u.tolist() if isinstance(pos_u, torch.Tensor) else pos_u
        pos_v_list = pos_v.tolist() if isinstance(pos_v, torch.Tensor) else pos_v
        neg_u_list = neg_u.tolist() if isinstance(neg_u, torch.Tensor) else neg_u
        neg_v_list = neg_v.tolist() if isinstance(neg_v, torch.Tensor) else neg_v
        
        features = []
        labels = []
        
        # Positive edges
        for u, v in zip(pos_u_list, pos_v_list):
            # Combine node embeddings (concatenation, element-wise product, etc.)
            emb_u = self.embeddings[int(u)]
            emb_v = self.embeddings[int(v)]
            edge_feat = np.concatenate([emb_u, emb_v, emb_u * emb_v, np.abs(emb_u - emb_v)])
            features.append(edge_feat)
            labels.append(1)
        
        # Negative edges
        for u, v in zip(neg_u_list, neg_v_list):
            emb_u = self.embeddings[int(u)]
            emb_v = self.embeddings[int(v)]
            edge_feat = np.concatenate([emb_u, emb_v, emb_u * emb_v, np.abs(emb_u - emb_v)])
            features.append(edge_feat)
            labels.append(0)
        
        return np.array(features), np.array(labels)
    
    def save_model(self):
        """
        Save the embeddings and classifier
        """
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Save embeddings
        embeddings_path = os.path.join(self.save_dir, f"{self.name}_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        # Save classifier
        classifier_path = os.path.join(self.save_dir, f"{self.name}_classifier.pkl")
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        print(f"Model saved to {self.save_dir}")
    
    def load_model(self):
        """
        Load the embeddings and classifier
        """
        embeddings_path = os.path.join(self.save_dir, f"{self.name}_embeddings.pkl")
        classifier_path = os.path.join(self.save_dir, f"{self.name}_classifier.pkl")
        
        if os.path.exists(embeddings_path) and os.path.exists(classifier_path):
            with open(embeddings_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            with open(classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            
            print(f"Model loaded from {self.save_dir}")
            return True
        else:
            print(f"Model files not found in {self.save_dir}")
            return False
    
    def forward(self, u, v):
        """
        Forward pass: predict link between nodes u and v
        """
        if self.embeddings is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Convert to lists if tensors
        u_list = u.tolist() if isinstance(u, torch.Tensor) else u
        v_list = v.tolist() if isinstance(v, torch.Tensor) else v
        
        features = []
        for u_idx, v_idx in zip(u_list, v_list):
            emb_u = self.embeddings[int(u_idx)]
            emb_v = self.embeddings[int(v_idx)]
            edge_feat = np.concatenate([emb_u, emb_v, emb_u * emb_v, np.abs(emb_u - emb_v)])
            features.append(edge_feat)
        
        # Predict probabilities
        probs = self.classifier.predict_proba(np.array(features))[:, 1]
        
        return torch.tensor(probs, dtype=torch.float32)
    
    def evaluate(self, pos_u, pos_v, neg_u, neg_v, device='cpu'):
        """
        Evaluate the model on positive and negative edges
        Returns AUC and Average Precision Score
        """
        if self.embeddings is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        with torch.no_grad():
            pos_score = self.forward(pos_u, pos_v)
            neg_score = self.forward(neg_u, neg_v)
            
            scores = torch.cat([pos_score, neg_score]).numpy()
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
            
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
        return auc, ap
    
    def get_metrics(self, pos_u, pos_v, neg_u, neg_v, device='cpu', threshold=0.5):
        """
        Get comprehensive metrics for the model
        """
        if self.embeddings is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        with torch.no_grad():
            pos_score = self.forward(pos_u, pos_v)
            neg_score = self.forward(neg_u, neg_v)
            
            scores = torch.cat([pos_score, neg_score]).numpy()
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
            
            # Threshold scores to get binary predictions
            predictions = (scores > threshold).astype(int)
            
            metrics = {
                'auc': roc_auc_score(labels, scores),
                'ap': average_precision_score(labels, scores),
                'f1': f1_score(labels, predictions),
                'precision': precision_score(labels, predictions),
                'recall': recall_score(labels, predictions)
            }
            
        return metrics


class DeepWalk(RandomWalkMethod):
    """
    DeepWalk: Online Learning of Social Representations
    Uses uniform random walks and Skip-gram model to learn node embeddings
    """
    def __init__(self, graph, embedding_dim=128, walk_length=80, num_walks=10, window_size=10, save_dir=None):
        super(DeepWalk, self).__init__(
            graph=graph, 
            embedding_dim=embedding_dim, 
            walk_length=walk_length, 
            num_walks=num_walks, 
            window_size=window_size, 
            method_name="DeepWalk",
            save_dir=save_dir
        )


class Node2Vec(RandomWalkMethod):
    """
    Node2Vec: Scalable Feature Learning for Networks
    Uses biased random walks controlled by parameters p and q
    """
    def __init__(self, graph, embedding_dim=128, walk_length=80, num_walks=10, 
                 window_size=10, p=1.0, q=1.0, save_dir=None):
        super(Node2Vec, self).__init__(
            graph=graph, 
            embedding_dim=embedding_dim, 
            walk_length=walk_length, 
            num_walks=num_walks, 
            window_size=window_size, 
            p=p,
            q=q,
            method_name="Node2Vec",
            save_dir=save_dir
        )
        
        # Precompute transition probabilities for efficiency
        self._precompute_probabilities()
    
    def _precompute_probabilities(self):
        """
        Precompute transition probabilities for biased random walks
        """
        self.alias_nodes = {}
        self.alias_edges = {}
        
        # Preprocess transition probabilities for each node
        for node in self.nx_graph.nodes():
            unnormalized_probs = [1.0 for _ in self.nx_graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self._alias_setup(normalized_probs)
        
        # Preprocess transition probabilities for each edge
        for edge in self.nx_graph.edges():
            self.alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
            self.alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])
    
    def _get_alias_edge(self, src, dst):
        """
        Get alias edge setup for edge (src, dst)
        """
        unnormalized_probs = []
        for dst_nbr in self.nx_graph.neighbors(dst):
            if dst_nbr == src:  # Return to the previous node
                unnormalized_probs.append(1.0 / self.p)
            elif self.nx_graph.has_edge(dst_nbr, src):  # Distance 1 from src
                unnormalized_probs.append(1.0)
            else:  # Distance 2 from src
                unnormalized_probs.append(1.0 / self.q)
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        
        return self._alias_setup(normalized_probs)
    
    def _alias_setup(self, probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        Implementation from https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)
        
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        # Process until we are out of small and large outcomes
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] - (1.0 - q[small])
            
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q
    
    def _alias_draw(self, J, q):
        """
        Draw sample from a non-uniform discrete distribution using alias sampling.
        """
        K = len(J)
        
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def _random_walk(self, start_node):
        """
        Simulate a biased random walk starting from start_node
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.nx_graph.neighbors(cur))
            
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    # First step: random uniform selection from neighbors
                    walk.append(cur_nbrs[self._alias_draw(*self.alias_nodes[cur])])
                else:
                    # Following steps: biased selection based on p and q
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[self._alias_draw(*self.alias_edges.get(edge, self.alias_nodes[cur]))]
                    walk.append(next_node)
            else:
                break
        
        return [str(node) for node in walk]


# Helper functions to make these models available
def create_deepwalk(graph, save_dir=None):
    """
    Create a DeepWalk model with default parameters
    """
    return DeepWalk(graph, save_dir=save_dir)

def create_node2vec(graph, p=1.0, q=1.0, save_dir=None):
    """
    Create a Node2Vec model with default parameters
    """
    return Node2Vec(graph, p=p, q=q, save_dir=save_dir)

# For backward compatibility with existing code
if __name__ == "__main__":
    import dgl
    
    # Load data
    with open('../link_prediction_data.pkl', 'rb') as f:
        data_split = pickle.load(f)
    
    # Create models
    deepwalk_model = DeepWalk(data_split['graph'])
    node2vec_model = Node2Vec(data_split['graph'], p=1.0, q=2.0)
    
    # Train and evaluate DeepWalk
    print("Training DeepWalk model...")
    deepwalk_model.train_model(data_split)
    
    test_auc, test_ap = deepwalk_model.evaluate(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    print(f"DeepWalk - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}")
    
    # Train and evaluate Node2Vec
    print("Training Node2Vec model...")
    node2vec_model.train_model(data_split)
    
    test_auc, test_ap = node2vec_model.evaluate(
        data_split['test_pos_u'], 
        data_split['test_pos_v'],
        data_split['test_neg_u'], 
        data_split['test_neg_v']
    )
    
    print(f"Node2Vec - Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}") 