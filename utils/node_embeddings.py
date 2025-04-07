import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import numpy as np
import random
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .data_preparation import load_graph, split_edges_into_parts

class Node2VecSampler:
    """Implementation of Node2Vec random walk sampling."""
    
    def __init__(self, graph, p=1.0, q=1.0, walk_length=80, num_walks=10):
        """
        Initialize the Node2Vec sampler.
        
        Args:
            graph: DGL graph
            p: Return parameter (1 = balanced)
            q: In-out parameter (1 = balanced, >1 = DFS-like, <1 = BFS-like)
            walk_length: Length of each random walk
            num_walks: Number of walks per node
        """
        self.graph = graph
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        
        # Convert to NetworkX for sampling
        self.nx_graph = dgl.to_networkx(graph, edge_attrs=None).to_undirected()
        
        # Precompute transition probabilities
        self._precompute_probs()
    
    def _precompute_probs(self):
        """Precompute transition probabilities for each node."""
        self.alias_nodes = {}
        self.alias_edges = {}
        
        # Calculate probabilities for nodes
        for node in self.nx_graph.nodes():
            unnormalized_probs = [1.0 for _ in self.nx_graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self._alias_setup(normalized_probs)
        
        # Calculate probabilities for edges
        for edge in self.nx_graph.edges():
            self._get_alias_edge(edge[0], edge[1])
            self._get_alias_edge(edge[1], edge[0])
    
    def _get_alias_edge(self, src, dst):
        """Get alias edge setup for edge src->dst."""
        edge_key = (src, dst)
        if edge_key in self.alias_edges:
            return self.alias_edges[edge_key]
        
        unnormalized_probs = []
        for dst_nbr in self.nx_graph.neighbors(dst):
            if dst_nbr == src:  # Return to source
                unnormalized_probs.append(1.0/self.p)
            elif self.nx_graph.has_edge(dst_nbr, src):  # Common neighbor
                unnormalized_probs.append(1.0)
            else:  # Not connected to source
                unnormalized_probs.append(1.0/self.q)
        
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]
        
        self.alias_edges[edge_key] = self._alias_setup(normalized_probs)
        return self.alias_edges[edge_key]
    
    def _alias_setup(self, probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        Referenced from https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int32)
        
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
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
        """Draw sample from a non-uniform discrete distribution using alias sampling."""
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def _node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start_node.
        
        Args:
            start_node: Starting node
            
        Returns:
            List of nodes in the walk
        """
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur_node = walk[-1]
            cur_neighbors = list(self.nx_graph.neighbors(cur_node))
            
            if len(cur_neighbors) > 0:
                if len(walk) == 1:  # First step: random neighbor
                    next_node = cur_neighbors[self._alias_draw(*self.alias_nodes[cur_node])]
                else:
                    prev_node = walk[-2]
                    edge = (prev_node, cur_node)
                    next_node = cur_neighbors[self._alias_draw(*self._get_alias_edge(prev_node, cur_node))]
                walk.append(next_node)
            else:
                break
        
        return walk
    
    def generate_walks(self):
        """
        Generate random walks for all nodes.
        
        Returns:
            List of walks, where each walk is a list of node IDs
        """
        walks = []
        nodes = list(self.nx_graph.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in tqdm(nodes, desc="Generating walks"):
                walks.append(self._node2vec_walk(node))
        
        return walks

class SkipGramModel(nn.Module):
    """Skip-gram model for learning node embeddings."""
    
    def __init__(self, num_nodes, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        
        # Define embeddings
        self.in_embeddings = nn.Embedding(num_nodes, embedding_dim)
        self.out_embeddings = nn.Embedding(num_nodes, embedding_dim)
        
        # Initialize embeddings
        self.in_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.out_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, input_nodes, output_nodes):
        """
        Forward pass: calculate the dot product of input and output embeddings.
        
        Args:
            input_nodes: Tensor of center node IDs
            output_nodes: Tensor of context node IDs
            
        Returns:
            Dot product scores
        """
        in_embeds = self.in_embeddings(input_nodes)
        out_embeds = self.out_embeddings(output_nodes)
        
        # Calculate dot product
        scores = torch.sum(in_embeds * out_embeds, dim=1)
        
        return scores
    
    def get_embeddings(self):
        """
        Get final node embeddings (average of input and output embeddings).
        
        Returns:
            Tensor of node embeddings
        """
        return (self.in_embeddings.weight.data + self.out_embeddings.weight.data) / 2

class NodeEmbeddingTrainer:
    """Trainer for node embedding models."""
    
    def __init__(self, graph, embedding_dim=128, window_size=5, num_negatives=5, 
                 p=1.0, q=1.0, walk_length=80, num_walks=10, batch_size=128, 
                 epochs=5, lr=0.01, method="node2vec"):
        """
        Initialize the trainer.
        
        Args:
            graph: DGL graph
            embedding_dim: Dimension of embeddings
            window_size: Context window size for skip-gram
            num_negatives: Number of negative samples per positive
            p, q: Node2Vec parameters
            walk_length, num_walks: Random walk parameters
            batch_size, epochs, lr: Training parameters
            method: "node2vec" or "deepwalk"
        """
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.method = method
        
        # Set up sampler (DeepWalk is Node2Vec with p=q=1)
        self.sampler = Node2VecSampler(
            graph, 
            p=p if method == "node2vec" else 1.0, 
            q=q if method == "node2vec" else 1.0,
            walk_length=walk_length, 
            num_walks=num_walks
        )
        
        # Initialize model
        self.model = SkipGramModel(graph.number_of_nodes(), embedding_dim)
        
        # Set up optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
    
    def _generate_training_data(self, walks):
        """
        Generate skip-gram training pairs from walks.
        
        Args:
            walks: List of walks
            
        Returns:
            sources, targets, labels
        """
        sources = []
        targets = []
        labels = []
        
        # For each walk
        for walk in walks:
            # For each node in the walk
            for i, node in enumerate(walk):
                # Get context window
                window_start = max(0, i - self.window_size)
                window_end = min(len(walk), i + self.window_size + 1)
                
                # For each node in the context window
                for j in range(window_start, window_end):
                    if i != j:  # Skip the center node
                        # Add positive pair
                        sources.append(node)
                        targets.append(walk[j])
                        labels.append(1)
                        
                        # Add negative pairs
                        for _ in range(self.num_negatives):
                            negative = random.randint(0, self.graph.number_of_nodes() - 1)
                            sources.append(node)
                            targets.append(negative)
                            labels.append(0)
        
        return (
            torch.LongTensor(sources),
            torch.LongTensor(targets),
            torch.FloatTensor(labels)
        )
    
    def train(self):
        """
        Train the node embedding model.
        
        Returns:
            Trained model
        """
        print(f"Generating {self.method} walks...")
        walks = self.sampler.generate_walks()
        
        print("Generating training pairs...")
        sources, targets, labels = self._generate_training_data(walks)
        
        # Create data loader
        dataset = TensorDataset(sources, targets, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        print(f"Training {self.method} model...")
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            total_examples = 0
            
            for batch_sources, batch_targets, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # Forward pass
                scores = self.model(batch_sources, batch_targets)
                loss = self.loss_fn(scores, batch_labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * len(batch_sources)
                total_examples += len(batch_sources)
            
            avg_loss = total_loss / total_examples
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        return self.model

class MLPLinkPredictor(nn.Module):
    """MLP model for link prediction using node embeddings."""
    
    def __init__(self, embedding_dim):
        super(MLPLinkPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, u_embeds, v_embeds):
        """
        Forward pass: concatenate embeddings and pass through MLP.
        
        Args:
            u_embeds: Embeddings of source nodes
            v_embeds: Embeddings of destination nodes
            
        Returns:
            Predictions
        """
        features = torch.cat([u_embeds, v_embeds], dim=1)
        return self.layers(features).squeeze()

def evaluate_embeddings(embeddings, pos_edges, neg_edges, method="dot_product"):
    """
    Evaluate link prediction performance using node embeddings.
    
    Args:
        embeddings: Node embeddings tensor
        pos_edges: Tuple of (u, v) tensors for positive edges
        neg_edges: Tuple of (u, v) tensors for negative edges
        method: "dot_product" or "mlp"
        
    Returns:
        Dictionary with AUC-ROC, AP, and accuracy metrics
    """
    pos_u, pos_v = pos_edges
    neg_u, neg_v = neg_edges
    
    if method == "dot_product":
        # Calculate scores using dot product
        pos_scores = torch.sum(embeddings[pos_u] * embeddings[pos_v], dim=1)
        neg_scores = torch.sum(embeddings[neg_u] * embeddings[neg_v], dim=1)
    elif method == "mlp":
        # Train MLP for scoring
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        embeddings = embeddings.to(device)
        
        # Combine positive and negative edges
        all_u = torch.cat([pos_u, neg_u]).to(device)
        all_v = torch.cat([pos_v, neg_v]).to(device)
        labels = torch.cat([torch.ones(len(pos_u)), torch.zeros(len(neg_u))]).to(device)
        
        # Split into train/test
        indices = torch.randperm(len(all_u))
        train_size = int(0.8 * len(indices))
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        
        # Create datasets
        train_u = all_u[train_indices]
        train_v = all_v[train_indices]
        train_labels = labels[train_indices]
        
        test_u = all_u[test_indices]
        test_v = all_v[test_indices]
        test_labels = labels[test_indices]
        
        # Initialize and train MLP
        mlp = MLPLinkPredictor(embeddings.size(1)).to(device)
        optimizer = optim.Adam(mlp.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Train MLP
        for epoch in range(10):
            mlp.train()
            optimizer.zero_grad()
            preds = mlp(embeddings[train_u], embeddings[train_v])
            loss = loss_fn(preds, train_labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate on test set
        mlp.eval()
        with torch.no_grad():
            pos_scores = mlp(embeddings[pos_u], embeddings[pos_v])
            neg_scores = mlp(embeddings[neg_u], embeddings[neg_v])
    else:
        raise ValueError(f"Method {method} not supported. Choose from ['dot_product', 'mlp']")
    
    # Combine scores and labels
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
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

def run_embedding_methods_evaluation(split_data, embedding_dim=128, methods=["node2vec", "deepwalk"]):
    """
    Run evaluation for node embedding methods.
    
    Args:
        split_data: Dictionary containing train/val/test splits
        embedding_dim: Dimension of embeddings
        methods: List of methods to evaluate
        
    Returns:
        Dictionary with results for each method
    """
    print("Evaluating node embedding methods...")
    
    results = []
    
    for method in methods:
        print(f"\nTraining {method} model...")
        
        # Initialize trainer
        trainer = NodeEmbeddingTrainer(
            split_data['train_graph'],
            embedding_dim=embedding_dim,
            method=method
        )
        
        # Train model
        model = trainer.train()
        
        # Get embeddings
        embeddings = model.get_embeddings()
        
        # Evaluate on validation set
        print(f"Validation set results for {method}:")
        val_result = evaluate_embeddings(
            embeddings,
            split_data['val_pos'],
            split_data['val_neg'],
            method="dot_product"
        )
        print(f"  Dot product: AUC = {val_result['auc']:.4f}, AP = {val_result['ap']:.4f}, Accuracy = {val_result['accuracy']:.4f}")
        val_result['split'] = 'validation'
        val_result['embedding_method'] = method
        results.append(val_result)
        
        val_result_mlp = evaluate_embeddings(
            embeddings,
            split_data['val_pos'],
            split_data['val_neg'],
            method="mlp"
        )
        print(f"  MLP: AUC = {val_result_mlp['auc']:.4f}, AP = {val_result_mlp['ap']:.4f}, Accuracy = {val_result_mlp['accuracy']:.4f}")
        val_result_mlp['split'] = 'validation'
        val_result_mlp['embedding_method'] = method
        results.append(val_result_mlp)
        
        # Evaluate on test set
        print(f"Test set results for {method}:")
        test_result = evaluate_embeddings(
            embeddings,
            split_data['test_pos'],
            split_data['test_neg'],
            method="dot_product"
        )
        print(f"  Dot product: AUC = {test_result['auc']:.4f}, AP = {test_result['ap']:.4f}, Accuracy = {test_result['accuracy']:.4f}")
        test_result['split'] = 'test'
        test_result['embedding_method'] = method
        results.append(test_result)
        
        test_result_mlp = evaluate_embeddings(
            embeddings,
            split_data['test_pos'],
            split_data['test_neg'],
            method="mlp"
        )
        print(f"  MLP: AUC = {test_result_mlp['auc']:.4f}, AP = {test_result_mlp['ap']:.4f}, Accuracy = {test_result_mlp['accuracy']:.4f}")
        test_result_mlp['split'] = 'test'
        test_result_mlp['embedding_method'] = method
        results.append(test_result_mlp)
    
    return results

if __name__ == "__main__":
    # Load data
    graph, encoder = load_graph()
    all_splits = split_edges_into_parts(graph, num_parts=5)
    
    # Run evaluation on first part
    results = run_embedding_methods_evaluation(all_splits[0]) 