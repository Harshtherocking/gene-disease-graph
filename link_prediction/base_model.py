import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import pickle
import time

class LinkPredictionModel(nn.Module):
    """
    Base class for link prediction models
    """
    def __init__(self, name, graph, hidden_dim=128, save_dir=None):
        super(LinkPredictionModel, self).__init__()
        self.name = name
        self.graph = graph
        self.hidden_dim = hidden_dim
        
        if save_dir is None:
            # Get the script directory and use models subdirectory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.save_dir = os.path.join(script_dir, "models")
        else:
            self.save_dir = save_dir
            
        os.makedirs(self.save_dir, exist_ok=True)

    def forward(self, u, v):
        """
        Forward pass to predict if an edge exists between nodes u and v
        This should be implemented by all subclasses
        """
        raise NotImplementedError

    def train_model(self, data_split, epochs=100, batch_size=1024, lr=0.001, device='cpu'):
        """
        Train the model using the provided data split
        """
        print(f"Training {self.name} on {device}")
        self.to(device)
        
        # Make sure the graph is on the correct device
        self.graph = self.graph.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Move data to device
        train_pos_u = data_split['train_pos_u'].to(device)
        train_pos_v = data_split['train_pos_v'].to(device)
        train_neg_u = data_split['train_neg_u'].to(device)
        train_neg_v = data_split['train_neg_v'].to(device)
        
        val_pos_u = data_split['val_pos_u'].to(device) if len(data_split['val_pos_u']) > 0 else None
        val_pos_v = data_split['val_pos_v'].to(device) if len(data_split['val_pos_v']) > 0 else None
        val_neg_u = data_split['val_neg_u'].to(device) if len(data_split['val_neg_u']) > 0 else None
        val_neg_v = data_split['val_neg_v'].to(device) if len(data_split['val_neg_v']) > 0 else None
        
        best_val_auc = 0
        start_time = time.time()
        
        train_losses = []
        val_metrics = {'auc': [], 'ap': []}
        
        for epoch in range(epochs):
            self.train()
            total_loss = 0
            
            # Train on batches
            for batch_idx in range(0, len(train_pos_u), batch_size):
                batch_end = min(batch_idx + batch_size, len(train_pos_u))
                pos_batch_size = batch_end - batch_idx
                
                pos_score = self.forward(
                    train_pos_u[batch_idx:batch_end],
                    train_pos_v[batch_idx:batch_end]
                )
                
                # Sample same number of negative edges
                neg_batch_idx = np.random.choice(len(train_neg_u), pos_batch_size)
                neg_score = self.forward(
                    train_neg_u[neg_batch_idx],
                    train_neg_v[neg_batch_idx]
                )
                
                # Combine positive and negative samples for binary classification
                labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
                scores = torch.cat([pos_score, neg_score])
                
                # Binary cross entropy loss
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * (batch_end - batch_idx)
            
            train_losses.append(total_loss / len(train_pos_u))
            
            # Validation
            if val_pos_u is not None and len(val_pos_u) > 0:
                val_auc, val_ap = self.evaluate(val_pos_u, val_pos_v, val_neg_u, val_neg_v, device)
                val_metrics['auc'].append(val_auc)
                val_metrics['ap'].append(val_ap)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    self.save_model()
                
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save training metrics
        training_info = {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'training_time': training_time
        }
        
        metrics_path = os.path.join(self.save_dir, f"{self.name}_training_metrics.pkl")
        with open(metrics_path, 'wb') as f:
            pickle.dump(training_info, f)
        
        return training_info

    def evaluate(self, pos_u, pos_v, neg_u, neg_v, device='cpu'):
        """
        Evaluate the model on positive and negative edges
        Returns AUC and Average Precision Score
        """
        self.eval()
        with torch.no_grad():
            pos_score = self.forward(pos_u, pos_v)
            neg_score = self.forward(neg_u, neg_v)
            
            scores = torch.cat([pos_score, neg_score]).cpu().numpy()
            labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
            
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
        return auc, ap

    def get_metrics(self, pos_u, pos_v, neg_u, neg_v, device='cpu', threshold=0.5):
        """
        Get comprehensive metrics for the model
        """
        self.eval()
        with torch.no_grad():
            pos_score = self.forward(pos_u, pos_v)
            neg_score = self.forward(neg_u, neg_v)
            
            scores = torch.cat([pos_score, neg_score]).cpu().numpy()
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

    def save_model(self):
        """
        Save the model
        """
        os.makedirs(self.save_dir, exist_ok=True)
        model_path = os.path.join(self.save_dir, f"{self.name}.pt")
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """
        Load the model
        """
        model_path = os.path.join(self.save_dir, f"{self.name}.pt")
        if os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} not found")
            return False 