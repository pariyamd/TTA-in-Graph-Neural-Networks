import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model import Model
from sklearn.metrics import f1_score
import random
import numpy as np
from model.Data import DomainData
import copy
import torch.optim as optim
import matplotlib.pyplot as plt

import os
from torch_geometric.utils import add_self_loops, degree, to_dense_adj
from sklearn.cluster import KMeans

from model.model_utilies import save_model, load_model, evaluate, test
# from eval_utils import evaluate, test, evaluate_node_classification

class HomoTTT:
    def __init__(self, model, num_clusters=7, 
                 drop_edge_ratio=0.2, temperature=0.1, update_layers=5,seed=0):
        """
        Initialize HomoTTT
        
        Args:
            model: Pre-trained GNN model
            num_clusters: Number of clusters for pseudo-labels
            drop_edge_ratio: Base ratio for edge dropping
            temperature: Temperature parameter for contrastive loss
            update_layers: Number of layers to update during test-time training
        """
        self.model = model
        self.num_clusters = num_clusters
        self.drop_edge_ratio = drop_edge_ratio
        self.temperature = temperature
        self.update_layers = model.Extractor.layer_count-1
        self.seed = seed
        
    def calculate_homophily(self, edge_index, pseudo_labels):
        """
        Calculate homophily scores for edges based on pseudo labels
        Args:
            edge_index: Tensor [2, num_edges] containing edge indices
            pseudo_labels: Tensor [num_nodes] containing pseudo labels for each node
        Returns:
            homophily_scores: Tensor [num_edges] containing homophily score for each edge
        """
        row, col = edge_index
        num_nodes = pseudo_labels.size(0)
        
        # For each node, find its neighbors
        neighbors_dict = {}
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            if src not in neighbors_dict:
                neighbors_dict[src] = []
            neighbors_dict[src].append(dst)
            
        # Calculate homophily scores for each edge
        homophily_scores = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            
            # Get source node's neighbors
            src_neighbors = neighbors_dict.get(src, [])
            if not src_neighbors:
                homophily_scores.append(0.0)
                continue
                
            # Count neighbors with same label as source node
            src_label = pseudo_labels[src].item()
            same_label_count = sum(1 for neighbor in src_neighbors 
                                 if pseudo_labels[neighbor].item() == src_label)
            
            # Calculate ratio
            homophily_score = same_label_count / len(src_neighbors)
            homophily_scores.append(homophily_score)
            
        homophily_scores = torch.tensor(homophily_scores, device=edge_index.device)
        
        # Normalize scores to [0,1]
        if homophily_scores.max() > homophily_scores.min():
            homophily_scores = (homophily_scores - homophily_scores.min()) / \
                              (homophily_scores.max() - homophily_scores.min())
        
        return homophily_scores
    
    def adaptive_edge_drop(self, edge_index, homophily_scores):
        """
        Perform adaptive edge dropping based on homophily scores
        Args:
            edge_index: Tensor [2, num_edges] containing edge indices
            homophily_scores: Tensor [num_edges] containing homophily score for each edge
        Returns:
            dropped_edge_index: Tensor [2, num_kept_edges] containing remaining edges
        """
        # Higher homophily = lower drop probability
        drop_probs = (1 - homophily_scores) * self.drop_edge_ratio
        
        # Sample edges to keep
        mask = torch.bernoulli(1 - drop_probs).bool()
        return edge_index[:, mask]
    
    def generate_augmented_view(self, edge_index, pseudo_labels):
        """
        Generate an augmented view of the graph using adaptive edge dropping
        Args:
            edge_index: Tensor [2, num_edges] containing edge indices
            pseudo_labels: Tensor [num_nodes] containing pseudo labels
        Returns:
            aug_edge_index: Tensor [2, num_kept_edges] for augmented graph
        """
        homophily_scores = self.calculate_homophily(edge_index, pseudo_labels)
        # plt.figure()
        # plt.scatter(range(len(homophily_scores)), homophily_scores.cpu().numpy())
        # plt.show()
        aug_edge_index = self.adaptive_edge_drop(edge_index, homophily_scores)
        return aug_edge_index
    
    def get_pseudo_labels(self, features):
        """
        Generate pseudo labels using k-means clustering
        Args:
            features: Tensor [num_nodes, feature_dim] of node features
        Returns:
            pseudo_labels: Tensor [num_nodes] of cluster assignments
        """
        features_np = features.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=self.seed)
        pseudo_labels = kmeans.fit_predict(features_np)
        print(np.unique(pseudo_labels,return_counts=True)[1])
        return torch.tensor(pseudo_labels, device=features.device)
    
    def contrastive_loss(self, anchor_features, positive_features, negative_features):
        """
        Calculate contrastive loss between anchor nodes and their augmented/negative views
        Args:
            anchor_features: Original node features [num_nodes, feature_dim]
            aug_features: Augmented node features [num_nodes, feature_dim]
        Returns:
            loss: Combined loss from positive and negative pairs
        """
        # Normalize features
        anchor_features = F.normalize(anchor_features, dim=1)
        positive_features = F.normalize(positive_features, dim=1)
        negative_features = F.normalize(negative_features, dim=1)
        
        pos_sim = torch.sum(anchor_features * positive_features, dim=1)
        pos_loss = torch.mean(1 - pos_sim)
        neg_sim = torch.sum(anchor_features * negative_features, dim=1)
        neg_loss = torch.mean(1 - neg_sim)
        
        # Total loss
        total_loss = pos_loss - neg_loss
        
        return total_loss
    
    def test_time_train(self, data, optimizer, args,logger,num_iterations=100):
        """
        Perform test-time training
        Args:
            data: PyG Data object containing:
                - x: node features [num_nodes, feature_dim]
                - edge_index: edge indices [2, num_edges]
            optimizer: PyTorch optimizer
            num_iterations: Number of training iterations
        """
        # Get initial features and pseudo-labels
        output = self.model(data)
        logger.info(f"accuracy of source model in target domain: { evaluate(output, data.y, args.metric)}")
        # Set layers to train
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            
        # Only enable gradient for first few layers
        for i in range(self.update_layers):
            for param in self.model.Extractor.conv_layers[i].parameters():
                param.requires_grad = True
                
        # Test-time training loop
        for iteration in range(num_iterations):
            self.model.enable_target()
            optimizer.zero_grad()
            
            
            # Get original and augmented features
            orig_features = self.model.Extractor(data.x, data.edge_index)
            pseudo_labels = self.get_pseudo_labels(orig_features)
            aug_edge_index = self.generate_augmented_view(data.edge_index, pseudo_labels)
            
            pos_features = self.model.Extractor(data.x, aug_edge_index)

            data_negative = data.x[torch.randperm(data.x.size(0))]
            neg_features = self.model.Extractor(data_negative, data.edge_index)
            
            # Calculate loss
            loss = self.contrastive_loss(orig_features, pos_features,neg_features)
            
            # Update model
            loss.backward()
            optimizer.step()
            
            output = self.model(data)
            micro = evaluate(output, data.y,"micro")
            macro = evaluate(output, data.y,"macro")

            logger.info(f'{loss.item():.4f}',key="Loss")
            logger.info(f'{macro:.4f}',key="Macro F1")
            logger.info(f'{micro:.4f}',key="Micro F1")
            logger.info(f'Iteration [{iteration+1}/{num_iterations}]: Loss = {loss.item():.4f}, Micro = {micro:.4f}')
            # print("accuracy",evaluate(output, data.y,args.metric))
            # print(f'Iteration [{iteration+1}/{num_iterations}]: Loss = {loss.item():.4f}')
            # val_acc, val_f1  = evaluate_node_classification(self.model, data, data.val_mask)
            # print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        
        # Reset model to eval mode
        self.model.eval()
        
        # Reset requires_grad for all parameters
        for param in self.model.parameters():
            param.requires_grad = True
            
        return self.model