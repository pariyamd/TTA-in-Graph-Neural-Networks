import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from model.model import Model
from sklearn.metrics import f1_score
from model.model_utilies import save_model, load_model, test, predict,evaluate
import random
import numpy as np
import tqdm



def node_pseudo_contrast_loss_efficient(anchor, augmented_views, anchor_pseudo_labels, augmented_pseudo_labels,temperature=0.1):
        # Normalize features
        anchor = F.normalize(anchor, dim=1)
        augmented_views = F.normalize(augmented_views, dim=1)
        
        # Compute all pairwise similarities [batch_size, batch_size]
        similarities = torch.matmul(anchor, augmented_views.T) / temperature
        exp_similarities = torch.exp(similarities)
        
        # Compute denominators for all anchors [batch_size]
        denominators = exp_similarities.sum(dim=1)
        
        # Create mask for positive pairs [batch_size, batch_size]
        # Each row i contains True for all samples with same label as anchor i
        labels_equal = anchor_pseudo_labels.unsqueeze(1) == augmented_pseudo_labels.unsqueeze(0)
        
        # Compute log(exp(similarity)/denominator) for all pairs
        log_probs = similarities - torch.log(denominators.unsqueeze(1) + 1e-8)
        
        # Mask to only positive pairs and compute mean for each anchor
        masked_log_probs = log_probs * labels_equal
        
        # Sum up non-zero elements and divide by number of positives for each anchor
        num_positives = labels_equal.sum(dim=1)
        loss_per_anchor = -masked_log_probs.sum(dim=1) / (num_positives + 1e-8)
        loss_per_anchor = loss_per_anchor[num_positives > 0]
        
        # plt.figure()
        # plt.scatter(range(len(loss_per_anchor)),loss_per_anchor.cpu().detach().numpy(),color=np.array(colors)[anchor_pseudo_labels.flatten().to(torch.int).numpy()])
        # counts = anchor_pseudo_labels.flatten().to(torch.int).unique(return_counts=True)[1]
        # plt.bar(range(len(counts)), counts.numpy(), color='r')
        # plt.show()
        # Return mean loss
        return loss_per_anchor.mean()



    

def test_time_node_adaptation(model, data, encoder_optimizer, augmenter_optimizer, args, logger,
                            val_mask=None, test_mask=None, epochs=1, 
                            log_every=1, lambda_reg=0.1, projection = True,num_augmentations = 4, batch_size = 64):
    """
    Perform test-time adaptation with min-max optimization
    
    Args:
        model: GraphTTA model
        data: Target domain data
        encoder_optimizer: Optimizer for GNN encoder and projector (minimizing)
        augmenter_optimizer: Optimizer for augmenter (maximizing)
        lambda_reg: Weight for edge regularization term
    """
    metrics = {
        'val_acc': [], 'val_f1': [],
        'augmenter_loss': [],'encoder_loss':[], 'edge_reg': [], f"{args.metric}":[]
    }
    def create_augmented_representations():
        augmented_features = []
        edge_mask_sums=[]
        
        for i in range(num_augmentations):
            masked_edge_index, masked_edge_weights, edge_mask = model.augmenter(data.x, data.edge_index)
            edge_mask_sums.append(edge_mask.mean())
            
            extracted_features_augmented = model.gnn.Extractor(data.x,masked_edge_index, edge_weight = masked_edge_weights)
                
            if projection:
                z_aug = model.projection(extracted_features_augmented)
            else:
                z_aug = extracted_features_augmented
            augmented_features.append(z_aug)
        
        z_aug_all = torch.concat(augmented_features,dim=0)
        # print("edge densities in augmented versions",edge_mask_sums)
        return z_aug_all, torch.stack(edge_mask_sums).mean()

    def get_confident_predictions(logits, threshold=0.7):
        probs = F.softmax(logits, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        confident_mask = max_probs > threshold
        return pseudo_labels, confident_mask

    def compute_loss():
        
        extracted_features = model.gnn.Extractor(data.x, data.edge_index)
        if projection:
            z = model.projection(extracted_features)
        else:
            z = extracted_features
        #create augmented representations
        z_aug_all, edge_reg = create_augmented_representations()
        
        with torch.no_grad():
            batch_logits = model.gnn.Classifier(z)
            batch_logits_aug = model.gnn.Classifier(z_aug_all)
            pseudo_labels = batch_logits.argmax(dim=1)
            augmented_pseudo_labels = batch_logits_aug.argmax(dim=1)
            # augmented_pseudo_labels, confident_mask = get_confident_predictions(batch_logits_aug)
        # print("counts",pseudo_labels.flatten().to(torch.int).unique(return_counts=True)[1])

    
        contrast_loss = node_pseudo_contrast_loss_efficient(
            z, z_aug_all, pseudo_labels, augmented_pseudo_labels
        )
        return contrast_loss, edge_reg
    
    def print_grad_norms(model):
        encoder_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.gnn.parameters() if p.grad is not None]))
        print(f"Encoder grad norm: {encoder_grad_norm:.4f}")
        augmenter_grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.augmenter.parameters() if p.grad is not None]))
        print(f"Augmenter grad norm: {augmenter_grad_norm:.4f}")

    for epoch in range(epochs):
        model.train()
        for p in model.gnn.Classifier.parameters():
            p.requires_grad = False
        # for i in tqdm.tqdm(range(0,data.x.size(0),batch_size)):
        # training augmenter
        augmenter_optimizer.zero_grad()
        contrast_loss, edge_reg = compute_loss()
        augmenter_loss = -(contrast_loss + lambda_reg * edge_reg)
        augmenter_loss.backward()
        # print_grad_norms(model)
        augmenter_optimizer.step()
        metrics['augmenter_loss'].append(augmenter_loss.item())
            # ---------------------------------------------------------------------------------------------
        # for i in tqdm.tqdm(range(0,data.x.size(0),batch_size)):
        # training encoder
        encoder_optimizer.zero_grad() 
        contrast_loss, edge_reg = compute_loss()
        encoder_loss = (contrast_loss + lambda_reg * edge_reg)
        encoder_loss.backward()
        # print_grad_norms(model)
        encoder_optimizer.step()
        metrics['encoder_loss'].append(encoder_loss.item())


        # Log metrics
        if (epoch + 1) % log_every == 0:
            metrics['encoder_loss'].append(encoder_loss.item())

            # metrics['augmenter_loss'].append(augmenter_loss.item())
            metrics['edge_reg'].append(edge_reg.item())
            outputs = model.gnn(data)
            micro = evaluate(outputs,data.y, "micro")
            macro = evaluate(outputs,data.y, "macro")
            logger.info(f"{micro:.4f}",key="Micro F1")
            logger.info(f"{macro:.4f}",key="Macro F1")
            logger.info(f"{encoder_loss.item():.4f}",key="Loss Encoder")
            logger.info(f"{augmenter_loss.item():.4f}",key="Loss Augmenter")
            
            print(f"Micro: {micro:.4f}")
            print("---")
            
    return metrics


class EdgeMaskingAugmenter(nn.Module):
    """Parameter-free edge masking augmenter using inner product scoring"""
    def __init__(self, gnn, temperature=1):
        super().__init__()
        self.gnn = gnn  # GNN encoder initialized from trained model
        self.temperature = temperature
    
    def forward(self, x, edge_index):
        # Get node representations using GNN encoder
        node_features = self.gnn.Extractor(x, edge_index)
        norms = torch.norm(node_features, p=2, dim=1)

        src_norms = norms[edge_index[0]]
        dst_norms = norms[edge_index[1]]
        inner_products = torch.sum(
            node_features[edge_index[0]] * node_features[edge_index[1]], 
            dim=1
        )
        edge_weights = inner_products / (src_norms * dst_norms + 1e-8)
        
        if self.training:
            delta = torch.rand_like(edge_weights)    
            edge_mask = torch.sigmoid(
                (torch.log(delta) - torch.log(1 - delta ) + edge_weights) / self.temperature
            )
            
        else:
            edge_mask = (edge_weights > 0).float()
            
        edge_mask = edge_mask.squeeze()
        keep_mask = edge_mask > 0.5
        masked_edge_index = edge_index[:, keep_mask]
        masked_edge_weights = edge_mask[keep_mask]
        
        return masked_edge_index, masked_edge_weights, edge_mask


            
class NodeGAPGC(nn.Module):
    """GraphTTA model for node classification"""
    def __init__(self, gnn_model, augmenter, hidden_dim, num_augmentations=4):
        super().__init__()
        self.gnn = gnn_model
        self.augmenter = augmenter
        self.num_augmentations = num_augmentations
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
