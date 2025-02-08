# Test Time Adaptation for Graph Neural Networks

This repository contains the implementation and evaluation of various Test-Time Adaptation (TTA) techniques for node classification using Graph Neural Networks (GNNs). The study compares three methods—GraphTTA, HomoTTT, and SOGA—across different domain adaptation scenarios.

## Overview

Test-time adaptation (TTA) addresses distribution shifts in GNNs without requiring access to source training data. This work presents a comprehensive comparison of different TTA methods, evaluating their effectiveness across various domain adaptation scenarios using citation network datasets.

## Methods Implemented

### GraphTTA
- Utilizes an adversarial learning approach with a GNN-based augmentor and classifier
- Generates structure-aware graph augmentations for contrastive learning
- Implemented a modified version of GraphTTA for node-level features with parameter-free augmentation

### HomoTTT
- Performs node-level contrastive learning using edge dropping based on homophily scores
- Creates negative samples through feature shuffling
- Implements homophily-based Graph Contrastive Learning and Adaptive Augmentation Strategy

### SOGA
- Combines information maximization and structure consistency objectives 
- Maintains prediction diversity while increasing prediction certainty
- Leverages target graph structure through local neighborhood relationships

## Datasets

| Dataset | # Nodes | # Edges | # Features | # Labels |
|---------|---------|---------|------------|----------|
| DBLP    | 5578    | 7341    | 7537       | 6        |
| ACM     | 7410    | 11135   | 7537       | 6        |
| CORA    | 2708    | 5429    | 1433       | 7        |

## Model Architecture

- GCN with 3 layers
- Hidden dimensions: 256 and 128
- Linear classifier layer
- Training: 100 epochs on source dataset
- Best validation accuracy model selected for adaptation

## Key Findings

1. **Method-Specific Performance**:
   - SOGA excels in natural domain shifts but struggles with artificial feature corruption
   - HomoTTT shows consistent but moderate improvements across scenarios
   - GraphTTA demonstrates stability but is sensitive to initial pseudo-labels

2. **Dataset Characteristics Impact**:
   - Source dataset size significantly affects adaptation success
   - Label distribution imbalance influences method effectiveness
   - ACM→DBLP adaptation consistently outperforms DBLP→ACM

3. **Training Duration Effects**:
   - Extended training doesn't guarantee better performance
   - Some methods show instability or performance degradation over time
   - Optimal adaptation duration varies by method and scenario

## Limitations and Future Work

1. **Label Imbalance**: Current methods don't explicitly address label imbalance during adaptation

2. **Method-Specific Challenges**:
   - GraphTTA: Sensitive to min-max optimization and initial pseudo-labels
   - HomoTTT: Requires careful tuning of homophily threshold
   - SOGA: Vulnerable to artificial feature corruption

3. **Future Directions**:
   - Develop methods addressing label imbalance
   - Create theoretical frameworks for convergence properties
   - Improve hyperparameter selection without target labels

## Acknowledgments

This implementation builds upon the excellent work from the SOGA repository (https://github.com/HaitaoMao/SOGA.git). We thank the authors for making their code publicly available.
