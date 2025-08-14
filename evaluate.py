# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 22:46:21 2025

@author: Akach
"""

import numpy as np

def precision_recall_at_k(sim_matrix, true_links, k=10):
    """
    Compute Precision@K and Recall@K for each entity.
    sim_matrix: NxN similarity scores
    true_links: dict mapping i -> set of true neighbor indices
    """
    n = sim_matrix.shape[0]
    precisions, recalls = [], []
    for i in range(n):
        # rank other indices by similarity
        inds = np.argsort(-sim_matrix[i])[:k]
        tp = len(set(inds) & true_links.get(i, set()))
        precisions.append(tp / k)
        recalls.append(tp / max(1, len(true_links.get(i, set()))))
    return np.mean(precisions), np.mean(recalls)

def mean_reciprocal_rank(ranks_list):
    """
    Compute MRR for a list of positive example ranks (1-based).
    ranks_list: list of ints (rank positions of the correct item)
    """
    return np.mean([1.0 / r for r in ranks_list])

def hits_at_k(ranks_list, k=10):
    """
    Fraction of ranks ≤ k.
    """
    return np.mean([1.0 if r <= k else 0.0 for r in ranks_list])

# Example usage
if __name__ == "__main__":
    # Load fused or individual similarity
    sim = np.load("output/chem_sim.npy")
    # Suppose true_links is precomputed from known drug–drug interactions
    true = {0: {1,2}, 1: {0}, 2: {0}, ...}

    p, r = precision_recall_at_k(sim, true, k=10)
    print(f"Precision@10: {p:.3f}, Recall@10: {r:.3f}")

    # For link prediction ranks (you’d get these from your KG model)
    ranks = [1, 5, 12, 3, ...]
    mrr = mean_reciprocal_rank(ranks)
    h10 = hits_at_k(ranks, k=10)
    print(f"MRR: {mrr:.3f}, Hits@10: {h10:.3f}")
