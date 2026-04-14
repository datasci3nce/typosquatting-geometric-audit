#!/usr/bin/env python3
"""
Subspace metrics: cosine similarity, projection norms.
"""
import numpy as np


def compute_direction_cosine(w_pre: np.ndarray, w_post: np.ndarray) -> float:
    norm_pre = np.linalg.norm(w_pre)
    norm_post = np.linalg.norm(w_post)
    if norm_pre * norm_post == 0:
        return 0.0
    return float(np.dot(w_pre, w_post) / (norm_pre * norm_post))


def compute_projection_norm(X: np.ndarray, w: np.ndarray) -> float:
    return float(np.mean(np.abs(X @ w)))