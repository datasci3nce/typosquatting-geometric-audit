#!/usr/bin/env python3
"""
Activation steering, projection depletion, and ablation hooks.
"""
import numpy as np
import torch


def steer_activations(X: np.ndarray, w: np.ndarray, alpha: float) -> np.ndarray:
    """h_steered = h - alpha * (projection of h onto w) * w"""
    proj = X @ w
    return X - alpha * np.outer(proj, w)


def depletion_projection(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Remove the component along w entirely."""
    w_unit = w / np.linalg.norm(w)
    projection = np.outer(X @ w_unit, w_unit)
    return X - projection


def make_head_ablation_hook(layer_idx: int, head_idx: int, num_heads: int, head_dim: int):
    """Creates a forward hook that zeros out a specific attention head."""
    def hook_fn(module, input, output):
        bsz, seq, hidden = output.shape
        output_reshaped = output.view(bsz, seq, num_heads, head_dim)
        output_reshaped[:, :, head_idx, :] = 0.0
        return output_reshaped.view(bsz, seq, hidden)
    return hook_fn


def make_layer_ablation_hook():
    """Creates a forward hook that zeros out an entire layer's output."""
    def hook_fn(module, input, output):
        return torch.zeros_like(output)
    return hook_fn