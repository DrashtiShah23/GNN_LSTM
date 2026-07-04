"""
Graph construction for HAR.

Each window is represented as a graph where nodes = sensor units.
Edges are initialised from physical proximity (fixed) or learned
from the data (learnable / attention-based), as described in the proposal.

For PAMAP2: nodes = [wrist, chest, ankle]  (3 IMUs)
For HHAR:   nodes = [accel x-axis, y-axis, z-axis]  (3 nodes; fully connected)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ── Fixed adjacency ───────────────────────────────────────────────────────────

def build_fixed_adj(edge_list: list[tuple[int, int]], n_nodes: int) -> Tensor:
    """
    Build a symmetric normalised adjacency matrix (with self-loops).

    Parameters
    ----------
    edge_list : list of (i, j) pairs
    n_nodes   : total number of nodes

    Returns
    -------
    Tensor of shape (n_nodes, n_nodes)
    """
    A = torch.eye(n_nodes)  # self-loops
    for i, j in edge_list:
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Symmetric normalisation: D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    return A_norm


def build_pamap2_adj() -> Tensor:
    """Fixed adjacency for PAMAP2 (wrist=0, chest=1, ankle=2)."""
    from src.config import PAMAP2_EDGES
    return build_fixed_adj(PAMAP2_EDGES, n_nodes=3)


def build_hhar_adj() -> Tensor:
    """Fixed adjacency for HHAR: nodes 0=x, 1=y, 2=z; edges connect all pairs (triangle)."""
    from src.config import HHAR_EDGES
    return build_fixed_adj(HHAR_EDGES, n_nodes=3)


# ── Learnable adjacency (used in ablation) ────────────────────────────────────

class LearnableAdjacency(torch.nn.Module):
    """
    A parameterised symmetric adjacency matrix.
    Initialised from a fixed adjacency and fine-tuned during training.
    """

    def __init__(self, n_nodes: int, init_adj: Tensor | None = None):
        super().__init__()
        if init_adj is not None:
            self.raw = torch.nn.Parameter(init_adj.clone())
        else:
            self.raw = torch.nn.Parameter(torch.eye(n_nodes))

    def forward(self) -> Tensor:
        # Symmetrise and apply softmax-normalisation
        A = (self.raw + self.raw.T) / 2.0
        A = torch.relu(A)  # keep non-negative
        # Row-normalise
        row_sum = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return A / row_sum


# ── Window → node feature mapping ────────────────────────────────────────────

def _node_stats(segment: np.ndarray) -> np.ndarray:
    """
    Compute rich statistical features for one node's time-series segment.

    Parameters
    ----------
    segment : (WINDOW_SIZE, n_ch)  — raw time-series for one sensor node

    Returns
    -------
    1-D array of length  6 * n_ch:
      [mean, std, min, max, rms, iqr]  per channel
    """
    mean  = segment.mean(axis=0)
    std   = segment.std(axis=0)
    mn    = segment.min(axis=0)
    mx    = segment.max(axis=0)
    rms   = np.sqrt((segment ** 2).mean(axis=0))
    q75, q25 = np.percentile(segment, [75, 25], axis=0)
    iqr   = q75 - q25
    return np.concatenate([mean, std, mn, mx, rms, iqr])


def window_to_node_features_pamap2(window: np.ndarray) -> np.ndarray:
    """
    Map a (WINDOW_SIZE, n_channels) window to node feature matrix (3, feat_dim).

    Assumes features are ordered: [wrist_channels..., chest_channels..., ankle_channels...]
    Each node gets 6 statistical descriptors × n_channels_per_node features.
    """
    n_total = window.shape[1]
    n_per_node = n_total // 3
    nodes = np.stack([
        _node_stats(window[:, : n_per_node]),                    # wrist
        _node_stats(window[:, n_per_node: 2 * n_per_node]),      # chest
        _node_stats(window[:, 2 * n_per_node:]),                  # ankle
    ])  # shape: (3, 6 * n_per_node)
    return nodes.astype(np.float32)


def window_to_node_features_hhar(window: np.ndarray) -> np.ndarray:
    """
    Map a (WINDOW_SIZE, 3) window (x, y, z accelerometer) to per-axis node
    features of shape (3, 6).

    Each spatial axis becomes one node (x=0, y=1, z=2) with 6 statistical
    features: [mean, std, min, max, rms, iqr].  The resulting 3-node graph
    captures cross-axis dependencies (e.g. gravity coupling x/y during walking,
    vertical vs horizontal motion during stairs) that a flat feature vector
    cannot represent explicitly.
    """
    return np.stack([
        _node_stats(window[:, i: i + 1])  # (6,) for 1-channel segment
        for i in range(window.shape[1])
    ]).astype(np.float32)  # (3, 6)
