"""Model registry with consistent publication names."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from src.config import (
    GCN_OUTPUT_DIM,
    HHAR_NODE_FEAT_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    DROPOUT,
    MLP_HIDDEN_DIM,
)
import numpy as np

from src.graph_construction import build_hhar_adj, build_pamap2_adj, window_to_node_features_pamap2
from src.models import CNN1DModel, GNNFlattenLSTMModel, ImprovedGNNLSTMModel


ModelFactory = Callable[[], nn.Module]


def get_model_factory(
    model_name: str,
    dataset: str,
    n_classes: int,
    input_shape: tuple[int, ...],
) -> tuple[ModelFactory, bool, str]:
    """
    Returns (factory, use_adj, model_type).
    model_type: 'window' | 'sequence'
    """
    T, C = input_shape[0], input_shape[1]
    if dataset == "pamap2":
        sample = np.zeros((T, C), dtype=np.float32)
        sample_nodes = window_to_node_features_pamap2(sample)
        n_nodes = int(sample_nodes.shape[0])
        node_feat = int(sample_nodes.shape[-1])
    else:
        n_nodes = 3
        node_feat = HHAR_NODE_FEAT_DIM

    if model_name == "CNN1D":
        def factory():
            return CNN1DModel(n_timesteps=T, n_channels=C, n_classes=n_classes)
        return factory, False, "window"

    if model_name == "Flatten_LSTM":
        def factory():
            return GNNFlattenLSTMModel(node_feat, n_nodes, n_classes)
        return factory, True, "sequence"

    if model_name == "Improved_GNN_LSTM":
        def factory():
            return ImprovedGNNLSTMModel(
                node_feat_dim=node_feat,
                n_nodes=n_nodes,
                n_classes=n_classes,
            )
        return factory, True, "sequence"

    raise ValueError(f"Unknown model: {model_name}")


def build_adj(dataset: str, n_channels: int | None = None):
    return build_pamap2_adj(n_channels) if dataset == "pamap2" else build_hhar_adj()
