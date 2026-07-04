"""Model registry with consistent publication names."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from src.config import (
    GCN_OUTPUT_DIM,
    HHAR_NODE_FEAT_DIM,
    PAMAP2_NODE_FEAT_DIM,
    LSTM_HIDDEN_DIM,
    LSTM_NUM_LAYERS,
    DROPOUT,
    MLP_HIDDEN_DIM,
)
from src.graph_construction import build_hhar_adj, build_pamap2_adj
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
    node_feat = PAMAP2_NODE_FEAT_DIM if dataset == "pamap2" else HHAR_NODE_FEAT_DIM
    n_nodes = 3

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


def build_adj(dataset: str):
    return build_pamap2_adj() if dataset == "pamap2" else build_hhar_adj()
