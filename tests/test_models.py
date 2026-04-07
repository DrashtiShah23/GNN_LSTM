"""
Unit tests for core project components.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest
import torch

from src.config import WINDOW_SIZE, GCN_OUTPUT_DIM, LSTM_HIDDEN_DIM


# ── Graph construction tests ──────────────────────────────────────────────────

class TestGraphConstruction:
    def test_fixed_adj_shape(self):
        from src.graph_construction import build_fixed_adj
        adj = build_fixed_adj([(0, 1), (1, 2)], n_nodes=3)
        assert adj.shape == (3, 3)

    def test_fixed_adj_symmetric(self):
        from src.graph_construction import build_fixed_adj
        adj = build_fixed_adj([(0, 1), (1, 2), (0, 2)], n_nodes=3)
        assert torch.allclose(adj, adj.T)

    def test_pamap2_adj(self):
        from src.graph_construction import build_pamap2_adj
        adj = build_pamap2_adj()
        assert adj.shape == (3, 3)

    def test_hhar_adj(self):
        from src.graph_construction import build_hhar_adj
        adj = build_hhar_adj()
        assert adj.shape == (2, 2)

    def test_node_features_pamap2_shape(self):
        from src.graph_construction import window_to_node_features_pamap2
        window = np.random.randn(WINDOW_SIZE, 12).astype(np.float32)
        feats = window_to_node_features_pamap2(window)
        # 3 nodes, 6 stats × (12/3)=4 channels each → 24 features per node
        assert feats.shape == (3, 24)

    def test_learnable_adjacency(self):
        from src.graph_construction import LearnableAdjacency
        la = LearnableAdjacency(n_nodes=3)
        adj = la()
        assert adj.shape == (3, 3)
        assert (adj >= 0).all()


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModels:
    def setup_method(self):
        self.n_nodes = 3
        self.node_feat = 4
        self.n_classes = 6
        self.batch = 8
        self.seq_len = 10
        self.adj = torch.eye(self.n_nodes)

    def test_gnn_lstm_forward(self):
        from src.models import GNNLSTMModel
        model = GNNLSTMModel(self.node_feat, self.n_nodes, self.n_classes)
        x = torch.randn(self.batch, self.seq_len, self.n_nodes, self.node_feat)
        out = model(x, self.adj)
        assert out.shape == (self.batch, self.n_classes)

    def test_lstm_only_forward(self):
        from src.models import LSTMOnlyModel
        input_dim = WINDOW_SIZE * self.node_feat
        model = LSTMOnlyModel(input_dim, self.n_classes)
        x = torch.randn(self.batch, self.seq_len, input_dim)
        out = model(x)
        assert out.shape == (self.batch, self.n_classes)

    def test_gnn_only_forward(self):
        from src.models import GNNOnlyModel
        model = GNNOnlyModel(self.node_feat, self.n_nodes, self.n_classes)
        x = torch.randn(self.batch, self.n_nodes, self.node_feat)
        out = model(x, self.adj)
        assert out.shape == (self.batch, self.n_classes)

    def test_gcn_layer_forward(self):
        from src.models import GCNLayer
        layer = GCNLayer(in_features=4, out_features=8)
        x = torch.randn(self.batch, self.n_nodes, 4)
        adj = torch.eye(self.n_nodes)
        out = layer(x, adj)
        assert out.shape == (self.batch, self.n_nodes, 8)


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_sliding_windows_shape(self):
        from src.preprocessing import _sliding_windows
        data = np.random.randn(500, 6)
        wins = _sliding_windows(data, window_size=128, overlap=0.5)
        assert wins.ndim == 3
        assert wins.shape[1] == 128
        assert wins.shape[2] == 6

    def test_normalise(self):
        from src.preprocessing import _normalise
        data = np.random.randn(200, 6) * 10 + 5
        normed = _normalise(data)
        assert np.abs(normed.mean(axis=0)).max() < 1e-5
        assert np.abs(normed.std(axis=0) - 1.0).max() < 0.1

    def test_resample_changes_length(self):
        from src.preprocessing import _resample_to_target
        data = np.random.randn(200, 6)
        resampled = _resample_to_target(data, orig_rate=100)
        # 50Hz target → length should be 100
        assert resampled.shape[0] == 100
        assert resampled.shape[1] == 6


# ── Baseline feature tests ────────────────────────────────────────────────────

class TestBaselines:
    def test_extract_features_shape(self):
        from src.baselines import extract_features
        X = np.random.randn(20, WINDOW_SIZE, 6).astype(np.float32)
        feats = extract_features(X)
        assert feats.shape == (20, 6 * 6)  # 6 feature types * 6 channels

    def test_extract_features_no_nan(self):
        from src.baselines import extract_features
        X = np.random.randn(10, WINDOW_SIZE, 4).astype(np.float32)
        feats = extract_features(X)
        assert not np.isnan(feats).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
