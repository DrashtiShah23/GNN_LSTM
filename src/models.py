"""
GNN + LSTM model for Human Activity Recognition.

Architecture (per proposal):
  1. GCN encoder  — GCN-style message passing over sensor graph
  2. Aggregation  — mean pooling of node embeddings → per-window vector
  3. LSTM         — models temporal transitions across windows
  4. MLP head     — predicts activity probabilities

Ablations supported:
  - GNN-only          (no LSTM)
  - LSTM-only         (no GNN, raw feature sequence)
  - GNN+LSTM          (full model)
  - CNN1D             (1D convolutional baseline)
  - GNNLearnableAdj   (GNN with trainable adjacency — graph ablation B)
  - GNNFlattenLSTM    (flatten nodes then LSTM — graph ablation C)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import (
    GCN_HIDDEN_DIM, GCN_OUTPUT_DIM,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS,
    DROPOUT, MLP_HIDDEN_DIM,
)


# ── GCN Layer ─────────────────────────────────────────────────────────────────

class GCNLayer(nn.Module):
    """Single Graph Convolutional layer: H' = σ(A_norm · H · W)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x   : (batch, n_nodes, in_features)
        adj : (n_nodes, n_nodes)  or  (batch, n_nodes, n_nodes)

        Returns
        -------
        (batch, n_nodes, out_features)
        """
        B = x.size(0)
        # Normalise adj to a single (n_nodes, n_nodes) matrix, then broadcast
        if adj.dim() == 4:
            # (batch, seq, n, n) — should not happen but guard anyway
            adj_2d = adj[0, 0]
        elif adj.dim() == 3:
            adj_2d = adj[0]          # take first (all are identical copies)
        else:
            adj_2d = adj             # (n_nodes, n_nodes)

        adj_b = adj_2d.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Message passing: aggregate neighbour features
        agg = torch.bmm(adj_b, x)
        out = self.linear(agg)  # (batch, n_nodes, out_features)

        # BatchNorm over (batch * n_nodes, out_features)
        b, n, f = out.shape
        out = self.bn(out.view(b * n, f)).view(b, n, f)
        return F.relu(out)


# ── GNN Encoder ──────────────────────────────────────────────────────────────

class GNNEncoder(nn.Module):
    """Two-layer GCN encoder."""

    def __init__(self, in_features: int, n_nodes: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, GCN_HIDDEN_DIM)
        self.gcn2 = GCNLayer(GCN_HIDDEN_DIM, GCN_OUTPUT_DIM)
        self.dropout = nn.Dropout(DROPOUT)
        self.n_nodes = n_nodes

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x   : (batch, n_nodes, in_features)
        adj : (n_nodes, n_nodes)

        Returns mean-pooled embedding: (batch, GCN_OUTPUT_DIM)
        """
        h = self.gcn1(x, adj)
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        return h.mean(dim=1)  # mean pool over nodes


# ── Full GNN + LSTM Model ─────────────────────────────────────────────────────

class GNNLSTMModel(nn.Module):
    """
    Hybrid GNN + LSTM model for HAR.

    Input per sample:
      x_seq : (seq_len, n_nodes, node_feat_dim)
      adj   : (n_nodes, n_nodes)

    Processing:
      For each time step t:  GNN(x_seq[t]) → embedding_t
      LSTM over {embedding_0, …, embedding_{T-1}} → hidden_T
      MLP(hidden_T) → class_logits
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_nodes: int,
        n_classes: int,
    ):
        super().__init__()
        self.gnn = GNNEncoder(node_feat_dim, n_nodes)
        self.lstm = nn.LSTM(
            input_size=GCN_OUTPUT_DIM,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def forward(self, x_seq: Tensor, adj: Tensor) -> Tensor:
        """
        x_seq : (batch, seq_len, n_nodes, node_feat_dim)
        adj   : (n_nodes, n_nodes)  or  (batch, n_nodes, n_nodes)

        Returns logits of shape (batch, n_classes).
        """
        batch, seq_len, n_nodes, feat_dim = x_seq.shape

        # Normalise adj to 2D for the GCN layer
        if adj.dim() == 3:
            adj_2d = adj[0]
        else:
            adj_2d = adj

        # Flatten batch & seq for GNN
        x_flat = x_seq.view(batch * seq_len, n_nodes, feat_dim)
        gnn_out = self.gnn(x_flat, adj_2d)                        # (B*T, GCN_OUT)
        gnn_seq = gnn_out.view(batch, seq_len, GCN_OUTPUT_DIM)    # (B, T, GCN_OUT)

        # LSTM temporal modelling
        lstm_out, _ = self.lstm(gnn_seq)                           # (B, T, LSTM_H)
        last_hidden = lstm_out[:, -1, :]                           # (B, LSTM_H)

        return self.classifier(last_hidden)


# ── LSTM-only Ablation ────────────────────────────────────────────────────────

class LSTMOnlyModel(nn.Module):
    """LSTM over raw flattened window features (no GNN)."""

    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x : (batch, seq_len, input_dim)  OR  (batch, input_dim) — auto-unsqueeze."""
        if x.dim() == 2:
            x = x.unsqueeze(1)   # treat as seq_len=1
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# ── GNN-only Ablation ─────────────────────────────────────────────────────────

class GNNOnlyModel(nn.Module):
    """Single-window GNN encoder + MLP head (no LSTM)."""

    def __init__(self, node_feat_dim: int, n_nodes: int, n_classes: int):
        super().__init__()
        self.gnn = GNNEncoder(node_feat_dim, n_nodes)
        self.classifier = nn.Sequential(
            nn.Linear(GCN_OUTPUT_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """x : (batch, n_nodes, node_feat_dim)"""
        emb = self.gnn(x, adj)
        return self.classifier(emb)


# ── 1D CNN Baseline ───────────────────────────────────────────────────────────

class CNN1DModel(nn.Module):
    """
    1D Convolutional baseline for time-series HAR.

    Input  : (batch, timesteps, n_channels)  — raw window
    Architecture:
      Conv1D(32) → MaxPool → Conv1D(64) → MaxPool → Conv1D(128) → MaxPool
      → Flatten → FC(128) → Dropout → FC(n_classes)
    """

    def __init__(self, n_timesteps: int, n_channels: int, n_classes: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            # Conv expects (batch, channels, length) so we transpose in forward
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        # Compute flattened size after pooling
        pooled_len = n_timesteps // 8   # 3 x MaxPool(2)
        self.flat_dim = 128 * pooled_len

        self.classifier = nn.Sequential(
            nn.Linear(self.flat_dim, MLP_HIDDEN_DIM * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM * 2, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (batch, timesteps * n_channels) flat  OR  (batch, timesteps, n_channels)
        We accept both since HARWindowDataset stores flat tensors.
        """
        if x.dim() == 2:
            # flat input — infer shape from flat_dim is not possible here,
            # so we rely on the model being constructed with correct dims.
            # Reshape to (batch, n_channels, timesteps) using stored flat_dim.
            # We just unsqueeze as (batch, 1, flat) and let conv handle it.
            # Better: caller should pass (batch, timesteps, channels).
            raise ValueError(
                "CNN1DModel requires 3D input (batch, timesteps, channels). "
                "Use HARWindowDataset2D instead of HARWindowDataset."
            )
        # x: (batch, timesteps, n_channels) → (batch, n_channels, timesteps)
        x = x.permute(0, 2, 1)
        x = self.conv_block(x)
        x = x.flatten(1)
        return self.classifier(x)


# ── GNN with Learnable Adjacency (Graph Ablation B) ──────────────────────────

class GNNLearnableAdjModel(nn.Module):
    """
    GNN-only model but with a TRAINABLE adjacency matrix (graph ablation B).
    The adjacency is initialised from the fixed topology and learned end-to-end.
    """

    def __init__(self, node_feat_dim: int, n_nodes: int, n_classes: int,
                 init_adj: Tensor | None = None):
        super().__init__()
        from src.graph_construction import LearnableAdjacency
        self.adj_module = LearnableAdjacency(n_nodes, init_adj=init_adj)
        self.gnn = GNNEncoder(node_feat_dim, n_nodes)
        self.classifier = nn.Sequential(
            nn.Linear(GCN_OUTPUT_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """adj argument is IGNORED — we use the learned adjacency instead."""
        learned_adj = self.adj_module()  # (n_nodes, n_nodes)
        emb = self.gnn(x, learned_adj)
        return self.classifier(emb)


# ── GNN Flatten + LSTM (Graph Ablation C) ─────────────────────────────────────

class GNNFlattenLSTMModel(nn.Module):
    """
    Graph ablation C: flatten node features (no message passing), then LSTM.
    Used to isolate whether the GCN message passing itself helps, vs just
    having structured node features fed into an LSTM.

    Input per sample:
      x_seq : (seq_len, n_nodes, node_feat_dim)  — same as GNNLSTMModel
    """

    def __init__(self, node_feat_dim: int, n_nodes: int, n_classes: int):
        super().__init__()
        flat_dim = n_nodes * node_feat_dim
        self.lstm = nn.LSTM(
            input_size=flat_dim,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_DIM, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def forward(self, x_seq: Tensor, adj: Tensor) -> Tensor:
        """
        x_seq : (batch, seq_len, n_nodes, node_feat_dim)
        adj   : ignored (no message passing)
        """
        batch, seq_len, n_nodes, feat_dim = x_seq.shape
        # Flatten nodes and features
        x_flat = x_seq.view(batch, seq_len, n_nodes * feat_dim)
        lstm_out, _ = self.lstm(x_flat)
        return self.classifier(lstm_out[:, -1, :])

