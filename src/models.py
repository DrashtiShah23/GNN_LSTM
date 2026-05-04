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
    """Stacked GCN encoder (depth configurable for hyperparameter search)."""

    def __init__(
        self,
        in_features: int,
        n_nodes: int,
        gcn_hidden: int | None = None,
        gcn_output: int | None = None,
        dropout: float | None = None,
        num_gcn_layers: int = 2,
    ):
        super().__init__()
        h = gcn_hidden if gcn_hidden is not None else GCN_HIDDEN_DIM
        o = gcn_output if gcn_output is not None else GCN_OUTPUT_DIM
        d = dropout if dropout is not None else DROPOUT
        nl = max(1, int(num_gcn_layers))
        self.gcn_out_dim = o
        self.dropout = nn.Dropout(d)
        self.n_nodes = n_nodes

        dims: list[int]
        if nl == 1:
            dims = [in_features, o]
        else:
            dims = [in_features] + [h] * (nl - 1) + [o]
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x   : (batch, n_nodes, in_features)
        adj : (n_nodes, n_nodes)

        Returns mean-pooled embedding: (batch, gcn_out_dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, adj)
            if i < len(self.layers) - 1:
                h = self.dropout(h)
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

    Optional constructor overrides (for hyperparameter sweeps) default to
    values in src.config.
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_nodes: int,
        n_classes: int,
        *,
        gcn_hidden: int | None = None,
        gcn_output: int | None = None,
        num_gcn_layers: int = 2,
        lstm_hidden: int | None = None,
        lstm_layers: int | None = None,
        dropout: float | None = None,
        mlp_hidden: int | None = None,
    ):
        super().__init__()
        dr = dropout if dropout is not None else DROPOUT
        self.gnn = GNNEncoder(
            node_feat_dim,
            n_nodes,
            gcn_hidden=gcn_hidden,
            gcn_output=gcn_output,
            dropout=dr,
            num_gcn_layers=num_gcn_layers,
        )
        g_out = self.gnn.gcn_out_dim
        lh = lstm_hidden if lstm_hidden is not None else LSTM_HIDDEN_DIM
        nl = lstm_layers if lstm_layers is not None else LSTM_NUM_LAYERS
        mh = mlp_hidden if mlp_hidden is not None else MLP_HIDDEN_DIM
        self.lstm = nn.LSTM(
            input_size=g_out,
            hidden_size=lh,
            num_layers=nl,
            batch_first=True,
            dropout=dr if nl > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lh, mh),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(mh, n_classes),
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

        g_out = self.gnn.gcn_out_dim
        # Flatten batch & seq for GNN
        x_flat = x_seq.view(batch * seq_len, n_nodes, feat_dim)
        gnn_out = self.gnn(x_flat, adj_2d)                        # (B*T, g_out)
        gnn_seq = gnn_out.view(batch, seq_len, g_out)             # (B, T, g_out)

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
            nn.Linear(self.gnn.gcn_out_dim, MLP_HIDDEN_DIM),
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
            nn.Linear(self.gnn.gcn_out_dim, MLP_HIDDEN_DIM),
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


# ── Improved GNN + LSTM (redesigned) ─────────────────────────────────────────

class ImprovedGNNLSTMModel(nn.Module):
    """
    Redesigned GNN+LSTM fixing three root causes of the original's underperformance.

    Root causes fixed:
      1. Mean pooling destroys node identity — replaced with concat pooling that
         preserves each sensor node's embedding separately.
      2. BatchNorm drifts in LOSO eval mode — replaced with LayerNorm which
         normalises per-sample and is stable across any batch size.
      3. Last-hidden-state discards temporal context — replaced with soft
         attention over all LSTM timesteps.

    Additional improvements:
      4. Skip connection — raw flattened node features are concatenated to the
         GCN output so the LSTM always has access to the original signal.
      5. Bidirectional LSTM — captures both causal and anti-causal temporal
         structure, doubling effective hidden capacity.
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_nodes: int,
        n_classes: int,
        *,
        gcn_hidden: int = 64,
        gcn_output: int = 64,
        proj_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_feat_dim = node_feat_dim
        self.gcn_output = gcn_output
        dr = dropout

        # GCN with LayerNorm (stable for variable LOSO batch sizes)
        self.gcn1_linear = nn.Linear(node_feat_dim, gcn_hidden)
        self.gcn1_ln     = nn.LayerNorm(gcn_hidden)
        self.gcn2_linear = nn.Linear(gcn_hidden, gcn_output)
        self.gcn2_ln     = nn.LayerNorm(gcn_output)
        self.gcn_dropout = nn.Dropout(dr)

        # Projection: concat(GCN output, raw skip) → proj_dim
        fused_dim = n_nodes * gcn_output + n_nodes * node_feat_dim
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dr),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dr if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2  # bidirectional

        # Temporal attention over sequence
        self.attn = nn.Linear(lstm_out_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def _gcn(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        x   : (B, n_nodes, node_feat_dim)
        adj : (n_nodes, n_nodes)
        Returns (B, n_nodes, gcn_output) — node embeddings, NOT pooled.
        """
        B = x.size(0)
        adj_b = adj.unsqueeze(0).expand(B, -1, -1)

        # Layer 1
        h = torch.bmm(adj_b, x)
        h = self.gcn1_linear(h)
        h = self.gcn1_ln(h)
        h = F.relu(h)
        h = self.gcn_dropout(h)

        # Layer 2
        h = torch.bmm(adj_b, h)
        h = self.gcn2_linear(h)
        h = self.gcn2_ln(h)
        h = F.relu(h)
        return h  # (B, n_nodes, gcn_output) — per-node embeddings preserved

    def forward(self, x_seq: Tensor, adj: Tensor) -> Tensor:
        """
        x_seq : (batch, seq_len, n_nodes, node_feat_dim)
        adj   : (n_nodes, n_nodes)  or  (batch, n_nodes, n_nodes)
        """
        B, T, N, F_dim = x_seq.shape

        if adj.dim() == 3:
            adj = adj[0]

        # Apply GCN to all (batch × timestep) frames at once
        x_flat = x_seq.view(B * T, N, F_dim)
        gcn_out = self._gcn(x_flat, adj)                          # (B*T, N, gcn_output)

        # Fix 1: concat pooling — preserve per-node identity
        gcn_concat = gcn_out.reshape(B * T, N * self.gcn_output)  # (B*T, N*gcn_output)

        # Fix 2: skip connection — preserve raw features alongside GCN output
        raw_flat = x_flat.reshape(B * T, N * F_dim)               # (B*T, N*F)

        fused = torch.cat([gcn_concat, raw_flat], dim=-1)          # (B*T, N*gcn_out+N*F)
        proj = self.proj(fused)                                    # (B*T, proj_dim)
        proj_seq = proj.view(B, T, -1)                             # (B, T, proj_dim)

        # Fix 5: bidirectional LSTM
        lstm_out, _ = self.lstm(proj_seq)                          # (B, T, 2*lstm_hidden)

        # Fix 3: temporal attention instead of last-step only
        attn_w = self.attn(lstm_out).squeeze(-1)                   # (B, T)
        attn_w = F.softmax(attn_w, dim=-1)
        context = (lstm_out * attn_w.unsqueeze(-1)).sum(dim=1)     # (B, 2*lstm_hidden)

        return self.classifier(context)


# ── ImprovedGNNLSTM with attention-based adjacency ───────────────────────────

class ImprovedGNNLSTMAttnAdj(nn.Module):
    """
    ImprovedGNNLSTMModel with an attention-gated adjacency matrix.

    The fixed anatomical adjacency is multiplied element-wise by a learned
    symmetric gate (sigmoid-activated), then row-normalised.  This lets the
    model down-weight edges that are not useful for temporal reasoning while
    keeping the inductive bias of the skeleton topology.

    All other architecture choices are identical to ImprovedGNNLSTMModel:
    concat pooling, LayerNorm, skip connection, bidirectional LSTM, temporal
    attention.
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_nodes: int,
        n_classes: int,
        init_adj: Tensor,
        *,
        gcn_hidden: int = 64,
        gcn_output: int = 64,
        proj_dim: int = 128,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_nodes    = n_nodes
        self.gcn_output = gcn_output
        dr = dropout

        # Learnable attention gate over fixed skeleton adjacency
        self.register_buffer("skeleton", init_adj.clone())
        self.edge_gate = nn.Parameter(torch.zeros(n_nodes, n_nodes))

        # GCN with LayerNorm
        self.gcn1_linear = nn.Linear(node_feat_dim, gcn_hidden)
        self.gcn1_ln     = nn.LayerNorm(gcn_hidden)
        self.gcn2_linear = nn.Linear(gcn_hidden, gcn_output)
        self.gcn2_ln     = nn.LayerNorm(gcn_output)
        self.gcn_dropout = nn.Dropout(dr)

        fused_dim = n_nodes * gcn_output + n_nodes * node_feat_dim
        self.proj = nn.Sequential(
            nn.Linear(fused_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(dr),
        )

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dr if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2
        self.attn = nn.Linear(lstm_out_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def _effective_adj(self) -> Tensor:
        g = torch.sigmoid((self.edge_gate + self.edge_gate.T) * 0.5)
        a = self.skeleton * g
        row_sum = a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return a / row_sum

    def _gcn(self, x: Tensor, adj: Tensor) -> Tensor:
        B = x.size(0)
        adj_b = adj.unsqueeze(0).expand(B, -1, -1)
        h = torch.bmm(adj_b, x)
        h = F.relu(self.gcn1_ln(self.gcn1_linear(h)))
        h = self.gcn_dropout(h)
        h = torch.bmm(adj_b, h)
        h = F.relu(self.gcn2_ln(self.gcn2_linear(h)))
        return h

    def forward(self, x_seq: Tensor, adj: Tensor) -> Tensor:
        """adj argument is ignored — learned attention gate is used instead."""
        B, T, N, F_dim = x_seq.shape
        eff_adj = self._effective_adj()

        x_flat   = x_seq.view(B * T, N, F_dim)
        gcn_out  = self._gcn(x_flat, eff_adj)
        gcn_cat  = gcn_out.reshape(B * T, N * self.gcn_output)
        raw_flat = x_flat.reshape(B * T, N * F_dim)

        proj     = self.proj(torch.cat([gcn_cat, raw_flat], dim=-1))
        proj_seq = proj.view(B, T, -1)

        lstm_out, _ = self.lstm(proj_seq)
        attn_w   = F.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
        context  = (lstm_out * attn_w.unsqueeze(-1)).sum(dim=1)
        return self.classifier(context)


# ── GNN with attention-style adaptive adjacency (skeleton × learned gate) ─────

class GNNAttentionAdjModel(nn.Module):
    """
    GNN-only where a fixed topology is multiplied by a learned symmetric gate
    per edge, then row-normalised. This is a lightweight form of
    attention-based adjacency learning (no transformer stack).
    """

    def __init__(
        self,
        node_feat_dim: int,
        n_nodes: int,
        n_classes: int,
        init_adj: Tensor,
    ):
        super().__init__()
        self.register_buffer("skeleton", init_adj.clone())
        self.edge_gate = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.gnn = GNNEncoder(node_feat_dim, n_nodes)
        self.classifier = nn.Sequential(
            nn.Linear(self.gnn.gcn_out_dim, MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(MLP_HIDDEN_DIM, n_classes),
        )

    def _effective_adj(self) -> Tensor:
        g = torch.sigmoid((self.edge_gate + self.edge_gate.T) * 0.5)
        a = self.skeleton * g
        rs = a.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return a / rs

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        a_eff = self._effective_adj()
        emb = self.gnn(x, a_eff)
        return self.classifier(emb)

