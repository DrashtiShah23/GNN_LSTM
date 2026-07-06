"""
Dataset classes (torch.utils.data.Dataset) for the HAR project.

HARWindowDataset    — flat windows, for LSTM-only / classical baselines
HARWindowDataset2D  — (timesteps, channels) windows, for CNN1D model
HARGraphDataset     — windows as graphs, for GNN and GNN+LSTM models  [LAZY]
HARSequenceDataset  — sequences of windows, for the full GNN+LSTM model [LAZY]

Design note — lazy loading
--------------------------
HARGraphDataset and HARSequenceDataset previously pre-built all node-feature
tensors at __init__ time. For HHAR (454 K windows) this caused multi-minute
hangs and ~4 GB of RAM usage before any training started.

Both classes now store only the raw numpy arrays and compute node features
inside __getitem__ on demand. An optional LRU-style cache can be enabled per
dataset instance to amortise repeated accesses (e.g. multiple epochs) at the
cost of memory proportional to the cache size.

  HARGraphDataset(X, y, dataset="hhar", cache=True)
      cache=True   → unlimited in-memory cache (all items, same RAM as before
                     but filled gradually rather than upfront)
      cache=False  → no caching; compute on every access (lower RAM, slower)
      cache=N      → keep the N most-recently-used items (LRU)
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from src.graph_construction import (
    window_to_node_features_pamap2,
    window_to_node_features_hhar,
    build_pamap2_adj,
    build_hhar_adj,
)


# ── Flat window dataset ───────────────────────────────────────────────────────

class HARWindowDataset(Dataset):
    """
    Returns (window_flat, label) where window_flat is (WINDOW_SIZE * n_channels,).
    Used for classical baselines and LSTM-only model.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, WINDOW_SIZE, n_channels)
        self.X = torch.tensor(X.reshape(len(X), -1), dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ── Graph window dataset ──────────────────────────────────────────────────────

class HARGraphDataset(Dataset):
    """
    Returns (node_features, adj, label).
    node_features : (n_nodes, node_feat_dim)  — computed lazily in __getitem__
    adj           : (n_nodes, n_nodes)

    Used for GNN-only ablation.

    Lazy loading
    ------------
    Node features are computed on-demand rather than pre-built for all N
    windows at construction time.  This reduces init time from O(N) to O(1)
    and avoids the multi-minute hang / ~4 GB RAM spike for HHAR (454 K windows).

    cache parameter
    ---------------
    cache=True  → unlimited dict cache — same total RAM as before, filled
                  lazily; fastest for multi-epoch training
    cache=False → no caching — constant RAM, recomputes every access
    cache=N     → LRU-bounded dict; keeps the N most-recently used items
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset: str = "pamap2",
        cache: Union[bool, int] = True,
    ):
        self.X_raw  = X                                    # (N, T, C) — raw numpy, no copy
        self.y      = torch.tensor(y, dtype=torch.long)
        self.dataset = dataset.lower()

        if self.dataset == "pamap2":
            self._node_fn = window_to_node_features_pamap2
            self.adj = build_pamap2_adj()
        else:
            self._node_fn = window_to_node_features_hhar
            self.adj = build_hhar_adj()

        # Cache setup
        self._cache_limit: Optional[int] = None          # None = unlimited
        self._cache: dict = {}
        if cache is True:
            self._cache_limit = None                      # unlimited
        elif cache is False:
            self._cache_limit = 0                         # disabled
        else:
            self._cache_limit = int(cache)                # bounded LRU

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self._cache_limit != 0 and idx in self._cache:
            node_feats = self._cache[idx]
        else:
            node_feats = torch.tensor(
                self._node_fn(self.X_raw[idx]), dtype=torch.float32
            )
            if self._cache_limit != 0:
                # Bounded LRU eviction: drop oldest entry when over limit
                if self._cache_limit is not None and len(self._cache) >= self._cache_limit:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                self._cache[idx] = node_feats
        return node_feats, self.adj, self.y[idx]


# ── Sequence dataset for GNN+LSTM ─────────────────────────────────────────────

class HARSequenceDataset(Dataset):
    """
    Groups consecutive windows into sequences of length `seq_len`,
    respecting subject boundaries so sequences never span two subjects.

    Returns (x_seq, adj, label) where
      x_seq : (seq_len, n_nodes, node_feat_dim)

    The label is the majority label of the sequence.

    Lazy loading
    ------------
    Node features are computed per-window inside __getitem__ rather than
    pre-built for all N windows at init.  Sequences store raw window indices
    into X; features are computed when a sequence is fetched.

    For PAMAP2 (15 K windows) the difference is modest, but it keeps the
    class consistent with HARGraphDataset and avoids surprises if PAMAP2
    ever grows or if the class is reused for HHAR.

    cache parameter — same semantics as HARGraphDataset.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray | None = None,
        dataset: str = "pamap2",
        seq_len: int = 10,
        cache: Union[bool, int] = True,
    ):
        self.X_raw   = X                                   # (N, T, C)
        self.seq_len = seq_len
        self.dataset = dataset.lower()

        if self.dataset == "pamap2":
            self._node_fn = window_to_node_features_pamap2
            self.adj = build_pamap2_adj()
        else:
            self._node_fn = window_to_node_features_hhar
            self.adj = build_hhar_adj()

        # Cache setup (per-window, not per-sequence)
        self._cache_limit: Optional[int] = None
        self._cache: dict = {}
        if cache is True:
            self._cache_limit = None
        elif cache is False:
            self._cache_limit = 0
        else:
            self._cache_limit = int(cache)

        # Build index lists (raw window indices per sequence) — O(N), fast
        if subjects is None:
            subjects = np.zeros(len(X), dtype=np.int64)

        seq_indices, seq_labels = [], []
        for subj in np.unique(subjects):
            mask = np.where(subjects == subj)[0]
            subj_y = y[mask]
            for start in range(0, len(mask) - seq_len + 1, seq_len):
                win_indices = mask[start: start + seq_len]   # absolute indices into X
                label_window = subj_y[start: start + seq_len]
                label = int(np.bincount(label_window).argmax())
                seq_indices.append(win_indices)
                seq_labels.append(label)

        # seq_indices: list of 1-D arrays of length seq_len
        self._seq_indices = seq_indices                    # list of np arrays
        self.labels = torch.tensor(seq_labels, dtype=torch.long)

    # -- helpers -------------------------------------------------------

    def _get_window_feat(self, win_idx: int) -> torch.Tensor:
        """Return node features for a single raw window index (with optional cache)."""
        if self._cache_limit != 0 and win_idx in self._cache:
            return self._cache[win_idx]
        feat = torch.tensor(
            self._node_fn(self.X_raw[win_idx]), dtype=torch.float32
        )
        if self._cache_limit != 0:
            if self._cache_limit is not None and len(self._cache) >= self._cache_limit:
                del self._cache[next(iter(self._cache))]
            self._cache[win_idx] = feat
        return feat

    # -- Dataset interface -------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        win_indices = self._seq_indices[idx]               # shape (seq_len,)
        frames = [self._get_window_feat(int(w)) for w in win_indices]
        x_seq  = torch.stack(frames, dim=0)                # (seq_len, n_nodes, feat_dim)
        return x_seq, self.adj, self.labels[idx]


# ── 2D window dataset (for CNN1D) ─────────────────────────────────────────────

class HARWindowDataset2D(Dataset):
    """
    Returns (window_2d, label) where window_2d is (timesteps, n_channels).
    Used for the CNN1D model which needs the temporal dimension preserved.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, WINDOW_SIZE, n_channels) — keep shape as-is
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
