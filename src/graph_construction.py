"""
Graph construction for HAR.

Each window is represented as a graph where nodes = sensor units.
Edges are initialised from physical proximity (fixed) or learned
from the data (learnable / attention-based), as described in the proposal.

For PAMAP2: default graph is hybrid:
  - 3 body-location aggregate nodes: hand/wrist, chest, ankle
  - one channel node per selected sensor channel at each location
  - optional global heart-rate node
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


def build_pamap2_body_location_adj() -> Tensor:
    """Legacy fixed adjacency for PAMAP2 body locations only (wrist=0, chest=1, ankle=2)."""
    from src.config import PAMAP2_EDGES
    return build_fixed_adj(PAMAP2_EDGES, n_nodes=3)


def build_pamap2_adj(n_channels: int | None = None) -> Tensor:
    """Fixed adjacency for PAMAP2.

    When n_channels is provided, build the hybrid body-location + channel graph.
    When omitted, return the legacy 3-node body-location graph for backwards
    compatibility with older scripts.
    """
    if n_channels is None:
        return build_pamap2_body_location_adj()
    return build_pamap2_hybrid_adj(n_channels)


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


def _pamap2_channel_metadata(n_channels: int) -> tuple[list[dict], bool]:
    """Infer PAMAP2 per-location channel metadata from canonical column order."""
    has_hr = n_channels % 3 == 1
    imu_channels = n_channels - 1 if has_hr else n_channels
    if imu_channels % 3 != 0:
        raise ValueError(
            "PAMAP2 graph features require IMU channels divisible by 3, "
            f"with optional trailing heart rate; got {n_channels}"
        )
    per_location = imu_channels // 3
    if per_location == 3:
        channel_defs = [
            ("acc16", "x"),
            ("acc16", "y"),
            ("acc16", "z"),
        ]
    elif per_location == 6:
        channel_defs = [
            ("acc16", "x"),
            ("acc16", "y"),
            ("acc16", "z"),
            ("gyro", "x"),
            ("gyro", "y"),
            ("gyro", "z"),
        ]
    elif per_location == 12:
        channel_defs = [
            ("acc16", "x"),
            ("acc16", "y"),
            ("acc16", "z"),
            ("acc6", "x"),
            ("acc6", "y"),
            ("acc6", "z"),
            ("gyro", "x"),
            ("gyro", "y"),
            ("gyro", "z"),
            ("mag", "x"),
            ("mag", "y"),
            ("mag", "z"),
        ]
    else:
        channel_defs = [(f"ch{i}", "none") for i in range(per_location)]

    locations = ["hand", "chest", "ankle"]
    meta: list[dict] = []
    raw_index = 0
    for loc_i, location in enumerate(locations):
        for local_i, (modality, axis) in enumerate(channel_defs):
            meta.append({
                "raw_index": raw_index,
                "location_index": loc_i,
                "location": location,
                "local_channel_index": local_i,
                "modality": modality,
                "axis": axis,
            })
            raw_index += 1
    return meta, has_hr


def _node_context(
    *,
    node_type: str,
    location: str,
    modality: str,
    axis: str,
) -> np.ndarray:
    """Small categorical context vector appended to statistical node features."""
    type_values = ["location", "channel", "global"]
    location_values = ["hand", "chest", "ankle", "global"]
    modality_values = ["aggregate", "acc16", "gyro", "hr", "other"]
    axis_values = ["aggregate", "x", "y", "z", "none"]

    def one_hot(value: str, values: list[str]) -> list[float]:
        if value not in values:
            value = values[-1]
        return [1.0 if value == v else 0.0 for v in values]

    return np.asarray(
        one_hot(node_type, type_values)
        + one_hot(location, location_values)
        + one_hot(modality, modality_values)
        + one_hot(axis, axis_values),
        dtype=np.float32,
    )


def _hybrid_feature(stats: np.ndarray, *, node_type: str, location: str, modality: str, axis: str) -> np.ndarray:
    return np.concatenate([
        stats.astype(np.float32),
        _node_context(node_type=node_type, location=location, modality=modality, axis=axis),
    ]).astype(np.float32)


def window_to_node_features_pamap2_body_location(window: np.ndarray) -> np.ndarray:
    """
    Map a (WINDOW_SIZE, n_channels) window to node feature matrix (3, feat_dim).

    Assumes features are ordered: [wrist_channels..., chest_channels..., ankle_channels...]
    Each node gets 6 statistical descriptors × n_channels_per_node features.
    If a trailing global channel is present, it is interpreted as heart rate and
    its statistics are appended to every node's feature vector.
    """
    n_total = window.shape[1]
    if n_total % 3 == 0:
        n_per_node = n_total // 3
        hr_stats = None
        imu = window
    elif n_total % 3 == 1:
        n_per_node = (n_total - 1) // 3
        imu = window[:, : 3 * n_per_node]
        hr_stats = _node_stats(window[:, -1:])
    else:
        raise ValueError(
            f"PAMAP2 graph features require channels divisible by 3, or divisible by 3 plus one trailing global channel; got {n_total}"
        )

    nodes = [
        _node_stats(imu[:, : n_per_node]),                    # wrist
        _node_stats(imu[:, n_per_node: 2 * n_per_node]),      # chest
        _node_stats(imu[:, 2 * n_per_node:]),                 # ankle
    ]
    if hr_stats is not None:
        nodes = [np.concatenate([node, hr_stats]) for node in nodes]
    nodes = np.stack(nodes)
    return nodes.astype(np.float32)


def window_to_node_features_pamap2_hybrid(window: np.ndarray) -> np.ndarray:
    """Map a PAMAP2 window to a hybrid body-location + channel graph.

    Node order:
      0..2: location aggregate nodes [hand, chest, ankle]
      next: one channel node per selected location channel
      final optional node: global heart-rate node

    Every node has the same feature dimension:
      6 statistical descriptors + categorical node context.
    """
    n_total = int(window.shape[1])
    channel_meta, has_hr = _pamap2_channel_metadata(n_total)
    locations = ["hand", "chest", "ankle"]
    nodes: list[np.ndarray] = []

    for loc_i, location in enumerate(locations):
        loc_indices = [m["raw_index"] for m in channel_meta if m["location_index"] == loc_i]
        loc_segment = window[:, loc_indices]
        loc_stats = _node_stats(loc_segment.reshape(-1, 1))
        nodes.append(_hybrid_feature(
            loc_stats,
            node_type="location",
            location=location,
            modality="aggregate",
            axis="aggregate",
        ))

    for meta in channel_meta:
        stats = _node_stats(window[:, int(meta["raw_index"]): int(meta["raw_index"]) + 1])
        nodes.append(_hybrid_feature(
            stats,
            node_type="channel",
            location=str(meta["location"]),
            modality=str(meta["modality"]),
            axis=str(meta["axis"]),
        ))

    if has_hr:
        hr_stats = _node_stats(window[:, -1:])
        nodes.append(_hybrid_feature(
            hr_stats,
            node_type="global",
            location="global",
            modality="hr",
            axis="none",
        ))

    return np.stack(nodes).astype(np.float32)


def build_pamap2_hybrid_adj(n_channels: int) -> Tensor:
    """Build normalized adjacency for the hybrid PAMAP2 graph."""
    channel_meta, has_hr = _pamap2_channel_metadata(int(n_channels))
    n_location_nodes = 3
    n_channel_nodes = len(channel_meta)
    hr_node = n_location_nodes + n_channel_nodes if has_hr else None
    n_nodes = n_location_nodes + n_channel_nodes + (1 if has_hr else 0)
    edges: set[tuple[int, int]] = set()

    # Body topology among location aggregate nodes.
    for edge in [(0, 1), (1, 2), (0, 2)]:
        edges.add(edge)

    # Membership edges: each channel belongs to one body location.
    for channel_i, meta in enumerate(channel_meta):
        node_i = n_location_nodes + channel_i
        loc_i = int(meta["location_index"])
        edges.add((loc_i, node_i))

    # Within-location channel interactions.
    for loc_i in range(3):
        loc_nodes = [
            n_location_nodes + i
            for i, meta in enumerate(channel_meta)
            if int(meta["location_index"]) == loc_i
        ]
        for i, src in enumerate(loc_nodes):
            for dst in loc_nodes[i + 1:]:
                edges.add((src, dst))

    # Same modality/axis across body locations, e.g. hand gyro_x <-> chest gyro_x.
    for i, left in enumerate(channel_meta):
        for j, right in enumerate(channel_meta[i + 1:], start=i + 1):
            if (
                left["modality"] == right["modality"]
                and left["axis"] == right["axis"]
                and left["location_index"] != right["location_index"]
            ):
                edges.add((n_location_nodes + i, n_location_nodes + j))

    # Heart rate is global physiology; connect it to body-location aggregate nodes.
    if hr_node is not None:
        for loc_i in range(3):
            edges.add((hr_node, loc_i))

    return build_fixed_adj(sorted(edges), n_nodes=n_nodes)


def window_to_node_features_pamap2(window: np.ndarray) -> np.ndarray:
    """Default PAMAP2 graph features: hybrid body-location + channel graph."""
    return window_to_node_features_pamap2_hybrid(window)


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
