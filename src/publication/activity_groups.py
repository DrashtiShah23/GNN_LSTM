"""Health-relevant activity group mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.publication.metrics import compute_full_metrics


def load_activity_groups(config_path: Path | str) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_group_mapping(dataset: str, label_names: list[str], cfg: dict) -> tuple[dict[int, str], list[str]]:
    """Map fine-grained class index → group name."""
    ds_groups = cfg.get(dataset, {})
    name_to_group = {}
    for group, activities in ds_groups.items():
        for act in activities:
            name_to_group[act.lower()] = group

    class_to_group = {}
    unmapped = []
    for i, name in enumerate(label_names):
        key = name.lower().replace(" ", "_")
        if key in name_to_group:
            class_to_group[i] = name_to_group[key]
        else:
            class_to_group[i] = "Unmapped_or_Other"
            unmapped.append(name)

    groups = sorted(set(class_to_group.values()))
    return class_to_group, unmapped


def collapse_to_groups(y_true: np.ndarray, y_pred: np.ndarray, class_to_group: dict[int, str], groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    g2i = {g: i for i, g in enumerate(groups)}

    def map_y(y):
        out = []
        for v in y:
            g = class_to_group.get(int(v), "Unmapped_or_Other")
            out.append(g2i[g])
        return np.array(out)

    return map_y(y_true), map_y(y_pred)


def dominant_group_confusion(cm: list | np.ndarray, group_idx: int, group_labels: list[str]) -> str:
    """Most common off-diagonal group misclassification for a given true group."""
    mat = np.array(cm)
    if group_idx >= mat.shape[0]:
        return ""
    row = mat[group_idx].astype(float).copy()
    row[group_idx] = 0.0
    if row.max() <= 0:
        return ""
    pred_j = int(np.argmax(row))
    return f"{group_labels[group_idx]}→{group_labels[pred_j]}"
