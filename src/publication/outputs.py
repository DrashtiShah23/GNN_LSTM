"""Output writers and logging for publication experiments."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_dir / f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def save_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or (list(rows[0].keys()) if rows else [])
    if not fields:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        if rows:
            w.writerows(rows)


def save_markdown_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    if not rows:
        return
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def save_predictions_bundle(
    out_dir: Path,
    tag: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray | None,
    subjects: np.ndarray | None,
    metadata: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{tag}_y_true.npy", y_true)
    np.save(out_dir / f"{tag}_y_pred.npy", y_pred)
    if probs is not None:
        np.save(out_dir / f"{tag}_probs.npy", probs)
    if subjects is not None:
        np.save(out_dir / f"{tag}_subjects.npy", subjects)
    save_json(out_dir / f"{tag}_metadata.json", metadata)


def copy_to_manuscript(src: Path, manuscript_dir: Path, name: str | None = None) -> Path:
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    dest = manuscript_dir / (name or src.name)
    dest.write_bytes(src.read_bytes())
    return dest


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
