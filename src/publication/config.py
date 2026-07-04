"""Load publication experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "publication_experiments.yaml"


def load_config(path: Path | str | None = None, smoke: bool = False) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    cfg["_root"] = str(ROOT)
    cfg["_smoke"] = smoke
    if smoke:
        cfg["datasets"] = cfg.get("smoke", {}).get("datasets", ["pamap2"])
        cfg["training"] = {**cfg["training"], "num_epochs": cfg["smoke"]["num_epochs"]}
        cfg["_max_folds"] = cfg["smoke"]["max_folds"]
        cfg["_max_windows_per_subject"] = cfg["smoke"]["max_windows_per_subject"]
        cfg["models"] = cfg["smoke"].get("models", cfg["models"])
    else:
        cfg["_max_folds"] = None
        cfg["_max_windows_per_subject"] = None
    return cfg


def results_dir(cfg: dict, experiment: str) -> Path:
    p = Path(cfg["_root"]) / cfg["results_root"] / experiment
    p.mkdir(parents=True, exist_ok=True)
    return p


def manuscript_dirs(cfg: dict) -> tuple[Path, Path]:
    root = Path(cfg["_root"]) / cfg["results_root"]
    tables = root / "manuscript_tables"
    figures = root / "manuscript_figures"
    logs = root / "logs"
    for d in (tables, figures, logs):
        d.mkdir(parents=True, exist_ok=True)
    return tables, figures
