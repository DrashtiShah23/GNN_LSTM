"""Shared CLI and helpers for publication experiment scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.publication.config import load_config, results_dir, manuscript_dirs
from src.publication.seeds import set_all_seeds


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="Path to publication_experiments.yaml")
    p.add_argument("--smoke", action="store_true", help="Fast subset run for pipeline verification")
    p.add_argument("--datasets", nargs="+", default=None, help="Override datasets list")
    p.add_argument("--resume", action="store_true", help="Skip runs whose prediction metadata already exists")
    return p


def init_experiment(name: str, args) -> tuple[dict, Path, object]:
    cfg = load_config(args.config, smoke=args.smoke)
    if args.datasets:
        cfg["datasets"] = args.datasets
    set_all_seeds(cfg["seed"])
    out = results_dir(cfg, name)
    tables, figures = manuscript_dirs(cfg)
    from src.publication.outputs import setup_logger
    logger = setup_logger(name, Path(cfg["_root"]) / cfg["results_root"] / "logs")
    logger.info("Config: %s | smoke=%s | seed=%s", cfg.get("_config_path"), cfg["_smoke"], cfg["seed"])
    return cfg, out, logger
