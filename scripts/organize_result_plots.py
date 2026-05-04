#!/usr/bin/env python3
"""
Move less-essential figures out of results/plots/ into results/plots/supplementary/
so the main folder stays small for browsing / slides.

Does not change metrics JSON — only PNG layout on disk. Safe to re-run.

Examples:
  .venv/bin/python scripts/organize_result_plots.py --dry-run
  .venv/bin/python scripts/organize_result_plots.py --preset talk --move
  .venv/bin/python scripts/organize_result_plots.py --undo

Presets:
  talk      — keep a tight set for a short talk; move the rest to supplementary/
  extra-cm  — move only LSTM/GNN CMs (keep GNN+LSTM + CNN1D + comparisons + summary)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "results" / "plots"
SUPP = PLOTS / "supplementary"

# Figures most decks need in the root plots/ folder.
KEEP_TALK: frozenset[str] = frozenset(
    {
        "model_comparison_pamap2.png",
        "model_comparison_hhar.png",
        "cross_dataset_comparison.png",
        "cm_gnnlstm_pamap2.png",
        "cm_gnnlstm_hhar.png",
        "cm_cnn1d_pamap2.png",
        "cm_cnn1d_hhar.png",
        "presentation_all_models_summary.png",
    }
)

MOVE_EXTRA_CM: frozenset[str] = frozenset(
    {
        "cm_lstm_pamap2.png",
        "cm_gnn_pamap2.png",
        "cm_lstm_hhar.png",
        "cm_gnn_hhar.png",
    }
)


def collect_moves(preset: str) -> list[Path]:
    if not PLOTS.is_dir():
        return []
    moves: list[Path] = []
    for p in sorted(PLOTS.glob("*.png")):
        if preset == "talk":
            if p.name not in KEEP_TALK:
                moves.append(p)
        elif preset == "extra-cm":
            if p.name in MOVE_EXTRA_CM:
                moves.append(p)
        else:
            raise ValueError(preset)
    return moves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preset",
        choices=["talk", "extra-cm"],
        default="talk",
        help="talk: keep 8 core figures; extra-cm: move only LSTM/GNN CMs",
    )
    ap.add_argument("--move", action="store_true", help="Apply moves (default is dry-run only)")
    ap.add_argument(
        "--undo",
        action="store_true",
        help="Move all PNGs from supplementary/ back to plots/ (skip if destination exists)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Never touch files (overrides --move for safety)",
    )
    args = ap.parse_args()

    apply = args.move and not args.dry_run

    if args.undo:
        if not SUPP.is_dir():
            print("Nothing to undo (no supplementary/).", file=sys.stderr)
            return
        PLOTS.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in sorted(SUPP.glob("*.png")):
            dest = PLOTS / p.name
            if dest.exists():
                print(f"  skip (exists in root): {p.name}")
                continue
            if apply:
                shutil.move(str(p), str(dest))
                print(f"  restored → plots/{p.name}")
            else:
                print(f"  would restore → plots/{p.name}")
            n += 1
        if not apply and n:
            print("\nDry-run: pass --move (and omit --dry-run) to restore.", file=sys.stderr)
        print(f"Done ({n} file(s)).")
        return

    moves = collect_moves(args.preset)
    if not moves:
        print(f"No PNGs to relocate under {PLOTS} for preset={args.preset!r}.")
        return

    SUPP.mkdir(parents=True, exist_ok=True)
    for p in moves:
        dest = SUPP / p.name
        if not apply:
            print(f"  would move: {p.name}  →  supplementary/{p.name}")
        else:
            if dest.exists():
                print(f"  skip (already in supplementary): {p.name}")
                continue
            shutil.move(str(p), str(dest))
            print(f"  moved: {p.name}")

    if not apply:
        print(f"\nDry-run: {len(moves)} file(s). Pass --move to apply.")
    else:
        print(f"\nMoved {len(moves)} file(s) → {SUPP}")


if __name__ == "__main__":
    main()
