"""Re-export tables and significance after run_full_pipeline.py finishes."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv" / "bin" / "python"


def main() -> None:
    for script in ("export_midway_comparison.py", "loso_significance_and_pairs.py"):
        subprocess.run([str(PY), str(ROOT / "scripts" / script)], check=True, cwd=ROOT)
    print("Done: midway_comparison.* + loso_significance.json (if folds present) + confusion_pairs_pamap2.json")


if __name__ == "__main__":
    main()
