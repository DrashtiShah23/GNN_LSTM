"""
Data download helpers for HHAR and PAMAP2 datasets.

Uses the official UCI ML Repo API (ucimlrepo) to fetch datasets reliably.
Falls back to direct ZIP download if the API is unavailable.

HHAR   — UCI dataset id=344
PAMAP2 — UCI dataset id=231
"""

import os
import urllib.request
import zipfile
from pathlib import Path

from src.config import HHAR_RAW_DIR, PAMAP2_RAW_DIR

# Updated direct-download URLs (new UCI static hosting)
HHAR_ZIP_URL   = "https://archive.ics.uci.edu/static/public/344/heterogeneity+activity+recognition.zip"
PAMAP2_ZIP_URL = "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip"


def _download(url: str, dest_path: str) -> None:
    """Download a file with a simple progress indicator."""
    print(f"  Downloading {url} …")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 256  # 256 KB chunks
        while True:
            block = response.read(chunk)
            if not block:
                break
            f.write(block)
            downloaded += len(block)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:5.1f}%  ({downloaded // 1_048_576} / {total // 1_048_576} MB)", end="", flush=True)
    print(f"\n  Saved → {dest_path}")


def _extract_zip(zip_path: str, extract_to: str) -> None:
    print(f"  Extracting {zip_path} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted → {extract_to}")


# ── ucimlrepo helpers ──────────────────────────────────────────────────────────

def _save_ucimlrepo_hhar(out_dir: Path) -> None:
    """Fetch HHAR via ucimlrepo and write CSVs to out_dir."""
    from ucimlrepo import fetch_ucirepo
    print("  Fetching HHAR via ucimlrepo (id=344) …")
    ds = fetch_ucirepo(id=344)
    df = ds.data.original if hasattr(ds.data, "original") else ds.data.features.join(ds.data.targets)
    csv_path = out_dir / "Activity recognition exp" 
    csv_path.mkdir(parents=True, exist_ok=True)
    # Split by source column if present, otherwise save whole dataframe
    if "source" in df.columns:
        for src, grp in df.groupby("source"):
            grp.to_csv(csv_path / f"{src}.csv", index=False)
    else:
        # Save as combined Phones_accelerometer.csv (rename gt/User if needed)
        df.to_csv(csv_path / "Phones_accelerometer.csv", index=False)
    print(f"  HHAR saved to {csv_path}")


def _save_ucimlrepo_pamap2(out_dir: Path) -> None:
    """Fetch PAMAP2 via ucimlrepo and write .dat files to out_dir."""
    from ucimlrepo import fetch_ucirepo
    print("  Fetching PAMAP2 via ucimlrepo (id=231) …")
    ds = fetch_ucirepo(id=231)
    df = ds.data.original if hasattr(ds.data, "original") else ds.data.features.join(ds.data.targets)
    protocol_dir = out_dir / "PAMAP2_Dataset" / "Protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    subject_col = next((c for c in df.columns if "subject" in c.lower()), None)
    if subject_col:
        for subj_id, grp in df.groupby(subject_col):
            grp.drop(columns=[subject_col]).to_csv(
                protocol_dir / f"subject10{int(subj_id)}.dat", sep=" ", index=False, header=False
            )
    else:
        df.to_csv(protocol_dir / "subject101.dat", sep=" ", index=False, header=False)
    print(f"  PAMAP2 saved to {protocol_dir}")


# ── Public API ────────────────────────────────────────────────────────────────

def download_hhar(force: bool = False) -> None:
    """Download and extract the HHAR dataset."""
    out_dir = Path(HHAR_RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.iterdir()):
        print("HHAR data already present — skipping download.")
        return

    # Try ucimlrepo first, fall back to ZIP
    try:
        import ucimlrepo  # noqa: F401
        _save_ucimlrepo_hhar(out_dir)
    except Exception as e:
        print(f"  ucimlrepo failed ({e}), trying ZIP download …")
        zip_path = out_dir / "HHAR.zip"
        _download(HHAR_ZIP_URL, str(zip_path))
        _extract_zip(str(zip_path), str(out_dir))
        zip_path.unlink(missing_ok=True)

    print("HHAR download complete.\n")


def download_pamap2(force: bool = False) -> None:
    """Download and extract the PAMAP2 dataset."""
    out_dir = Path(PAMAP2_RAW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not force and any(out_dir.iterdir()):
        print("PAMAP2 data already present — skipping download.")
        return

    # Try ucimlrepo first, fall back to ZIP
    try:
        import ucimlrepo  # noqa: F401
        _save_ucimlrepo_pamap2(out_dir)
    except Exception as e:
        print(f"  ucimlrepo failed ({e}), trying ZIP download …")
        zip_path = out_dir / "PAMAP2.zip"
        _download(PAMAP2_ZIP_URL, str(zip_path))
        _extract_zip(str(zip_path), str(out_dir))
        zip_path.unlink(missing_ok=True)

    print("PAMAP2 download complete.\n")


if __name__ == "__main__":
    download_hhar()
    download_pamap2()
