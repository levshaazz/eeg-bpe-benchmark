"""
EEG During Mental Arithmetic Tasks (PhysioNet)
================================================
- 36 subjects, 19 EEG channels (10-20 system), 500 Hz
- Tasks: baseline rest vs. serial subtraction (cognitive load)
- Source: PhysioNet — direct download via wfdb or HTTP

Reference:
    Zyma et al. (2019). Electroencephalograms during Mental Arithmetic Task Performance.
"""

import os
import sys
# Force UTF-8 output so Unicode chars (─, ✓, ✗, etc.) work on Windows cp1251 consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import logging
import shutil
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "mental_arithmetic"

# PhysioNet direct download base URL
PHYSIONET_BASE = "https://physionet.org/files/eegmat/1.0.0/"


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def download_mental_arithmetic(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Download EEG During Mental Arithmetic dataset from PhysioNet.

    Uses wfdb if available, otherwise falls back to HTTP download.

    Parameters
    ----------
    data_dir : Path
        Directory to store downloaded data.

    Returns
    -------
    Path
        Path to data directory.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EEG During Mental Arithmetic Tasks")
    print(f"  36 subjects · 19 EEG · 500 Hz · rest vs. cognitive load")
    print(f"  Estimated size: ~250 MB")
    print(f"{'='*60}\n")

    # Check if already downloaded
    edf_files = list(data_dir.rglob("*.edf"))
    if len(edf_files) >= 36:
        logger.info(f"Data already downloaded: {len(edf_files)} EDF files found in {data_dir}")
        data_size = _dir_size_mb(data_dir)
        print(f"  ✓ Already downloaded: {len(edf_files)} EDF files ({data_size:.1f} MB)")
        return data_dir

    t0 = time.time()

    # Try wfdb first (cleanest approach)
    try:
        import wfdb

        print("  Method: wfdb.dl_database()")
        print("  Downloading all files from PhysioNet...\n")

        # wfdb.dl_database doesn't have built-in progress, so we wrap it
        # by showing an indeterminate progress bar
        pbar = tqdm(
            total=None,
            desc="Mental-Arith (wfdb)",
            unit="file",
            bar_format="{l_bar}{bar}| {n_fmt} files [{elapsed}]",
            colour="yellow",
        )

        # Count files before/after to update progress
        files_before = sum(1 for _ in data_dir.rglob("*") if _.is_file())

        wfdb.dl_database(
            "eegmat",
            dl_dir=str(data_dir),
            keep_subdirs=True,
        )

        files_after = sum(1 for _ in data_dir.rglob("*") if _.is_file())
        pbar.update(files_after - files_before)
        pbar.close()

        elapsed = time.time() - t0
        data_size = _dir_size_mb(data_dir)
        edf_files = list(data_dir.rglob("*.edf"))

        print(f"\n{'─'*60}")
        print(f"  ✓ Downloaded: {len(edf_files)} EDF files")
        print(f"  ⏱ Time: {elapsed:.1f}s")
        print(f"  💾 Data size: {data_size:.1f} MB")
        print(f"{'─'*60}\n")

        _validate_and_write_metadata(data_dir, elapsed)
        return data_dir

    except ImportError:
        logger.info("wfdb not available, trying HTTP download...")
    except Exception as e:
        logger.warning(f"wfdb download failed ({e}), trying HTTP...")

    # Fallback: HTTP download
    _download_via_http(data_dir)
    elapsed = time.time() - t0
    _validate_and_write_metadata(data_dir, elapsed)
    return data_dir


def _download_via_http(data_dir: Path) -> None:
    """Download dataset files via HTTP from PhysioNet."""
    try:
        import requests
    except ImportError:
        logger.error("requests required. Install with:\n  pip install requests")
        sys.exit(1)

    print("  Method: HTTP download (fallback)")
    print("  Fetching file listing from PhysioNet...\n")

    # PhysioNet provides RECORDS file listing all records
    # Note: for eegmat, RECORDS entries already include the .edf extension
    records_url = PHYSIONET_BASE + "RECORDS"
    resp = requests.get(records_url)
    resp.raise_for_status()
    # Each line is a filename like "Subject00_1.edf" — download as-is
    records = [line.strip() for line in resp.text.strip().split('\n') if line.strip()]

    # Also grab aux metadata files
    extra_files = ["subject-info.csv", "README.txt", "SHA256SUMS.txt"]

    all_files = records + extra_files
    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    pbar = tqdm(
        all_files,
        desc="Mental-Arith (HTTP)",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]",
        colour="yellow",
    )
    for filename in pbar:
        file_url = PHYSIONET_BASE + filename
        file_path = data_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.exists() and file_path.stat().st_size > 0:
            skipped += 1
            continue

        try:
            resp = requests.get(file_url, stream=True)
            if resp.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        f.write(chunk)
                        total_bytes += len(chunk)
                downloaded += 1
                pbar.set_postfix_str(
                    f"{downloaded} dl, {_format_bytes(total_bytes)}",
                    refresh=True,
                )
            elif resp.status_code == 404 and filename in extra_files:
                pass  # optional metadata — ignore 404
            else:
                failed += 1
                logger.warning(f"HTTP {resp.status_code} for {filename}")
        except Exception as e:
            logger.warning(f"Failed to download {file_url}: {e}")
            failed += 1

    print(f"\n  HTTP results: {downloaded} downloaded, {skipped} cached, {failed} failed")
    print(f"  Total transferred: {_format_bytes(total_bytes)}")


def _format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _validate_and_write_metadata(data_dir: Path, elapsed: float = 0) -> None:
    """Validate downloaded data and write metadata."""
    edf_files = list(data_dir.rglob("*.edf"))
    data_size = _dir_size_mb(data_dir)
    logger.info(f"Found {len(edf_files)} EDF files ({data_size:.1f} MB)")

    # Try reading a sample file
    sample_info = ""
    try:
        import mne
        if edf_files:
            raw = mne.io.read_raw_edf(str(edf_files[0]), preload=False, verbose=False)
            sample_info = (
                f"\nSample file: {edf_files[0].name}\n"
                f"  Channels: {len(raw.ch_names)} ({', '.join(raw.ch_names[:5])}...)\n"
                f"  Sampling rate: {raw.info['sfreq']} Hz\n"
                f"  Duration: {raw.times[-1]:.1f} s\n"
            )
    except Exception:
        pass

    meta_path = data_dir / "README.txt"
    with open(meta_path, "w") as f:
        f.write("EEG During Mental Arithmetic Tasks\n")
        f.write("=" * 45 + "\n")
        f.write(f"Subjects: 36\n")
        f.write(f"Channels: 19 EEG (10-20 system)\n")
        f.write(f"Sampling rate: 500 Hz\n")
        f.write(f"Task: baseline rest vs. serial subtraction (cognitive load)\n")
        f.write(f"Format: EDF\n")
        f.write(f"Source: PhysioNet (eeg-during-mental-arithmetic/1.0.0)\n")
        f.write(f"\nFiles found: {len(edf_files)} EDF\n")
        f.write(f"Data size: {data_size:.1f} MB\n")
        if elapsed > 0:
            f.write(f"Download time: {elapsed:.1f}s\n")
        f.write(sample_info)

    logger.info(f"Metadata written to {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download EEG Mental Arithmetic dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    download_mental_arithmetic(data_dir=args.data_dir)
