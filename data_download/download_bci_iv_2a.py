"""
BCI Competition IV Dataset 2a — Motor Imagery (4 classes)
==========================================================
- 9 subjects, 22 EEG channels + 3 EOG, 250 Hz
- 4 classes: left hand, right hand, both feet, tongue
- Source: MOABB (auto-download from BNCI Horizon 2020)

Reference:
    Brunner et al. (2008). BCI Competition 2008 – Graz data set A.
"""

import os
import sys
# Force UTF-8 output so Unicode chars (─, ✓, ✗, etc.) work on Windows cp1251 consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import logging
from pathlib import Path

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default save location
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "bci_iv_2a"


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def download_bci_iv_2a(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Download BCI Competition IV 2a dataset via MOABB.

    MOABB handles caching automatically — re-running this script
    will not re-download if data already exists.

    Parameters
    ----------
    data_dir : Path
        Directory to store downloaded data. MOABB stores in its own cache,
        but we also save a metadata summary here.

    Returns
    -------
    Path
        Path to the directory containing the data.
    """
    try:
        import moabb
        from moabb.datasets import BNCI2014_001
        from moabb.utils import set_download_dir
    except ImportError:
        logger.error(
            "MOABB is required. Install with:\n"
            "  pip install moabb\n"
            "or:\n"
            "  pip install -r requirements.txt"
        )
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Set MOABB cache to our raw directory
    moabb_cache = data_dir.parent / "moabb_cache"
    moabb_cache.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(moabb_cache))

    logger.info("Initializing BCI Competition IV 2a dataset via MOABB...")
    dataset = BNCI2014_001()

    subjects = dataset.subject_list
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"  BCI Competition IV 2a — Motor Imagery")
    print(f"  {n_subjects} subjects · 22 EEG + 3 EOG · 250 Hz · 4 classes")
    print(f"  Estimated size: ~300 MB")
    print(f"{'='*60}\n")

    # Download data for all subjects with progress bar
    t0 = time.time()
    successful = 0
    failed_subjects = []

    pbar = tqdm(
        subjects,
        desc="BCI-IV-2a",
        unit="subj",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} subjects [{elapsed}<{remaining}, {rate_fmt}]",
        colour="green",
    )
    for subject_id in pbar:
        pbar.set_postfix_str(f"S{subject_id:02d}", refresh=True)
        try:
            data = dataset.get_data(subjects=[subject_id])
            n_sessions = len(data[subject_id])
            n_runs = sum(len(runs) for runs in data[subject_id].values())
            successful += 1
            pbar.set_postfix_str(f"S{subject_id:02d} ✓ {n_sessions}sess/{n_runs}runs", refresh=True)
        except Exception as e:
            failed_subjects.append(subject_id)
            pbar.set_postfix_str(f"S{subject_id:02d} ✗ {e}", refresh=True)
            logger.error(f"  Subject {subject_id} FAILED: {e}")

    elapsed = time.time() - t0
    cache_size = _dir_size_mb(moabb_cache)

    # Summary
    print(f"\n{'─'*60}")
    print(f"  ✓ Downloaded: {successful}/{n_subjects} subjects")
    if failed_subjects:
        print(f"  ✗ Failed: {failed_subjects}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/n_subjects:.1f}s/subject)")
    print(f"  💾 Cache size: {cache_size:.1f} MB")
    print(f"{'─'*60}\n")

    # Write metadata
    meta_path = data_dir / "README.txt"
    with open(meta_path, "w") as f:
        f.write("BCI Competition IV Dataset 2a\n")
        f.write("=" * 40 + "\n")
        f.write(f"Subjects: {subjects}\n")
        f.write(f"Channels: 22 EEG + 3 EOG\n")
        f.write(f"Sampling rate: 250 Hz\n")
        f.write(f"Classes: left hand, right hand, both feet, tongue\n")
        f.write(f"Sessions: 2 per subject\n")
        f.write(f"\nData is cached by MOABB in: {moabb_cache}\n")
        f.write(f"Use MOABB API to load: moabb.datasets.BNCI2014_001()\n")
        f.write(f"\nDownloaded: {successful}/{n_subjects} subjects in {elapsed:.1f}s\n")
        f.write(f"Cache size: {cache_size:.1f} MB\n")
        if failed_subjects:
            f.write(f"Failed subjects: {failed_subjects}\n")

    logger.info(f"Metadata written to {meta_path}")
    return data_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download BCI Competition IV 2a dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory to store data metadata")
    args = parser.parse_args()

    download_bci_iv_2a(data_dir=args.data_dir)
