"""
SSVEP Nakanishi 2015 Dataset (via MOABB)
==========================================
- 10 subjects, 8 EEG channels, 256 Hz
- 12-class SSVEP (12 flickering frequencies)
- 15 blocks per subject, 12 trials per block
- Source: MOABB (auto-download)

Reference:
    Nakanishi et al. (2015). A comparison study of canonical correlation analysis
    based methods for detecting steady-state visual evoked potentials.
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

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "ssvep_nakanishi"


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def download_ssvep_nakanishi(data_dir: Path = DEFAULT_DATA_DIR) -> Path:
    """
    Download SSVEP Nakanishi 2015 dataset via MOABB.

    MOABB handles caching — re-running is safe.

    Parameters
    ----------
    data_dir : Path
        Directory to store metadata.

    Returns
    -------
    Path
        Path to metadata directory.
    """
    try:
        import moabb
        from moabb.datasets import Nakanishi2015
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

    logger.info("Initializing SSVEP Nakanishi 2015 dataset via MOABB...")
    dataset = Nakanishi2015()

    subjects = dataset.subject_list
    n_subjects = len(subjects)

    print(f"\n{'='*60}")
    print(f"  SSVEP Nakanishi 2015")
    print(f"  {n_subjects} subjects · 8 EEG · 256 Hz · 12-class SSVEP")
    print(f"  15 blocks × 12 trials · Estimated size: ~15 MB")
    print(f"{'='*60}\n")

    t0 = time.time()
    successful = 0
    failed_subjects = []

    # Expected .mat file location in MOABB cache
    mat_cache_dir = moabb_cache / "MNE-nakanishi-data" / "mnakanishi" / "12JFPM_SSVEP" / "raw" / "master" / "data"

    # Direct download URL (GitHub) used as fallback when get_data() fails (e.g. NumPy 2.0 incompatibility)
    GITHUB_BASE = "https://github.com/mnakanishi/12JFPM_SSVEP/raw/master/data/"

    pbar = tqdm(
        subjects,
        desc="SSVEP-Nakanishi",
        unit="subj",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} subjects [{elapsed}<{remaining}, {rate_fmt}]",
        colour="red",
    )
    for subject_id in pbar:
        pbar.set_postfix_str(f"S{subject_id:02d}", refresh=True)

        # Check if .mat file already in cache (no need to call get_data())
        mat_file = mat_cache_dir / f"s{subject_id}.mat"
        if mat_file.exists():
            successful += 1
            pbar.set_postfix_str(f"S{subject_id:02d} ✓ cached", refresh=True)
            continue

        try:
            data = dataset.get_data(subjects=[subject_id])
            n_sessions = len(data[subject_id])
            n_runs = sum(len(runs) for runs in data[subject_id].values())
            successful += 1
            pbar.set_postfix_str(f"S{subject_id:02d} ✓ {n_sessions}sess/{n_runs}runs", refresh=True)
        except Exception as e:
            # Fallback: direct HTTP download from GitHub
            try:
                import urllib.request
                mat_cache_dir.mkdir(parents=True, exist_ok=True)
                url = GITHUB_BASE + f"s{subject_id}.mat"
                urllib.request.urlretrieve(url, mat_file)
                successful += 1
                pbar.set_postfix_str(f"S{subject_id:02d} ✓ direct", refresh=True)
                logger.info(f"  Subject {subject_id}: downloaded directly from GitHub")
            except Exception as e2:
                failed_subjects.append(subject_id)
                pbar.set_postfix_str(f"S{subject_id:02d} ✗", refresh=True)
                logger.error(f"  Subject {subject_id} FAILED: {e} | fallback: {e2}")

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
        f.write("SSVEP Nakanishi 2015 Dataset\n")
        f.write("=" * 35 + "\n")
        f.write(f"Subjects: {subjects}\n")
        f.write(f"Channels: 8 EEG (occipital)\n")
        f.write(f"Sampling rate: 256 Hz\n")
        f.write(f"Paradigm: SSVEP, 12 flickering frequencies\n")
        f.write(f"Blocks: 15 per subject\n")
        f.write(f"Trials: 12 per block (one per frequency)\n")
        f.write(f"\nData cached by MOABB in: {moabb_cache}\n")
        f.write(f"Use MOABB API to load: moabb.datasets.Nakanishi2015()\n")
        f.write(f"\nDownloaded: {successful}/{n_subjects} subjects in {elapsed:.1f}s\n")
        f.write(f"Cache size: {cache_size:.1f} MB\n")
        if failed_subjects:
            f.write(f"Failed subjects: {failed_subjects}\n")

    logger.info(f"Metadata written to {meta_path}")
    return data_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SSVEP Nakanishi 2015 dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    download_ssvep_nakanishi(data_dir=args.data_dir)
