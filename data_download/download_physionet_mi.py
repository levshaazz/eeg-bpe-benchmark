"""
PhysioNet EEG Motor Movement/Imagery Dataset
==============================================
- 109 subjects, 64 EEG channels (10-10 system), 160 Hz
- Tasks: open/close fists (real + imagined), open/close feet (real + imagined)
- ~2 min per run, 14 runs per subject
- Source: PhysioNet (via MNE-Python auto-download)

Reference:
    Schalk et al. (2004). BCI2000: A General-Purpose Brain-Computer Interface (BCI) System.
    Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
"""

import os
import sys
# Force UTF-8 output so Unicode chars (─, ✓, ✗, etc.) work on Windows cp1251 consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Max simultaneous HTTP connections to PhysioNet — be polite to the server.
MAX_DOWNLOAD_WORKERS = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "physionet_mi"


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def download_physionet_mi(
    data_dir: Path = DEFAULT_DATA_DIR,
    subjects: list[int] | None = None,
) -> Path:
    """
    Download PhysioNet EEG Motor Movement/Imagery Dataset via MNE.

    MNE caches downloads automatically — re-running is safe.

    Parameters
    ----------
    data_dir : Path
        Directory to store metadata.
    subjects : list[int] | None
        List of subject IDs (1-109). None = all subjects.

    Returns
    -------
    Path
        Path to metadata directory.
    """
    try:
        import mne
        from mne.datasets import eegbci
    except ImportError:
        logger.error("MNE-Python is required. Install with:\n  pip install mne")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Pre-set MNE config to avoid interactive prompt
    mne.set_config("MNE_DATASETS_EEGBCI_PATH", str(data_dir), set_env=False)

    if subjects is None:
        subjects = list(range(1, 110))  # 1-109

    # Run descriptions (from PhysioNet documentation):
    # Runs 1:  Baseline, eyes open
    # Runs 2:  Baseline, eyes closed
    # Runs 3, 7, 11:  Motor execution — open and close left or right fist
    # Runs 4, 8, 12:  Motor imagery — imagine opening and closing left or right fist
    # Runs 5, 9, 13:  Motor execution — open and close both fists or both feet
    # Runs 6, 10, 14: Motor imagery — imagine opening and closing both fists or both feet
    all_runs = list(range(1, 15))  # runs 1-14

    total_files = len(subjects) * len(all_runs)

    print(f"\n{'='*60}")
    print(f"  PhysioNet Motor Movement/Imagery")
    print(f"  {len(subjects)} subjects · 64 EEG · 160 Hz · 14 runs each")
    print(f"  Total files: {total_files} · Estimated size: ~1.5 GB")
    print(f"{'='*60}\n")

    t0 = time.time()
    failed_subjects = []
    successful = 0
    total_downloaded_files = 0
    _lock = threading.Lock()

    def _download_subject(subject_id: int):
        """Worker: download all runs for one subject."""
        fnames = eegbci.load_data(subject_id, all_runs, path=str(data_dir))
        return subject_id, len(fnames)

    n_workers = min(len(subjects), MAX_DOWNLOAD_WORKERS)
    logger.info(f"Downloading {len(subjects)} subjects with {n_workers} parallel workers")

    pbar = tqdm(
        total=len(subjects),
        desc="PhysioNet-MI",
        unit="subj",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} subjects [{elapsed}<{remaining}, {rate_fmt}]",
        colour="blue",
    )
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_subject, s): s for s in subjects}
        for future in as_completed(futures):
            subj = futures[future]
            try:
                _, n_files = future.result()
                with _lock:
                    successful += 1
                    total_downloaded_files += n_files
                pbar.set_postfix_str(f"S{subj:03d} ✓ ({successful} done)", refresh=True)
            except Exception as e:
                logger.error(f"  S{subj:03d} FAILED: {e}")
                with _lock:
                    failed_subjects.append(subj)
                pbar.set_postfix_str(f"S{subj:03d} ✗", refresh=True)
            pbar.update(1)
    pbar.close()

    elapsed = time.time() - t0
    data_size = _dir_size_mb(data_dir)

    # Summary
    print(f"\n{'─'*60}")
    print(f"  ✓ Downloaded: {successful}/{len(subjects)} subjects ({total_downloaded_files} files)")
    if failed_subjects:
        print(f"  ✗ Failed: {failed_subjects}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/max(len(subjects),1):.1f}s/subject)")
    print(f"  💾 Data size: {data_size:.1f} MB")
    print(f"{'─'*60}\n")

    # Write metadata
    meta_path = data_dir / "README.txt"
    with open(meta_path, "w") as f:
        f.write("PhysioNet EEG Motor Movement/Imagery Dataset\n")
        f.write("=" * 50 + "\n")
        f.write(f"Subjects: {len(subjects)} (IDs 1-109)\n")
        f.write(f"Channels: 64 EEG (10-10 system)\n")
        f.write(f"Sampling rate: 160 Hz\n")
        f.write(f"Runs per subject: 14\n")
        f.write(f"  Runs 1, 2: baseline (eyes open / closed)\n")
        f.write(f"  Runs 3, 7, 11: motor execution (fists)\n")
        f.write(f"  Runs 4, 8, 12: motor imagery (fists)\n")
        f.write(f"  Runs 5, 9, 13: motor execution (fists/feet)\n")
        f.write(f"  Runs 6, 10, 14: motor imagery (fists/feet)\n")
        f.write(f"\nFormat: EDF (European Data Format)\n")
        f.write(f"\nDownloaded: {successful}/{len(subjects)} subjects in {elapsed:.1f}s\n")
        f.write(f"Total files: {total_downloaded_files}\n")
        f.write(f"Data size: {data_size:.1f} MB\n")
        if failed_subjects:
            f.write(f"\nFailed subjects: {failed_subjects}\n")

    logger.info(f"Metadata written to {meta_path}")
    return data_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download PhysioNet Motor Imagery dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs (1-109). Default: all.")
    args = parser.parse_args()

    download_physionet_mi(data_dir=args.data_dir, subjects=args.subjects)
