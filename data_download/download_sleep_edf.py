"""
Sleep-EDF Expanded Dataset
============================
- 78 subjects (SC: 78 healthy, ST: 22 with sleep disorders)
- 2 EEG channels: Fpz-Cz, Pz-Oz (+ 1 EOG, 1 EMG)
- 100 Hz sampling rate
- 5 sleep stages: W, N1, N2, N3, REM
- Source: PhysioNet (via MNE-Python auto-download)

Reference:
    Kemp et al. (2000). Analysis of a sleep-dependent neuronal feedback loop.
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
# 4 is safer than 8: PhysioNet throttles aggressively above ~4 parallel connections,
# causing cascading read timeouts on the full remaining batch.
MAX_DOWNLOAD_WORKERS = 4

# Error substrings that indicate a recording/subject is permanently absent from the
# corpus (not transient network failures) — these should NOT be retried.
_PERMANENT_ABSENT_MSGS = (
    "not available in corpus",  # night 1 absent for some subjects (e.g. 36, 52)
    "Unknown subjects",          # subject ID not in dataset (e.g. 39, 68, 69, 78, 79)
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "raw" / "sleep_edf"


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def download_sleep_edf(
    data_dir: Path = DEFAULT_DATA_DIR,
    subjects: list[int] | None = None,
) -> Path:
    """
    Download Sleep-EDF Expanded dataset via MNE.

    Downloads the 'Sleep Cassette' (SC) subset by default — 78 healthy subjects,
    two full nights each. This is the standard benchmark subset.

    Parameters
    ----------
    data_dir : Path
        Directory to store metadata and pointer files.
    subjects : list[int] | None
        List of subject IDs (0-82). None = all available.

    Returns
    -------
    Path
        Path to metadata directory.
    """
    try:
        import mne
        from mne.datasets.sleep_physionet import age as sleep_age
    except ImportError:
        logger.error("MNE-Python is required. Install with:\n  pip install mne")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)

    # Pre-set MNE config to avoid interactive prompt
    mne.set_config("MNE_DATASETS_SLEEP_PHYSIONET_PATH", str(data_dir), set_env=False)

    if subjects is None:
        subjects = list(range(0, 83))  # SC subjects: 0-82

    print(f"\n{'='*60}")
    print(f"  Sleep-EDF Expanded (Sleep Cassette)")
    print(f"  {len(subjects)} subjects · 2 EEG + EOG + EMG · 100 Hz")
    print(f"  Up to 2 nights per subject · Estimated size: ~5 GB")
    print(f"{'='*60}\n")

    t0 = time.time()
    failed_subjects = []
    successful_recordings = 0
    successful_subjects = 0
    _lock = threading.Lock()

    def _fetch_with_retry(subject_id: int, night: int, max_retries: int = 3) -> None:
        """Fetch one recording, retrying transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                sleep_age.fetch_data(
                    subjects=[subject_id],
                    recording=[night],
                    path=str(data_dir),
                )
                return  # success
            except Exception as e:
                if any(msg in str(e) for msg in _PERMANENT_ABSENT_MSGS):
                    raise  # corpus absence — retrying won't help
                if attempt < max_retries - 1:
                    wait = 5 * (2 ** attempt)  # 5 s → 10 s → 20 s
                    logger.warning(
                        f"  SC{subject_id:02d} night {night}: transient error, "
                        f"retry {attempt+1}/{max_retries-1} in {wait}s — {e}"
                    )
                    time.sleep(wait)
                else:
                    raise  # exhausted retries

    def _download_subject(subject_id: int):
        """Worker: download up to 2 nights for one Sleep-EDF subject.

        Handles three outcome types:
        - "absent": subject/recording genuinely not in corpus → skip, don't count as failure
        - "failed": transient error exhausted retries → mark as failed
        - "ok": ≥1 recording downloaded
        Returns (subject_id, n_recordings_downloaded, absent: bool).
        """
        n_rec = 0
        night1_absent = False
        for night in [1, 2]:
            try:
                _fetch_with_retry(subject_id, night)
                n_rec += 1
            except Exception as e:
                err_str = str(e)
                if "Unknown subjects" in err_str:
                    # Subject does not exist in dataset at all — silent skip
                    return subject_id, 0, True  # absent
                if night == 1 and "not available in corpus" in err_str:
                    # Night 1 absent for this subject (e.g. SC36, SC52) — try night 2
                    night1_absent = True
                    continue
                if night == 2:
                    pass  # night 2 always optional
                else:
                    raise  # hard failure for night 1 on transient errors
        # If only night 2 existed (night1_absent) and it also failed → 0 recordings
        # but that's fine, not a corpus error
        if night1_absent and n_rec == 0:
            return subject_id, 0, True  # nothing available
        return subject_id, n_rec, False

    n_workers = min(len(subjects), MAX_DOWNLOAD_WORKERS)
    logger.info(f"Downloading {len(subjects)} Sleep-EDF subjects with {n_workers} parallel workers")

    pbar = tqdm(
        total=len(subjects),
        desc="Sleep-EDF",
        unit="subj",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} subjects [{elapsed}<{remaining}, {rate_fmt}]",
        colour="magenta",
    )
    absent_subjects = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_download_subject, s): s for s in subjects}
        for future in as_completed(futures):
            subj = futures[future]
            try:
                _, n_rec, absent = future.result()
                if absent:
                    with _lock:
                        absent_subjects.append(subj)
                    pbar.set_postfix_str(f"SC{subj:02d} – absent", refresh=True)
                else:
                    with _lock:
                        successful_subjects += 1
                        successful_recordings += n_rec
                    pbar.set_postfix_str(
                        f"SC{subj:02d} ✓ {n_rec}night(s) ({successful_subjects} done)",
                        refresh=True,
                    )
            except Exception as e:
                logger.error(f"  SC{subj:02d} FAILED: {e}")
                with _lock:
                    failed_subjects.append(subj)
                pbar.set_postfix_str(f"SC{subj:02d} ✗", refresh=True)
            pbar.update(1)
    pbar.close()

    elapsed = time.time() - t0
    data_size = _dir_size_mb(data_dir)

    # Summary
    print(f"\n{'─'*60}")
    print(f"  ✓ Subjects: {successful_subjects}/{len(subjects)}")
    print(f"  ✓ Recordings: {successful_recordings} (nights)")
    if absent_subjects:
        print(f"  – Absent from corpus: {sorted(absent_subjects)}")
    if failed_subjects:
        print(f"  ✗ Failed (retry if transient): {failed_subjects}")
    print(f"  ⏱ Time: {elapsed:.1f}s ({elapsed/max(len(subjects),1):.1f}s/subject)")
    print(f"  💾 Data size: {data_size:.1f} MB")
    print(f"{'─'*60}\n")

    # Write metadata
    meta_path = data_dir / "README.txt"
    with open(meta_path, "w") as f:
        f.write("Sleep-EDF Expanded Dataset (Sleep Cassette subset)\n")
        f.write("=" * 55 + "\n")
        f.write(f"Subjects: up to 83 (healthy volunteers, ages 25-101)\n")
        f.write(f"Channels: 2 EEG (Fpz-Cz, Pz-Oz) + 1 EOG + 1 EMG\n")
        f.write(f"Sampling rate: 100 Hz\n")
        f.write(f"Annotations: 30-second epochs labeled as:\n")
        f.write(f"  W (Wake), 1 (N1), 2 (N2), 3 (N3), R (REM), ? (Movement/Unknown)\n")
        f.write(f"\nFormat: EDF + annotations\n")
        f.write(f"\nDownloaded: {successful_subjects}/{len(subjects)} subjects, {successful_recordings} recordings\n")
        f.write(f"Time: {elapsed:.1f}s\n")
        f.write(f"Data size: {data_size:.1f} MB\n")
        if absent_subjects:
            f.write(f"Absent from corpus (expected): {sorted(absent_subjects)}\n")
        if failed_subjects:
            f.write(f"Failed subjects: {failed_subjects}\n")

    logger.info(f"Metadata written to {meta_path}")
    return data_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Sleep-EDF Expanded dataset")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="+", default=None,
                        help="Subject IDs (0-82). Default: all.")
    args = parser.parse_args()

    download_sleep_edf(data_dir=args.data_dir, subjects=args.subjects)
