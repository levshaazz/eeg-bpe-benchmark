"""
Unified data loaders for all 6 EEG datasets.
Returns standardised (data, labels, metadata) tuples.
All loaders use vectorised MNE / scipy / MOABB APIs where possible.
"""
from __future__ import annotations

import numpy as np
import mne
import pickle
import hashlib
from pathlib import Path
from typing import Optional
from joblib import Parallel, delayed

from .config import (
    DATASET_PATHS, DATASET_INFO, RAW_DATA_DIR,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQS, N_JOBS, N_IO_JOBS, CACHE_DIR,
)
from .utils import get_logger, timed

mne.set_log_level("ERROR")
logger = get_logger("data_loading")

# ─── In-memory dataset cache (avoids repeated disk I/O) ──────────────────────
_DATASET_CACHE: dict[str, dict] = {}


# ─── Minimal preprocessing (Step 0) ──────────────────────────────────────────

def preprocess_raw(raw: mne.io.BaseRaw, sfreq: float | None = None) -> mne.io.BaseRaw:
    """
    Minimal preprocessing (Step 0): bandpass 0.5–45 Hz + notch 50/60 Hz.

    Does NOT remove artifact channels or run ICA.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw MNE object (will be copied internally).
    sfreq : float or None, optional
        Target sampling frequency for optional resampling.

    Returns
    -------
    mne.io.BaseRaw
        Preprocessed raw object.
    """
    raw = raw.copy()
    raw.load_data()

    nyquist = raw.info["sfreq"] / 2.0

    # Bandpass — clamp h_freq to below Nyquist
    h_freq = min(BANDPASS_HIGH, nyquist - 1.0)
    if h_freq > BANDPASS_LOW:
        raw.filter(l_freq=BANDPASS_LOW, h_freq=h_freq,
                   n_jobs=N_JOBS, verbose=False)

    # Notch — only apply frequencies below Nyquist
    valid_notch = [f for f in NOTCH_FREQS if f < nyquist]
    if valid_notch:
        raw.notch_filter(freqs=valid_notch, n_jobs=N_JOBS, verbose=False)
    # Optional resample
    if sfreq is not None and raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, n_jobs=N_JOBS, verbose=False)
    return raw


# ─── BCI IV 2a ────────────────────────────────────────────────────────────────

def _load_bci_iv_2a_subject(subj: int, dataset, paradigm) -> tuple[int, dict | None]:
    """Load a single BCI IV 2a subject and encode string labels to integers."""
    try:
        logger.info(f"Loading BCI-IV-2a subject {subj}")
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
        # MOABB returns strings like "left_hand", "right_hand", "feet", "tongue".
        # Encode to integers 0..n_classes-1 in sorted order for reproducibility.
        classes = sorted(set(y))
        label_map = {c: i for i, c in enumerate(classes)}
        y_int = np.array([label_map[c] for c in y], dtype=np.int64)
        return subj, {
            "epochs": X, "labels": y_int, "sfreq": 250.0,
            "meta": meta, "class_names": classes,
        }
    except Exception as e:
        logger.warning(f"BCI IV 2a subject {subj} failed: {e}")
        return subj, None


@timed("data_loading")
def load_bci_iv_2a(subjects: list[int] | None = None):
    """
    Load BCI Competition IV 2a via MOABB. Parallelised across subjects.

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject IDs to load (default: 1–9).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float, "meta": DataFrame}}``.  Epochs shape:
        ``(n_trials, n_channels, n_times)``.
    """
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery

    mne.set_config("MNE_DATASETS_BNCI_PATH",
                    str(DATASET_PATHS["bci_iv_2a"]), set_env=False)

    dataset  = BNCI2014_001()
    paradigm = MotorImagery(n_classes=4)

    if subjects is None:
        subjects = list(range(1, 10))

    # prefer="threads": MOABB/MNE releases the GIL for file I/O.
    # BCI IV 2a has only 9 subjects so N_IO_JOBS is naturally capped.
    results_list = Parallel(n_jobs=N_IO_JOBS, prefer="threads", verbose=0)(
        delayed(_load_bci_iv_2a_subject)(s, dataset, paradigm) for s in subjects
    )
    return {subj: res for subj, res in results_list if res is not None}


# ─── PhysioNet MI ─────────────────────────────────────────────────────────────

# Run-aware label mapping: each run's T1/T2 encodes different motor classes.
# Runs 4,8,12  → imagine left fist (T1=0) / right fist (T2=1)
# Runs 6,10,14 → imagine both fists (T1=2) / both feet  (T2=3)
_PHYSIONET_RUN_LABELS: dict[int, dict[str, int]] = {
    4:  {"T1": 0, "T2": 1},
    6:  {"T1": 2, "T2": 3},
    8:  {"T1": 0, "T2": 1},
    10: {"T1": 2, "T2": 3},
    12: {"T1": 0, "T2": 1},
    14: {"T1": 2, "T2": 3},
}


def _load_physionet_subject(subj: int, runs: list[int]) -> dict | None:
    """
    Load a single PhysioNet MI subject with run-aware label mapping.

    Each run is processed independently so that T1/T2 annotations are
    mapped to the correct 4-class labels (left=0, right=1, both=2, feet=3)
    rather than being conflated across runs.
    """
    data_path = DATASET_PATHS["physionet_mi"]
    mne.set_config("MNE_DATASETS_EEGBCI_PATH",
                    str(data_path.parent), set_env=False)

    all_epochs_list: list[np.ndarray] = []
    all_labels_list: list[int] = []
    last_sfreq: float | None = None

    for run in runs:
        label_map = _PHYSIONET_RUN_LABELS.get(run)
        if label_map is None:
            continue
        try:
            fnames = mne.datasets.eegbci.load_data(
                subj, [run], path=str(data_path.parent))
            if not fnames:
                continue
            raw = mne.io.read_raw_edf(fnames[0], preload=True, verbose=False)
            mne.datasets.eegbci.standardize(raw)
            raw = preprocess_raw(raw)
            last_sfreq = raw.info["sfreq"]

            events, ev_id = mne.events_from_annotations(raw, verbose=False)

            for annot_name, class_id in label_map.items():
                if annot_name not in ev_id:
                    continue
                code = ev_id[annot_name]
                ev_t = events[events[:, 2] == code]
                if len(ev_t) == 0:
                    continue
                ep = mne.Epochs(raw, ev_t, {annot_name: code},
                                tmin=-0.5, tmax=4.0, baseline=None,
                                preload=True, verbose=False)
                d = ep.get_data()
                if len(d) > 0:
                    all_epochs_list.append(d)
                    all_labels_list.extend([class_id] * len(d))

        except Exception as e:
            logger.debug(f"PhysioNet MI subj={subj} run={run}: {e}")
            continue

    if not all_epochs_list or last_sfreq is None:
        logger.warning(f"PhysioNet MI subject {subj}: no data loaded")
        return None

    return {
        "epochs": np.concatenate(all_epochs_list, axis=0),
        "labels": np.array(all_labels_list, dtype=np.int64),
        "sfreq": last_sfreq,
    }


@timed("data_loading")
def load_physionet_mi(subjects: list[int] | None = None):
    """
    Load PhysioNet MI (4-class). Parallelised across subjects.

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject IDs to load (default: 1–109).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float}}``.
    """
    # 4-class MI runs:
    #   runs 4,8,12 = imagine left fist (T1) / right fist (T2)
    #   runs 6,10,14 = imagine both fists (T1) / both feet (T2)
    # Using all 6 runs gives 4 classes (left, right, both-fists, both-feet)
    runs = [4, 6, 8, 10, 12, 14]
    if subjects is None:
        subjects = list(range(1, 110))

    # prefer="threads": file I/O + MNE/NumPy release the GIL → threads > processes
    results_list = Parallel(n_jobs=N_IO_JOBS, prefer="threads", verbose=0)(
        delayed(_load_physionet_subject)(s, runs) for s in subjects
    )
    return {
        subj: res for subj, res in zip(subjects, results_list)
        if res is not None
    }


# ─── Sleep-EDF ────────────────────────────────────────────────────────────────

def _load_sleep_edf_subject(subj_idx: int) -> dict | None:
    """Load one Sleep-EDF subject."""
    try:
        data_path = DATASET_PATHS["sleep_edf"]
        mne.set_config("MNE_DATASETS_SLEEP_PHYSIONET_PATH",
                        str(data_path.parent), set_env=False)

        [psg_file] = mne.datasets.sleep_physionet.age.fetch_data(
            subjects=[subj_idx], recording=[1],
            path=str(data_path.parent)
        )
        psg_path, hyp_path = psg_file

        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot)

        # Keep only EEG channels
        eeg_channels = [ch for ch in raw.ch_names
                        if ch.startswith("EEG")]
        if not eeg_channels:
            return None
        raw.pick(eeg_channels)
        raw = preprocess_raw(raw)

        # Create 30-s epochs from annotations
        mapping = {
            "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
            "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
        }
        events, event_id = mne.events_from_annotations(
            raw, event_id=mapping, chunk_duration=30.0, verbose=False
        )
        epochs = mne.Epochs(raw, events, event_id=None,
                            tmin=0.0, tmax=30.0 - 1.0 / raw.info["sfreq"],
                            baseline=None, preload=True, verbose=False)
        return {
            "epochs": epochs.get_data(),
            "labels": epochs.events[:, -1],
            "sfreq": raw.info["sfreq"],
        }
    except Exception as e:
        logger.warning(f"Sleep-EDF subject {subj_idx} failed: {e}")
        return None


@timed("data_loading")
def load_sleep_edf(subjects: list[int] | None = None):
    """
    Load Sleep-EDF dataset. Parallelised across subjects.

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject indices to load (default: 0–19).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float}}``.
    """
    if subjects is None:
        subjects = list(range(0, 20))  # first 20 subjects for speed

    results_list = Parallel(n_jobs=N_IO_JOBS, prefer="threads", verbose=0)(
        delayed(_load_sleep_edf_subject)(s) for s in subjects
    )
    return {
        subj: res for subj, res in zip(subjects, results_list)
        if res is not None
    }


# ─── Mental Arithmetic ───────────────────────────────────────────────────────

def _load_mental_arithmetic_subject(subj: int, data_dir: Path) -> dict | None:
    """Load one Mental Arithmetic subject (baseline + task EDF files)."""
    # PhysioNet EEGMAT files are named Subject01_1.edf … Subject36_2.edf (1-indexed).
    subj_str = f"Subject{subj:02d}"
    f1 = data_dir / f"{subj_str}_1.edf"  # baseline
    f2 = data_dir / f"{subj_str}_2.edf"  # task
    if not f1.exists() or not f2.exists():
        return None

    try:
        raw1 = mne.io.read_raw_edf(str(f1), preload=True, verbose=False)
        raw2 = mne.io.read_raw_edf(str(f2), preload=True, verbose=False)

        # Derive EEG channel list from each file independently, then intersect.
        # This guards against the two EDF files having slightly different channel
        # names / ordering (common in practice for mental-arithmetic recordings).
        eeg_ch1 = [ch for ch in raw1.ch_names if not ch.startswith("Status")][:19]
        eeg_ch2_set = {ch for ch in raw2.ch_names if not ch.startswith("Status")}
        # Preserve order from raw1; keep only channels present in both files.
        eeg_ch = [ch for ch in eeg_ch1 if ch in eeg_ch2_set]
        if not eeg_ch:
            logger.warning(f"Mental Arithmetic subject {subj}: no common EEG channels")
            return None

        raw1.pick(eeg_ch)
        raw2.pick(eeg_ch)

        raw1 = preprocess_raw(raw1)
        raw2 = preprocess_raw(raw2)

        # Segment into 4-second epochs (vectorised windowing)
        epoch_dur = 4.0  # seconds
        sfreq = raw1.info["sfreq"]
        n_samples = int(epoch_dur * sfreq)

        data1 = raw1.get_data()  # (ch, time)
        data2 = raw2.get_data()

        n_epochs1 = data1.shape[1] // n_samples
        n_epochs2 = data2.shape[1] // n_samples
        epochs1 = data1[:, :n_epochs1 * n_samples].reshape(
            data1.shape[0], n_epochs1, n_samples
        ).transpose(1, 0, 2)  # (n_trials, ch, time)
        epochs2 = data2[:, :n_epochs2 * n_samples].reshape(
            data2.shape[0], n_epochs2, n_samples
        ).transpose(1, 0, 2)

        # Balance classes by undersampling the majority class.
        # All subjects have rest≈45 epochs vs task≈15 epochs (3:1 structural imbalance
        # from file durations: f1≈182s vs f2≈62s). Without balancing, kappa=0 because
        # the classifier predicts the majority class (rest) for all trials.
        # Deterministic RNG per subject ensures reproducibility across runs.
        min_n = min(n_epochs1, n_epochs2)
        if min_n > 0 and n_epochs1 != n_epochs2:
            rng = np.random.default_rng(42 + subj)
            if n_epochs1 > min_n:
                idx1 = np.sort(rng.choice(n_epochs1, min_n, replace=False))
                epochs1   = epochs1[idx1]
                n_epochs1 = min_n
            if n_epochs2 > min_n:
                idx2 = np.sort(rng.choice(n_epochs2, min_n, replace=False))
                epochs2   = epochs2[idx2]
                n_epochs2 = min_n
            logger.info(
                f"Mental Arithmetic subj {subj}: balanced to {min_n} epochs/class"
            )

        all_epochs = np.concatenate([epochs1, epochs2], axis=0)
        labels = np.concatenate([
            np.zeros(n_epochs1, dtype=np.int64),
            np.ones(n_epochs2,  dtype=np.int64),
        ])

        logger.debug(
            f"Mental Arithmetic subj {subj}: "
            f"rest={n_epochs1}, task={n_epochs2}"
        )

        return {"epochs": all_epochs, "labels": labels, "sfreq": sfreq}

    except Exception as e:
        logger.warning(f"Mental Arithmetic subject {subj} failed: {e}")
        return None


@timed("data_loading")
def load_mental_arithmetic(subjects: list[int] | None = None):
    """
    Load Mental Arithmetic dataset (EDF files with _1=baseline, _2=task).

    PhysioNet EEGMAT files are 0-indexed: Subject00_1.edf … Subject35_2.edf.
    (Confirmed from PhysioNet RECORDS file: 36 subjects, IDs 00–35.)

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject IDs to load (default: 0–35, matching Subject00…Subject35 files).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float}}``.
    """
    data_dir = DATASET_PATHS["mental_arithmetic"]
    if subjects is None:
        subjects = list(range(0, 36))  # Subject00 … Subject35 (0-indexed)

    results_list = Parallel(n_jobs=N_IO_JOBS, prefer="threads", verbose=0)(
        delayed(_load_mental_arithmetic_subject)(s, data_dir) for s in subjects
    )
    return {
        subj: res for subj, res in zip(subjects, results_list)
        if res is not None
    }


# ─── EPFL P300 ────────────────────────────────────────────────────────────────

def _load_epfl_p300_subject(subj: int, dataset, paradigm) -> tuple[int, dict | None]:
    """Load a single EPFL P300 subject and encode string labels to integers."""
    try:
        logger.info(f"Loading EPFL P300 subject {subj}")
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=[subj])
        # MOABB P300 paradigm returns string labels: "Target" / "NonTarget".
        # Encode to integers: NonTarget=0, Target=1 (sorted order).
        classes = sorted(set(y))
        label_map = {c: i for i, c in enumerate(classes)}
        y_int = np.array([label_map[c] for c in y], dtype=np.int64)
        return subj, {
            "epochs": X, "labels": y_int, "sfreq": 2048.0,
            "meta": meta, "class_names": classes,
        }
    except Exception as e:
        logger.warning(f"EPFL P300 subject {subj} failed: {e}")
        return subj, None


@timed("data_loading")
def load_epfl_p300(subjects: list[int] | None = None):
    """
    Load EPFL P300 dataset via MOABB (BNCI2014-009). Parallelised across subjects.

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject IDs to load (default: 1–10).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float, "meta": DataFrame}}``.
    """
    from moabb.datasets import BNCI2014_009
    from moabb.paradigms import P300

    mne.set_config("MNE_DATASETS_BNCI_PATH",
                    str(DATASET_PATHS["epfl_p300"]), set_env=False)

    dataset  = BNCI2014_009()
    paradigm = P300()

    if subjects is None:
        subjects = list(range(1, 11))

    results_list = Parallel(n_jobs=N_IO_JOBS, prefer="threads", verbose=0)(
        delayed(_load_epfl_p300_subject)(s, dataset, paradigm) for s in subjects
    )
    return {subj: res for subj, res in results_list if res is not None}


# ─── SSVEP Nakanishi ──────────────────────────────────────────────────────────

@timed("data_loading")
def load_ssvep_nakanishi(subjects: list[int] | None = None):
    """
    Load SSVEP Nakanishi ``.mat`` files.

    Parameters
    ----------
    subjects : list of int or None, optional
        Subject indices to load (default: all available).

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float}}``.
    """
    import scipy.io as sio

    data_dir = DATASET_PATHS["ssvep_nakanishi"]
    mat_files = sorted(data_dir.glob("*.mat"))

    if not mat_files:
        # Try MOABB cache
        moabb_dir = RAW_DATA_DIR / "moabb_cache" / "MNE-nakanishi-data"
        mat_files = sorted(moabb_dir.rglob("*.mat"))

    if subjects is None:
        subjects = list(range(len(mat_files)))

    result = {}
    for subj_idx in subjects:
        if subj_idx >= len(mat_files):
            continue
        fpath = mat_files[subj_idx]
        try:
            mat = sio.loadmat(str(fpath))
            eeg = mat.get("eeg", mat.get("data", None))
            if eeg is None:
                # Some MOABB caches store differently
                for key in mat:
                    if not key.startswith("_"):
                        eeg = mat[key]
                        break

            if eeg is None:
                continue

            # Actual .mat shape from GitHub: (n_classes, n_channels, n_times, n_trials)
            # e.g. (12, 8, 1114, 15) for 12 classes × 8 ch × 1114 samples × 15 trials.
            if eeg.ndim == 4:
                n_classes, n_channels, n_times, n_trials = eeg.shape
                # → transpose to (n_classes, n_trials, n_channels, n_times)
                # → reshape to (n_classes * n_trials, n_channels, n_times)
                all_epochs = eeg.transpose(0, 3, 1, 2).reshape(-1, n_channels, n_times)
                labels = np.repeat(np.arange(n_classes), n_trials)
            else:
                logger.warning(f"Unexpected eeg shape {eeg.shape} in {fpath}")
                continue

            result[subj_idx] = {
                "epochs": all_epochs,
                "labels": labels,
                "sfreq": 256.0,
            }
        except Exception as e:
            logger.warning(f"SSVEP subject {subj_idx} failed: {e}")

    return result


# ─── Unified loader ───────────────────────────────────────────────────────────

LOADERS = {
    "bci_iv_2a": load_bci_iv_2a,
    "physionet_mi": load_physionet_mi,
    "sleep_edf": load_sleep_edf,
    "mental_arithmetic": load_mental_arithmetic,
    "epfl_p300": load_epfl_p300,
    "ssvep_nakanishi": load_ssvep_nakanishi,
}


def load_dataset(name: str, subjects: list[int] | None = None,
                 max_subjects: int | None = None) -> dict:
    """
    Load any dataset by name.

    Uses in-memory cache to avoid re-loading across experiments.
    Optionally persists to disk cache (pickle) for cross-session reuse.

    Parameters
    ----------
    name : str
        Dataset name (one of ``LOADERS.keys()``).
    subjects : list of int or None, optional
        Specific subject IDs to load.
    max_subjects : int or None, optional
        Limit the number of subjects returned.

    Returns
    -------
    dict
        ``{subject_id: {"epochs": np.ndarray, "labels": np.ndarray,
        "sfreq": float}}``.
    """
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(LOADERS)}")

    # Build a stable subjects tag for both memory and disk keys.
    # This prevents returning wrong cached data when subjects list differs.
    _subj_tag = hashlib.md5(
        str(sorted(subjects) if subjects is not None else "all").encode()
    ).hexdigest()[:8]
    # Version suffix: bump when loader behaviour changes (forces disk cache rebuild).
    _CACHE_VERSIONS: dict[str, str] = {
        "mental_arithmetic": "_b1",  # b1 = class-balanced (2026-03-03)
    }
    _ver = _CACHE_VERSIONS.get(name, "")
    cache_key = f"{name}_s{_subj_tag}_m{max_subjects}{_ver}"

    # Check in-memory cache first
    if cache_key in _DATASET_CACHE:
        logger.info(f"Cache hit (memory): {name} ({len(_DATASET_CACHE[cache_key])} subjects)")
        return _DATASET_CACHE[cache_key]

    # Check disk cache — key now includes subjects hash + version, preventing stale hits
    disk_cache = CACHE_DIR / f"{name}_s{_subj_tag}_max{max_subjects}{_ver}.pkl"
    if disk_cache.exists():
        try:
            with open(disk_cache, "rb") as f:
                data = pickle.load(f)
            logger.info(f"Cache hit (disk): {name} ({len(data)} subjects)")
            _DATASET_CACHE[cache_key] = data
            return data
        except Exception:
            pass  # corrupted cache, reload

    # Load from source
    data = LOADERS[name](subjects=subjects)

    if max_subjects is not None:
        keys = sorted(data.keys())[:max_subjects]
        data = {k: data[k] for k in keys}

    logger.info(f"Loaded {name}: {len(data)} subjects")

    # Log summary: trial count, shape, label distribution
    if data:
        first_subj = next(iter(data.values()))
        all_labels = np.concatenate([v["labels"] for v in data.values()])
        sample_epochs = first_subj["epochs"]
        uniq, counts = np.unique(all_labels, return_counts=True)
        label_dist = dict(zip(uniq.tolist(), counts.tolist()))
        logger.info(
            f"  {name}: {len(all_labels)} trials, {len(data)} subjects, "
            f"shape=({sample_epochs.shape[1]}, {sample_epochs.shape[2]}), "
            f"labels={label_dist}"
        )

    # Persist to caches
    _DATASET_CACHE[cache_key] = data
    try:
        with open(disk_cache, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Cached to disk: {disk_cache}")
    except Exception as e:
        logger.error(f"Disk cache write failed: {e}")

    return data


def get_fragment(data: dict, subject_id: int,
                 duration_s: float = 600.0) -> np.ndarray:
    """
    Extract a fragment of given duration from a subject's epochs.

    Concatenates all trials along the time axis and truncates.

    Parameters
    ----------
    data : dict
        Dataset dict (must contain ``subject_id`` key).
    subject_id : int
        Subject to extract from.
    duration_s : float, optional
        Duration in seconds (default 600).

    Returns
    -------
    np.ndarray
        Signal fragment, shape ``(n_channels, n_samples)``.
    """
    subj = data[subject_id]
    sfreq = subj["sfreq"]
    epochs = subj["epochs"]  # (n_trials, ch, time)
    # Transpose to (ch, n_trials, time) BEFORE flattening so that each
    # row of the result corresponds to one channel's continuous timeline.
    # epochs.reshape(ch, -1) is WRONG: it mixes channels across trials in
    # C-order memory layout.
    concat = epochs.transpose(1, 0, 2).reshape(epochs.shape[1], -1)  # (ch, n_trials*time)
    n_samples = min(int(duration_s * sfreq), concat.shape[1])
    return concat[:, :n_samples]
