"""
Experiment 6: Alternative Tokenization Strategies (H5–H7)
==========================================================
Tests three preprocessing approaches designed to make BPE effective
for frequency-coded EEG paradigms (Motor Imagery, SSVEP).

Approaches
----------
raw          — Baseline: BPE on original signal at native sfreq (same as Exp 2).
downsample   — Decimate to 64 Hz before BPE (H5: temporal scale alignment).
envelope     — Hilbert envelope of 8–30 Hz bandpass (H6: direct ERD/ERS encoding).
spectral     — STFT log-power sequence (H7: spectral shape tokenisation).

For each approach × dataset:
  1. Apply preprocessing (or skip for 'raw').
  2. Train BPE vocab (vocab_size=1024 default; cached per approach).
  3. Extract BPE histogram features.
  4. Classify with LogReg (LOSO or 5-fold).
  5. Log accuracy, compression ratio, mean token length.

Output
------
  results/logs/exp6_per_fold.csv
  results/logs/exp6_results.json
  results/plots/exp6/  ← comparison charts
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed

from .config import (
    DATASET_INFO, RANDOM_SEEDS, N_JOBS, N_BOOTSTRAP,
    LOGS_DIR, PLOTS_DIR, MODELS_DIR, DEVICE, CACHE_DIR,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
    DEFAULT_QUANT_METHOD,
)
from .quantization import quantize
from .bpe_engine import train_bpe, apply_bpe_batch, BPEVocab
from .data_loading import load_dataset
from .tokenization import downsample_epochs, envelope_epochs, spectral_epochs, delta_epochs
from .utils import (
    ExperimentLogger, save_json, append_csv, get_logger, bootstrap_ci,
    pca_reduce,
)

logger = get_logger("exp6")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ─── Approach definitions ─────────────────────────────────────────────────────

APPROACHES = {
    "raw": {
        "label":       "Raw BPE (baseline)",
        "target_sfreq": None,   # unchanged
        "preprocess":  None,    # no preprocessing
    },
    "downsample_64hz": {
        "label":       "Downsampled 64 Hz (H5)",
        "target_sfreq": 64.0,
        "preprocess":  "downsample",
    },
    "envelope_8_30hz": {
        "label":       "Envelope adaptive bands (H6)",
        "target_sfreq": 64.0,
        "preprocess":  "envelope",
    },
    "spectral_500ms": {
        "label":       "STFT log-PSD 1000 ms (H7)",
        "target_sfreq": None,   # effective sfreq = 1/step_sec = 2 Hz (windows)
        "preprocess":  "spectral",
    },
    "delta_quant": {
        "label":       "Delta (Δ amplitude) BPE (H8)",
        "target_sfreq": None,   # same sfreq as input
        "preprocess":  "delta",
    },
    "multiband_alpha_beta": {
        "label":       "Multi-band Envelope θ/α/β (H9)",
        "target_sfreq": 64.0,
        "preprocess":  "multiband",
    },
}

# Spectral approach parameters — 1000 ms window for 1 Hz freq resolution
# (needed to distinguish SSVEP harmonics as close as 1.25 Hz apart)
_SPEC_WIN_SEC  = 1.0
_SPEC_STEP_SEC = 0.5
_SPEC_FMIN     = 1.0
_SPEC_FMAX     = 45.0

# Envelope parameters (defaults; overridden per-paradigm in apply_preprocessing)
_ENV_FMIN = 8.0
_ENV_FMAX = 30.0
_ENV_TARGET_SFREQ = 64.0

# Per-paradigm envelope bandpass frequencies
# Each paradigm uses the frequency band most relevant to its neural signature
_PARADIGM_ENVELOPE_BANDS: dict[str, tuple[float, float]] = {
    "Motor Imagery": (8.0,  30.0),   # mu (8–12 Hz) + beta (13–30 Hz): ERD/ERS
    "Sleep":         (0.5, 30.0),    # broadband: preserve delta, spindles, alpha, beta
    "SSVEP":         (6.0,  20.0),   # SSVEP fundamental freqs (6.25–15 Hz)
    "P300":          (1.0,  20.0),   # broadband including P300 deflection
    "Cognitive":     (8.0,  30.0),   # same as MI (frontal alpha/beta load effects)
}

# Downsample target
_DS_TARGET_SFREQ = 64.0

# Paradigms that benefit from a dataset-specific (rather than shared) BPE vocab
# because their signal statistics differ radically from other paradigms
_DATASET_SPECIFIC_VOCAB_PARADIGMS = {"P300"}


# ─── Preprocessing dispatcher ─────────────────────────────────────────────────

def apply_preprocessing(epochs: np.ndarray, orig_sfreq: float,
                        approach_key: str,
                        paradigm: str | None = None) -> np.ndarray:
    """
    Apply the tokenisation preprocessing for *approach_key* to *epochs*.

    Returns preprocessed epochs with the same (n_trials, n_ch, n_times_new)
    shape convention, or the original array for the ``'raw'`` approach.

    Parameters
    ----------
    paradigm : str or None
        EEG paradigm name (e.g. ``"Motor Imagery"``, ``"Sleep"``, ``"SSVEP"``).
        Used to select appropriate envelope bandpass frequencies when
        ``approach_key == 'envelope_8_30hz'``.  If *None*, falls back to the
        default 8–30 Hz band.
    """
    kind = APPROACHES[approach_key]["preprocess"]
    if kind is None:
        return epochs

    if kind == "downsample":
        return downsample_epochs(epochs, orig_sfreq, _DS_TARGET_SFREQ)

    if kind == "envelope":
        # Select frequency band appropriate for this paradigm
        fmin, fmax = _PARADIGM_ENVELOPE_BANDS.get(
            paradigm or "", (_ENV_FMIN, _ENV_FMAX)
        )
        nyq = orig_sfreq / 2.0
        fmax_safe = min(fmax, nyq * 0.9)
        if fmin >= fmax_safe:
            # sfreq too low for this bandpass (e.g., already-downsampled data)
            logger.warning(
                f"[{approach_key}] sfreq={orig_sfreq} too low for "
                f"[{fmin},{fmax}] Hz envelope ({paradigm}); using raw signal."
            )
            return epochs
        logger.debug(
            f"[{approach_key}] paradigm={paradigm}: "
            f"envelope [{fmin},{fmax_safe:.1f}] Hz"
        )
        return envelope_epochs(epochs, orig_sfreq,
                               fmin=fmin, fmax=fmax_safe,
                               target_sfreq=_ENV_TARGET_SFREQ)

    if kind == "spectral":
        return spectral_epochs(epochs, orig_sfreq,
                               win_sec=_SPEC_WIN_SEC,
                               step_sec=_SPEC_STEP_SEC,
                               fmin=_SPEC_FMIN, fmax=_SPEC_FMAX)

    if kind == "delta":
        return delta_epochs(epochs)  # first-order amplitude differences, same shape

    if kind == "multiband":
        from .tokenization import multiband_envelope_epochs
        return multiband_envelope_epochs(epochs, orig_sfreq, target_sfreq=64.0)

    raise ValueError(f"Unknown preprocessing kind: {kind!r}")


# ─── BPE vocab (cached per approach) ─────────────────────────────────────────

def _vocab_path(vocab_size: int, n_bins: int, approach_key: str,
               suffix: str | None = None, method: str = "adaptive") -> Path:
    # For spectral approach: embed window duration in filename so that changing
    # _SPEC_WIN_SEC automatically invalidates stale cached vocabs.
    extra = ""
    if approach_key == "spectral_500ms":
        extra = f"_w{int(_SPEC_WIN_SEC * 1000)}ms_adapt"  # _adapt = adaptive window cap
    if suffix:
        return MODELS_DIR / f"bpe_vocab_V{vocab_size}_B{n_bins}_{method}_{approach_key}{extra}_{suffix}.json"
    return MODELS_DIR / f"bpe_vocab_V{vocab_size}_B{n_bins}_{method}_{approach_key}{extra}.json"


def get_or_train_vocab(
    approach_key: str,
    datasets: list[str],
    vocab_size: int,
    n_bins: int,
    method: str,
    max_subjects_train: int = 5,
    suffix: str | None = None,
) -> BPEVocab:
    """
    Load cached BPE vocab for *approach_key*, or train a new one.

    The vocab is trained on a small subset of each dataset (to keep
    exp6 self-contained and fast).  For 'raw', the vocab from Exp 1/2 is
    reused if available.

    Parameters
    ----------
    suffix : str or None
        Optional dataset-name suffix for the cache path.  When set, the vocab
        is trained using only *datasets* (which should be a single dataset)
        and stored under a distinct filename, e.g.
        ``bpe_vocab_V1024_B64_adaptive_envelope_8_30hz_epfl_p300.json``.
        This allows paradigm-specific vocabs that capture only the target
        dataset's signal statistics.
    """
    vpath = _vocab_path(vocab_size, n_bins, approach_key, suffix, method)

    # For 'raw' (no suffix), also try the standard exp1/exp2 vocab path as fallback
    if approach_key == "raw" and suffix is None and not vpath.exists():
        std_path = MODELS_DIR / f"bpe_vocab_V{vocab_size}_B{n_bins}_{method}.json"
        if std_path.exists():
            logger.info(f"[raw] reusing exp1/2 vocab: {std_path.name}")
            return BPEVocab.load(str(std_path))

    if vpath.exists():
        logger.info(f"  Loaded cached vocab: {vpath.name}")
        return BPEVocab.load(str(vpath))

    ds_label = suffix or "all-datasets"
    logger.info(
        f"  Training BPE vocab (V={vocab_size}, B={n_bins}, "
        f"approach={approach_key}, ds={ds_label}) "
        f"on {max_subjects_train} subj/dataset …"
    )
    all_seqs: list[list[int]] = []
    for ds_name in datasets:
        sfreq = DATASET_INFO[ds_name]["sfreq"]
        paradigm = DATASET_INFO[ds_name].get("paradigm")
        try:
            data = load_dataset(ds_name, max_subjects=max_subjects_train)
        except Exception as exc:
            logger.warning(f"  Skip {ds_name} for vocab training: {exc}")
            continue

        epoch_list = [d["epochs"] for d in data.values() if "epochs" in d]
        if not epoch_list:
            continue

        # Pad channels/time to common shape
        n_ch_max   = max(e.shape[1] for e in epoch_list)
        n_time_max = max(e.shape[2] for e in epoch_list)
        padded = []
        for e in epoch_list:
            p = np.zeros((e.shape[0], n_ch_max, n_time_max), dtype=np.float32)
            p[:, :e.shape[1], :e.shape[2]] = e
            padded.append(p)
        X = np.concatenate(padded, axis=0)

        # Preprocess — use paradigm-aware bands for envelope approach
        try:
            X_pp = apply_preprocessing(X, sfreq, approach_key, paradigm=paradigm)
        except Exception as exc:
            logger.warning(f"  Preprocessing failed for {ds_name}: {exc}")
            continue

        # Quantize → sequences
        n_tr, n_ch, n_t = X_pp.shape
        flat = X_pp.reshape(n_tr * n_ch, n_t)
        codes, _ = quantize(flat, method, n_bins, normalize=True)
        for i in range(codes.shape[0]):
            all_seqs.append(codes[i].tolist())

        # Cap total tokens to avoid OOM during BPE training
        total_tokens = sum(len(s) for s in all_seqs)
        if total_tokens > 5_000_000:
            logger.info(f"  Token budget reached ({total_tokens:,}); stopping.")
            break

    if not all_seqs:
        raise RuntimeError(f"No sequences collected for approach={approach_key}")

    logger.info(
        f"  Training BPE on {len(all_seqs):,} sequences, "
        f"{sum(len(s) for s in all_seqs):,} tokens …"
    )
    vocab = train_bpe(all_seqs, vocab_size=vocab_size, base_vocab_size=n_bins)
    vocab.save(str(vpath))
    logger.info(f"  Saved vocab: {vpath.name}")
    return vocab


# ─── BPE histogram features ───────────────────────────────────────────────────

def epochs_to_bpe_histograms(
    epochs_preproc: np.ndarray,
    vocab: BPEVocab,
    method: str,
    n_bins: int,
) -> np.ndarray:
    """
    BPE histogram features for *preprocessed* epochs.

    Identical to the Exp 2 version but operates on already-preprocessed data.
    Applies TruncatedSVD when feature dimensionality > 2048.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_ch * vocab_size)`` or ``(n_trials, 2048)``
        after SVD.
    """
    n_trials, n_ch, n_time = epochs_preproc.shape
    V = vocab.vocab_size
    n_total = n_trials * n_ch

    flat = epochs_preproc.reshape(n_total, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)

    seq_list = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # COO-style vectorised histogram
    row_list, col_list = [], []
    for idx, seq in enumerate(bpe_seqs):
        if not seq:
            continue
        arr = np.asarray(seq, dtype=np.int32)
        arr = arr[arr < V]
        if len(arr):
            row_list.append(np.full(len(arr), idx, dtype=np.int32))
            col_list.append(arr)

    if row_list:
        rows = np.concatenate(row_list).astype(np.int64)
        cols = np.concatenate(col_list).astype(np.int64)
        flat_idx = rows * V + cols
        counts_flat = np.bincount(flat_idx, minlength=n_total * V)
        histograms = counts_flat[: n_total * V].reshape(n_trials, n_ch, V).astype(np.float32)
    else:
        histograms = np.zeros((n_trials, n_ch, V), dtype=np.float32)

    sums = histograms.sum(axis=-1, keepdims=True)
    histograms = histograms / (sums + 1e-10)
    X_hist = histograms.reshape(n_trials, -1)

    # Dimensionality reduction if needed
    if X_hist.shape[1] > 2048:
        n_comp = min(2048, X_hist.shape[0] - 1)
        logger.info(f"    pca_reduce {X_hist.shape[1]} → {n_comp} …")
        X_hist = pca_reduce(X_hist, n_comp, device=DEVICE)

    return X_hist


def _cached_bpe_histograms_exp6(epochs_preproc: np.ndarray,
                                 vocab: BPEVocab,
                                 method: str,
                                 n_bins: int,
                                 approach_key: str) -> np.ndarray:
    """Disk-cached version of the local ``epochs_to_bpe_histograms``."""
    cache_dir = CACHE_DIR / "hist_cache"
    n, c, t = epochs_preproc.shape
    n_sample = min(8, n)
    sample = np.concatenate([epochs_preproc[:n_sample].ravel(),
                              epochs_preproc[-n_sample:].ravel()])
    data_hash  = hashlib.md5(sample.tobytes()).hexdigest()[:12]
    vocab_hash = hashlib.md5(str(vocab.merges).encode()).hexdigest()[:12]
    # Include preprocessing parameters in cache key so changing constants
    # (window sizes, bandpass frequencies, etc.) invalidates stale caches.
    _params_str = (f"{_SPEC_WIN_SEC}_{_SPEC_STEP_SEC}_{_SPEC_FMIN}_{_SPEC_FMAX}"
                   f"_{_ENV_FMIN}_{_ENV_FMAX}_{_ENV_TARGET_SFREQ}"
                   f"_{_DS_TARGET_SFREQ}")
    params_hash = hashlib.md5(_params_str.encode()).hexdigest()[:8]
    key        = (f"exp6_{approach_key}_v{vocab_hash}_d{data_hash}"
                  f"_{method}_b{n_bins}_n{n}_c{c}_t{t}_p{params_hash}")
    cache_path = cache_dir / f"{key}.npz"

    if cache_path.exists():
        logger.info(f"    [hist cache hit]  {cache_path.name}")
        return np.load(str(cache_path))["X"]

    logger.info(f"    [hist cache miss] computing → {cache_path.name}")
    X = epochs_to_bpe_histograms(epochs_preproc, vocab, method, n_bins)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X)
    logger.info(f"    [hist cache save] {cache_path.name} "
                f"({cache_path.stat().st_size // 1024} KB)")
    return X


def compute_compression_ratio(epochs_preproc, vocab, method, n_bins):
    """
    Mean BPE compression ratio (original tokens / BPE tokens).

    Samples up to 200 (trial, channel) pairs for speed.
    """
    n_trials, n_ch, n_time = epochs_preproc.shape
    flat = epochs_preproc.reshape(n_trials * n_ch, n_time)

    # Sample a subset
    n_sample = min(200, flat.shape[0])
    idx = np.random.choice(flat.shape[0], n_sample, replace=False)
    codes, _ = quantize(flat[idx], method, n_bins, normalize=True)
    seq_list = [codes[i].tolist() for i in range(codes.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    ratios = []
    for orig, bpe_seq in zip(seq_list, bpe_seqs):
        if bpe_seq:
            ratios.append(len(orig) / len(bpe_seq))
    return float(np.mean(ratios)) if ratios else 1.0


def mean_token_length_ms(epochs_preproc, vocab, method, n_bins, effective_sfreq):
    """
    Mean BPE token duration in milliseconds.

    Token length in samples × 1000 / effective_sfreq.
    """
    n_trials, n_ch, n_time = epochs_preproc.shape
    flat = epochs_preproc.reshape(n_trials * n_ch, n_time)

    n_sample = min(100, flat.shape[0])
    idx = np.random.choice(flat.shape[0], n_sample, replace=False)
    codes, _ = quantize(flat[idx], method, n_bins, normalize=True)
    seq_list = [codes[i].tolist() for i in range(codes.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    lengths_ms = []
    for orig, bpe_seq in zip(seq_list, bpe_seqs):
        if bpe_seq:
            mean_tok_len_samples = len(orig) / len(bpe_seq)
            lengths_ms.append(mean_tok_len_samples * 1000.0 / effective_sfreq)
    return float(np.mean(lengths_ms)) if lengths_ms else 0.0


# ─── CV helpers ───────────────────────────────────────────────────────────────

def _logreg():
    return LogisticRegression(
        solver=LOGREG_SOLVER, max_iter=LOGREG_MAX_ITER, C=LOGREG_C,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL, n_jobs=1,
    )


def _run_fold(X_train, y_train, X_test, y_test, fold_id, seed):
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, f1_score, cohen_kappa_score,
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = _logreg()
    clf.set_params(random_state=seed)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    acc  = float(accuracy_score(y_test, y_pred))
    f1   = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_test, y_pred))
    return {"fold": fold_id, "accuracy": acc, "macro_f1": f1, "kappa": kappa}


def run_cv(X, y, groups, cv_strategy, seed):
    """Run LOSO or subject-stratified 5-fold CV; return list of fold result dicts."""
    results = []
    if cv_strategy == "LOSO":
        logo = LeaveOneGroupOut()
        for fold_id, (tr, te) in enumerate(logo.split(X, y, groups)):
            results.append(_run_fold(X[tr], y[tr], X[te], y[te], fold_id, seed))
    else:
        # Subject-stratified 5-fold: split unique subjects into folds to prevent
        # within-subject data leakage (matches exp2's run_kfold_cv).
        unique_subj = np.unique(groups)
        rng = np.random.default_rng(seed)
        unique_subj = rng.permutation(unique_subj)
        n_splits = min(5, len(unique_subj))
        fold_size = max(1, len(unique_subj) // n_splits)
        for fold_id in range(n_splits):
            start = fold_id * fold_size
            test_subj = (unique_subj[start:] if fold_id == n_splits - 1
                         else unique_subj[start:start + fold_size])
            mask = np.isin(groups, test_subj)
            tr, te = np.where(~mask)[0], np.where(mask)[0]
            if len(tr) == 0 or len(te) == 0:
                continue
            results.append(_run_fold(X[tr], y[tr], X[te], y[te], fold_id, seed))
    return results


# ─── Main experiment function ─────────────────────────────────────────────────

def run_experiment_6(
    datasets: list[str] | None = None,
    vocab_size: int = 1024,
    n_bins: int = 64,
    method: str = DEFAULT_QUANT_METHOD,
    max_subjects: int | None = None,
    approaches: list[str] | None = None,
    max_subjects_vocab_train: int = 5,
) -> list[dict]:
    """
    Run Experiment 6: Alternative Tokenization Strategies (H5–H7).

    Parameters
    ----------
    datasets : list of str or None
        Datasets to test.  Default: all 6.
    vocab_size : int, optional
        BPE vocabulary size (per approach).  Default 1024.
    n_bins : int, optional
        Quantisation bins.  Default 64.
    method : str, optional
        Quantisation method.  Default ``"uniform"``.
    max_subjects : int or None
        Max subjects per dataset for classification.
    approaches : list of str or None
        Subset of approach keys to run.  Default: all 4.
    max_subjects_vocab_train : int, optional
        Subjects per dataset used for vocab training.  Default 5.

    Returns
    -------
    list of dict
        Per-seed aggregate result dicts.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    if approaches is None:
        approaches = list(APPROACHES.keys())

    exp_log = ExperimentLogger("exp6_alt_tokenization")
    all_results: list[dict] = []

    exp_log.info(
        f"Exp 6: approaches={approaches}, vocab_size={vocab_size}, "
        f"n_bins={n_bins}, method={method}"
    )

    # ─── Step 1: train/load BPE vocab for each approach ──────────────────
    # Shared vocab (trained on all datasets)
    vocabs: dict[str, BPEVocab] = {}
    for ap_key in approaches:
        exp_log.info(f"--- Vocab (shared): approach={ap_key} ---")
        try:
            vocabs[ap_key] = get_or_train_vocab(
                ap_key, datasets, vocab_size, n_bins, method,
                max_subjects_train=max_subjects_vocab_train,
            )
        except Exception as exc:
            exp_log.error(f"Vocab training failed for {ap_key}: {exc}")

    if not vocabs:
        exp_log.error("No vocabs available — aborting.")
        exp_log.finalize()
        return []

    # Dataset-specific vocabs for paradigms that benefit from them (e.g. P300)
    # Key: (ds_name, ap_key) → BPEVocab
    ds_specific_vocabs: dict[tuple[str, str], BPEVocab] = {}
    for ds_name in datasets:
        ds_paradigm = DATASET_INFO[ds_name].get("paradigm", "")
        if ds_paradigm in _DATASET_SPECIFIC_VOCAB_PARADIGMS:
            # High-sfreq paradigms (P300 at 2048 Hz) need V≥1024 to learn
            # meaningful merge patterns (each merge ≈ 0.5 ms at 512 tokens).
            ds_vocab_size = max(vocab_size, 1024)
            for ap_key in approaches:
                exp_log.info(
                    f"--- Vocab (dataset-specific: {ds_name}, V={ds_vocab_size}): "
                    f"approach={ap_key} ---"
                )
                try:
                    ds_specific_vocabs[(ds_name, ap_key)] = get_or_train_vocab(
                        ap_key, [ds_name], ds_vocab_size, n_bins, method,
                        max_subjects_train=max_subjects_vocab_train,
                        suffix=ds_name,
                    )
                except Exception as exc:
                    exp_log.warning(
                        f"Dataset-specific vocab failed for {ds_name}/{ap_key}: {exc}"
                    )

    # ─── Step 2: per dataset × approach ──────────────────────────────────
    # Read CSV once before the loop (avoids 36× re-reads per run)
    _df6_cache = None
    if exp_log.csv_path.exists():
        try:
            import pandas as _pd6
            _df6_cache = _pd6.read_csv(exp_log.csv_path)
        except Exception:
            _df6_cache = None

    ds_bar = (
        _tqdm(datasets, desc="Exp6 datasets", unit="ds", leave=True)
        if _HAS_TQDM else datasets
    )
    for ds_name in ds_bar:
        if _HAS_TQDM and hasattr(ds_bar, "set_description"):
            ds_bar.set_description(f"Exp6 [{ds_name}]")

        info    = DATASET_INFO[ds_name]
        sfreq   = info["sfreq"]
        cv_strat = info["cv_strategy"]
        exp_log.info(f"=== Dataset: {ds_name} (sfreq={sfreq}) ===")

        # Dataset-level skip: if all (approach, seed) combinations are already done,
        # skip data loading entirely.
        if _df6_cache is not None:
            try:
                _ds_vocab_src = (
                    "ds-specific"
                    if DATASET_INFO[ds_name].get("paradigm", "") in _DATASET_SPECIFIC_VOCAB_PARADIGMS
                    else "shared"
                )
                _all_ap_done = all(
                    len(set(
                        _df6_cache.loc[
                            (_df6_cache["dataset"].astype(str) == str(ds_name)) &
                            (_df6_cache["approach"].astype(str) == str(ap_key)) &
                            (_df6_cache["vocab_size"].astype(str) == str(vocab_size)) &
                            (_df6_cache["vocab_source"].astype(str) == _ds_vocab_src),
                            "seed"
                        ].astype(str).unique()
                    )) >= len(RANDOM_SEEDS)
                    for ap_key in approaches
                    if ap_key in vocabs
                )
                if _all_ap_done:
                    exp_log.info(f"  {ds_name}: all approaches × seeds done — skipping")
                    continue
            except Exception:
                pass

        try:
            data = load_dataset(ds_name, max_subjects=max_subjects)
        except Exception as exc:
            exp_log.error(f"Failed to load {ds_name}: {exc}")
            continue
        if not data:
            continue

        all_epochs, all_labels, all_groups = [], [], []
        for subj_id, subj_data in sorted(data.items()):
            all_epochs.append(subj_data["epochs"])
            all_labels.extend(subj_data["labels"])
            all_groups.extend([subj_id] * len(subj_data["labels"]))

        if not all_epochs:
            continue

        n_ch_max   = max(e.shape[1] for e in all_epochs)
        n_time_max = max(e.shape[2] for e in all_epochs)
        padded = []
        for e in all_epochs:
            p = np.zeros((e.shape[0], n_ch_max, n_time_max), dtype=np.float32)
            p[:, :e.shape[1], :e.shape[2]] = e
            padded.append(p)
        X_epochs = np.concatenate(padded, axis=0)

        y_raw  = np.array(all_labels)
        groups = np.array(all_groups)
        le     = LabelEncoder()
        y_enc  = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        chance = 1.0 / n_classes

        paradigm = info.get("paradigm", "")

        # ─── Per approach ──────────────────────────────────────────────
        for ap_key in approaches:
            if ap_key not in vocabs:
                exp_log.warning(f"  Skipping {ap_key}: no vocab.")
                continue

            ap_info = APPROACHES[ap_key]
            # Prefer dataset-specific vocab (e.g. for P300), else use shared
            vocab = ds_specific_vocabs.get((ds_name, ap_key), vocabs[ap_key])
            vocab_src = "ds-specific" if (ds_name, ap_key) in ds_specific_vocabs else "shared"
            exp_log.info(f"  Approach: {ap_key} ({ap_info['label']})  vocab={vocab_src}")

            # Resume: skip if all seeds already completed for this config.
            # Note: CSV writes vocab_size (function param), NOT vocab.vocab_size
            # (which may be larger for ds-specific P300 vocabs). Match accordingly.
            _pending_seeds = list(RANDOM_SEEDS)
            if _df6_cache is not None:
                try:
                    _mask6 = (
                        (_df6_cache["dataset"].astype(str) == str(ds_name)) &
                        (_df6_cache["approach"].astype(str) == str(ap_key)) &
                        (_df6_cache["vocab_size"].astype(str) == str(vocab_size)) &
                        (_df6_cache["vocab_source"].astype(str) == str(vocab_src))
                    )
                    _done6 = set(_df6_cache.loc[_mask6, "seed"].astype(str).unique())
                    _pending_seeds = [s for s in RANDOM_SEEDS if str(s) not in _done6]
                except Exception:
                    pass
            if not _pending_seeds:
                exp_log.info(f"    [resume] All seeds done — skipping")
                continue

            # 1. Preprocess — pass paradigm for adaptive envelope bands
            try:
                t_pp = time.perf_counter()
                X_pp = apply_preprocessing(X_epochs, sfreq, ap_key, paradigm=paradigm)
                exp_log.info(
                    f"    Preprocessing: {X_epochs.shape} → {X_pp.shape} "
                    f"({time.perf_counter()-t_pp:.1f}s)"
                )
            except Exception as exc:
                exp_log.error(f"    Preprocessing failed: {exc}")
                continue

            # Effective sfreq after preprocessing (for ms calculation)
            if ap_key == "downsample_64hz":
                eff_sfreq = _DS_TARGET_SFREQ
            elif ap_key == "envelope_8_30hz":
                eff_sfreq = _ENV_TARGET_SFREQ
            elif ap_key == "spectral_500ms":
                eff_sfreq = 1.0 / _SPEC_STEP_SEC   # "time steps" are windows
            else:
                eff_sfreq = sfreq

            # 2. BPE histogram features
            try:
                t_feat = time.perf_counter()
                X_hist = _cached_bpe_histograms_exp6(
                    X_pp, vocab, method, n_bins, ap_key)
                exp_log.info(
                    f"    Histogram features: {X_hist.shape} "
                    f"({time.perf_counter()-t_feat:.1f}s)"
                )
            except Exception as exc:
                exp_log.error(f"    Histogram extraction failed: {exc}")
                continue

            # 3. Compression ratio & token length
            try:
                comp_ratio = compute_compression_ratio(X_pp, vocab, method, n_bins)
                tok_len_ms = mean_token_length_ms(X_pp, vocab, method, n_bins, eff_sfreq)
                exp_log.info(
                    f"    Compression: {comp_ratio:.2f}x, "
                    f"mean token: {tok_len_ms:.1f} ms"
                )
            except Exception as exc:
                exp_log.warning(f"    Compression metric failed: {exc}")
                comp_ratio = 1.0
                tok_len_ms = 0.0

            # 4. Classification (parallel seeds)
            def _one_seed(seed):
                return run_cv(X_hist, y_enc, groups, cv_strat, seed)

            n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
            seed_results = Parallel(n_jobs=n_sjobs, prefer="threads")(
                delayed(_one_seed)(seed) for seed in _pending_seeds
            )

            for seed, fold_results in zip(_pending_seeds, seed_results):
                accs   = [f["accuracy"] for f in fold_results]
                f1s    = [f["macro_f1"] for f in fold_results]
                kappas = [f["kappa"]    for f in fold_results]
                _, ci_lo, ci_hi = bootstrap_ci(np.array(accs), n_boot=N_BOOTSTRAP)

                result = {
                    "dataset":          ds_name,
                    "approach":         ap_key,
                    "approach_label":   ap_info["label"],
                    "vocab_size":       vocab_size,
                    "vocab_source":     vocab_src,
                    "seed":             seed,
                    "paradigm":         paradigm,
                    "n_classes":        n_classes,
                    "chance_level":     round(chance, 4),
                    "accuracy_mean":    float(np.mean(accs)),
                    "accuracy_std":     float(np.std(accs)),
                    "accuracy_ci_low":  float(ci_lo),
                    "accuracy_ci_high": float(ci_hi),
                    "macro_f1_mean":    float(np.mean(f1s)),
                    "kappa_mean":       float(np.mean(kappas)),
                    "compression_ratio": round(comp_ratio, 3),
                    "mean_token_len_ms": round(tok_len_ms, 2),
                    "n_folds":          len(fold_results),
                    "n_trials":         len(y_enc),
                    "n_subjects":       int(np.unique(groups).size),
                }

                exp_log.log_result(result)
                all_results.append(result)

                # Per-fold CSV
                for f in fold_results:
                    append_csv({
                        "dataset":   ds_name,
                        "approach":  ap_key,
                        "seed":      seed,
                        "fold":      f["fold"],
                        "accuracy":  f["accuracy"],
                        "macro_f1":  f["macro_f1"],
                        "kappa":     f["kappa"],
                        "compression_ratio": comp_ratio,
                        "mean_token_len_ms": tok_len_ms,
                    }, LOGS_DIR / "exp6_per_fold.csv")

                exp_log.info(
                    f"    [{ap_key}] seed={seed} vocab={vocab_src}: "
                    f"acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f} "
                    f"(chance={chance:.3f}), "
                    f"F1={result['macro_f1_mean']:.3f}, "
                    f"κ={result['kappa_mean']:.3f}"
                )

    # ─── Save & plot ──────────────────────────────────────────────────────
    from .utils import save_json
    save_json(all_results, LOGS_DIR / "exp6_results.json")

    _plot_exp6(all_results, PLOTS_DIR / "exp6")

    exp_log.finalize()
    logger.info(f"Experiment 6 complete: {len(all_results)} result entries")
    return all_results


# ─── Visualisations ───────────────────────────────────────────────────────────

def _plot_exp6(results: list[dict], out_dir: Path):
    """Generate all Exp 6 comparison plots."""
    if not results:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate: mean metrics per (dataset, approach) across seeds
    from collections import defaultdict
    agg:       dict[tuple, list] = defaultdict(list)   # accuracy
    agg_f1:    dict[tuple, list] = defaultdict(list)   # macro F1
    agg_kappa: dict[tuple, list] = defaultdict(list)   # Cohen's κ
    comp: dict[tuple, float] = {}
    tok_ms: dict[tuple, float] = {}
    for r in results:
        key = (r["dataset"], r["approach"])
        agg[key].append(r["accuracy_mean"])
        agg_f1[key].append(r.get("macro_f1_mean", 0.0))
        agg_kappa[key].append(r.get("kappa_mean", 0.0))
        comp[key]   = r["compression_ratio"]
        tok_ms[key] = r["mean_token_len_ms"]

    datasets  = sorted(set(r["dataset"]  for r in results))
    approaches = sorted(set(r["approach"] for r in results),
                        key=lambda a: list(APPROACHES.keys()).index(a)
                        if a in APPROACHES else 99)

    approach_labels = [APPROACHES[a]["label"] if a in APPROACHES else a
                       for a in approaches]

    n_ds = len(datasets)
    n_ap = len(approaches)

    # ── 1. Accuracy comparison (grouped bar) ──────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n_ds * 2), 5))
    x = np.arange(n_ds)
    width = 0.8 / n_ap

    colors = plt.cm.tab10(np.linspace(0, 1, n_ap))
    for i, (ap_key, ap_label) in enumerate(zip(approaches, approach_labels)):
        means = [float(np.mean(agg.get((ds, ap_key), [0]))) for ds in datasets]
        bars  = ax.bar(x + i * width - 0.4 + width / 2, means,
                       width, label=ap_label, color=colors[i], alpha=0.85)

    # Chance levels
    for j, ds in enumerate(datasets):
        n_cls = DATASET_INFO[ds]["n_classes"]
        chance = 1.0 / n_cls
        ax.axhline(y=chance, color="grey", linestyle="--", linewidth=0.6,
                   xmin=(j) / n_ds, xmax=(j + 1) / n_ds, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 6: Classification Accuracy by Tokenisation Approach")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exp6_accuracy.png", dpi=150)
    plt.close(fig)

    # ── 2. Compression ratio (grouped bar) ────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n_ds * 2), 4))
    for i, (ap_key, ap_label) in enumerate(zip(approaches, approach_labels)):
        ratios = [comp.get((ds, ap_key), 1.0) for ds in datasets]
        ax.bar(x + i * width - 0.4 + width / 2, ratios,
               width, label=ap_label, color=colors[i], alpha=0.85)

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=0.8,
               label="No compression (1×)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Compression ratio")
    ax.set_title("Exp 6: BPE Compression Ratio by Approach")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exp6_compression.png", dpi=150)
    plt.close(fig)

    # ── 3. Mean token length in ms (grouped bar) ──────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, n_ds * 2), 4))
    for i, (ap_key, ap_label) in enumerate(zip(approaches, approach_labels)):
        lengths = [tok_ms.get((ds, ap_key), 0.0) for ds in datasets]
        ax.bar(x + i * width - 0.4 + width / 2, lengths,
               width, label=ap_label, color=colors[i], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylabel("Mean token length (ms)")
    ax.set_title("Exp 6: Mean BPE Token Duration by Approach")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exp6_token_lengths.png", dpi=150)
    plt.close(fig)

    # ── 4. Scatter: compression ratio vs accuracy (per dataset×approach) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (ap_key, ap_label) in enumerate(zip(approaches, approach_labels)):
        xs, ys, labels_scatter = [], [], []
        for ds in datasets:
            key = (ds, ap_key)
            if key in agg:
                xs.append(comp.get(key, 1.0))
                ys.append(float(np.mean(agg[key])))
                labels_scatter.append(ds[:6])
        if xs:
            sc = ax.scatter(xs, ys, label=ap_label, color=colors[i],
                            s=60, alpha=0.9, zorder=3)
            for lx, ly, lt in zip(xs, ys, labels_scatter):
                ax.annotate(lt, (lx, ly), textcoords="offset points",
                            xytext=(4, 2), fontsize=6, color=colors[i])

    ax.set_xlabel("BPE Compression Ratio")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 6: Compression vs Accuracy")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "exp6_compression_vs_accuracy.png", dpi=150)
    plt.close(fig)

    # ── 5. MI datasets focus: per-approach bar ─────────────────────────────
    mi_datasets = [ds for ds in datasets
                   if DATASET_INFO[ds]["paradigm"] == "Motor Imagery"]
    if mi_datasets:
        fig, ax = plt.subplots(figsize=(max(6, len(mi_datasets) * 2.5), 4))
        x_mi = np.arange(len(mi_datasets))
        for i, (ap_key, ap_label) in enumerate(zip(approaches, approach_labels)):
            means = [float(np.mean(agg.get((ds, ap_key), [0])))
                     for ds in mi_datasets]
            ax.bar(x_mi + i * width - 0.4 + width / 2, means,
                   width, label=ap_label, color=colors[i], alpha=0.85)
        ax.axhline(0.25, color="grey", linestyle="--", label="Chance (4-class)")
        ax.set_xticks(x_mi)
        ax.set_xticklabels(mi_datasets)
        ax.set_ylabel("Accuracy")
        ax.set_title("Exp 6: Motor Imagery — Approach Comparison (H5, H6, H7)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "exp6_mi_focus.png", dpi=150)
        plt.close(fig)

    # ── 6. Kappa heatmap (datasets × approaches) ──────────────────────────
    _plot_metric_heatmap(
        agg_kappa, datasets, approaches, approach_labels,
        metric_name="Cohen's κ", vmin=-0.1, vmax=1.0,
        out_path=out_dir / "exp6_kappa_heatmap.png",
        title="Exp 6: Cohen's κ by Dataset × Approach",
        fmt=".2f",
    )

    # ── 7. Macro-F1 heatmap (datasets × approaches) ───────────────────────
    _plot_metric_heatmap(
        agg_f1, datasets, approaches, approach_labels,
        metric_name="Macro F1", vmin=0.0, vmax=1.0,
        out_path=out_dir / "exp6_f1_heatmap.png",
        title="Exp 6: Macro-F1 by Dataset × Approach",
        fmt=".2f",
    )

    logger.info(f"Exp 6 plots saved to {out_dir}")


def _plot_metric_heatmap(
    agg_dict: dict,
    datasets: list[str],
    approaches: list[str],
    approach_labels: list[str],
    metric_name: str,
    vmin: float,
    vmax: float,
    out_path,
    title: str,
    fmt: str = ".2f",
):
    """Render a (dataset × approach) heatmap for any scalar metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_ds = len(datasets)
    n_ap = len(approaches)
    data_mat = np.full((n_ds, n_ap), np.nan)
    for i, ds in enumerate(datasets):
        for j, ap in enumerate(approaches):
            vals = agg_dict.get((ds, ap), [])
            if vals:
                data_mat[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(5, n_ap * 1.5), max(4, n_ds * 0.8)))
    im = ax.imshow(data_mat, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label=metric_name)

    # Annotate cells
    for i in range(n_ds):
        for j in range(n_ap):
            if not np.isnan(data_mat[i, j]):
                ax.text(j, i, f"{data_mat[i, j]:{fmt}}",
                        ha="center", va="center", fontsize=9,
                        color="black" if vmin + 0.2 * (vmax - vmin) < data_mat[i, j] < vmax - 0.2 * (vmax - vmin) else "white")

    ax.set_xticks(np.arange(n_ap))
    ax.set_xticklabels(approach_labels, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(np.arange(n_ds))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exp 6: Alternative Tokenisation")
    parser.add_argument("--datasets",  nargs="*", default=None)
    parser.add_argument("--approaches", nargs="*", default=None,
                        choices=list(APPROACHES.keys()))
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-bins",    type=int, default=64)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    run_experiment_6(
        datasets=args.datasets,
        vocab_size=args.vocab_size,
        n_bins=args.n_bins,
        max_subjects=args.max_subjects,
        approaches=args.approaches,
    )
