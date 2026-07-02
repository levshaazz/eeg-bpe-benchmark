"""
Experiment 2: Downstream Classification (H1)
=============================================
Shows that BPE-tokens are informative for classification tasks.
6 datasets × 4 classifiers × 5 baselines × 5 seeds.

Classifiers on BPE:
  - Histogram + LogReg
  - Histogram + RandomForest
  - Sequence + 1D-CNN
  - Sequence + small Transformer

Baselines:
  - CSP + LDA (MI only)
  - EEGNet (raw)
  - Fixed-size patching + Transformer
  - VQ-VAE + Transformer (approximation)
  - Chronos-binning + Transformer

Validation: LOSO-CV or 5-fold subject-split.
Stats: mean±std, 95% CI, Wilcoxon signed-rank, Holm-Bonferroni.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from scipy import stats as scipy_stats
from collections import Counter
from joblib import Parallel, delayed
from pathlib import Path

import hashlib
import time

from .config import (
    DATASET_INFO, DEVICE, RANDOM_SEEDS, N_JOBS, N_BOOTSTRAP,
    LOGS_DIR, PLOTS_DIR, MODELS_DIR, CACHE_DIR,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
    PRIMARY_METRIC_PER_DATASET, PRIMARY_METRIC_DEFAULT, VOCAB_SWEEP_SIZES,
)
from .quantization import quantize
from .bpe_engine import train_bpe, apply_bpe, apply_bpe_batch, BPEVocab
from .data_loading import load_dataset
from .utils import (
    ExperimentLogger, save_json, save_csv, append_csv, timed, get_logger,
    bootstrap_ci, wilcoxon_holm, pca_reduce,
)

logger = get_logger("exp2")

# Reference threshold: datasets with more than this many subjects should use
# subject-stratified k-fold rather than LOSO (set explicitly in config.DATASET_INFO).
# physionet_mi (109 subjects) is now explicitly "5fold-subject" in config.
MAX_LOSO_SUBJECTS: int = 20

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ─── Feature extraction ──────────────────────────────────────────────────────

def epochs_to_bpe_histograms(epochs: np.ndarray,
                              vocab: BPEVocab,
                              method: str = "mu_law",
                              n_bins: int = 256) -> np.ndarray:
    """
    Convert epochs to BPE histogram features.

    Each trial is represented as the concatenation of per-channel
    BPE token histograms (L1-normalised).

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method (default ``"mu_law"``).
    n_bins : int, optional
        Number of quantization bins (default 256).

    Returns
    -------
    np.ndarray
        Histogram features, shape ``(n_trials, n_ch * vocab_size)``.
    """
    n_trials, n_ch, n_time = epochs.shape
    V = vocab.vocab_size
    n_total = n_trials * n_ch

    # Quantize all channels at once (fully vectorised)
    flat = epochs.reshape(-1, n_time)          # (n_trials * n_ch, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)

    seq_list = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # ── Vectorised histogram via padded matrix + np.where ─────────────────
    if bpe_seqs:
        lengths = np.array([len(s) for s in bpe_seqs], dtype=np.int32)
        max_len = int(lengths.max()) if lengths.size > 0 else 0
    else:
        max_len = 0

    if max_len > 0:
        padded = np.full((n_total, max_len), V, dtype=np.int32)
        for i, seq in enumerate(bpe_seqs):
            if seq:
                l = len(seq)
                padded[i, :l] = seq[:l]
        valid = padded < V
        row_idx, col_pos = np.where(valid)
        tok_vals = padded[row_idx, col_pos]
        flat_idx = row_idx.astype(np.int64) * V + tok_vals.astype(np.int64)
        counts_flat = np.bincount(flat_idx, minlength=n_total * V)
        histograms = counts_flat[:n_total * V].reshape(
            n_trials, n_ch, V).astype(np.float32)
    else:
        histograms = np.zeros((n_trials, n_ch, V), dtype=np.float32)

    # L1-normalise per channel
    sums = histograms.sum(axis=-1, keepdims=True)
    histograms = histograms / (sums + 1e-10)

    return histograms.reshape(n_trials, -1)     # (n_trials, n_ch * V)


# ─── Disk-cached histogram wrapper ───────────────────────────────────────────

_HIST_DISK_CACHE_DIR = CACHE_DIR / "hist_cache"

# In-memory cache: avoids repeated disk reads when the same .npz is loaded
# multiple times (e.g., 5 seeds × same dataset × same vocab).
# Key: str(cache_path), Value: np.ndarray
_MEM_CACHE: dict[str, np.ndarray] = {}
_SEQ_MEM_CACHE: dict[str, list] = {}


def _hist_cache_key(all_epochs: np.ndarray, vocab: BPEVocab,
                    method: str, n_bins: int) -> str:
    """
    Build a short cache key for the (epochs, vocab, method, n_bins) combination.

    Vocab identity  → md5 of its merge list (fast, deterministic).
    Epochs identity → shape + md5 of a small sample (first+last 8 trials).
    """
    # Hash vocab merges
    vocab_hash = hashlib.md5(str(vocab.merges).encode()).hexdigest()[:12]
    # Hash data: shape + boundary rows (avoids hashing the full 100-MB array)
    n, c, t = all_epochs.shape
    n_sample = min(8, n)
    sample = np.concatenate([all_epochs[:n_sample].ravel(),
                              all_epochs[-n_sample:].ravel()])
    data_hash = hashlib.md5(sample.tobytes()).hexdigest()[:12]
    return f"v{vocab_hash}_d{data_hash}_{method}_b{n_bins}_n{n}_c{c}_t{t}"


def cached_epochs_to_bpe_histograms(all_epochs: np.ndarray,
                                     vocab: BPEVocab,
                                     method: str = "uniform",
                                     n_bins: int = 64,
                                     cache_dir: Path | None = None,
                                     pre_computed_seqs: list | None = None) -> np.ndarray:
    """
    ``epochs_to_bpe_histograms`` with transparent disk and in-memory caching.

    On the first call for a given (epochs, vocab, method, n_bins) combination
    the histogram matrix is computed and saved as a compressed ``.npz`` file
    under *cache_dir* (default ``results/cache/hist_cache/``).  Subsequent
    calls with identical inputs load directly from memory (if this process has
    already loaded the file) or disk, making re-runs and multi-seed ablation
    sweeps near-instant.

    Parameters
    ----------
    all_epochs : np.ndarray, shape (n_trials, n_ch, n_time)
    vocab : BPEVocab
    method : str
    n_bins : int
    cache_dir : Path or None
        Directory for cached ``.npz`` files.  Created automatically.
    pre_computed_seqs : list of list[int] or None
        If provided, skip BPE re-computation and build histogram directly from
        these pre-computed token sequences (avoids double BPE application when
        ``cached_bpe_token_sequences`` has already been called).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_ch * vocab_size)``, same as
        :func:`epochs_to_bpe_histograms`.
    """
    if cache_dir is None:
        cache_dir = _HIST_DISK_CACHE_DIR
    cache_dir = Path(cache_dir)

    key        = _hist_cache_key(all_epochs, vocab, method, n_bins)
    cache_path = cache_dir / f"{key}.npz"
    mem_key    = str(cache_path)

    # 1. In-memory cache (fastest: avoids disk I/O on repeated loads)
    if mem_key in _MEM_CACHE:
        logger.info(f"  [hist mem  hit]   {cache_path.name}")
        return _MEM_CACHE[mem_key]

    # 2. Disk cache
    if cache_path.exists():
        try:
            X = np.load(str(cache_path))["X"]
            logger.info(f"  [hist cache hit]  {cache_path.name}")
            _MEM_CACHE[mem_key] = X
            return X
        except Exception as _cache_err:
            logger.warning(f"  [hist cache corrupt] {cache_path.name}: {_cache_err} — rebuilding")
            cache_path.unlink(missing_ok=True)

    # 3. Compute — use pre_computed_seqs if available to avoid double BPE pass
    logger.info(f"  [hist cache miss] computing histograms → {cache_path.name}")
    if pre_computed_seqs is not None:
        n, c = all_epochs.shape[:2]
        V = vocab.vocab_size
        X = _hist_from_bpe_seqs(pre_computed_seqs, n, c, V)
    else:
        X = epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X)
    logger.info(f"  [hist cache save] {cache_path.name} "
                f"({cache_path.stat().st_size // 1024} KB)")
    _MEM_CACHE[mem_key] = X
    return X


# ─── Shared BPE token sequence cache ─────────────────────────────────────────
# Quantize + apply_bpe_batch is the most expensive step in exp2 (O(N·M) for N
# trials × C channels × M merges).  By caching the resulting token sequences
# we avoid repeating this work for every feature type (hist, bigram, sequences)
# in the same experiment run.

def cached_bpe_token_sequences(
    all_epochs: np.ndarray,
    vocab: BPEVocab,
    method: str = "uniform",
    n_bins: int = 64,
    cache_dir: Path | None = None,
) -> list[list[int]]:
    """
    Quantize + apply BPE to all (trial, channel) pairs, with disk caching.

    Returns a list of ``n_trials * n_ch`` BPE token sequences in row-major
    order (same order as ``all_epochs.reshape(-1, n_time)``).  On cache hit
    the result is loaded from a compressed ``.npz`` file in < 1 second.

    The cache uses the same key as :func:`cached_epochs_to_bpe_histograms`
    (vocab merges hash + data boundary hash + shape), so the two caches stay
    in sync when the vocab or data changes.

    Parameters
    ----------
    all_epochs : np.ndarray, shape (n_trials, n_ch, n_time)
    vocab : BPEVocab
    method, n_bins : quantization settings
    cache_dir : directory for cached ``.npz`` files

    Returns
    -------
    list of list[int]
        Length ``n_trials * n_ch``.  Each inner list is a BPE token sequence.
    """
    if cache_dir is None:
        cache_dir = _HIST_DISK_CACHE_DIR
    cache_dir = Path(cache_dir)

    key = _hist_cache_key(all_epochs, vocab, method, n_bins)
    cache_path = cache_dir / f"seqs_{key}.npz"
    mem_key = f"seq_{cache_path}"

    # 1. In-memory cache (avoids disk I/O on repeated calls within same process)
    if mem_key in _SEQ_MEM_CACHE:
        logger.info(f"  [seq  mem  hit]   {cache_path.name}")
        return _SEQ_MEM_CACHE[mem_key]

    # 2. Disk cache
    if cache_path.exists():
        logger.info(f"  [seq  cache hit]  {cache_path.name}")
        npz = np.load(str(cache_path))
        data, lengths = npz["data"], npz["lengths"]
        seqs: list[list[int]] = []
        offset = 0
        for l in lengths:
            seqs.append(data[offset:offset + l].tolist())
            offset += l
        _SEQ_MEM_CACHE[mem_key] = seqs
        return seqs

    logger.info(f"  [seq  cache miss] computing BPE seqs → {cache_path.name}")
    n_trials, n_ch, n_time = all_epochs.shape
    flat = all_epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
    seq_list = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # Serialise ragged list as concatenated array + lengths
    lengths_arr = np.array([len(s) for s in bpe_seqs], dtype=np.int32)
    data_arr    = np.concatenate(
        [np.array(s, dtype=np.int32) for s in bpe_seqs]
    ) if any(bpe_seqs) else np.empty(0, dtype=np.int32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), data=data_arr, lengths=lengths_arr)
    logger.info(f"  [seq  cache save] {cache_path.name} "
                f"({cache_path.stat().st_size // 1024} KB)")
    _SEQ_MEM_CACHE[mem_key] = bpe_seqs
    return bpe_seqs


def _hist_from_bpe_seqs(bpe_seqs: list[list[int]],
                         n_trials: int, n_ch: int, V: int) -> np.ndarray:
    """Build L1-normalised (n_trials, n_ch*V) histogram from pre-computed BPE seqs.

    Vectorised: pad all seqs into one 2-D array, mask invalids, single bincount.
    """
    n_total = n_trials * n_ch

    # Pad all seqs into (n_total, max_len) array with sentinel = V (out-of-vocab)
    if bpe_seqs:
        lengths = np.array([len(s) for s in bpe_seqs], dtype=np.int32)
        max_len = int(lengths.max()) if lengths.size > 0 else 0
    else:
        max_len = 0

    if max_len == 0:
        histograms = np.zeros((n_trials, n_ch, V), dtype=np.float32)
        return histograms.reshape(n_trials, -1)

    # Fill padded matrix — sentinel V will be filtered by the < V mask
    padded = np.full((n_total, max_len), V, dtype=np.int32)
    for i, seq in enumerate(bpe_seqs):
        if seq:
            l = len(seq)
            padded[i, :l] = seq[:l]

    # Mask valid tokens (< V), build flat COO indices in one shot
    valid = padded < V  # (n_total, max_len) bool
    row_idx, col_pos = np.where(valid)
    tok_vals = padded[row_idx, col_pos]  # token IDs
    flat_idx = row_idx.astype(np.int64) * V + tok_vals.astype(np.int64)

    counts_flat = np.bincount(flat_idx, minlength=n_total * V)
    histograms = counts_flat[:n_total * V].reshape(
        n_trials, n_ch, V).astype(np.float32)

    sums = histograms.sum(axis=-1, keepdims=True)
    histograms = histograms / (sums + 1e-10)
    return histograms.reshape(n_trials, -1)


def _bigram_from_bpe_seqs(bpe_seqs: list[list[int]],
                            n_trials: int, n_ch: int, V: int) -> np.ndarray:
    """Build L1-normalised bigram matrix from pre-computed BPE seqs.

    Vectorised: pad seqs, clip to V_bg, compute bigram pairs across all
    seqs in one pass, aggregate per trial via np.add.at.
    """
    V_bg = min(V, 128)
    X = np.zeros((n_trials, V_bg * V_bg), dtype=np.float32)

    # Collect all valid bigram pairs across all sequences
    trial_ids: list[np.ndarray] = []
    bigram_ids: list[np.ndarray] = []
    for idx, seq in enumerate(bpe_seqs):
        if len(seq) < 2:
            continue
        arr = np.asarray(seq, dtype=np.int32)
        arr = arr[arr < V_bg]
        if len(arr) < 2:
            continue
        trial_idx = idx // n_ch
        bg = arr[:-1].astype(np.int64) * V_bg + arr[1:].astype(np.int64)
        trial_ids.append(np.full(len(bg), trial_idx, dtype=np.int32))
        bigram_ids.append(bg)

    if trial_ids:
        all_trials = np.concatenate(trial_ids)
        all_bigrams = np.concatenate(bigram_ids)
        # Single vectorised scatter-add
        np.add.at(X, (all_trials, all_bigrams), 1.0)

    sums = X.sum(axis=1, keepdims=True)
    X = X / (sums + 1e-10)
    return X


def _flat_seqs_from_bpe_seqs(bpe_seqs: list[list[int]],
                               n_trials: int, n_ch: int, V: int,
                               max_len: int = 512) -> np.ndarray:
    """Build (n_trials, max_len) flat token sequence array from pre-computed BPE seqs."""
    SEP = V
    PAD = V + 1
    sep_slots  = n_ch - 1
    max_per_ch = max(4, (max_len - sep_slots) // n_ch)

    ch_tokens = np.full((n_trials * n_ch, max_per_ch), PAD, dtype=np.int32)
    for i, seq in enumerate(bpe_seqs):
        l = min(len(seq), max_per_ch)
        if l > 0:
            ch_tokens[i, :l] = seq[:l]
    ch_tokens = ch_tokens.reshape(n_trials, n_ch, max_per_ch)

    sep_col = np.full((n_trials, n_ch - 1, 1), SEP, dtype=np.int32)
    ch_with_sep = np.concatenate([ch_tokens[:, :-1, :], sep_col], axis=2)
    combined = np.concatenate(
        [ch_with_sep.reshape(n_trials, (n_ch - 1) * (max_per_ch + 1)),
         ch_tokens[:, -1, :]],
        axis=1,
    )
    result = np.full((n_trials, max_len), PAD, dtype=np.int32)
    take = min(combined.shape[1], max_len)
    result[:, :take] = combined[:, :take]
    return result


def _cw_seqs_from_bpe_seqs(bpe_seqs: list[list[int]],
                             n_trials: int, n_ch: int,
                             max_len_per_ch: int = 128,
                             pad_id: int = 0) -> np.ndarray:
    """Build (n_trials, n_ch, max_len_per_ch) channelwise sequences from pre-computed seqs.

    Parameters
    ----------
    pad_id : int
        Padding token ID (must be ``vocab_size + 1`` to match ChannelwiseTransformer).
    """
    result = np.full((n_trials, n_ch, max_len_per_ch), pad_id, dtype=np.int32)
    for i, seq in enumerate(bpe_seqs):
        trial_idx = i // n_ch
        ch_idx    = i %  n_ch
        l = min(len(seq), max_len_per_ch)
        if l > 0:
            result[trial_idx, ch_idx, :l] = seq[:l]
    return result


def epochs_to_windowed_bpe_histograms(epochs: np.ndarray,
                                       vocab: BPEVocab,
                                       method: str = "uniform",
                                       n_bins: int = 64,
                                       n_windows: int = 5) -> np.ndarray:
    """
    BPE histogram features with temporal windowing (H9).

    Quantises each channel over the *full* epoch (consistent normalisation),
    then splits the quantised codes into ``n_windows`` equal-length windows
    before applying BPE and computing per-window histograms.  The window
    histograms are concatenated, giving the classifier access to *when* in
    the epoch each token appeared.

    For a 4-second Motor Imagery trial at 250 Hz: window = 0.8 s, capturing
    ERD-onset (~window 1–2), sustained ERD (windows 2–4), and ERS-recovery
    (~window 5).

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method (default ``"uniform"``).
    n_bins : int, optional
        Number of quantization bins (default 64).
    n_windows : int, optional
        Number of temporal windows (default 5).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_windows * n_ch * vocab_size)``,
        L1-normalised per (window, channel).
    """
    n_trials, n_ch, n_time = epochs.shape
    V = vocab.vocab_size
    n_total = n_trials * n_ch
    window_len = max(1, n_time // n_windows)

    # Quantise full epoch once for consistent per-channel normalisation
    flat = epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)

    # Build all window sequences in one batched apply_bpe_batch call.
    # Interleave: [win0_seq0, win1_seq0, ..., winW-1_seq0, win0_seq1, ...]
    # so that window index = global_idx % n_windows, seq index = global_idx // n_windows.
    # This gives apply_bpe_batch a larger pool to parallelise and avoids N_JOBS
    # overhead per window.
    all_win_seqlists: list[list[int]] = []
    window_boundaries: list[tuple[int, int]] = []  # (t_start, t_end) per window
    for w in range(n_windows):
        t_start = w * window_len
        t_end = n_time if w == n_windows - 1 else (w + 1) * window_len
        window_boundaries.append((t_start, t_end))
        win_codes = codes_flat[:, t_start:t_end]
        for i in range(n_total):
            all_win_seqlists.append(win_codes[i].tolist())

    all_bpe_seqs = apply_bpe_batch(all_win_seqlists, vocab, n_jobs=N_JOBS)

    window_hists = []
    for w, (t_start, t_end) in enumerate(window_boundaries):
        # Extract the n_total sequences for this window
        bpe_seqs_w = all_bpe_seqs[w * n_total:(w + 1) * n_total]

        # Vectorised histogram (COO-style bincount)
        row_idx_list: list[np.ndarray] = []
        col_idx_list: list[np.ndarray] = []
        for idx, bpe_seq in enumerate(bpe_seqs_w):
            if not bpe_seq:
                continue
            arr = np.asarray(bpe_seq, dtype=np.int32)
            arr = arr[arr < V]
            if len(arr) > 0:
                row_idx_list.append(np.full(len(arr), idx, dtype=np.int32))
                col_idx_list.append(arr)

        if row_idx_list:
            rows = np.concatenate(row_idx_list).astype(np.int64)
            cols = np.concatenate(col_idx_list).astype(np.int64)
            flat_idx = rows * V + cols
            counts = np.bincount(flat_idx, minlength=n_total * V)
            h = counts[:n_total * V].reshape(n_trials, n_ch, V).astype(np.float32)
        else:
            h = np.zeros((n_trials, n_ch, V), dtype=np.float32)

        sums = h.sum(axis=-1, keepdims=True)
        h = h / (sums + 1e-10)
        window_hists.append(h.reshape(n_trials, n_ch * V))

    # Memory guard: if total array > 8 GB, PCA-reduce each window before concat.
    # Same budget as main histogram (also capped at _MAX_HIST_FEATURES via pca_reduce).
    _total_bytes = int(n_trials) * int(n_windows) * int(n_ch) * int(V) * 4
    if _total_bytes > 8_000_000_000:
        _target = min(2048, n_ch * V)
        window_hists = [pca_reduce(_wh, _target, device=DEVICE) for _wh in window_hists]

    return np.concatenate(window_hists, axis=1)   # (n_trials, n_windows * n_ch * V)


def cached_epochs_to_windowed_bpe_histograms(
    all_epochs: np.ndarray,
    vocab: BPEVocab,
    method: str = "uniform",
    n_bins: int = 64,
    n_windows: int = 5,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """Disk-cached wrapper for :func:`epochs_to_windowed_bpe_histograms`."""
    if cache_dir is None:
        cache_dir = _HIST_DISK_CACHE_DIR
    cache_dir = Path(cache_dir)

    key = _hist_cache_key(all_epochs, vocab, method, n_bins)
    cache_path = cache_dir / f"windowed{n_windows}_{key}.npz"
    mem_key = f"win_{cache_path}"

    if mem_key in _MEM_CACHE:
        logger.info(f"  [win  mem  hit]   {cache_path.name}")
        return _MEM_CACHE[mem_key]

    if cache_path.exists():
        logger.info(f"  [win  cache hit]  {cache_path.name}")
        X = np.load(str(cache_path))["X"]
        _MEM_CACHE[mem_key] = X
        return X

    logger.info(f"  [win  cache miss] computing windowed hists → {cache_path.name}")
    X = epochs_to_windowed_bpe_histograms(all_epochs, vocab, method, n_bins, n_windows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(cache_path), X=X)
    logger.info(f"  [win  cache save] {cache_path.name} "
                f"({cache_path.stat().st_size // 1024} KB)")
    _MEM_CACHE[mem_key] = X
    return X


def epochs_to_bigram_histograms(epochs: np.ndarray,
                                 vocab: BPEVocab,
                                 method: str = "uniform",
                                 n_bins: int = 64) -> np.ndarray:
    """
    Bigram (adjacent-pair) histogram features from BPE token sequences (H10).

    For each (trial, channel) sequence, counts adjacent token pairs
    (t_i, t_{i+1}); counts are aggregated across channels per trial.
    Unlike bag-of-tokens, bigrams preserve *local sequential order*:
    two sequences with the same tokens in different order yield different
    bigram histograms.

    The bigram vocabulary is capped at ``V_bg = min(vocab_size, 128)`` to
    keep the V_bg² feature space tractable (128² = 16 384 features).
    Tokens above this cap are discarded before counting.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method (default ``"uniform"``).
    n_bins : int, optional
        Number of quantization bins (default 64).

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, V_bg * V_bg)``, L1-normalised per trial.
    """
    n_trials, n_ch, n_time = epochs.shape
    V = vocab.vocab_size
    V_bg = min(V, 128)   # bigram vocab cap: V_bg² ≤ 16 384 features
    n_total = n_trials * n_ch

    flat = epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
    seq_list = [codes_flat[i].tolist() for i in range(n_total)]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # Accumulate bigram counts per trial (summed across channels)
    X = np.zeros((n_trials, V_bg * V_bg), dtype=np.float32)
    for idx, seq in enumerate(bpe_seqs):
        trial_idx = idx // n_ch
        arr = np.asarray(seq, dtype=np.int32)
        arr = arr[arr < V_bg]   # keep tokens within bigram vocab
        if len(arr) < 2:
            continue
        t_i = arr[:-1].astype(np.int64)
        t_j = arr[1:].astype(np.int64)
        counts = np.bincount(t_i * V_bg + t_j,
                             minlength=V_bg * V_bg).astype(np.float32)
        X[trial_idx] += counts

    # L1-normalise per trial
    sums = X.sum(axis=1, keepdims=True)
    X = X / (sums + 1e-10)
    return X   # (n_trials, V_bg²)


def epochs_to_bpe_sequences(epochs: np.ndarray,
                             vocab: BPEVocab,
                             method: str = "mu_law",
                             n_bins: int = 256,
                             max_len: int = 512) -> np.ndarray:
    """
    Convert epochs to flat BPE token sequences (for 1D-CNN / flat Transformer).

    Channels are concatenated with SEP tokens, with *equal allocation* per
    channel so all channels are represented even when the combined sequence
    would exceed *max_len*.

    PAD_TOKEN = vocab_size + 1  (NOT 0, which is a valid amplitude bin).
    SEP_TOKEN = vocab_size.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method (default ``"mu_law"``).
    n_bins : int, optional
        Number of quantization bins (default 256).
    max_len : int, optional
        Total sequence length including SEP tokens (default 512).

    Returns
    -------
    np.ndarray
        Integer token array, shape ``(n_trials, max_len)``,
        padded with ``vocab_size + 1``.
    """
    n_trials, n_ch, n_time = epochs.shape
    V   = vocab.vocab_size
    SEP = V          # channel separator
    PAD = V + 1      # dedicated padding — avoids conflating token 0 with padding

    flat = epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
    seq_list = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # Equal token budget per channel: reserve 1 position per SEP
    sep_slots    = n_ch - 1
    max_per_ch   = max(4, (max_len - sep_slots) // n_ch)

    # Pre-fill a flat (n_total, max_per_ch) array with PAD, then place tokens
    ch_tokens = np.full((n_trials * n_ch, max_per_ch), PAD, dtype=np.int32)
    for i, seq in enumerate(bpe_seqs):
        l = min(len(seq), max_per_ch)
        if l > 0:
            ch_tokens[i, :l] = seq[:l]
    ch_tokens = ch_tokens.reshape(n_trials, n_ch, max_per_ch)

    # Interleave channels + SEP tokens into result via numpy concatenation per trial
    # sep column: shape (n_trials, n_ch-1, 1) filled with SEP
    sep_col = np.full((n_trials, n_ch - 1, 1), SEP, dtype=np.int32)
    # Build (n_trials, n_ch, max_per_ch+1) by appending SEP to all but last channel
    ch_with_sep = np.concatenate(
        [ch_tokens[:, :-1, :], sep_col], axis=2
    )  # (n_trials, n_ch-1, max_per_ch+1)
    # Last channel has no trailing SEP
    combined = np.concatenate(
        [ch_with_sep.reshape(n_trials, (n_ch - 1) * (max_per_ch + 1)),
         ch_tokens[:, -1, :]],
        axis=1,
    )  # (n_trials, (n_ch-1)*(max_per_ch+1) + max_per_ch)

    result = np.full((n_trials, max_len), PAD, dtype=np.int32)
    take = min(combined.shape[1], max_len)
    result[:, :take] = combined[:, :take]
    return result


def epochs_to_bpe_sequences_channelwise(epochs: np.ndarray,
                                         vocab: BPEVocab,
                                         method: str = "mu_law",
                                         n_bins: int = 256,
                                         max_len_per_ch: int = 128) -> np.ndarray:
    """
    Convert epochs to per-channel BPE token sequences.

    Unlike ``epochs_to_bpe_sequences``, each channel is kept as an
    independent sequence of length *max_len_per_ch*.  This is the correct
    input format for ``classify_seq_transformer_channelwise``.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method.
    n_bins : int, optional
        Number of quantization bins.
    max_len_per_ch : int, optional
        Maximum tokens per channel (default 128).

    Returns
    -------
    np.ndarray
        Integer token array, shape ``(n_trials, n_ch, max_len_per_ch)``,
        padded with ``vocab_size + 1``.
    """
    n_trials, n_ch, n_time = epochs.shape
    V   = vocab.vocab_size
    PAD = V + 1

    flat = epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
    seq_list = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    bpe_seqs = apply_bpe_batch(seq_list, vocab, n_jobs=N_JOBS)

    # Vectorised fill: pre-allocate flat (n_total, max_len_per_ch), then reshape
    n_total = n_trials * n_ch
    result_flat = np.full((n_total, max_len_per_ch), PAD, dtype=np.int32)
    for i, seq in enumerate(bpe_seqs):
        l = min(len(seq), max_len_per_ch)
        if l > 0:
            result_flat[i, :l] = seq[:l]
    return result_flat.reshape(n_trials, n_ch, max_len_per_ch)


# ─── Classifiers ──────────────────────────────────────────────────────────────

def classify_histogram_logreg(X_train, y_train, X_test, y_test):
    """
    Logistic Regression classifier on BPE histograms.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels (unused; for API consistency).

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Class probabilities.
    """
    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER, C=LOGREG_C, solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    return y_pred, y_proba


def classify_histogram_rf(X_train, y_train, X_test, y_test):
    """
    Random Forest classifier on BPE histograms.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels (unused; for API consistency).

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Class probabilities.
    """
    clf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                  n_jobs=N_JOBS, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    return y_pred, y_proba


# ─── Baselines ────────────────────────────────────────────────────────────────

def baseline_raw_bins_logreg(epochs: np.ndarray, labels: np.ndarray,
                              train_idx, test_idx,
                              method: str = "mu_law", n_bins: int = 256):
    """
    Baseline: quantized bins (no BPE) + histogram + Logistic Regression.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_ch, n_time)``.
    labels : np.ndarray
        Trial labels.
    train_idx : array-like
        Training indices.
    test_idx : array-like
        Test indices.
    method : str, optional
        Quantization method (default ``"mu_law"``).
    n_bins : int, optional
        Number of quantization bins (default 256).

    Returns
    -------
    np.ndarray
        Predicted labels for the test set.
    """
    n_trials, n_ch, n_time = epochs.shape
    flat = epochs.reshape(-1, n_time)
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)

    # ── Vectorised batch bincount via offset trick ────────────────────────
    # codes_flat: (n_trials * n_ch, n_time) — treat each row as one histogram
    n_rows   = codes_flat.shape[0]
    offsets  = np.arange(n_rows, dtype=np.int64)[:, None] * n_bins
    shifted  = (codes_flat.astype(np.int64) + offsets).ravel()
    counts   = np.bincount(shifted, minlength=n_rows * n_bins)
    hist     = counts[:n_rows * n_bins].reshape(
                   n_trials, n_ch, n_bins).astype(np.float32)
    sums     = hist.sum(axis=-1, keepdims=True)
    hist     = hist / (sums + 1e-10)
    X        = hist.reshape(n_trials, -1)

    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER, C=LOGREG_C, solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
    )
    clf.fit(X[train_idx], labels[train_idx])
    y_pred = clf.predict(X[test_idx])
    return y_pred


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute classification metrics: accuracy, macro-F1, kappa, AUC-ROC.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities, shape (n_samples, n_classes).

    Returns
    -------
    dict
        Metrics including accuracy, macro_f1, kappa, and auc_roc if available.
    """
    from sklearn.metrics import confusion_matrix as _cm

    le = LabelEncoder()
    y_enc = le.fit_transform(y_true)
    try:
        p_enc = le.transform(y_pred)
    except ValueError:
        p_enc = y_pred

    metrics = {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa":             float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix":  _cm(y_true, y_pred).tolist(),
    }

    # AUC-ROC: binary or multiclass (OVR)
    if y_proba is not None:
        try:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics["auc_roc"] = float(roc_auc_score(y_enc, y_proba[:, 1]))
            else:
                metrics["auc_roc"] = float(
                    roc_auc_score(y_enc, y_proba, multi_class="ovr", average="macro")
                )
        except Exception:
            pass

    return metrics


# ─── Cross-validation strategies ─────────────────────────────────────────────

def run_loso_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                clf_fn, seed: int = 42) -> list[dict]:
    """
    Leave-One-Subject-Out Cross-Validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    groups : np.ndarray
        Subject group IDs.
    clf_fn : callable
        Classifier function ``(X_train, y_train, X_test, y_test) -> (y_pred, y_proba)``.
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    list of dict
        Per-fold metric dicts.
    """
    logo = LeaveOneGroupOut()
    fold_results = []
    n_folds = logo.get_n_splits(X, y, groups)
    _fold_bar = (
        _tqdm(enumerate(logo.split(X, y, groups)),
              total=n_folds, desc="LOSO", unit="fold",
              leave=False, dynamic_ncols=True)
        if _HAS_TQDM else enumerate(logo.split(X, y, groups))
    )

    for fold_idx, (train_idx, test_idx) in _fold_bar:
        y_pred, y_proba = clf_fn(X[train_idx], y[train_idx],
                                  X[test_idx], y[test_idx])
        metrics = compute_classification_metrics(y[test_idx], y_pred, y_proba)
        metrics["fold"] = fold_idx
        metrics["test_subject"] = int(groups[test_idx[0]])
        fold_results.append(metrics)

    return fold_results


def run_kfold_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 clf_fn, n_splits: int = 5, seed: int = 42) -> list[dict]:
    """
    K-fold CV with subject-level splits (no data leakage).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    groups : np.ndarray
        Subject group IDs.
    clf_fn : callable
        Classifier function ``(X_train, y_train, X_test, y_test) -> (y_pred, y_proba)``.
    n_splits : int, optional
        Number of folds (default 5).
    seed : int, optional
        Random seed (default 42).

    Returns
    -------
    list of dict
        Per-fold metric dicts.
    """
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        logger.warning(f"run_kfold_cv: only {len(unique_groups)} groups, "
                       f"falling back to LOSO")
        return run_loso_cv(X, y, groups, clf_fn, seed=seed)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    fold_size = len(unique_groups) // n_splits
    fold_results = []
    _fold_bar = (
        _tqdm(range(n_splits), desc="KFold", unit="fold",
              leave=False, dynamic_ncols=True)
        if _HAS_TQDM else range(n_splits)
    )

    for fold_idx in _fold_bar:
        start = fold_idx * fold_size
        if fold_idx == n_splits - 1:
            test_groups = unique_groups[start:]
        else:
            test_groups = unique_groups[start:start + fold_size]

        test_mask = np.isin(groups, test_groups)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]

        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        y_pred, y_proba = clf_fn(X[train_idx], y[train_idx],
                                  X[test_idx], y[test_idx])
        metrics = compute_classification_metrics(y[test_idx], y_pred, y_proba)
        metrics["fold"] = fold_idx
        fold_results.append(metrics)

    return fold_results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_exp2_results(results: list[dict], save_dir: Path) -> None:
    """
    Bar chart: accuracy per dataset × classifier.

    Parameters
    ----------
    results : list of dict
        Experiment 2 result dicts.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in results))
    classifiers = sorted(set(r["classifier"] for r in results))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(datasets))
    width = 0.8 / max(len(classifiers), 1)

    for i, clf_name in enumerate(classifiers):
        accs = []
        stds = []
        for ds in datasets:
            subset = [r for r in results
                      if r["dataset"] == ds and r["classifier"] == clf_name]
            if subset:
                accs.append(np.mean([r["accuracy_mean"] for r in subset]))
                stds.append(np.mean([r["accuracy_std"] for r in subset]))
            else:
                accs.append(0)
                stds.append(0)
        ax.bar(x + i * width, accs, width, yerr=stds, label=clf_name, alpha=0.8)

    ax.set_xticks(x + width * len(classifiers) / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 2: Downstream Classification")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_dir / "exp2_accuracy_overview.png", dpi=150)
    plt.close(fig)


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp2")
def run_experiment_2(datasets: list[str] | None = None,
                     vocab_size: int = 4096,
                     n_bins: int = 64,
                     method: str = "uniform",
                     max_subjects: int | None = None) -> list[dict]:
    """
    Run Experiment 2: Downstream classification.

    Parameters
    ----------
    datasets : list of str or None, optional
        Datasets to evaluate (default: all).
    vocab_size : int, optional
        BPE vocabulary size (default 4096).
    n_bins : int, optional
        Number of quantization bins (default 64; A2 ablation shows 64 > 256).
    method : str, optional
        Quantization method (default ``"uniform"``; A1 ablation shows uniform >= adaptive > mu_law).
    max_subjects : int or None, optional
        Max subjects per dataset.

    Returns
    -------
    list of dict
        Classification result dicts.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())

    exp_log = ExperimentLogger("exp2_downstream")
    all_results = []

    # ── Resume: load already-completed (dataset, classifier, seed) ────────────
    # Previously saved results are reloaded into all_results so that the
    # final Wilcoxon comparisons and plots cover the complete picture.
    _results_csv = LOGS_DIR / "exp2_downstream_results.csv"
    _done_keys: set[tuple] = set()
    if _results_csv.exists():
        try:
            import csv as _csv
            import json as _json
            with open(_results_csv, newline="") as _f:
                for _row in _csv.DictReader(_f):
                    _k = (_row["dataset"], _row["classifier"],
                          str(_row.get("vocab_size", vocab_size)), str(_row["seed"]))
                    _done_keys.add(_k)
                    # Reconstruct result dict for Wilcoxon / plots
                    try:
                        _res = {
                            "dataset":                _row["dataset"],
                            "classifier":             _row["classifier"],
                            "vocab_size":             int(_row["vocab_size"]),
                            "seed":                   int(_row["seed"]),
                            "cv_strategy":            _row.get("cv_strategy", "unknown"),
                            "accuracy_mean":          float(_row["accuracy_mean"]),
                            "accuracy_std":           float(_row["accuracy_std"]),
                            "accuracy_ci_low":        float(_row["accuracy_ci_low"]),
                            "accuracy_ci_high":       float(_row["accuracy_ci_high"]),
                            "balanced_accuracy_mean": (float(_row["balanced_accuracy_mean"])
                                                       if _row.get("balanced_accuracy_mean") else None),
                            "macro_f1_mean":          float(_row["macro_f1_mean"]),
                            "kappa_mean":             float(_row["kappa_mean"]),
                            "auc_roc_mean":           (float(_row["auc_roc_mean"])
                                                       if _row.get("auc_roc_mean") else None),
                            "n_folds":                int(_row["n_folds"]),
                            "n_trials":               int(_row["n_trials"]),
                            "n_subjects":             int(_row["n_subjects"]),
                        }
                        all_results.append(_res)
                    except Exception:
                        pass  # skip malformed rows
            if _done_keys:
                exp_log.info(
                    f"Resume mode: {len(_done_keys)} (dataset, classifier, vocab_size, seed) "
                    f"combos already done — will skip them."
                )
        except Exception as _re:
            exp_log.warning(f"Could not load resume state: {_re}")

    # ── Load or train BPE vocabulary ──────────────────────────────────────────
    # NOTE — Vocabulary and transductive preprocessing:
    #   The vocabulary is trained ONCE on all subjects (including those that
    #   will later serve as test subjects in LOSO-CV).  BPE training is purely
    #   unsupervised (co-occurrence frequency counting; no class labels are
    #   used), so this constitutes *transductive preprocessing* rather than
    #   label leakage.  A fully strict pipeline would retrain the vocabulary
    #   inside each fold, at significant computational cost and with a smaller
    #   corpus per fold.  We adopt the shared-vocabulary approach and document
    #   it here for transparency.
    # Cache key includes n_bins so different quantisation settings don't collide.
    vocab_path = MODELS_DIR / f"bpe_vocab_V{vocab_size}_B{n_bins}_{method}.json"
    if vocab_path.exists():
        vocab = BPEVocab.load(str(vocab_path))
        exp_log.info(f"Loaded cached BPE vocab: V={vocab.vocab_size}")
    else:
        exp_log.info("No cached vocab found; train with exp1 first or inline here")
        # Quick inline training
        from .exp1_vocab_analysis import dataset_to_sequences
        all_seqs = []
        for ds_name in datasets:
            try:
                seqs = dataset_to_sequences(ds_name, max_subjects=5,
                                             max_hours=2, n_bins=n_bins,
                                             method=method)
                all_seqs.extend(seqs)
            except Exception:
                pass
        vocab = train_bpe(all_seqs, vocab_size=vocab_size,
                          base_vocab_size=n_bins)
        vocab.save(str(vocab_path))

    # Histogram-based classifiers
    hist_classifiers = {
        "BPE_Hist_LogReg": classify_histogram_logreg,
        "BPE_Hist_RF": classify_histogram_rf,
    }

    # Temporal-windowed histogram classifiers (H9 — preserves temporal dynamics)
    windowed_classifiers = {
        "BPE_Windowed_LogReg": classify_histogram_logreg,
    }

    # Bigram histogram classifiers (H10 — preserves local token order)
    bigram_classifiers = {
        "BPE_Bigram_LogReg": classify_histogram_logreg,
    }

    # Sequence-based classifiers
    seq_classifiers_flat = {}       # operate on (n_trials, max_len)
    seq_classifiers_cw   = {}       # operate on (n_trials, n_ch, max_len_per_ch)
    _windowed_seq_cnn_fn = None     # set below when classifiers available
    try:
        from .classifiers import (classify_seq_cnn, classify_seq_transformer,
                                   classify_seq_transformer_channelwise,
                                   classify_windowed_hist_cnn)
        seq_classifiers_flat = {
            "BPE_Seq_CNN":         classify_seq_cnn,
            "BPE_Seq_Transformer": classify_seq_transformer,
        }
        seq_classifiers_cw = {
            "BPE_Seq_CW_Transformer": classify_seq_transformer_channelwise,
        }
        _windowed_seq_cnn_fn = classify_windowed_hist_cnn
    except Exception as e:
        logger.warning(f"Sequence classifiers unavailable: {e}")

    _per_fold_accs: dict[tuple, np.ndarray] = {}
    _MAX_HIST_FEATURES = 2048   # TruncatedSVD target dim for oversized histograms

    # Build the complete set of BPE classifier names used in this run — used for
    # dataset-level early-exit when all seeds are already in _done_keys.
    _all_bpe_clfs_set = (
        set(hist_classifiers) | set(windowed_classifiers) | set(bigram_classifiers) |
        set(seq_classifiers_flat) | set(seq_classifiers_cw) |
        ({"BPE_WindowedSeq_CNN"} if _windowed_seq_cnn_fn is not None else set())
    )
    try:
        from .baselines import BASELINE_CLASSIFIERS as _BLC_main
    except ImportError:
        _BLC_main = {}

    _ds_bar = (
        _tqdm(datasets, desc="Exp2 datasets", unit="ds",
              leave=True, dynamic_ncols=True)
        if _HAS_TQDM else datasets
    )
    for ds_name in _ds_bar:
        if _HAS_TQDM and hasattr(_ds_bar, "set_description"):
            _ds_bar.set_description(f"Exp2 [{ds_name}]")
        info = DATASET_INFO[ds_name]
        exp_log.info(f"=== Dataset: {ds_name} ===")

        # SSVEP is frequency-coded — bag-of-tokens cannot capture frequency structure.
        # Results will be at/below chance; included for completeness only.
        if info["paradigm"] == "SSVEP":
            exp_log.warning(
                f"{ds_name}: SSVEP paradigm is fundamentally incompatible with "
                "amplitude-based bag-of-tokens. Expect chance-level results."
            )

        # ── Dataset-level early-exit: skip ALL feature computation (BPE seqs,
        # histograms, PCA, windowed, bigrams) when every expected classifier has
        # already written results for every seed.  Saves ~50s per dataset on
        # warm-cache runs where all work was done in a previous iteration.
        _bl_clfs_ds = {
            bl for bl, bl_i in _BLC_main.items()
            if not (bl_i.get("mi_only") and info["paradigm"] != "Motor Imagery")
            and not (bl_i.get("ssvep_only") and info["paradigm"] != "SSVEP")
        }
        _all_clfs_ds = _all_bpe_clfs_set | _bl_clfs_ds
        if _all_clfs_ds and all(
            (ds_name, clf, str(vocab_size), str(seed)) in _done_keys
            for clf in _all_clfs_ds for seed in RANDOM_SEEDS
        ):
            exp_log.info(
                f"  {ds_name}: all {len(_all_clfs_ds)} classifiers × "
                f"{len(RANDOM_SEEDS)} seeds done — skipping"
            )
            continue

        try:
            data = load_dataset(ds_name, max_subjects=max_subjects)
        except Exception as e:
            exp_log.error(f"Failed to load {ds_name}: {e}")
            continue

        if not data:
            continue

        # Assemble all epochs + labels + groups (subject IDs)
        all_epochs = []
        all_labels = []
        all_groups = []
        for subj_id, subj_data in sorted(data.items()):
            epochs = subj_data["epochs"]
            labels = subj_data["labels"]
            all_epochs.append(epochs)
            all_labels.extend(labels)
            all_groups.extend([subj_id] * len(labels))

        if not all_epochs:
            continue

        # Pad/truncate channels to match across subjects
        n_ch_max = max(e.shape[1] for e in all_epochs)
        n_time_max = max(e.shape[2] for e in all_epochs)
        exp_log.info(f"  Padding: n_ch_max={n_ch_max}, n_time_max={n_time_max}")

        padded = []
        for e in all_epochs:
            p = np.zeros((e.shape[0], n_ch_max, n_time_max))
            p[:, :e.shape[1], :e.shape[2]] = e
            padded.append(p)

        X_epochs = np.concatenate(padded, axis=0)
        y = np.array(all_labels)
        groups = np.array(all_groups)

        # Encode labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # ── Shared BPE token sequences (quantize + apply_bpe ONCE per dataset) ─
        # All histogram / sequence feature types below reuse the same cached seqs,
        # avoiding repeated quantization and BPE application (which would otherwise
        # run 9× per dataset: 1 hist + 5 windows + 1 bigram + 1 flat + 1 cw).
        n_trials_all, n_ch_all = X_epochs.shape[:2]
        V = vocab.vocab_size
        exp_log.info(f"  Computing BPE token sequences (V={vocab_size}, cached)…")
        bpe_seqs = cached_bpe_token_sequences(X_epochs, vocab, method, n_bins)

        # BPE histogram features — pass pre_computed_seqs to avoid double BPE pass
        exp_log.info(f"  Building BPE histograms (V={vocab_size})...")
        X_hist = cached_epochs_to_bpe_histograms(
            X_epochs, vocab, method, n_bins, pre_computed_seqs=bpe_seqs)

        # ── Dimensionality reduction for large histogram feature spaces ───────
        # BPE histograms = n_ch × vocab_size features.  For high-channel datasets
        # (physionet_mi: 64ch × 4096V = 262 144) this is both huge RAM and slow
        # for SAGA-LogReg (O(n_features) per gradient step).
        # Fix: TruncatedSVD on the sparse histogram matrix (BPE histos are
        # ~2–5% dense) — unsupervised so negligible data-leakage concern.
        if X_hist.shape[1] > _MAX_HIST_FEATURES:
            n_comp = min(_MAX_HIST_FEATURES, X_hist.shape[0] - 1)
            exp_log.info(
                f"  High-dim histograms ({X_hist.shape[1]} features): "
                f"pca_reduce({n_comp}) …"
            )
            X_hist = pca_reduce(X_hist, n_comp, device=DEVICE)
            exp_log.info(f"  → X_hist after SVD: {X_hist.shape}")

        # ── Temporal-windowed BPE histograms (H9, disk-cached) ────────────
        _N_WINDOWS = 5
        exp_log.info(f"  Computing windowed BPE histograms (n_windows={_N_WINDOWS})...")
        X_windowed = cached_epochs_to_windowed_bpe_histograms(
            X_epochs, vocab, method, n_bins, n_windows=_N_WINDOWS)
        # Keep unreduced version for WindSeqCNN (needs window structure intact)
        X_windowed_raw = X_windowed
        if X_windowed.shape[1] > _MAX_HIST_FEATURES:
            n_comp_w = min(_MAX_HIST_FEATURES, X_windowed.shape[0] - 1)
            exp_log.info(
                f"  Windowed histograms ({X_windowed.shape[1]} features): "
                f"pca_reduce({n_comp_w}) …"
            )
            X_windowed = pca_reduce(X_windowed, n_comp_w, device=DEVICE)

        # ── Bigram BPE histograms (H10, from shared seqs) ─────────────────
        exp_log.info(f"  Computing BPE bigram histograms...")
        t0_bigram = time.perf_counter()
        X_bigram = _bigram_from_bpe_seqs(bpe_seqs, n_trials_all, n_ch_all, V)
        dt_bigram = time.perf_counter() - t0_bigram
        exp_log.info(f"  Bigram histograms computed in {dt_bigram:.1f}s, shape={X_bigram.shape}")
        if X_bigram.shape[1] > _MAX_HIST_FEATURES:
            n_comp_b = min(_MAX_HIST_FEATURES, X_bigram.shape[0] - 1)
            exp_log.info(
                f"  Bigram histograms ({X_bigram.shape[1]} features): "
                f"pca_reduce({n_comp_b}) …"
            )
            X_bigram = pca_reduce(X_bigram, n_comp_b, device=DEVICE)

        # BPE flat sequences (for CNN + flat Transformer, from shared seqs)
        X_seq = None
        if seq_classifiers_flat:
            exp_log.info(f"  Computing BPE flat sequences (V={vocab_size})...")
            X_seq = _flat_seqs_from_bpe_seqs(bpe_seqs, n_trials_all, n_ch_all, V)

        # BPE channelwise sequences (for CW Transformer, from shared seqs)
        n_ch_dataset = X_epochs.shape[1]
        # Allocate 128 tokens per channel; reduce for datasets with many channels
        cw_max_per_ch = max(32, min(128, 2048 // max(n_ch_dataset, 1)))
        X_seq_cw = None
        if seq_classifiers_cw:
            exp_log.info(
                f"  Computing BPE channelwise sequences "
                f"(V={vocab_size}, {n_ch_dataset}ch × {cw_max_per_ch}tok)..."
            )
            PAD_ID = V + 1
            X_seq_cw = _cw_seqs_from_bpe_seqs(
                bpe_seqs, n_trials_all, n_ch_all, cw_max_per_ch,
                pad_id=PAD_ID)

        # ─── Run histogram-based classifiers ──────────────────────────────
        # Seeds are fully independent → parallelise with threads.
        # (SAGA / RF both release the GIL in their C extensions.)
        for clf_name, clf_fn in hist_classifiers.items():
            _cv_str = info["cv_strategy"]
            _pending_seeds = [s for s in RANDOM_SEEDS
                              if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
            if not _pending_seeds:
                exp_log.info(f"  {clf_name}: all seeds already done — skipping")
                continue

            def _run_one_hist_seed(seed, _fn=clf_fn, _cv=_cv_str):
                if _cv == "LOSO":
                    return run_loso_cv(X_hist, y_enc, groups, _fn, seed=seed)
                return run_kfold_cv(X_hist, y_enc, groups, _fn,
                                    n_splits=5, seed=seed)

            _n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
            exp_log.info(
                f"  {clf_name}: running {len(_pending_seeds)} seeds "
                f"(n_jobs={_n_sjobs}) …"
            )
            seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                delayed(_run_one_hist_seed)(seed) for seed in _pending_seeds
            )

            for seed, fold_results in zip(_pending_seeds, seed_fold_results):
                _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                 vocab_size, seed, y, groups,
                                 all_results, _per_fold_accs)

        # ─── Run temporal-windowed histogram classifiers (H9) ─────────────
        for clf_name, clf_fn in windowed_classifiers.items():
            _cv_str = info["cv_strategy"]
            _pending_seeds = [s for s in RANDOM_SEEDS
                              if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
            if not _pending_seeds:
                exp_log.info(f"  {clf_name}: all seeds already done — skipping")
                continue

            def _run_one_windowed_seed(seed, _fn=clf_fn, _cv=_cv_str):
                if _cv == "LOSO":
                    return run_loso_cv(X_windowed, y_enc, groups, _fn, seed=seed)
                return run_kfold_cv(X_windowed, y_enc, groups, _fn,
                                    n_splits=5, seed=seed)

            _n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
            exp_log.info(
                f"  {clf_name}: running {len(_pending_seeds)} seeds "
                f"(n_jobs={_n_sjobs}) …"
            )
            seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                delayed(_run_one_windowed_seed)(seed) for seed in _pending_seeds
            )
            for seed, fold_results in zip(_pending_seeds, seed_fold_results):
                _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                 vocab_size, seed, y, groups,
                                 all_results, _per_fold_accs)

        # ─── Run BPE_WindowedSeq_CNN (Conv1D over window sequence) ────────
        if _windowed_seq_cnn_fn is not None:
            clf_name = "BPE_WindowedSeq_CNN"
            _cv_str = info["cv_strategy"]
            _pending_seeds = [s for s in RANDOM_SEEDS
                              if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
            if not _pending_seeds:
                exp_log.info(f"  {clf_name}: all seeds already done — skipping")
            else:
                _nw = _N_WINDOWS

                def _run_one_wscnn_seed(seed, _fn=_windowed_seq_cnn_fn,
                                        _cv=_cv_str, _nw=_nw):
                    def _wscnn_clf(X_tr, y_tr, X_te, y_te):
                        return _fn(X_tr, y_tr, X_te, y_te, n_windows=_nw)
                    if _cv == "LOSO":
                        return run_loso_cv(X_windowed_raw, y_enc, groups,
                                           _wscnn_clf, seed=seed)
                    return run_kfold_cv(X_windowed_raw, y_enc, groups,
                                        _wscnn_clf, n_splits=5, seed=seed)

                exp_log.info(
                    f"  {clf_name}: running {len(_pending_seeds)} seeds …"
                )
                for seed in _pending_seeds:
                    fold_results = _run_one_wscnn_seed(seed)
                    _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                     vocab_size, seed, y, groups,
                                     all_results, _per_fold_accs)

        # ─── Run bigram histogram classifiers (H10) ───────────────────────
        for clf_name, clf_fn in bigram_classifiers.items():
            _cv_str = info["cv_strategy"]
            _pending_seeds = [s for s in RANDOM_SEEDS
                              if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
            if not _pending_seeds:
                exp_log.info(f"  {clf_name}: all seeds already done — skipping")
                continue

            def _run_one_bigram_seed(seed, _fn=clf_fn, _cv=_cv_str):
                if _cv == "LOSO":
                    return run_loso_cv(X_bigram, y_enc, groups, _fn, seed=seed)
                return run_kfold_cv(X_bigram, y_enc, groups, _fn,
                                    n_splits=5, seed=seed)

            _n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
            exp_log.info(
                f"  {clf_name}: running {len(_pending_seeds)} seeds "
                f"(n_jobs={_n_sjobs}) …"
            )
            seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                delayed(_run_one_bigram_seed)(seed) for seed in _pending_seeds
            )
            for seed, fold_results in zip(_pending_seeds, seed_fold_results):
                _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                 vocab_size, seed, y, groups,
                                 all_results, _per_fold_accs)

        # ─── Run flat-sequence classifiers (CNN, flat Transformer) ────────
        if X_seq is not None:
            for clf_name, clf_fn in seq_classifiers_flat.items():
                _pending_seeds = [s for s in RANDOM_SEEDS
                                  if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
                if not _pending_seeds:
                    exp_log.info(f"  {clf_name}: all seeds already done — skipping")
                    continue

                def _run_one_seq_flat_seed(seed, _fn=clf_fn, _cv=info["cv_strategy"],
                                           _vs=vocab_size):
                    def _seq_clf_fn(X_tr, y_tr, X_te, y_te):
                        return _fn(X_tr, y_tr, X_te, y_te,
                                   vocab_size=_vs, device=DEVICE)
                    if _cv == "LOSO":
                        return run_loso_cv(X_seq, y_enc, groups, _seq_clf_fn,
                                           seed=seed)
                    return run_kfold_cv(X_seq, y_enc, groups, _seq_clf_fn,
                                        n_splits=5, seed=seed)

                _n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
                exp_log.info(
                    f"  {clf_name}: running {len(_pending_seeds)} seeds "
                    f"(n_jobs={_n_sjobs}) …"
                )
                seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                    delayed(_run_one_seq_flat_seed)(seed) for seed in _pending_seeds
                )
                for seed, fold_results in zip(_pending_seeds, seed_fold_results):
                    _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                     vocab_size, seed, y, groups,
                                     all_results, _per_fold_accs)

        # ─── Run channelwise-sequence Transformer ─────────────────────────
        if X_seq_cw is not None:
            for clf_name, clf_fn in seq_classifiers_cw.items():
                _pending_seeds = [s for s in RANDOM_SEEDS
                                  if (ds_name, clf_name, str(vocab_size), str(s)) not in _done_keys]
                if not _pending_seeds:
                    exp_log.info(f"  {clf_name}: all seeds already done — skipping")
                    continue

                def _run_one_cw_seed(seed, _fn=clf_fn, _cv=info["cv_strategy"],
                                     _vs=vocab_size):
                    def _cw_clf_fn(X_tr, y_tr, X_te, y_te):
                        return _fn(X_tr, y_tr, X_te, y_te,
                                   vocab_size=_vs, device=DEVICE)
                    if _cv == "LOSO":
                        return run_loso_cv(X_seq_cw, y_enc, groups, _cw_clf_fn,
                                           seed=seed)
                    return run_kfold_cv(X_seq_cw, y_enc, groups, _cw_clf_fn,
                                        n_splits=5, seed=seed)

                _n_sjobs = min(len(_pending_seeds), max(1, N_JOBS))
                exp_log.info(
                    f"  {clf_name}: running {len(_pending_seeds)} seeds "
                    f"(n_jobs={_n_sjobs}) …"
                )
                try:
                    seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                        delayed(_run_one_cw_seed)(seed) for seed in _pending_seeds
                    )
                    for seed, fold_results in zip(_pending_seeds, seed_fold_results):
                        _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                                         vocab_size, seed, y, groups,
                                         all_results, _per_fold_accs)
                except Exception as _cw_err:
                    exp_log.error(
                        f"  {clf_name} FAILED: {_cw_err}",
                        exc_info=True,
                    )
                    # Free VRAM and continue with remaining seeds/datasets
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass

        # ─── Run baseline classifiers (D2: CSP+LDA, EEGNet, etc.) ────────
        try:
            from .baselines import BASELINE_CLASSIFIERS
        except ImportError:
            BASELINE_CLASSIFIERS = {}

        for bl_name, bl_info in BASELINE_CLASSIFIERS.items():
            # Skip MI-only baselines on non-MI datasets
            if bl_info.get("mi_only") and info["paradigm"] != "Motor Imagery":
                continue
            # Skip SSVEP-only baselines (e.g. FFT) on non-SSVEP datasets
            if bl_info.get("ssvep_only") and info["paradigm"] != "SSVEP":
                continue

            bl_fn   = bl_info["fn"]
            _cv_str = info["cv_strategy"]
            _sfreq  = float(info["sfreq"])   # captured in closure for PSD/FFT baselines
            _pending_bl_seeds = [s for s in RANDOM_SEEDS
                                  if (ds_name, bl_name, str(vocab_size), str(s)) not in _done_keys]
            if not _pending_bl_seeds:
                exp_log.info(f"  Baseline {bl_name}: all seeds already done — skipping")
                continue

            def _run_one_bl_seed(seed, _fn=bl_fn, _cv=_cv_str, _sf=_sfreq):
                def _clf(X_tr, y_tr, X_te, y_te):
                    return _fn(X_tr, y_tr, X_te, y_te, sfreq=_sf)
                if _cv == "LOSO":
                    return run_loso_cv(X_epochs, y_enc, groups, _clf, seed=seed)
                return run_kfold_cv(X_epochs, y_enc, groups, _clf,
                                    n_splits=5, seed=seed)

            # EEGNet: single GPU context → serial; Chronos: capped at 3 for RAM
            if bl_info.get("no_parallel_seeds"):
                _n_sjobs = 1
            elif "max_parallel_seeds" in bl_info:
                _n_sjobs = min(len(_pending_bl_seeds),
                               bl_info["max_parallel_seeds"])
            else:
                _n_sjobs = min(len(_pending_bl_seeds), max(1, N_JOBS))
            exp_log.info(
                f"  Baseline {bl_name}: running {len(_pending_bl_seeds)} seeds "
                f"(n_jobs={_n_sjobs}) …"
            )
            seed_fold_results = Parallel(n_jobs=_n_sjobs, prefer="threads")(
                delayed(_run_one_bl_seed)(seed) for seed in _pending_bl_seeds
            )

            for seed, fold_results in zip(_pending_bl_seeds, seed_fold_results):
                _log_clf_results(exp_log, fold_results, ds_name, bl_name,
                                 vocab_size, seed, y, groups,
                                 all_results, _per_fold_accs)

    # ─── Statistical comparisons (D7/D8): Wilcoxon + Holm-Bonferroni ─────
    _wilcoxon_csv = LOGS_DIR / "exp2_wilcoxon_comparisons.csv"
    if _wilcoxon_csv.exists():
        exp_log.info("Wilcoxon comparisons CSV already exists — skipping recomputation")
        comparisons = []
    else:
        comparisons = _compute_pairwise_comparisons(all_results, _per_fold_accs)
    if comparisons:
        save_csv(comparisons, LOGS_DIR / "exp2_wilcoxon_comparisons.csv")
        save_json(comparisons, LOGS_DIR / "exp2_wilcoxon_comparisons.json")
        for c in comparisons:
            sig_str = "***" if c.get("significant") else "n.s."
            exp_log.info(
                f"  {c['method_a']} vs {c['method_b']} on {c.get('dataset','')}: "
                f"Δ={c.get('mean_diff',float('nan')):.4f}, d_z={c.get('cohens_dz',float('nan')):.2f} {sig_str}"
            )

    # Plots — use all_results if available, otherwise load from CSV
    _plot_results = all_results
    if not _plot_results and exp_log.csv_path.exists():
        try:
            import pandas as pd
            _df_plot = pd.read_csv(exp_log.csv_path)
            _plot_results = _df_plot.to_dict("records")
            exp_log.info(f"Loaded {len(_plot_results)} results from CSV for plotting")
        except Exception:
            pass
    plot_exp2_results(_plot_results, PLOTS_DIR / "exp2")
    _plot_confusion_matrices(_plot_results, _per_fold_accs, PLOTS_DIR / "exp2")
    _plot_bpe_vs_baselines(_plot_results, PLOTS_DIR / "exp2")

    # ─── Vocab-size sensitivity sweep (BPE_Hist_LogReg only) ─────────────────
    # Tests how downstream accuracy changes across vocab sizes [1024, 2048, 4096].
    # Separate CSV so the main results file stays clean.
    _sweep_csv = LOGS_DIR / "exp2_vocab_sweep.csv"
    _sweep_done_set: set[tuple] = set()
    if _sweep_csv.exists():
        try:
            import csv as _csv_s
            with open(_sweep_csv, newline="") as _f_s:
                for _r_s in _csv_s.DictReader(_f_s):
                    _sweep_done_set.add(
                        (_r_s["dataset"], str(_r_s["vocab_size"]), str(_r_s["seed"]))
                    )
        except Exception:
            pass

    exp_log.info("=== Vocab-size sensitivity sweep (BPE_Hist_LogReg) ===")
    sweep_all_results: list[dict] = []
    for _sv in VOCAB_SWEEP_SIZES:
        _sv_path = MODELS_DIR / f"bpe_vocab_V{_sv}_B{n_bins}_{method}.json"
        if not _sv_path.exists():
            exp_log.info(f"  Sweep V={_sv}: no cached vocab — skip (run exp1 first)")
            continue
        _sv_vocab = BPEVocab.load(str(_sv_path))
        exp_log.info(f"  Sweep V={_sv}: vocab loaded")
        for _ds_sw in datasets:
            # Skip entire dataset if all seeds already done for this (ds, V).
            if all((_ds_sw, str(_sv), str(_s)) in _sweep_done_set
                   for _s in RANDOM_SEEDS):
                exp_log.info(f"  Sweep V={_sv}, {_ds_sw}: all seeds done — skipping")
                continue
            _info_sw = DATASET_INFO[_ds_sw]
            try:
                _data_sw = load_dataset(_ds_sw, max_subjects=max_subjects)
            except Exception as _e_sw:
                exp_log.warning(f"  Sweep V={_sv}, {_ds_sw}: load failed: {_e_sw}")
                continue
            if not _data_sw:
                continue
            # Assemble + pad epochs
            _ep_list, _lb_list, _gr_list = [], [], []
            for _sid_s, _sdat_s in sorted(_data_sw.items()):
                _ep_list.append(_sdat_s["epochs"])
                _lb_list.extend(_sdat_s["labels"])
                _gr_list.extend([_sid_s] * len(_sdat_s["labels"]))
            _n_ch_sw = max(e.shape[1] for e in _ep_list)
            _n_t_sw  = max(e.shape[2] for e in _ep_list)
            _padded_sw = []
            for _e_s in _ep_list:
                _pp = np.zeros((_e_s.shape[0], _n_ch_sw, _n_t_sw), dtype=np.float32)
                _pp[:, :_e_s.shape[1], :_e_s.shape[2]] = _e_s
                _padded_sw.append(_pp)
            _X_sw  = np.concatenate(_padded_sw, axis=0)
            _le_sw = LabelEncoder()
            _y_sw  = _le_sw.fit_transform(np.array(_lb_list))
            _grp_sw = np.array(_gr_list)
            # BPE histograms + optional PCA reduce (cached to avoid recompute)
            exp_log.info(f"    {_ds_sw}: histograms V={_sv}…")
            _Xh_sw = cached_epochs_to_bpe_histograms(_X_sw, _sv_vocab, method, n_bins)
            if _Xh_sw.shape[1] > _MAX_HIST_FEATURES:
                _Xh_sw = pca_reduce(
                    _Xh_sw, min(_MAX_HIST_FEATURES, _Xh_sw.shape[0] - 1),
                    device=DEVICE,
                )
            for _seed_sw in RANDOM_SEEDS:
                _sk_sw = (_ds_sw, str(_sv), str(_seed_sw))
                if _sk_sw in _sweep_done_set:
                    continue

                def _clf_sw(X_tr, y_tr, X_te, y_te):
                    return classify_histogram_logreg(X_tr, y_tr, X_te, y_te)

                if _info_sw["cv_strategy"] == "LOSO":
                    _folds_sw = run_loso_cv(_Xh_sw, _y_sw, _grp_sw,
                                            _clf_sw, seed=_seed_sw)
                else:
                    _folds_sw = run_kfold_cv(_Xh_sw, _y_sw, _grp_sw,
                                             _clf_sw, seed=_seed_sw)

                _accs_sw  = [f["accuracy"]                             for f in _folds_sw]
                _bal_sw   = [f.get("balanced_accuracy", f["accuracy"]) for f in _folds_sw]
                _kap_sw   = [f["kappa"]                                for f in _folds_sw]
                for _f_sw in _folds_sw:
                    append_csv({
                        "dataset":           _ds_sw,
                        "vocab_size":        _sv,
                        "seed":              _seed_sw,
                        "fold":              _f_sw["fold"],
                        "accuracy":          _f_sw["accuracy"],
                        "balanced_accuracy": _f_sw.get("balanced_accuracy", ""),
                        "kappa":             _f_sw["kappa"],
                    }, _sweep_csv)
                sweep_all_results.append({
                    "dataset":                _ds_sw,
                    "vocab_size":             _sv,
                    "seed":                   _seed_sw,
                    "accuracy_mean":          float(np.mean(_accs_sw)),
                    "balanced_accuracy_mean": float(np.mean(_bal_sw)),
                    "kappa_mean":             float(np.mean(_kap_sw)),
                })
                exp_log.info(
                    f"    {_ds_sw} V={_sv}: bal_acc={np.mean(_bal_sw):.3f}"
                )

    if sweep_all_results:
        save_json(sweep_all_results, LOGS_DIR / "exp2_vocab_sweep.json")
        _plot_vocab_sweep(sweep_all_results, PLOTS_DIR / "exp2")

    exp_log.finalize()
    logger.info(f"Experiment 2 complete: {len(all_results)} configurations")
    return all_results


def _plot_vocab_sweep(sweep_results: list[dict], save_dir: Path) -> None:
    """
    Line chart: balanced_accuracy vs vocab_size, one line per dataset.

    Saved to *save_dir*/exp2_vocab_sweep_balanced_acc.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    datasets_sw   = sorted(set(r["dataset"]    for r in sweep_results))
    vocab_sizes_sw = sorted(set(r["vocab_size"] for r in sweep_results))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for di, ds in enumerate(datasets_sw):
        means, stds = [], []
        for v in vocab_sizes_sw:
            subset = [r for r in sweep_results
                      if r["dataset"] == ds and r["vocab_size"] == v]
            if subset:
                vals = [r["balanced_accuracy_mean"] for r in subset]
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
            else:
                means.append(None)
                stds.append(0.0)
        valid = [(v, m, s) for v, m, s in zip(vocab_sizes_sw, means, stds)
                 if m is not None]
        if valid:
            vs, ms, ss = zip(*valid)
            ax.plot(vs, ms, marker="o", label=ds, color=colors[di % len(colors)])
            ax.fill_between(vs,
                            np.array(ms) - np.array(ss),
                            np.array(ms) + np.array(ss),
                            alpha=0.12, color=colors[di % len(colors)])

    ax.set_xscale("log", base=2)
    ax.set_xticks(vocab_sizes_sw)
    ax.set_xticklabels([str(v) for v in vocab_sizes_sw])
    ax.set_xlabel("Vocabulary size")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Vocab-size sensitivity: BPE_Hist_LogReg (shaded = ±std across seeds)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "exp2_vocab_sweep_balanced_acc.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrices(all_results: list[dict], per_fold_accs: dict,
                             output_dir: Path) -> None:
    """
    Plot confusion matrix heatmaps for the top classifiers per dataset.

    For each dataset, picks the top 4 classifiers by mean accuracy and
    renders their aggregated confusion matrices as subplot heatmaps.
    Saves to ``output_dir / "exp2_confusion_matrices.png"``.

    Parameters
    ----------
    all_results : list of dict
        Full result dicts (must contain 'confusion_matrix' key).
    per_fold_accs : dict
        Per-fold accuracy dict (unused, kept for interface consistency).
    output_dir : Path
        Directory where the plot is saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))

    # Aggregate: average confusion matrix across seeds for each (ds, clf)
    from collections import defaultdict
    cm_agg: dict[tuple[str, str], list] = defaultdict(list)
    acc_agg: dict[tuple[str, str], list] = defaultdict(list)
    for r in all_results:
        key = (r["dataset"], r["classifier"])
        acc_agg[key].append(r["accuracy_mean"])
        cm = r.get("confusion_matrix")
        if cm is not None:
            cm_agg[key].append(np.array(cm))

    if not cm_agg:
        # No confusion matrices stored — nothing to plot.
        # Fold-level y_true/y_pred are aggregated after all folds finish; per-fold
        # confusion matrices are not stored to keep the log size bounded.
        # when confusion_matrix is not aggregated in _log_clf_results.
        return

    n_top = 4
    n_ds = len(datasets)
    if n_ds == 0:
        return

    fig, axes = plt.subplots(n_ds, n_top, figsize=(4 * n_top, 3.5 * n_ds),
                             squeeze=False)

    for di, ds in enumerate(datasets):
        # Rank classifiers by mean accuracy for this dataset
        ds_clfs = [(clf, float(np.mean(accs)))
                   for (d, clf), accs in acc_agg.items() if d == ds]
        ds_clfs.sort(key=lambda x: x[1], reverse=True)
        top_clfs = [c for c, _ in ds_clfs[:n_top]]

        for ci in range(n_top):
            ax = axes[di][ci]
            if ci < len(top_clfs):
                clf = top_clfs[ci]
                key = (ds, clf)
                if key in cm_agg and cm_agg[key]:
                    # Sum confusion matrices across seeds
                    cm_sum = sum(cm_agg[key])
                    # Normalise rows to proportions
                    row_sums = cm_sum.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    cm_norm = cm_sum / row_sums

                    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1,
                                   aspect="auto")
                    ax.set_title(f"{clf}\n({ds})", fontsize=7)
                    ax.set_xlabel("Predicted", fontsize=7)
                    ax.set_ylabel("True", fontsize=7)
                    ax.tick_params(labelsize=6)
                else:
                    ax.set_visible(False)
            else:
                ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "exp2_confusion_matrices.png", dpi=150)
    plt.close(fig)


def _plot_bpe_vs_baselines(all_results: list[dict], output_dir: Path) -> None:
    """
    Grouped bar chart comparing BPE classifiers vs baselines per dataset.

    BPE classifiers are those whose name starts with ``"BPE_"``; everything
    else is treated as a baseline.  Bars are coloured by group.
    Saves to ``output_dir / "exp2_bpe_vs_baselines.png"``.

    Parameters
    ----------
    all_results : list of dict
        Full result dicts.
    output_dir : Path
        Directory where the plot is saved.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))
    classifiers = sorted(set(r["classifier"] for r in all_results))
    if not datasets or not classifiers:
        return

    # Group classifiers
    bpe_clfs = [c for c in classifiers if c.startswith("BPE_")]
    bl_clfs = [c for c in classifiers if not c.startswith("BPE_")]

    # Compute mean accuracy per (dataset, classifier) across seeds
    acc_map: dict[tuple[str, str], float] = defaultdict(float)
    cnt_map: dict[tuple[str, str], int] = defaultdict(int)
    for r in all_results:
        key = (r["dataset"], r["classifier"])
        acc_map[key] += r["accuracy_mean"]
        cnt_map[key] += 1
    for key in acc_map:
        if cnt_map[key] > 0:
            acc_map[key] /= cnt_map[key]

    # Interleave: BPE first, then baselines
    ordered_clfs = bpe_clfs + bl_clfs
    n_clfs = len(ordered_clfs)
    n_bpe = len(bpe_clfs)

    fig, ax = plt.subplots(figsize=(max(14, 2 * len(datasets)), 6))
    x = np.arange(len(datasets))
    width = 0.8 / max(n_clfs, 1)

    for i, clf in enumerate(ordered_clfs):
        accs = [acc_map.get((ds, clf), 0.0) for ds in datasets]
        colour = "#1f77b4" if i < n_bpe else "#ff7f0e"
        label = clf
        ax.bar(x + i * width, accs, width, color=colour, alpha=0.8,
               label=label)

    ax.set_xticks(x + width * n_clfs / 2)
    ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("BPE vs Baselines: Downstream Classification")

    # Create legend with group labels only (avoid duplicate entries)
    from matplotlib.patches import Patch
    legend_elements = []
    if bpe_clfs:
        legend_elements.append(Patch(facecolor="#1f77b4", alpha=0.8, label="BPE"))
    if bl_clfs:
        legend_elements.append(Patch(facecolor="#ff7f0e", alpha=0.8, label="Baseline"))
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "exp2_bpe_vs_baselines.png", dpi=150)
    plt.close(fig)


def _log_clf_results(exp_log, fold_results, ds_name, clf_name,
                     vocab_size, seed, y, groups,
                     all_results, per_fold_accs):
    """
    Extract metrics from fold results, log them, save per-fold CSV, and
    accumulate into all_results / per_fold_accs.

    Parameters
    ----------
    exp_log : ExperimentLogger
        Logger instance.
    fold_results : list of dict
        Per-fold metric dicts from run_loso_cv / run_kfold_cv.
    ds_name : str
        Dataset name.
    clf_name : str
        Classifier name.
    vocab_size : int
        BPE vocabulary size.
    seed : int
        Random seed.
    y : np.ndarray
        True labels.
    groups : np.ndarray
        Subject group ids.
    all_results : list
        Accumulator for result dicts.
    per_fold_accs : dict
        Accumulator for per-fold accuracy arrays (keyed by (ds, clf, seed)).
    """
    accs      = [f["accuracy"]                             for f in fold_results]
    bal_accs  = [f.get("balanced_accuracy", f["accuracy"]) for f in fold_results]
    f1s       = [f["macro_f1"]                             for f in fold_results]
    kappas    = [f["kappa"]                                for f in fold_results]
    auc_rocs  = [f.get("auc_roc") for f in fold_results]
    auc_rocs  = [a for a in auc_rocs if a is not None]

    cv_strategy = DATASET_INFO.get(ds_name, {}).get("cv_strategy", "unknown")

    # Per-fold CSV logging
    for f in fold_results:
        append_csv({
            "dataset":           ds_name,
            "classifier":        clf_name,
            "vocab_size":        vocab_size,
            "seed":              seed,
            "cv_strategy":       cv_strategy,
            "fold":              f["fold"],
            "accuracy":          f["accuracy"],
            "balanced_accuracy": f.get("balanced_accuracy", ""),
            "macro_f1":          f["macro_f1"],
            "kappa":             f["kappa"],
            "auc_roc":           f.get("auc_roc", ""),
        }, LOGS_DIR / "exp2_per_fold.csv")

    # Bootstrap CIs on accuracy
    _mean, ci_low, ci_high = bootstrap_ci(np.array(accs), n_boot=N_BOOTSTRAP)

    # I1: Aggregate confusion matrices across folds
    cm_total = None
    for f in fold_results:
        cm = f.get("confusion_matrix")
        if cm is not None:
            cm_arr = np.array(cm)
            cm_total = cm_arr if cm_total is None else cm_total + cm_arr

    result = {
        "dataset":                ds_name,
        "classifier":             clf_name,
        "vocab_size":             vocab_size,
        "seed":                   seed,
        "cv_strategy":            cv_strategy,
        "accuracy_mean":          float(np.mean(accs)),
        "accuracy_std":           float(np.std(accs)),
        "accuracy_ci_low":        float(ci_low),
        "accuracy_ci_high":       float(ci_high),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "macro_f1_mean":          float(np.mean(f1s)),
        "kappa_mean":             float(np.mean(kappas)),
        "auc_roc_mean":           float(np.mean(auc_rocs)) if auc_rocs else None,
        "n_folds":                len(fold_results),
        "n_trials":               len(y),
        "n_subjects":             len(np.unique(groups)),
        "confusion_matrix":       cm_total.tolist() if cm_total is not None else None,
    }
    exp_log.log_result(result)
    all_results.append(result)

    # Store per-fold accuracies for Wilcoxon test later
    _per_fold_key = (ds_name, clf_name, seed)
    per_fold_accs[_per_fold_key] = np.array(accs)

    # Log primary metric (dataset-specific)
    primary_metric = PRIMARY_METRIC_PER_DATASET.get(ds_name, PRIMARY_METRIC_DEFAULT)
    primary_val = result["kappa_mean"] if primary_metric == "kappa" else result["balanced_accuracy_mean"]
    exp_log.info(
        f"    → {primary_metric}={primary_val:.3f}  "
        f"acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f} "
        f"[{ci_low:.3f},{ci_high:.3f}]  F1={result['macro_f1_mean']:.3f}"
    )


def _compute_pairwise_comparisons(all_results, per_fold_accs) -> list[dict]:
    """Wilcoxon signed-rank between all classifier pairs per dataset."""
    from itertools import combinations
    comparisons_all = []

    datasets = sorted(set(r["dataset"] for r in all_results))
    classifiers = sorted(set(r["classifier"] for r in all_results))

    for ds in datasets:
        # Average fold-accs across seeds for each classifier
        clf_accs = {}
        for clf in classifiers:
            seed_accs = []
            for seed_key, accs in per_fold_accs.items():
                if seed_key[0] == ds and seed_key[1] == clf:
                    seed_accs.append(accs)
            if seed_accs:
                # Pool fold-level accuracies across all seeds for higher
                # statistical power in Wilcoxon signed-rank test.
                clf_accs[clf] = np.concatenate(seed_accs)

        pairs = []
        for ca, cb in combinations(clf_accs.keys(), 2):
            min_len = min(len(clf_accs[ca]), len(clf_accs[cb]))
            if min_len >= 5:
                pairs.append((
                    clf_accs[ca][:min_len], clf_accs[cb][:min_len], ca, cb
                ))

        if pairs:
            results = wilcoxon_holm(pairs)
            for r in results:
                r["dataset"] = ds
            comparisons_all.extend(results)

    return comparisons_all


if __name__ == "__main__":
    run_experiment_2()
