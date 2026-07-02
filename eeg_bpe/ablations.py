"""
Ablation Studies A1-A15, K7
===========================
A1:  Quantization method (uniform / μ-law / adaptive)
A2:  Number of bins (64 / 128 / 256 / 512)
A3:  Frequency-aware strategy (resample / temporal norm)
A4:  Channel strategy (independent / spatial+VQ)
A5:  Artifact tokens (with / without)
A6:  BPE vocabulary size (512-64K)  — already covered by exp5
A7:  BPE vs no-BPE vs random merges (critical ablation)
A8:  Discriminative BPE (Fisher-ratio merge selection)
A9:  B×V grid search (bins × vocab size)
A10: PhysioNet-MI scale ablation (9 vs 109 subjects)
A11: Histogram feature transform (raw / TF-IDF / log / binary)
A12: BPE training corpus size (500–all sequences)
A13: Window size for windowed histograms (n_windows ∈ {2,5,10,20})
A14: Normalization strategy (global z-score / per-channel / per-trial / robust / none)
A15: Epoch segmentation (sub-epoch lengths)
K7:  BPE+PSD ensemble
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from joblib import Parallel, delayed
from pathlib import Path
import random as _random

from .config import (
    DATASET_INFO, QUANT_METHODS, QUANT_BINS, N_JOBS, RANDOM_SEEDS,
    TARGET_SFREQ, LOGS_DIR, PLOTS_DIR, MODELS_DIR,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
    PRIMARY_METRIC_PER_DATASET, PRIMARY_METRIC_DEFAULT,
    DEFAULT_QUANT_METHOD,
)
from .quantization import quantize
from .bpe_engine import (
    train_bpe, apply_bpe, apply_bpe_batch, BPEVocab,
    apply_merge_to_batch, train_bpe_discriminative,
)
from .data_loading import load_dataset
from .exp2_downstream import (
    epochs_to_bpe_histograms, cached_epochs_to_bpe_histograms,
    classify_histogram_logreg,
    epochs_to_bpe_sequences,
    run_loso_cv, run_kfold_cv,
)
from .utils import (
    ExperimentLogger, save_json, save_csv, timed, get_logger, bootstrap_ci,
    pca_reduce,
)

logger = get_logger("ablations")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _make_ablation_result(base: dict, fold_results: list[dict]) -> dict:
    """
    Build ablation result dict with accuracy, balanced_accuracy, and kappa
    stats (mean, std, 95% bootstrap CI).

    Parameters
    ----------
    base : dict
        Base fields for the result (ablation, dataset, etc.).
    fold_results : list of dict
        Per-fold metric dicts from run_loso_cv / run_kfold_cv.  Each dict
        must contain ``"accuracy"``; ``"balanced_accuracy"`` and ``"kappa"``
        are used when present.

    Returns
    -------
    dict
        Result dict with accuracy_mean/std/ci, balanced_accuracy_mean, kappa_mean.
    """
    accs     = [f["accuracy"]                             for f in fold_results]
    bal_accs = [f.get("balanced_accuracy", f["accuracy"]) for f in fold_results]
    kappas   = [f.get("kappa", 0.0)                       for f in fold_results]
    _mean, ci_low, ci_high = bootstrap_ci(np.array(accs))
    return {
        **base,
        "accuracy_mean":          float(np.mean(accs)),
        "accuracy_std":           float(np.std(accs)),
        "accuracy_ci_low":        float(ci_low),
        "accuracy_ci_high":       float(ci_high),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "kappa_mean":             float(np.mean(kappas)),
    }


def _load_csv_done_keys(csv_path, base_dict: dict) -> set:
    """Return set of seed strings already computed for the given base_dict config."""
    from pathlib import Path as _Path
    if not _Path(csv_path).exists():
        return set()
    try:
        import pandas as pd
        df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
        if df.empty or "seed" not in df.columns:
            return set()
        mask = pd.Series([True] * len(df), index=df.index)
        for k, v in base_dict.items():
            if k in df.columns:
                mask &= (df[k].astype(str) == str(v))
        return set(df.loc[mask, "seed"].astype(str).unique())
    except Exception:
        return set()


def _all_seeds_done(csv_path, base_dict: dict) -> bool:
    """Return True if every seed in RANDOM_SEEDS is already in the CSV for base_dict."""
    done = _load_csv_done_keys(csv_path, base_dict)
    return all(str(s) in done for s in RANDOM_SEEDS)


def _parallel_seeds(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    clf_fn, info: dict, base_dict: dict,
                    seeds: list | None = None,
                    n_jobs: int | None = None,
                    csv_path=None) -> list[dict]:
    """
    Run cross-validation for multiple seeds in parallel (threads).

    Seeds are fully independent computations — parallelising them with
    ``prefer="threads"`` is safe because sklearn's ``saga`` solver and
    numpy operations release the GIL during heavy computation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y, groups : np.ndarray
        Labels and subject groups.
    clf_fn : callable
        ``(X_tr, y_tr, X_te, y_te) -> (y_pred, y_proba)``
    info : dict
        Dataset info dict (needs ``cv_strategy``).
    base_dict : dict
        Metadata fields merged into every result (ablation, condition, …).
        ``seed`` is added automatically.
    seeds : list of int, optional
        Default: ``RANDOM_SEEDS``.
    n_jobs : int or None
        Parallel workers. None → ``min(len(seeds), N_JOBS)``.

    Returns
    -------
    list of dict
        One result dict per seed.
    """
    if seeds is None:
        seeds = RANDOM_SEEDS

    # Resume: skip seeds whose results are already in the CSV.
    if csv_path is not None:
        done = _load_csv_done_keys(csv_path, base_dict)
        if done:
            seeds = [s for s in seeds if str(s) not in done]
            if not seeds:
                return []

    _nj = n_jobs if n_jobs is not None else min(len(seeds), N_JOBS)

    def _one(seed: int) -> dict:
        if info["cv_strategy"] == "LOSO":
            folds = run_loso_cv(X, y, groups, clf_fn, seed=seed)
        else:
            folds = run_kfold_cv(X, y, groups, clf_fn, seed=seed)
        return _make_ablation_result({**base_dict, "seed": seed}, folds)

    return Parallel(n_jobs=_nj, prefer="threads")(
        delayed(_one)(seed) for seed in seeds
    )


def _reduce_hist_if_large(X_hist: np.ndarray,
                           max_features: int = 2048) -> np.ndarray:
    """
    Apply TruncatedSVD when the histogram feature space is very large.

    BPE histograms are ~2-5% dense (most tokens unseen per channel), so
    converting to a sparse matrix and using TruncatedSVD is fast and
    memory-efficient.  SVD is unsupervised — no label leakage.

    Parameters
    ----------
    X_hist : np.ndarray
        Dense histogram matrix, shape (n_trials, n_ch * vocab_size).
    max_features : int
        Target dimensionality. Applied only when current dim > max_features.

    Returns
    -------
    np.ndarray
        Possibly reduced matrix, shape (n_trials, ≤max_features).
    """
    if X_hist.shape[1] <= max_features:
        return X_hist
    from .config import DEVICE
    n_comp = min(max_features, X_hist.shape[0] - 1)
    logger.info(
        f"  Hist features {X_hist.shape[1]} → pca_reduce({n_comp}) …"
    )
    return pca_reduce(X_hist, n_comp, device=DEVICE)


# ─── Tokenization cache ───────────────────────────────────────────────────────
# Shared across A1 / A4 / A5 / A7 within one run_all_ablations call.
# Key: (ds_name, method, n_bins, vocab_size)
# Value: (BPEVocab, X_hist_raw) where X_hist_raw is the histogram BEFORE
#   _reduce_hist_if_large — stored raw so callers can mask columns (A5)
#   or inspect vocab (A7) before dimensionality reduction.
_TOKENIZATION_CACHE: dict = {}

# Sequence cache: avoids re-quantizing the same (ds, method, bins) combo
# when multiple vocab_sizes share the same quantization (e.g., A9 B×V grid).
# Key: (ds_name, method, n_bins)
# Value: list[list[int]] — quantized sequences
_SEQ_CACHE: dict = {}


def _get_bpe_vocab_and_hist(ds_name: str, all_epochs: np.ndarray,
                             method: str, n_bins: int,
                             vocab_size: int) -> tuple:
    """
    Return (vocab, X_hist_raw) for the given parameters, training BPE only
    on the first call and returning cached results on subsequent calls.

    Caching is two-level:
    1. In-memory (_TOKENIZATION_CACHE): shared across all ablations within one run.
    2. Disk (MODELS_DIR): vocab JSON saved after training so subsequent runs
       skip BPE training entirely (BPE is deterministic for the same input).

    X_hist_raw is the flat per-channel histogram BEFORE ``_reduce_hist_if_large``.
    Callers must apply ``_reduce_hist_if_large`` themselves (allows A5 to do
    per-channel artifact masking on the raw histogram first).

    Vocab is trained on *this dataset only* (unlike exp2 cross-dataset corpus);
    intentional for ablations so each variable is isolated.
    """
    key = (ds_name, method, n_bins, vocab_size)
    if key not in _TOKENIZATION_CACHE:
        vocab_path = MODELS_DIR / f"ablation_vocab_{ds_name}_{method}_B{n_bins}_V{vocab_size}.json"
        if vocab_path.exists():
            logger.info(f"  [vocab disk hit]  {vocab_path.name}")
            vocab = BPEVocab.load(str(vocab_path))
        else:
            logger.info(f"  [BPE cache miss] ds={ds_name} method={method} "
                        f"bins={n_bins} V={vocab_size} — training vocab")
            seq_key = (ds_name, method, n_bins)
            if seq_key not in _SEQ_CACHE:
                _SEQ_CACHE[seq_key] = _epochs_to_seqs(all_epochs, method, n_bins)
            seqs = _SEQ_CACHE[seq_key]
            vocab = train_bpe(seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))
            logger.info(f"  [vocab saved]    {vocab_path.name}")
        X_hist_raw = cached_epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins)
        _TOKENIZATION_CACHE[key] = (vocab, X_hist_raw)
    else:
        logger.info(f"  [BPE cache hit]  ds={ds_name} method={method} "
                    f"bins={n_bins} V={vocab_size}")
    return _TOKENIZATION_CACHE[key]


# ─── A1: Quantization method comparison ──────────────────────────────────────

@timed("ablations")
def run_a1(ds_name: str = "bci_iv_2a",
           vocab_size: int = 4096,
           n_bins: int = 64,
           max_subjects: int = 9) -> list[dict]:
    """
    A1: Compare quantization methods (uniform, μ-law, adaptive)
    on downstream accuracy.

    Parameters
    ----------
    ds_name : str, optional
        Dataset name (default ``"bci_iv_2a"``).
    vocab_size : int, optional
        BPE vocabulary size (default 4096).
    n_bins : int, optional
        Number of quantization bins (default 256).
    max_subjects : int, optional
        Max subjects to load (default 9).

    Returns
    -------
    list of dict
        Ablation results per method × seed.
    """
    exp_log = ExperimentLogger("ablation_A1")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    _mbar = (
        _tqdm(QUANT_METHODS, desc="A1 methods", unit="method", leave=False)
        if _HAS_TQDM else QUANT_METHODS
    )
    for method in _mbar:
        if _HAS_TQDM and hasattr(_mbar, "set_description"):
            _mbar.set_description(f"A1 [{method}]")
        exp_log.info(f"A1: method={method}")

        _a1_base = {"ablation": "A1", "dataset": ds_name, "method": method,
                    "n_bins": n_bins, "vocab_size": vocab_size}
        if _all_seeds_done(exp_log.csv_path, _a1_base):
            exp_log.info(f"  A1 method={method}: all seeds done — skipping")
            continue

        # Vocab trained on this dataset only (unlike exp2 cross-dataset corpus).
        # Intentional for ablations: isolates one variable at a time.
        # Cache hit when the same (ds, method, n_bins, vocab_size) is reused
        # by A4 / A5 / A7 later in the same run.
        vocab, X_hist_raw = _get_bpe_vocab_and_hist(
            ds_name, all_epochs, method, n_bins, vocab_size)
        X_hist = _reduce_hist_if_large(X_hist_raw)

        seed_results = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg, info,
            _a1_base,
            csv_path=exp_log.csv_path,
        )
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

    exp_log.finalize()
    return results


# ─── A2: Number of bins ──────────────────────────────────────────────────────

@timed("ablations")
def run_a2(ds_name: str = "bci_iv_2a",
           vocab_size: int = 4096,
           method: str = "mu_law",
           max_subjects: int = 9,
           fixed_vocab_size: int | None = None) -> list[dict]:
    """
    A2: Compare number of quantization bins on downstream accuracy.

    Runs two conditions per B value:
      - ``proportional``: V = min(vocab_size, B*8) — original formula
      - ``fixed_V``      : V = fixed_vocab_size (default 512) — deconfounds B from V

    Parameters
    ----------
    ds_name : str, optional
        Dataset name (default ``"bci_iv_2a"``).
    vocab_size : int, optional
        Upper bound for proportional-V condition (default 4096).
    method : str, optional
        Quantization method (default ``"mu_law"``).
    max_subjects : int, optional
        Max subjects to load (default 9).
    fixed_vocab_size : int or None, optional
        Fixed total vocab size for the control condition.
        Defaults to 512 (the minimum across all B values in proportional mode).

    Returns
    -------
    list of dict
        Ablation results per n_bins × condition × seed.
    """
    if fixed_vocab_size is None:
        fixed_vocab_size = 512  # ensures all B values have a valid fixed target

    exp_log = ExperimentLogger("ablation_A2")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    bins_list = [64, 128, 256, 512]
    _bbar = (
        _tqdm(bins_list, desc="A2 bins", unit="B", leave=False)
        if _HAS_TQDM else bins_list
    )
    for n_bins in _bbar:
        if _HAS_TQDM and hasattr(_bbar, "set_description"):
            _bbar.set_description(f"A2 [B={n_bins}]")
        exp_log.info(f"A2: n_bins={n_bins}")

        # ── Condition 1: proportional V (original formula, may confound B and V)
        v_prop = min(vocab_size, n_bins * 8)
        v_fixed_val = max(fixed_vocab_size if fixed_vocab_size is not None else 512, n_bins + 1)
        _a2_prop_base  = {"ablation": "A2", "dataset": ds_name, "method": method,
                          "n_bins": n_bins, "vocab_size": v_prop, "condition": "proportional_V"}
        _a2_fixed_base = {"ablation": "A2", "dataset": ds_name, "method": method,
                          "n_bins": n_bins, "vocab_size": v_fixed_val, "condition": "fixed_V"}
        if _all_seeds_done(exp_log.csv_path, _a2_prop_base) and \
                _all_seeds_done(exp_log.csv_path, _a2_fixed_base):
            exp_log.info(f"  A2 n_bins={n_bins}: all conditions done — skipping")
            continue
        # _get_bpe_vocab_and_hist caches vocab to disk and histogram to disk,
        # avoiding repeated BPE training on subsequent warm-cache runs.
        vocab_prop, X_hist_prop_raw = _get_bpe_vocab_and_hist(
            ds_name, all_epochs, method, n_bins, v_prop)
        X_hist_prop = _reduce_hist_if_large(X_hist_prop_raw)
        for r in _parallel_seeds(
            X_hist_prop, y, groups, classify_histogram_logreg, info,
            {"ablation": "A2", "dataset": ds_name, "method": method,
             "n_bins": n_bins, "vocab_size": vocab_prop.vocab_size,
             "condition": "proportional_V"},
            csv_path=exp_log.csv_path,
        ):
            exp_log.log_result(r)
            results.append(r)

        # ── Condition 2: fixed V (deconfounds B from V)
        v_fixed = max(fixed_vocab_size, n_bins + 1)  # must exceed base size
        vocab_fixed, X_hist_fixed_raw = _get_bpe_vocab_and_hist(
            ds_name, all_epochs, method, n_bins, v_fixed)
        X_hist_fixed = _reduce_hist_if_large(X_hist_fixed_raw)
        for r in _parallel_seeds(
            X_hist_fixed, y, groups, classify_histogram_logreg, info,
            {"ablation": "A2", "dataset": ds_name, "method": method,
             "n_bins": n_bins, "vocab_size": vocab_fixed.vocab_size,
             "condition": "fixed_V"},
            csv_path=exp_log.csv_path,
        ):
            exp_log.log_result(r)
            results.append(r)

    exp_log.finalize()
    return results


# ─── A7: BPE vs no-BPE vs random merges (CRITICAL) ───────────────────────────

def random_merge_vocab(base_vocab_size: int, n_merges: int,
                       seed: int = 42) -> BPEVocab:
    """
    Create a vocabulary with random (not frequency-based) merges.

    Parameters
    ----------
    base_vocab_size : int
        Number of base tokens (quantization bins).
    n_merges : int
        Number of random merges to perform.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BPEVocab
        Vocabulary with random merge rules.
    """
    rng = _random.Random(seed)
    vocab = BPEVocab(base_vocab_size=base_vocab_size)
    all_tokens = list(range(base_vocab_size))

    for _ in range(n_merges):
        if len(all_tokens) < 2:
            break
        a = rng.choice(all_tokens)
        b = rng.choice(all_tokens)
        new_id = vocab.add_merge((a, b))
        all_tokens.append(new_id)

    return vocab


@timed("ablations")
def run_a7(ds_name: str = "bci_iv_2a",
           vocab_size: int = 4096,
           n_bins: int = 64,
           method: str = "uniform",
           max_subjects: int = 9) -> list[dict]:
    """
    A7: Critical ablation isolating BPE's contribution.

    Compare: BPE / Raw bins / Random merges.

    Parameters
    ----------
    ds_name : str, optional
        Dataset name (default ``"bci_iv_2a"``).
    vocab_size : int, optional
        BPE vocabulary size (default 4096).
    n_bins : int, optional
        Number of quantization bins (default 256).
    method : str, optional
        Quantization method (default ``"mu_law"``).
    max_subjects : int, optional
        Max subjects to load (default 9).

    Returns
    -------
    list of dict
        Ablation results per condition × seed.
    """
    exp_log = ExperimentLogger("ablation_A7")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]
    n_trials = all_epochs.shape[0]

    # Vocab trained on this dataset only (unlike exp2 cross-dataset corpus).
    # All three A7 conditions use the same training corpus for fair comparison.
    # BPE vocab + its histogram come from the tokenization cache (populated by
    # A1 when method/n_bins/vocab_size match the "uniform" default condition).

    # 1) Real BPE — use cache; avoids re-training if A1 already ran
    bpe_vocab, _bpe_hist_raw = _get_bpe_vocab_and_hist(
        ds_name, all_epochs, method, n_bins, vocab_size)

    # 2) Random merges (same vocab size)
    rand_vocab = random_merge_vocab(n_bins, vocab_size - n_bins, seed=42)

    # 3) No BPE (raw bins) — represented as identity vocab
    raw_vocab = BPEVocab(base_vocab_size=n_bins)

    # Map: condition name → (vocab, precomputed_X_hist_raw or None)
    # None means we must call epochs_to_bpe_histograms at classification time.
    conditions = {
        "BPE":           (bpe_vocab,  _bpe_hist_raw),
        "Random_Merges": (rand_vocab, None),
        "Raw_Bins":      (raw_vocab,  None),
    }

    # Try to import sequential CNN (requires PyTorch — may be unavailable)
    try:
        from .classifiers import classify_seq_cnn as _classify_seq_cnn
        _has_seq_cnn = True
    except Exception as _e:
        logger.warning(f"A7: Sequential CNN unavailable ({_e}), skipping seq conditions")
        _has_seq_cnn = False

    _cbar7 = (
        _tqdm(list(conditions.items()), desc="A7 conditions", unit="cond", leave=False)
        if _HAS_TQDM else conditions.items()
    )
    for cond_name, (vocab, _cached_hist_raw) in _cbar7:
        if _HAS_TQDM and hasattr(_cbar7, "set_description"):
            _cbar7.set_description(f"A7 [{cond_name}]")
        exp_log.info(f"A7: {cond_name}")

        # ── Histogram + LogReg (bag-of-tokens, loses order) ──────────────
        _a7_hist_base = {"ablation": "A7", "dataset": ds_name, "condition": cond_name,
                         "classifier": "Hist_LogReg", "vocab_size": vocab.vocab_size}
        _skip_hist = _all_seeds_done(exp_log.csv_path, _a7_hist_base)
        if not _skip_hist:
            if _cached_hist_raw is not None:
                X_hist = _reduce_hist_if_large(_cached_hist_raw)
            else:
                X_hist = _reduce_hist_if_large(
                    cached_epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins))
        else:
            exp_log.info(f"  A7 Hist_LogReg {cond_name}: all seeds done — skipping PCA")
        if not _skip_hist:
            hist_results = _parallel_seeds(
                X_hist, y, groups, classify_histogram_logreg, info,
                _a7_hist_base,
                csv_path=exp_log.csv_path,
            )
            for r in hist_results:
                exp_log.log_result(r)
            results.extend(hist_results)

        # ── Sequential CNN (preserves token order — BPE value should show) ─
        if _has_seq_cnn:
            # Adaptive max_len: give each channel ~64 tokens so the CNN has
            # meaningful context even for high-channel datasets (bci_iv_2a=22ch).
            # Default 512 would give only ~23 tokens/ch for 22ch, which is too short.
            _n_ch = all_epochs.shape[1]
            _flat_max_len = max(512, _n_ch * 64 + (_n_ch - 1))  # 64 tok/ch + SEP slots
            _cnn_base = {"ablation": "A7", "dataset": ds_name,
                         "condition": cond_name, "classifier": "Seq_CNN",
                         "vocab_size": vocab.vocab_size}
            _cnn_done = _load_csv_done_keys(exp_log.csv_path, _cnn_base)
            _cnn_seeds = [s for s in RANDOM_SEEDS if str(s) not in _cnn_done]
            if _cnn_seeds:
                X_seq = epochs_to_bpe_sequences(all_epochs, vocab, method, n_bins,
                                                 max_len=_flat_max_len)
                _vs = vocab.vocab_size

                def _seq_clf(X_tr, y_tr, X_te, y_te, _vs=_vs):
                    return _classify_seq_cnn(X_tr, y_tr, X_te, y_te,
                                             vocab_size=_vs)

                # Sequential CNN uses PyTorch — safer to run seeds sequentially
                _sbar = (
                    _tqdm(_cnn_seeds, desc=f"A7/{cond_name} CNN seeds",
                          unit="seed", leave=False)
                    if _HAS_TQDM else _cnn_seeds
                )
                for seed in _sbar:
                    if info["cv_strategy"] == "LOSO":
                        folds = run_loso_cv(X_seq, y, groups, _seq_clf, seed=seed)
                    else:
                        folds = run_kfold_cv(X_seq, y, groups, _seq_clf, seed=seed)
                    r = _make_ablation_result({**_cnn_base, "seed": seed}, folds)
                    exp_log.log_result(r)
                    results.append(r)

    exp_log.finalize()
    return results


# ─── A3: Frequency-aware strategy ────────────────────────────────────────────

@timed("ablations")
def run_a3(ds_name: str = "mental_arithmetic",
           vocab_size: int = 4096,
           n_bins: int = 64,
           method: str = "uniform",
           max_subjects: int = 9) -> list[dict]:
    """
    A3: Compare frequency-handling strategies on downstream accuracy.

    Conditions:
        - resample: resample to TARGET_SFREQ before quantization (Approach A)
        - temporal_norm: group samples by temporal bin (Approach B approximation)
        - original: no frequency correction (raw sfreq)

    Parameters
    ----------
    ds_name : str
        Dataset name (ideally one with high sfreq, e.g. mental_arithmetic=500Hz).
    vocab_size : int
        BPE vocabulary size.
    n_bins : int
        Number of quantization bins.
    method : str
        Quantization method.
    max_subjects : int
        Max subjects to load.

    Returns
    -------
    list of dict
        Ablation results per condition × seed.
    """
    from scipy.signal import resample as scipy_resample

    exp_log = ExperimentLogger("ablation_A3")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]
    orig_sfreq = info["sfreq"]

    conditions = {}

    # 1) Original (no correction)
    conditions["original"] = all_epochs

    # 2) Resample to TARGET_SFREQ (Approach A)
    if orig_sfreq != TARGET_SFREQ:
        n_times_new = int(all_epochs.shape[-1] * TARGET_SFREQ / orig_sfreq)
        resampled = scipy_resample(all_epochs, n_times_new, axis=-1)
        conditions["resample"] = resampled.astype(np.float64)
    else:
        conditions["resample"] = all_epochs

    # 3) Temporal normalization (Approach B approximation):
    #    Group consecutive samples to match TARGET_SFREQ time resolution,
    #    then average within groups before quantization.
    ratio = orig_sfreq / TARGET_SFREQ
    if ratio > 1:
        group_size = max(1, int(round(ratio)))
        n_groups = all_epochs.shape[-1] // group_size
        trimmed = all_epochs[:, :, :n_groups * group_size]
        reshaped = trimmed.reshape(*trimmed.shape[:-1], n_groups, group_size)
        temp_normed = reshaped.mean(axis=-1)
        conditions["temporal_norm"] = temp_normed
    else:
        conditions["temporal_norm"] = all_epochs

    _cbar = (
        _tqdm(list(conditions.items()), desc="A3 conditions", unit="cond", leave=False)
        if _HAS_TQDM else conditions.items()
    )
    for cond_name, X_cond in _cbar:
        if _HAS_TQDM and hasattr(_cbar, "set_description"):
            _cbar.set_description(f"A3 [{cond_name}]")
        exp_log.info(f"A3: {cond_name}")

        _a3_base = {"ablation": "A3", "dataset": ds_name, "condition": cond_name,
                    "orig_sfreq": orig_sfreq, "target_sfreq": TARGET_SFREQ,
                    "n_times": X_cond.shape[-1]}
        if _all_seeds_done(exp_log.csv_path, _a3_base):
            exp_log.info(f"  A3 {cond_name}: all seeds done — skipping")
            continue

        # Vocab trained on this dataset's preprocessed data (condition-specific).
        # Intentional: isolates the effect of each frequency-handling strategy.
        # Use a condition-specific key to cache each preprocessed variant's vocab.
        _a3_key = (f"a3_{ds_name}_{cond_name}", method, n_bins, vocab_size)
        if _a3_key not in _TOKENIZATION_CACHE:
            vocab_path = MODELS_DIR / f"ablation_vocab_a3_{ds_name}_{cond_name}_{method}_B{n_bins}_V{vocab_size}.json"
            if vocab_path.exists():
                logger.info(f"  [vocab disk hit]  {vocab_path.name}")
                vocab = BPEVocab.load(str(vocab_path))
            else:
                seqs = _epochs_to_seqs(X_cond, method, n_bins)
                vocab = train_bpe(seqs, vocab_size=vocab_size,
                                  base_vocab_size=n_bins, verbose=False,
                                  max_train_tokens=5_000_000)
                vocab.save(str(vocab_path))
            X_hist_raw_a3 = cached_epochs_to_bpe_histograms(X_cond, vocab, method, n_bins)
            _TOKENIZATION_CACHE[_a3_key] = (vocab, X_hist_raw_a3)
        else:
            vocab, X_hist_raw_a3 = _TOKENIZATION_CACHE[_a3_key]
        X_hist = _reduce_hist_if_large(X_hist_raw_a3)

        seed_results = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg, info,
            _a3_base,
            csv_path=exp_log.csv_path,
        )
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

    exp_log.finalize()
    return results


# ─── A4: Channel strategy (independent vs spatial+VQ) ────────────────────────

@timed("ablations")
def run_a4(ds_name: str = "bci_iv_2a",
           vocab_size: int = 4096,
           n_bins: int = 64,
           method: str = "uniform",
           n_spatial_codes: int = 256,
           max_subjects: int = 9) -> list[dict]:
    """
    A4: Compare channel-independent vs spatial VQ tokenization.

    Conditions:
        - independent: BPE trained per-channel (current default)
        - spatial_vq: VQ-encode spatial vectors, then BPE on VQ codes

    Parameters
    ----------
    ds_name : str
        Dataset name.
    vocab_size : int
        BPE vocabulary size.
    n_bins : int
        Number of quantization bins.
    method : str
        Quantization method.
    n_spatial_codes : int
        Number of spatial VQ codebook entries.
    max_subjects : int
        Max subjects to load.

    Returns
    -------
    list of dict
        Ablation results per condition × seed.
    """
    from sklearn.cluster import MiniBatchKMeans

    exp_log = ExperimentLogger("ablation_A4")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]
    n_trials, n_ch, n_time = all_epochs.shape

    # ─── Condition 1: Independent (default) ───────────────────────────────
    exp_log.info("A4: channel_independent")
    _a4_ind_base = {"ablation": "A4", "dataset": ds_name, "condition": "channel_independent"}
    if not _all_seeds_done(exp_log.csv_path, _a4_ind_base):
        # Vocab trained on this dataset only (unlike exp2 cross-dataset corpus).
        # Intentional for ablations: isolates one variable at a time.
        # Cache hit when A1 already ran (uniform, n_bins, vocab_size) for this ds.
        _, X_hist_ind_raw = _get_bpe_vocab_and_hist(
            ds_name, all_epochs, method, n_bins, vocab_size)
        X_hist_ind = _reduce_hist_if_large(X_hist_ind_raw)
        results.extend(_parallel_seeds(
            X_hist_ind, y, groups, classify_histogram_logreg, info,
            _a4_ind_base,
            csv_path=exp_log.csv_path,
        ))
    else:
        exp_log.info("  A4 channel_independent: all seeds done — skipping")

    # ─── Condition 2: Spatial VQ + BPE ────────────────────────────────────
    exp_log.info("A4: spatial_vq")
    _a4_svq_base = {"ablation": "A4", "dataset": ds_name, "condition": "spatial_vq",
                    "n_spatial_codes": n_spatial_codes}
    if not _all_seeds_done(exp_log.csv_path, _a4_svq_base):
        # Quantize all channels independently
        flat = all_epochs.reshape(-1, n_time)
        codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
        codes = codes_flat.reshape(n_trials, n_ch, n_time)

        # At each time step: spatial vector of n_ch quantized values → VQ code
        spatial_vectors = codes.transpose(0, 2, 1).reshape(-1, n_ch)  # (n_trials*n_time, n_ch)

        # Sub-sample for k-means
        rng = np.random.RandomState(42)
        n_total = len(spatial_vectors)
        if n_total > 100000:
            idx = rng.choice(n_total, 100000, replace=False)
            train_sv = spatial_vectors[idx].astype(np.float32)
        else:
            train_sv = spatial_vectors.astype(np.float32)

        km = MiniBatchKMeans(n_clusters=n_spatial_codes, batch_size=2048,
                             n_init=3, random_state=42)
        km.fit(train_sv)
        spatial_codes = km.predict(spatial_vectors.astype(np.float32))
        spatial_codes = spatial_codes.reshape(n_trials, n_time)

        # BPE on spatial code sequences
        spatial_seqs = [spatial_codes[i].tolist() for i in range(n_trials)]
        vocab_sp = train_bpe(spatial_seqs, vocab_size=vocab_size,
                              base_vocab_size=n_spatial_codes, verbose=False,
                              max_train_tokens=5_000_000)

        # Histograms
        V_sp = vocab_sp.vocab_size
        bpe_seqs = apply_bpe_batch(spatial_seqs, vocab_sp, n_jobs=N_JOBS)
        hist_sp = np.zeros((n_trials, V_sp), dtype=np.float32)
        for i, seq in enumerate(bpe_seqs):
            arr = np.array(seq, dtype=np.intp)
            arr = arr[arr < V_sp]
            if len(arr) > 0:
                hist_sp[i] = np.bincount(arr, minlength=V_sp)[:V_sp].astype(np.float32)
        hist_sp = hist_sp / (hist_sp.sum(axis=1, keepdims=True) + 1e-10)

        results.extend(_parallel_seeds(
            hist_sp, y, groups, classify_histogram_logreg, info,
            _a4_svq_base,
            csv_path=exp_log.csv_path,
        ))
    else:
        exp_log.info("  A4 spatial_vq: all seeds done — skipping")

    for r in results:
        exp_log.log_result(r)
    exp_log.finalize()
    return results


# ─── A5: Artifact tokens (with vs without) ───────────────────────────────────

@timed("ablations")
def run_a5(ds_name: str = "bci_iv_2a",
           vocab_size: int = 4096,
           n_bins: int = 64,
           method: str = "uniform",
           artifact_percentile: float = 95.0,
           max_subjects: int = 9) -> list[dict]:
    """
    A5: Compare classification with and without artifact tokens.

    Artifact identification: tokens whose decoded waveforms have extreme
    amplitude (> artifact_percentile of overall amplitude distribution)
    are masked to zero in the histogram feature.

    Parameters
    ----------
    ds_name : str
        Dataset name.
    vocab_size : int
        BPE vocabulary size.
    n_bins : int
        Number of quantization bins.
    method : str
        Quantization method.
    artifact_percentile : float
        Percentile threshold for artifact detection (default 95).
    max_subjects : int
        Max subjects to load.

    Returns
    -------
    list of dict
        Ablation results per condition × seed.
    """
    exp_log = ExperimentLogger("ablation_A5")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    # Vocab trained on this dataset only (unlike exp2 cross-dataset corpus).
    # Intentional for ablations: isolates one variable at a time.
    # Cache hit when A1 already ran (method, n_bins, vocab_size) for this ds.
    # X_hist is raw (pre-PCA) so artifact columns can be masked correctly
    # before dimensionality reduction.
    vocab, X_hist = _get_bpe_vocab_and_hist(
        ds_name, all_epochs, method, n_bins, vocab_size)

    # Identify artifact tokens:
    # For each token in the vocab, compute the mean absolute amplitude
    # of the constituent quantization bins. Tokens with high amplitude
    # (quantization bins near the extremes: 0 or n_bins-1) are artifacts.
    V = vocab.vocab_size
    token_max_bin = np.zeros(V)
    for tok_id in range(V):
        # Get constituent base tokens (bins) for this BPE token
        expanded = vocab.decode_token(tok_id)
        if expanded:
            # Distance from center bin = proxy for amplitude extremeness
            center = n_bins / 2.0
            distances = [abs(b - center) / center for b in expanded if b < n_bins]
            token_max_bin[tok_id] = max(distances) if distances else 0.0

    threshold = np.percentile(token_max_bin[token_max_bin > 0], artifact_percentile)
    artifact_mask = token_max_bin > threshold
    n_artifact = int(artifact_mask.sum())
    exp_log.info(f"A5: Identified {n_artifact}/{V} artifact tokens (threshold={threshold:.3f})")

    # Condition 1: With artifact tokens (original)
    _a5_with_base = {"ablation": "A5", "dataset": ds_name, "condition": "with_artifacts",
                     "n_artifact_tokens": n_artifact}
    if not _all_seeds_done(exp_log.csv_path, _a5_with_base):
        results.extend(_parallel_seeds(
            _reduce_hist_if_large(X_hist), y, groups, classify_histogram_logreg, info,
            _a5_with_base,
            csv_path=exp_log.csv_path,
        ))
    else:
        exp_log.info("  A5 with_artifacts: all seeds done — skipping PCA")

    # Condition 2: Without artifact tokens (masked)
    _a5_without_base = {"ablation": "A5", "dataset": ds_name, "condition": "without_artifacts",
                        "n_artifact_tokens": n_artifact}
    if not _all_seeds_done(exp_log.csv_path, _a5_without_base):
        # Keep X_hist dense for column-level masking, then reduce before classification
        X_hist_clean = X_hist.copy()
        n_ch = all_epochs.shape[1]
        # Mask artifact columns in each channel's histogram
        for ch in range(n_ch):
            col_start = ch * V
            col_end = (ch + 1) * V
            if col_end <= X_hist_clean.shape[1]:
                X_hist_clean[:, col_start:col_end][:, artifact_mask] = 0.0
        # Re-normalize
        for ch in range(n_ch):
            col_start = ch * V
            col_end = (ch + 1) * V
            if col_end <= X_hist_clean.shape[1]:
                ch_sums = X_hist_clean[:, col_start:col_end].sum(axis=1, keepdims=True)
                X_hist_clean[:, col_start:col_end] /= (ch_sums + 1e-10)
        results.extend(_parallel_seeds(
            _reduce_hist_if_large(X_hist_clean), y, groups, classify_histogram_logreg, info,
            _a5_without_base,
            csv_path=exp_log.csv_path,
        ))
    else:
        exp_log.info("  A5 without_artifacts: all seeds done — skipping PCA")

    for r in results:
        exp_log.log_result(r)
    exp_log.finalize()
    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_ablation_bar(results: list[dict], ablation_name: str,
                      group_col: str, save_dir: Path) -> None:
    """
    Generic bar chart for ablation results.

    Parameters
    ----------
    results : list of dict
        Ablation result dicts.
    ablation_name : str
        Ablation identifier for the plot title.
    group_col : str
        Column name used to group bars on the x-axis.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    groups = sorted(set(r[group_col] for r in results))
    means = []
    stds = []
    for g in groups:
        subset = [r for r in results if r[group_col] == g]
        accs = [r["accuracy_mean"] for r in subset]
        means.append(np.mean(accs))
        stds.append(np.std(accs))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(g) for g in groups], means, yerr=stds, alpha=0.8,
           color="#1f77b4")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Ablation {ablation_name}")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_dir / f"ablation_{ablation_name}.png", dpi=150)
    plt.close(fig)


def plot_a7_grouped(results: list[dict], save_dir: Path) -> None:
    """
    A7 grouped bar chart: x-axis = BPE condition, hue = classifier type.

    For each (condition × classifier) pair the bar shows mean accuracy ± std.
    This makes it easy to see whether BPE adds value over Random / Raw when
    the classifier can exploit token order (Seq_CNN) vs not (Hist_LogReg).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(set(r["condition"] for r in results))
    classifiers = sorted(set(r.get("classifier", "Hist_LogReg") for r in results))

    n_conds = len(conditions)
    n_clfs = len(classifiers)
    width = 0.8 / max(n_clfs, 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig, ax = plt.subplots(figsize=(max(8, n_conds * 2.5), 5))
    x = np.arange(n_conds)

    for ci, clf_name in enumerate(classifiers):
        means, stds, cis = [], [], []
        for cond in conditions:
            subset = [r for r in results
                      if r["condition"] == cond
                      and r.get("classifier", "Hist_LogReg") == clf_name]
            if subset:
                accs = [r["accuracy_mean"] for r in subset]
                means.append(float(np.mean(accs)))
                stds.append(float(np.std(accs)))
                # Use CI from individual records if available
                lo = np.mean([r.get("accuracy_ci_low", accs[0]) for r in subset])
                hi = np.mean([r.get("accuracy_ci_high", accs[0]) for r in subset])
                cis.append((means[-1] - lo, hi - means[-1]))
            else:
                means.append(0.0)
                stds.append(0.0)
                cis.append((0.0, 0.0))

        offsets = x + (ci - n_clfs / 2 + 0.5) * width
        yerr = np.array([[c[0] for c in cis], [c[1] for c in cis]])
        bars = ax.bar(offsets, means, width=width * 0.9,
                      label=clf_name, color=colors[ci % len(colors)],
                      alpha=0.85, yerr=yerr, capsize=4, error_kw={"linewidth": 1})

        # Annotate bars with value
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("A7: BPE vs Random Merges vs Raw Bins\n"
                 "(Hist=bag-of-tokens loses order; Seq_CNN=preserves order)")
    ax.legend(title="Classifier", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_dir / "ablation_A7_grouped.png", dpi=150)
    plt.close(fig)


def plot_ablation_grouped_datasets(
    results: list[dict],
    ablation_name: str,
    group_col: str,
    save_dir: "Path",
    metric: str = "balanced_accuracy_mean",
) -> None:
    """
    Grouped bar chart: x-axis = condition, hue = dataset.

    Uses *metric* on the y-axis (default ``balanced_accuracy_mean``).
    Falls back to ``accuracy_mean`` when the primary metric is absent.
    Saves ``ablation_{ablation_name}_multi_dataset.png``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)
    conditions = sorted(set(r[group_col] for r in results))
    ds_list    = sorted(set(r.get("dataset", "") for r in results))
    colors     = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    n_conds    = len(conditions)
    n_ds       = len(ds_list)
    width      = 0.8 / max(n_ds, 1)

    fig, ax = plt.subplots(figsize=(max(7, n_conds * 2.5), 5))
    x = np.arange(n_conds)

    for di, ds in enumerate(ds_list):
        means, errs = [], []
        for cond in conditions:
            subset = [r for r in results
                      if r[group_col] == cond and r.get("dataset", "") == ds]
            vals = [r.get(metric, r["accuracy_mean"]) for r in subset]
            means.append(float(np.mean(vals)) if vals else 0.0)
            errs.append(float(np.std(vals)) if vals else 0.0)

        offsets = x + (di - n_ds / 2 + 0.5) * width
        ax.bar(offsets, means, width * 0.9, label=ds,
               color=colors[di % len(colors)], alpha=0.85,
               yerr=errs, capsize=4, error_kw={"linewidth": 1})
        for xpos, m in zip(offsets, means):
            if m > 0:
                ax.text(xpos, m + 0.005, f"{m:.3f}",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in conditions], fontsize=9)
    metric_label = metric.replace("_mean", "").replace("_", " ")
    ax.set_ylabel(metric_label.capitalize())
    ax.set_title(f"Ablation {ablation_name}  (y = {metric_label})")
    ax.legend(title="Dataset", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_dir / f"ablation_{ablation_name}_multi_dataset.png", dpi=150)
    plt.close(fig)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _assemble_data(data: dict) -> tuple[np.ndarray, list, list]:
    """Assemble epochs from loaded dataset dict."""
    all_epochs = []
    all_labels = []
    all_groups = []
    for subj_id, subj_data in sorted(data.items()):
        all_epochs.append(subj_data["epochs"])
        all_labels.extend(subj_data["labels"])
        all_groups.extend([subj_id] * len(subj_data["labels"]))

    n_ch_max = max(e.shape[1] for e in all_epochs)
    n_time_max = max(e.shape[2] for e in all_epochs)
    padded = []
    for e in all_epochs:
        p = np.zeros((e.shape[0], n_ch_max, n_time_max))
        p[:, :e.shape[1], :e.shape[2]] = e
        padded.append(p)

    return np.concatenate(padded, axis=0), all_labels, all_groups


def _epochs_to_seqs(epochs: np.ndarray, method: str,
                    n_bins: int) -> list[list[int]]:
    """Quantize epochs → list of int sequences."""
    flat = epochs.reshape(-1, epochs.shape[-1])
    codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
    return [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]


# ─── A8: Discriminative BPE ──────────────────────────────────────────────────

@timed("ablations")
def run_a8(ds_name: str = "bci_iv_2a",
           vocab_size: int = 512,
           n_bins: int = 64,
           max_subjects: int = 9) -> list[dict]:
    """
    A8: Discriminative BPE — compare standard vs Fisher-ratio-guided merge selection.

    Tests alpha ∈ {0.0, 0.5, 1.0} where alpha=0 is standard BPE and alpha=1
    selects merges purely by class-discriminative score.  Expected to help on
    MI (frequency-coded, BPE near chance) and be neutral on Sleep.
    """
    from .config import DEFAULT_QUANT_METHOD
    method = DEFAULT_QUANT_METHOD
    exp_log = ExperimentLogger("ablation_A8")
    results = []

    # Fast path: skip data loading if all conditions × seeds already done
    alphas = [
        (0.0, "Standard_BPE"),
        (0.5, "DiscBPE_alpha0.5"),
        (1.0, "DiscBPE_alpha1.0"),
    ]
    if all(
        _all_seeds_done(exp_log.csv_path,
                        {"ablation": "A8", "dataset": ds_name, "condition": cond_name,
                         "alpha": alpha, "vocab_size": vocab_size})
        for alpha, cond_name in alphas
    ):
        exp_log.info(f"A8 {ds_name}: all conditions × seeds done — skipping")
        exp_log.finalize()
        return results

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        exp_log.finalize()
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    n_classes = len(np.unique(y))
    seq_key = (ds_name, method, n_bins)
    if seq_key not in _SEQ_CACHE:
        _SEQ_CACHE[seq_key] = _epochs_to_seqs(all_epochs, method, n_bins)
    seqs = _SEQ_CACHE[seq_key]
    # seqs has n_trials * n_channels entries (one per channel).
    # y has n_trials entries.  Expand to match by repeating each label n_ch times.
    n_ch = all_epochs.shape[1]
    y_expanded = np.repeat(y, n_ch)

    for alpha, cond_name in alphas:
        exp_log.info(f"A8: {cond_name} (α={alpha})")
        _a8_base = {"ablation": "A8", "dataset": ds_name, "condition": cond_name,
                    "alpha": alpha, "vocab_size": vocab_size}
        if _all_seeds_done(exp_log.csv_path, _a8_base):
            exp_log.info(f"  A8 {cond_name}: all seeds done — skipping")
            continue

        vocab_path = MODELS_DIR / f"ablation_vocab_a8_{ds_name}_{cond_name}_{method}_B{n_bins}_V{vocab_size}.json"
        if vocab_path.exists():
            logger.info(f"  [vocab disk hit]  {vocab_path.name}")
            vocab = BPEVocab.load(str(vocab_path))
        elif alpha == 0.0:
            vocab = train_bpe(seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))
        else:
            vocab = train_bpe_discriminative(
                seqs, vocab_size=vocab_size, base_vocab_size=n_bins,
                y_labels=list(y_expanded), n_classes=n_classes,
                alpha=alpha, verbose=False,
                max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))

        X_hist = _reduce_hist_if_large(
            cached_epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins))

        cond_results = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg, info,
            _a8_base,
            csv_path=exp_log.csv_path,
        )
        for r in cond_results:
            exp_log.log_result(r)
        results.extend(cond_results)

    return results


# ─── A9: B×V grid with best downstream classifier ────────────────────────────

@timed("ablations")
def run_a9(ds_name: str = "bci_iv_2a",
           max_subjects: int = 9) -> list[dict]:
    """
    A9: B × V grid with both LogReg and RF classifiers.

    Motivation: Figure 5 (vocab sweep) used BPE_Hist_LogReg only.
    For Sleep-EDF, RF reaches 87.6% while LogReg peaks at 75.0% at V=128.
    The "V=128 optimal" finding is a LogReg regularisation artefact.
    A9 tests B in {32, 64, 128} and V in {128, 256, 512, 1024} with both
    classifiers, revealing the true optimal (B, V) pair per dataset.
    """
    from .config import DEFAULT_QUANT_METHOD
    from sklearn.ensemble import RandomForestClassifier as RF_

    method = DEFAULT_QUANT_METHOD
    exp_log = ExperimentLogger("ablation_A9")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    bins_list  = [32, 64, 128]
    vocab_list = [128, 256, 512, 1024]

    for n_bins in bins_list:
        for vocab_size in vocab_list:
            _a9_lr_base = {"ablation": "A9", "dataset": ds_name,
                           "n_bins": n_bins, "vocab_size": vocab_size,
                           "classifier": "LogReg"}
            _a9_rf_base = {"ablation": "A9", "dataset": ds_name,
                           "n_bins": n_bins, "vocab_size": vocab_size,
                           "classifier": "RF"}
            _lr_done = _all_seeds_done(exp_log.csv_path, _a9_lr_base)
            _rf_done = _all_seeds_done(exp_log.csv_path, _a9_rf_base)
            if _lr_done and _rf_done:
                exp_log.info(f"  A9 B={n_bins} V={vocab_size}: all done — skipping")
                continue

            _, X_hist_raw = _get_bpe_vocab_and_hist(
                ds_name, all_epochs, method, n_bins, vocab_size)
            X_hist = _reduce_hist_if_large(X_hist_raw)

            # LogReg
            lr_rows = [] if _lr_done else _parallel_seeds(
                X_hist, y, groups, classify_histogram_logreg, info,
                _a9_lr_base,
                csv_path=exp_log.csv_path,
            )
            # RF — use reduced features for speed
            X_rf = X_hist
            if X_rf.shape[1] > 2048:
                from sklearn.decomposition import TruncatedSVD
                svd = TruncatedSVD(n_components=512, random_state=42)
                X_rf = svd.fit_transform(X_rf)

            def _rf_classify(X_train, y_train, X_test, y_test):
                # n_jobs=1: _parallel_seeds runs 5 seeds in parallel threads;
                # using n_jobs=N_JOBS here would spawn 5×N_JOBS processes,
                # over-subscribing the CPU. Let thread-level parallelism win.
                clf = RF_(n_estimators=200, n_jobs=1,
                          class_weight="balanced", random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_proba = clf.predict_proba(X_test)
                return y_pred, y_proba

            rf_rows = [] if _rf_done else _parallel_seeds(
                X_rf, y, groups, _rf_classify, info,
                _a9_rf_base,
                csv_path=exp_log.csv_path,
            )

            for r in lr_rows + rf_rows:
                exp_log.log_result(r)
            results.extend(lr_rows + rf_rows)

            logger.info(
                "A9 %s B=%d V=%d: LR=%.3f RF=%.3f" % (
                    ds_name, n_bins, vocab_size,
                    float(np.mean([r["accuracy_mean"] for r in lr_rows])) if lr_rows else 0,
                    float(np.mean([r["accuracy_mean"] for r in rf_rows])) if rf_rows else 0,
                )
            )

    exp_log.finalize()
    return results


def _plot_a9_heatmap(results: list[dict], plot_dir: Path) -> None:
    """Heatmap: B × V for LogReg vs RF side by side."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(results)
        datasets = df["dataset"].unique()
        classifiers = ["LogReg", "RF"]

        for ds in datasets:
            sub = df[df["dataset"] == ds]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                                     sharey=True, sharex=True)
            for ax, clf in zip(axes, classifiers):
                sub_clf = sub[sub["classifier"] == clf]
                piv = sub_clf.pivot_table(
                    index="n_bins", columns="vocab_size",
                    values="accuracy_mean", aggfunc="mean",
                )
                if piv.empty:
                    continue
                im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn",
                               vmin=0.0, vmax=1.0)
                ax.set_xticks(range(len(piv.columns)))
                ax.set_xticklabels([str(v) for v in piv.columns])
                ax.set_yticks(range(len(piv.index)))
                ax.set_yticklabels([str(b) for b in piv.index])
                ax.set_xlabel("Vocab size V")
                ax.set_ylabel("Bins B")
                ax.set_title(f"{clf}")
                for i in range(piv.values.shape[0]):
                    for j in range(piv.values.shape[1]):
                        v = piv.values[i, j]
                        if not np.isnan(v):
                            ax.text(j, i, f"{v:.2f}", ha="center",
                                    va="center", fontsize=7, color="black")
                plt.colorbar(im, ax=ax)
            fig.suptitle(f"A9: B x V grid — {ds}")
            plt.tight_layout()
            out = plot_dir / f"ablation_A9_bv_grid_{ds}.png"
            plt.savefig(str(out), dpi=150)
            plt.close(fig)
    except Exception as exc:
        logger.warning(f"A9 plot failed: {exc}")


# ─── A10: PhysioNet-MI scale ablation ────────────────────────────────────────

@timed("ablations")
def run_a10_physionet_scale(vocab_size: int = 1024,
                             n_bins: int = 64) -> list[dict]:
    """
    A10: PhysioNet-MI scale ablation.

    Hypothesis: BPE achieves 42.7% on PhysioNet-MI (full, 109 subjects) because
    the large sample size gives BPE enough data to learn meaningful co-occurrences.
    When subsampled to 9 subjects (same scale as BCI-IV-2a), BPE accuracy may
    collapse to near-chance — isolating scale as the key variable.

    Conditions:
      - ``subj9``   : max_subjects=9 (same scale as BCI-IV-2a)
      - ``all_subj``: all 109 subjects (replicates main-experiment setting)
    """
    from .config import DEFAULT_QUANT_METHOD
    method = DEFAULT_QUANT_METHOD
    exp_log = ExperimentLogger("ablation_A10")
    results = []

    conditions = [("subj9", 9), ("all_subj", None)]

    # Fast path: skip data loading if all conditions × seeds already done
    if all(
        _all_seeds_done(exp_log.csv_path,
                        {"ablation": "A10", "dataset": "physionet_mi",
                         "condition": cond, "vocab_size": vocab_size})
        for cond, _ in conditions
    ):
        exp_log.info("A10: all conditions × seeds done — skipping")
        exp_log.finalize()
        return results

    info = DATASET_INFO["physionet_mi"]

    for cond, max_subj in conditions:
        base = {"ablation": "A10", "dataset": "physionet_mi",
                "condition": cond, "vocab_size": vocab_size}
        if _all_seeds_done(exp_log.csv_path, base):
            exp_log.info(f"  A10 {cond}: all seeds done — skipping")
            continue

        data = load_dataset("physionet_mi", max_subjects=max_subj)
        if not data:
            continue

        all_epochs, all_labels, all_groups = _assemble_data(data)
        y      = LabelEncoder().fit_transform(np.array(all_labels))
        groups = np.array(all_groups)
        n_subj = len(np.unique(groups))
        exp_log.info(f"  A10 {cond}: {n_subj} subjects loaded")

        # Separate vocab path per condition — avoids _TOKENIZATION_CACHE collision
        vocab_path = (MODELS_DIR
                      / f"ablation_vocab_physionet_mi_{cond}_{method}_B{n_bins}_V{vocab_size}.json")
        if vocab_path.exists():
            logger.info(f"  [vocab disk hit]  {vocab_path.name}")
            vocab = BPEVocab.load(str(vocab_path))
        else:
            seq_key = (f"physionet_mi_{cond}", method, n_bins)
            if seq_key not in _SEQ_CACHE:
                _SEQ_CACHE[seq_key] = _epochs_to_seqs(all_epochs, method, n_bins)
            seqs = _SEQ_CACHE[seq_key]
            vocab = train_bpe(seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))

        X_hist_raw = cached_epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins)
        X_hist = _reduce_hist_if_large(X_hist_raw)

        rows = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg, info, base,
            csv_path=exp_log.csv_path,
        )
        for r in rows:
            exp_log.log_result(r)
        results.extend(rows)

        if results:
            save_csv(results, LOGS_DIR / "ablation_A10_physionet_scale.csv")
        exp_log.info(
            "A10 %s: acc=%.3f" % (
                cond,
                float(np.mean([r["accuracy_mean"] for r in rows])) if rows else 0.0,
            )
        )

    exp_log.finalize()
    return results


# ─── K7: BPE + PSD ensemble on Mental Arithmetic ─────────────────────────────

@timed("ablations")
def run_k7_bpe_psd_ensemble(ds_name: str = "mental_arithmetic",
                             vocab_size: int = 1024,
                             n_bins: int = 64,
                             max_subjects: int | None = None) -> list[dict]:
    """
    K7: BPE + Welch-PSD feature ensemble on Mental Arithmetic.

    Tests whether combining BPE histogram features (amplitude/temporal patterns)
    with PSD band-power features (δ/θ/α/β/γ) outperforms either alone on the
    mental-arithmetic paradigm (which sits between amplitude- and frequency-coded).

    Three conditions (all use LogReg):
      - ``BPE_Hist_LogReg``  : BPE histograms only  (baseline, same as exp2)
      - ``PSD_LogReg``       : Welch PSD features only
      - ``BPE_PSD_Ensemble`` : concatenated BPE + PSD features
    """
    from .config import DEFAULT_QUANT_METHOD
    from scipy.signal import welch as _welch
    from sklearn.preprocessing import StandardScaler

    method  = DEFAULT_QUANT_METHOD
    exp_log = ExperimentLogger("ablation_K7")
    results = []

    clf_names = ["BPE_Hist_LogReg", "PSD_LogReg", "BPE_PSD_Ensemble"]

    # Fast path
    if all(
        _all_seeds_done(exp_log.csv_path,
                        {"ablation": "K7", "dataset": ds_name,
                         "classifier": c, "vocab_size": vocab_size})
        for c in clf_names
    ):
        exp_log.info(f"K7 {ds_name}: all conditions × seeds done — skipping")
        exp_log.finalize()
        return results

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        exp_log.finalize()
        return results

    all_epochs, all_labels, all_groups = _assemble_data(data)
    y      = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info   = DATASET_INFO[ds_name]
    sfreq  = float(info.get("sfreq", 256.0))

    # ── BPE histograms ────────────────────────────────────────────────────────
    _, X_bpe_raw = _get_bpe_vocab_and_hist(ds_name, all_epochs, method, n_bins, vocab_size)
    X_bpe = _reduce_hist_if_large(X_bpe_raw)          # (n_trials, n_feat_bpe)

    # ── PSD band features (deterministic per-trial; no fold fitting) ──────────
    BANDS = [(1., 4.), (4., 8.), (8., 13.), (13., 30.), (30., 45.)]

    def _extract_psd_bands(X: np.ndarray) -> np.ndarray:
        """(n_tr, n_ch, n_t) → (n_tr, n_ch×5) log-power band means."""
        n_tr, n_ch, n_t = X.shape
        Xf = X.reshape(n_tr * n_ch, n_t)
        nperseg = min(256, n_t)
        freqs, pxx = _welch(Xf, fs=sfreq, nperseg=nperseg)
        pxx_log = np.log1p(pxx).reshape(n_tr, n_ch, -1)
        feats = []
        for lo, hi in BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                feats.append(np.zeros((n_tr, n_ch), dtype=np.float32))
            else:
                feats.append(pxx_log[:, :, mask].mean(axis=-1))
        return np.concatenate(feats, axis=-1).astype(np.float32)  # (n_tr, n_ch*5)

    X_psd_raw = _extract_psd_bands(all_epochs)        # (n_trials, n_ch*5)
    n_bpe_feats = X_bpe.shape[1]

    # Ensemble: BPE (unscaled) ‖ PSD (StandardScaler per fold)
    X_combo = np.hstack([X_bpe, X_psd_raw])           # (n_trials, n_bpe+n_psd)

    # ── Per-condition classify functions ──────────────────────────────────────
    def _psd_classify(X_train, y_train, X_test, y_test):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_train)
        X_te = sc.transform(X_test)
        clf = LogisticRegression(
            max_iter=LOGREG_MAX_ITER, C=LOGREG_C, solver=LOGREG_SOLVER,
            class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
        )
        clf.fit(X_tr, y_train)
        return clf.predict(X_te), clf.predict_proba(X_te)

    def _ensemble_classify(X_train, y_train, X_test, y_test):
        # Scale PSD part within fold; BPE histograms are already in [0,1]
        X_bpe_tr = X_train[:, :n_bpe_feats]
        X_psd_tr = X_train[:, n_bpe_feats:]
        X_bpe_te = X_test[:, :n_bpe_feats]
        X_psd_te = X_test[:, n_bpe_feats:]
        sc = StandardScaler()
        X_psd_tr = sc.fit_transform(X_psd_tr)
        X_psd_te = sc.transform(X_psd_te)
        X_tr = np.hstack([X_bpe_tr, X_psd_tr])
        X_te = np.hstack([X_bpe_te, X_psd_te])
        clf = LogisticRegression(
            max_iter=LOGREG_MAX_ITER, C=LOGREG_C, solver=LOGREG_SOLVER,
            class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
        )
        clf.fit(X_tr, y_train)
        return clf.predict(X_te), clf.predict_proba(X_te)

    # ── Run each condition ────────────────────────────────────────────────────
    for X_feat, clf_fn, clf_name in [
        (X_bpe,   classify_histogram_logreg, "BPE_Hist_LogReg"),
        (X_psd_raw, _psd_classify,           "PSD_LogReg"),
        (X_combo,   _ensemble_classify,      "BPE_PSD_Ensemble"),
    ]:
        base = {"ablation": "K7", "dataset": ds_name,
                "classifier": clf_name, "vocab_size": vocab_size}
        if _all_seeds_done(exp_log.csv_path, base):
            exp_log.info(f"  K7 {clf_name}: all seeds done — skipping")
            continue

        rows = _parallel_seeds(
            X_feat, y, groups, clf_fn, info, base,
            csv_path=exp_log.csv_path,
        )
        for r in rows:
            exp_log.log_result(r)
        results.extend(rows)

    if results:
        save_csv(results, LOGS_DIR / "ablation_K7_bpe_psd_ensemble.csv")
        for clf_name in clf_names:
            acc = float(np.mean([r["accuracy_mean"] for r in results
                                 if r.get("classifier") == clf_name]))
            exp_log.info(f"K7 {clf_name}: acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A11: Histogram Transform Ablation (TF-IDF, log-freq, binary)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_hist_transform(X_hist: np.ndarray, transform: str) -> np.ndarray:
    """Apply histogram feature transform.

    Parameters
    ----------
    X_hist : np.ndarray
        L1-normalised histogram features (n_trials, n_features).
    transform : str
        One of "raw", "tfidf", "log", "binary".
    """
    if transform == "raw":
        return X_hist
    elif transform == "log":
        return np.log1p(X_hist * 1000)  # scale up before log to spread values
    elif transform == "binary":
        return (X_hist > 0).astype(np.float32)
    elif transform == "tfidf":
        # TF = X_hist (already L1-normalised = term frequency)
        # IDF = log(N / (1 + df))  where df = number of trials with token > 0
        n_trials = X_hist.shape[0]
        df = np.sum(X_hist > 0, axis=0).astype(np.float64)  # document frequency
        idf = np.log(n_trials / (1.0 + df)).astype(np.float32)
        return X_hist * idf[np.newaxis, :]
    else:
        raise ValueError(f"Unknown transform: {transform}")


@timed("ablations")
def run_a11(ds_name: str = "sleep_edf",
            vocab_size: int = 1024,
            n_bins: int = 64,
            method: str = "adaptive",
            max_subjects: int = 9) -> list[dict]:
    """
    A11: Histogram feature transform ablation.

    Compares raw L1-normalised histograms against TF-IDF, log-frequency,
    and binary (presence/absence) representations.
    """
    exp_log = ExperimentLogger("ablation_A11")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results
    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    vocab, X_hist_raw = _get_bpe_vocab_and_hist(
        ds_name, all_epochs, method, n_bins, vocab_size)

    transforms = ["raw", "tfidf", "log", "binary"]
    for transform in transforms:
        base_dict = {
            "ablation": "A11", "dataset": ds_name,
            "condition": transform, "vocab_size": vocab_size,
            "n_bins": n_bins, "method": method,
        }
        if _all_seeds_done(exp_log.csv_path, base_dict):
            exp_log.info(f"  A11 {transform}: all seeds done — skipping")
            continue

        X_transformed = _apply_hist_transform(X_hist_raw, transform)
        X_reduced = _reduce_hist_if_large(X_transformed)

        seed_results = _parallel_seeds(
            X_reduced, y, groups, classify_histogram_logreg,
            info, base_dict, csv_path=exp_log.csv_path)
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

        if seed_results:
            acc = np.mean([r["accuracy_mean"] for r in seed_results])
            exp_log.info(f"  A11 {transform}: acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A12: BPE Corpus Size Ablation
# ═══════════════════════════════════════════════════════════════════════════════

@timed("ablations")
def run_a12(ds_name: str = "sleep_edf",
            vocab_size: int = 1024,
            n_bins: int = 64,
            method: str = "adaptive",
            max_subjects: int = 9) -> list[dict]:
    """
    A12: Effect of BPE training corpus size.

    Trains BPE on {500, 1000, 2000, 5000, all} sequences and measures
    downstream accuracy to test vocabulary stability.
    """
    exp_log = ExperimentLogger("ablation_A12")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results
    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    # Get quantized sequences
    seq_key = (ds_name, method, n_bins)
    if seq_key not in _SEQ_CACHE:
        _SEQ_CACHE[seq_key] = _epochs_to_seqs(all_epochs, method, n_bins)
    all_seqs = _SEQ_CACHE[seq_key]

    corpus_sizes = [500, 1000, 2000, 5000, len(all_seqs)]
    corpus_labels = ["500", "1000", "2000", "5000", "all"]

    for n_corpus, label in zip(corpus_sizes, corpus_labels):
        n_corpus = min(n_corpus, len(all_seqs))
        base_dict = {
            "ablation": "A12", "dataset": ds_name,
            "condition": f"corpus_{label}", "corpus_size": n_corpus,
            "vocab_size": vocab_size, "n_bins": n_bins, "method": method,
        }
        if _all_seeds_done(exp_log.csv_path, base_dict):
            exp_log.info(f"  A12 corpus={label}: all seeds done — skipping")
            continue

        # Train vocab with limited corpus
        train_seqs = all_seqs[:n_corpus]
        vocab_path = (MODELS_DIR /
                      f"ablation_vocab_{ds_name}_{method}_B{n_bins}"
                      f"_V{vocab_size}_corpus{n_corpus}.json")
        if vocab_path.exists():
            vocab = BPEVocab.load(str(vocab_path))
        else:
            vocab = train_bpe(train_seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))

        # Build histograms with this vocab on ALL data
        X_hist = cached_epochs_to_bpe_histograms(all_epochs, vocab, method, n_bins)
        X_hist = _reduce_hist_if_large(X_hist)

        seed_results = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg,
            info, base_dict, csv_path=exp_log.csv_path)
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

        if seed_results:
            acc = np.mean([r["accuracy_mean"] for r in seed_results])
            exp_log.info(f"  A12 corpus={label}: acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A13: Window Size Ablation (temporal resolution of windowed histograms)
# ═══════════════════════════════════════════════════════════════════════════════

@timed("ablations")
def run_a13(ds_name: str = "sleep_edf",
            vocab_size: int = 1024,
            n_bins: int = 64,
            method: str = "adaptive",
            max_subjects: int = 9) -> list[dict]:
    """
    A13: Window size for windowed BPE histograms.

    Tests n_windows ∈ {2, 5, 10, 20} (corresponding to different temporal
    resolutions depending on epoch length).
    """
    from .exp2_downstream import epochs_to_windowed_bpe_histograms

    exp_log = ExperimentLogger("ablation_A13")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results
    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    # Get vocab (shared across window sizes)
    vocab, _ = _get_bpe_vocab_and_hist(
        ds_name, all_epochs, method, n_bins, vocab_size)

    sfreq = info.get("sfreq", 250)
    n_time = all_epochs.shape[-1]
    epoch_sec = n_time / sfreq

    n_windows_list = [2, 5, 10, 20]

    for n_windows in n_windows_list:
        win_sec = round(epoch_sec / n_windows, 2)
        # Skip if window would be < 10 samples
        win_samples = n_time // n_windows
        if win_samples < 10:
            exp_log.info(f"  A13 n_win={n_windows}: window too short "
                        f"({win_samples} samples) — skipping")
            continue

        base_dict = {
            "ablation": "A13", "dataset": ds_name,
            "condition": f"nwin_{n_windows}",
            "n_windows": n_windows, "window_sec": win_sec,
            "vocab_size": vocab_size, "n_bins": n_bins, "method": method,
        }
        if _all_seeds_done(exp_log.csv_path, base_dict):
            exp_log.info(f"  A13 n_win={n_windows}: all seeds done — skipping")
            continue

        X_win = epochs_to_windowed_bpe_histograms(
            all_epochs, vocab, method, n_bins, n_windows=n_windows)
        X_win = _reduce_hist_if_large(X_win)

        seed_results = _parallel_seeds(
            X_win, y, groups, classify_histogram_logreg,
            info, base_dict, csv_path=exp_log.csv_path)
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

        if seed_results:
            acc = np.mean([r["accuracy_mean"] for r in seed_results])
            exp_log.info(f"  A13 n_win={n_windows} ({win_sec}s): acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A14: Normalization Strategy Ablation
# ═══════════════════════════════════════════════════════════════════════════════

def _quantize_with_norm(epochs: np.ndarray, method: str, n_bins: int,
                        norm_strategy: str) -> np.ndarray:
    """Quantize with different normalization strategies.

    Parameters
    ----------
    norm_strategy : str
        "global_zscore" — z-score over all samples (current default)
        "per_channel"   — z-score per channel (across trials)
        "per_trial"     — z-score per trial (across channels)
        "robust"        — median/IQR scaling per channel
        "none"          — no normalization
    """
    n_trials, n_ch, n_time = epochs.shape

    if norm_strategy == "global_zscore":
        # Current default: quantize(normalize=True) does per-row z-score
        flat = epochs.reshape(-1, n_time)
        codes, _ = quantize(flat, method, n_bins, normalize=True)
        return codes

    elif norm_strategy == "per_channel":
        # Z-score per channel across ALL trials: each channel gets its own μ,σ
        # shape: compute stats over (n_trials, n_time) for each channel
        normalized = np.empty_like(epochs)
        for ch in range(n_ch):
            ch_data = epochs[:, ch, :]  # (n_trials, n_time)
            mu = ch_data.mean()
            sigma = max(ch_data.std(), 1e-8)
            normalized[:, ch, :] = (ch_data - mu) / sigma
        flat = normalized.reshape(-1, n_time)
        codes, _ = quantize(flat, method, n_bins, normalize=False)
        return codes

    elif norm_strategy == "per_trial":
        # Z-score per trial across all channels
        normalized = np.empty_like(epochs)
        for t in range(n_trials):
            trial = epochs[t]  # (n_ch, n_time)
            mu = trial.mean()
            sigma = max(trial.std(), 1e-8)
            normalized[t] = (trial - mu) / sigma
        flat = normalized.reshape(-1, n_time)
        codes, _ = quantize(flat, method, n_bins, normalize=False)
        return codes

    elif norm_strategy == "robust":
        # Median/IQR per channel-row (robust to outliers/artifacts)
        flat = epochs.reshape(-1, n_time)
        medians = np.median(flat, axis=-1, keepdims=True)
        q75 = np.percentile(flat, 75, axis=-1, keepdims=True)
        q25 = np.percentile(flat, 25, axis=-1, keepdims=True)
        iqr = np.maximum(q75 - q25, 1e-8)
        normalized = (flat - medians) / iqr
        codes, _ = quantize(normalized, method, n_bins, normalize=False)
        return codes

    elif norm_strategy == "none":
        flat = epochs.reshape(-1, n_time)
        codes, _ = quantize(flat, method, n_bins, normalize=False)
        return codes

    else:
        raise ValueError(f"Unknown norm_strategy: {norm_strategy}")


@timed("ablations")
def run_a14(ds_name: str = "sleep_edf",
            vocab_size: int = 1024,
            n_bins: int = 64,
            method: str = "adaptive",
            max_subjects: int = 9) -> list[dict]:
    """
    A14: Normalization strategy ablation.

    Tests global_zscore (default), per_channel, per_trial, robust, and none.
    Full pipeline: normalize → quantize → BPE → histogram → LogReg.
    """
    exp_log = ExperimentLogger("ablation_A14")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results
    all_epochs, all_labels, all_groups = _assemble_data(data)
    y = LabelEncoder().fit_transform(np.array(all_labels))
    groups = np.array(all_groups)
    info = DATASET_INFO[ds_name]

    norm_strategies = ["global_zscore", "per_channel", "per_trial", "robust", "none"]

    for norm in norm_strategies:
        base_dict = {
            "ablation": "A14", "dataset": ds_name,
            "condition": norm, "vocab_size": vocab_size,
            "n_bins": n_bins, "method": method,
        }
        if _all_seeds_done(exp_log.csv_path, base_dict):
            exp_log.info(f"  A14 {norm}: all seeds done — skipping")
            continue

        # Quantize with chosen normalization
        codes = _quantize_with_norm(all_epochs, method, n_bins, norm)
        seqs = [codes[i].tolist() for i in range(len(codes))]

        # Train BPE vocab on these sequences
        vocab_path = (MODELS_DIR /
                      f"ablation_vocab_{ds_name}_{method}_B{n_bins}"
                      f"_V{vocab_size}_norm_{norm}.json")
        if vocab_path.exists():
            vocab = BPEVocab.load(str(vocab_path))
        else:
            train_seqs = seqs[:min(5000, len(seqs))]
            vocab = train_bpe(train_seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))

        # Apply BPE and build histograms
        bpe_seqs = apply_bpe_batch(seqs, vocab, n_jobs=N_JOBS)
        n_trials, n_ch = all_epochs.shape[0], all_epochs.shape[1]
        V = vocab.vocab_size
        n_total = n_trials * n_ch

        # Vectorised histogram
        if bpe_seqs:
            lengths = np.array([len(s) for s in bpe_seqs], dtype=np.int32)
            max_len = int(lengths.max()) if lengths.size > 0 else 0
        else:
            max_len = 0
        if max_len > 0:
            padded = np.full((n_total, max_len), V, dtype=np.int32)
            for i, seq in enumerate(bpe_seqs):
                if seq:
                    padded[i, :len(seq)] = seq[:len(seq)]
            valid = padded < V
            row_idx, col_pos = np.where(valid)
            tok_vals = padded[row_idx, col_pos]
            flat_idx = row_idx.astype(np.int64) * V + tok_vals.astype(np.int64)
            counts = np.bincount(flat_idx, minlength=n_total * V)
            histograms = counts[:n_total * V].reshape(
                n_trials, n_ch, V).astype(np.float32)
        else:
            histograms = np.zeros((n_trials, n_ch, V), dtype=np.float32)

        sums = histograms.sum(axis=-1, keepdims=True)
        histograms = histograms / (sums + 1e-10)
        X_hist = histograms.reshape(n_trials, -1)
        X_hist = _reduce_hist_if_large(X_hist)

        seed_results = _parallel_seeds(
            X_hist, y, groups, classify_histogram_logreg,
            info, base_dict, csv_path=exp_log.csv_path)
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

        if seed_results:
            acc = np.mean([r["accuracy_mean"] for r in seed_results])
            exp_log.info(f"  A14 {norm}: acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# A15: Epoch Segmentation Ablation (sub-epoch lengths)
# ═══════════════════════════════════════════════════════════════════════════════

def _segment_epochs(epochs: np.ndarray, labels: np.ndarray,
                    groups: np.ndarray, sfreq: float,
                    sub_epoch_sec: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each epoch into non-overlapping sub-epochs.

    If sub_epoch_sec >= epoch length, returns original data unchanged.
    Each sub-epoch inherits the parent's label and group.
    """
    n_trials, n_ch, n_time = epochs.shape
    epoch_sec = n_time / sfreq
    if sub_epoch_sec >= epoch_sec:
        return epochs, labels, groups

    sub_samples = int(sub_epoch_sec * sfreq)
    n_sub = n_time // sub_samples  # integer division, discard remainder

    sub_epochs = []
    sub_labels = []
    sub_groups = []
    for t in range(n_trials):
        for s in range(n_sub):
            start = s * sub_samples
            sub_epochs.append(epochs[t, :, start:start + sub_samples])
            sub_labels.append(labels[t])
            sub_groups.append(groups[t])

    return (np.array(sub_epochs), np.array(sub_labels), np.array(sub_groups))


@timed("ablations")
def run_a15(ds_name: str = "sleep_edf",
            vocab_size: int = 1024,
            n_bins: int = 64,
            method: str = "adaptive",
            max_subjects: int = 9) -> list[dict]:
    """
    A15: Epoch segmentation ablation.

    Tests different sub-epoch lengths to find optimal temporal granularity.
    Sleep-EDF (30s): tests 5s, 10s, 15s, 30s.
    MI (4s): tests 1s, 2s, 4s.
    """
    exp_log = ExperimentLogger("ablation_A15")
    results = []

    data = load_dataset(ds_name, max_subjects=max_subjects)
    if not data:
        return results
    all_epochs, all_labels, all_groups = _assemble_data(data)
    y_orig = LabelEncoder().fit_transform(np.array(all_labels))
    groups_orig = np.array(all_groups)
    info = DATASET_INFO[ds_name]
    sfreq = info.get("sfreq", 250)
    epoch_sec = all_epochs.shape[-1] / sfreq

    # Choose sub-epoch lengths based on epoch duration
    if epoch_sec >= 25:  # Sleep-like (30s)
        sub_secs = [5.0, 10.0, 15.0, epoch_sec]
    elif epoch_sec >= 3:   # MI-like (4s)
        sub_secs = [1.0, 2.0, epoch_sec]
    else:
        sub_secs = [epoch_sec]

    for sub_sec in sub_secs:
        label = f"{sub_sec:.0f}s" if sub_sec == int(sub_sec) else f"{sub_sec:.1f}s"
        if sub_sec >= epoch_sec:
            label = f"full_{label}"

        base_dict = {
            "ablation": "A15", "dataset": ds_name,
            "condition": label, "sub_epoch_sec": sub_sec,
            "vocab_size": vocab_size, "n_bins": n_bins, "method": method,
        }
        if _all_seeds_done(exp_log.csv_path, base_dict):
            exp_log.info(f"  A15 {label}: all seeds done — skipping")
            continue

        # Segment epochs
        seg_epochs, seg_labels, seg_groups = _segment_epochs(
            all_epochs, y_orig, groups_orig, sfreq, sub_sec)

        exp_log.info(f"  A15 {label}: {seg_epochs.shape[0]} segments "
                    f"({seg_epochs.shape[-1]} samples each)")

        # Train or reuse BPE vocab (train on segmented data)
        vocab_path = (MODELS_DIR /
                      f"ablation_vocab_{ds_name}_{method}_B{n_bins}"
                      f"_V{vocab_size}_seg_{label}.json")
        if vocab_path.exists():
            vocab = BPEVocab.load(str(vocab_path))
        else:
            flat = seg_epochs.reshape(-1, seg_epochs.shape[-1])
            codes, _ = quantize(flat, method, n_bins, normalize=True)
            train_seqs = [codes[i].tolist() for i in range(min(5000, len(codes)))]
            vocab = train_bpe(train_seqs, vocab_size=vocab_size,
                              base_vocab_size=n_bins, verbose=False,
                              max_train_tokens=5_000_000)
            vocab.save(str(vocab_path))

        # Build histograms
        X_hist = epochs_to_bpe_histograms(seg_epochs, vocab, method, n_bins)
        X_hist = _reduce_hist_if_large(X_hist)

        seed_results = _parallel_seeds(
            X_hist, seg_labels, seg_groups, classify_histogram_logreg,
            info, base_dict, csv_path=exp_log.csv_path)
        for r in seed_results:
            exp_log.log_result(r)
        results.extend(seed_results)

        if seed_results:
            acc = np.mean([r["accuracy_mean"] for r in seed_results])
            exp_log.info(f"  A15 {label}: acc={acc:.3f}")

    exp_log.finalize()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plots for A11-A15
# ═══════════════════════════════════════════════════════════════════════════════

def _plot_a11_a15(all_results: list[dict], plot_dir: Path) -> None:
    """Generate plots for new ablations A11-A15."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    ablation_groups = {}
    for r in all_results:
        abl = r.get("ablation", "")
        if abl.startswith("A1") and abl in ("A11", "A12", "A13", "A14", "A15"):
            ablation_groups.setdefault(abl, []).append(r)

    configs = {
        "A11": ("Histogram Transform", "condition"),
        "A12": ("BPE Corpus Size", "condition"),
        "A13": ("Window Count", "condition"),
        "A14": ("Normalization Strategy", "condition"),
        "A15": ("Epoch Segmentation", "condition"),
    }

    for abl, (title, x_field) in configs.items():
        data = ablation_groups.get(abl, [])
        if not data:
            continue

        # Group by (dataset, condition) → mean accuracy
        import pandas as pd
        df = pd.DataFrame(data)
        datasets = sorted(df["dataset"].unique())

        fig, ax = plt.subplots(figsize=(max(8, len(df[x_field].unique()) * 1.5), 5))
        conditions = sorted(df[x_field].unique(), key=str)
        x = np.arange(len(conditions))
        n_ds = len(datasets)
        bar_w = 0.7 / max(n_ds, 1)
        colors = plt.cm.Set2(np.linspace(0, 1, max(n_ds, 2)))

        for di, ds in enumerate(datasets):
            ds_df = df[df["dataset"] == ds]
            means, stds = [], []
            for cond in conditions:
                cond_df = ds_df[ds_df[x_field] == cond]
                means.append(cond_df["accuracy_mean"].mean() if len(cond_df) > 0 else 0)
                stds.append(cond_df["accuracy_mean"].std() if len(cond_df) > 1 else 0)
            ax.bar(x + di * bar_w, means, bar_w, yerr=stds,
                   label=ds, color=colors[di], edgecolor="white",
                   capsize=3)

        ax.set_xticks(x + bar_w * n_ds / 2)
        ax.set_xticklabels([str(c) for c in conditions], fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Ablation {abl}: {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(plot_dir / f"ablation_{abl}_{title.replace(' ', '_').lower()}.png",
                    dpi=150)
        fig.savefig(plot_dir / f"ablation_{abl}_{title.replace(' ', '_').lower()}.pdf")
        plt.close(fig)
        logger.info(f"  Saved {abl} plot")


# ─── Cross-ablation summary plot ─────────────────────────────────────────────

def _plot_ablation_summary(output_dir: Path) -> None:
    """Horizontal bar chart showing accuracy range (min to max) per ablation.

    Reads all ``ablation_A*_results.csv`` files from LOGS_DIR.  For each
    ablation, extracts the best and worst condition by ``accuracy_mean``
    and draws a horizontal span from min to max.  Robust: silently skips
    ablations whose CSV is missing or malformed.
    """
    try:
        import glob as _glob
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        # Collect per-ablation min/max accuracy
        # Try individual CSVs first, then fall back to the combined CSV
        ablation_ranges: dict[str, tuple[float, float]] = {}

        # -- individual CSVs: ablation_A*_results.csv
        pattern = str(LOGS_DIR / "ablation_A*_results.csv")
        for csv_path in sorted(_glob.glob(pattern)):
            try:
                df = pd.read_csv(csv_path)
                if "accuracy_mean" not in df.columns:
                    continue
                label = Path(csv_path).stem.replace("ablation_", "").replace("_results", "")
                ablation_ranges[label] = (
                    float(df["accuracy_mean"].min()),
                    float(df["accuracy_mean"].max()),
                )
            except Exception:
                continue

        # -- combined CSV: ablation_all_results.csv (fills gaps)
        all_csv = LOGS_DIR / "ablation_all_results.csv"
        if all_csv.exists():
            try:
                df_all = pd.read_csv(all_csv)
                if "ablation" in df_all.columns and "accuracy_mean" in df_all.columns:
                    for abl, grp in df_all.groupby("ablation"):
                        label = str(abl)
                        if label not in ablation_ranges:
                            ablation_ranges[label] = (
                                float(grp["accuracy_mean"].min()),
                                float(grp["accuracy_mean"].max()),
                            )
            except Exception:
                pass

        if not ablation_ranges:
            logger.info("No ablation CSVs found — skipping summary plot")
            return

        # Sort by ablation name
        labels = sorted(ablation_ranges.keys())
        mins = [ablation_ranges[l][0] for l in labels]
        maxs = [ablation_ranges[l][1] for l in labels]

        fig, ax = plt.subplots(figsize=(8, max(3, len(labels) * 0.5 + 1)))
        y_pos = np.arange(len(labels))

        # Draw horizontal bars spanning from min to max accuracy
        bar_heights = 0.5
        for i, (lo, hi) in enumerate(zip(mins, maxs)):
            ax.barh(i, hi - lo, left=lo, height=bar_heights,
                    color="#4472C4", alpha=0.75, edgecolor="white")
            # Annotate min and max values
            ax.text(lo - 0.01, i, f"{lo:.1%}", va="center", ha="right",
                    fontsize=7, color="#666666")
            ax.text(hi + 0.01, i, f"{hi:.1%}", va="center", ha="left",
                    fontsize=7, color="#333333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Accuracy")
        ax.set_title("Ablation Summary: Accuracy Range per Ablation")
        ax.set_xlim(
            max(0, min(mins) - 0.08),
            min(1.0, max(maxs) + 0.08),
        )
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()  # top-to-bottom order
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "ablation_summary.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved ablation summary plot -> {save_path}")

    except Exception as exc:
        logger.warning(f"Ablation summary plot failed: {exc}")


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("ablations")
def run_all_ablations(ds_name: str = "bci_iv_2a",
                      max_subjects: int = 9,
                      vocab_size: int | None = None,
                      n_bins: int = 64,
                      ablation_datasets: list | None = None) -> dict:
    """
    Run all ablation studies A1, A2, A3, A4, A5, A7, A8.

    A1, A2, A4, A5, A7, A8 run on *ablation_datasets* (default:
    ``["bci_iv_2a", "sleep_edf"]``).  Running on both a near-chance
    dataset (bci_iv_2a) and a high-accuracy dataset (sleep_edf) is
    important: differences between ablation conditions are only
    interpretable when the absolute accuracy is well above chance.

    A3 (frequency-aware strategy) runs on *ds_name* only (default
    ``"bci_iv_2a"``; switch to ``"mental_arithmetic"`` for high-sfreq
    data).

    Parameters
    ----------
    ds_name : str
        Dataset for A3 (frequency-aware ablation).
    max_subjects : int
        Maximum number of subjects to load per dataset.
    vocab_size : int or None
        BPE vocabulary size override. If None, each ablation uses its default.
    n_bins : int
        Quantisation bins.
    ablation_datasets : list of str or None
        Datasets for A1, A2, A4, A5, A7. Default: ``["bci_iv_2a", "sleep_edf"]``.

    Returns
    -------
    dict
        Summary with result count and ablation names.
    """
    if ablation_datasets is None:
        ablation_datasets = ["bci_iv_2a", "sleep_edf"]

    logger.info(f"=== Running Ablation Studies on: {ablation_datasets} ===")
    vs_kw   = {"vocab_size": vocab_size} if vocab_size is not None else {}
    bins_kw = {"n_bins": n_bins}

    a1, a2, a4, a5, a7, a8, a9 = [], [], [], [], [], [], []
    for _ds in ablation_datasets:
        a1 += run_a1(ds_name=_ds, max_subjects=max_subjects, **vs_kw, **bins_kw)
        a2 += run_a2(ds_name=_ds, max_subjects=max_subjects, **vs_kw)
        a4 += run_a4(ds_name=_ds, max_subjects=max_subjects, **vs_kw, **bins_kw)
        a5 += run_a5(ds_name=_ds, max_subjects=max_subjects, **vs_kw, **bins_kw)
        a7 += run_a7(ds_name=_ds, max_subjects=max_subjects, **vs_kw, **bins_kw)
        a8 += run_a8(ds_name=_ds, max_subjects=max_subjects, **vs_kw, **bins_kw)
        a9 += run_a9(ds_name=_ds, max_subjects=max_subjects)

    # A3 is paradigm-specific (frequency-aware); keep on single dataset
    a3 = run_a3(ds_name=ds_name, max_subjects=max_subjects, **vs_kw, **bins_kw)

    # A10: PhysioNet-MI scale ablation (always physionet_mi, fixed conditions)
    a10 = run_a10_physionet_scale(**{k: v for k, v in {**vs_kw, **bins_kw}.items()
                                      if k in ("vocab_size", "n_bins")})

    # K7: BPE+PSD ensemble on mental_arithmetic
    k7 = run_k7_bpe_psd_ensemble(**{k: v for k, v in {**vs_kw, **bins_kw}.items()
                                     if k in ("vocab_size", "n_bins")})

    # A11-A15: New ablations (run on ablation_datasets)
    _new_vs = vocab_size if vocab_size is not None else 1024
    _new_method = DEFAULT_QUANT_METHOD
    a11, a12, a13, a14, a15 = [], [], [], [], []
    for _ds in ablation_datasets:
        a11 += run_a11(ds_name=_ds, max_subjects=max_subjects,
                       vocab_size=_new_vs, n_bins=n_bins, method=_new_method)
        a12 += run_a12(ds_name=_ds, max_subjects=max_subjects,
                       vocab_size=_new_vs, n_bins=n_bins, method=_new_method)
        a13 += run_a13(ds_name=_ds, max_subjects=max_subjects,
                       vocab_size=_new_vs, n_bins=n_bins, method=_new_method)
        a14 += run_a14(ds_name=_ds, max_subjects=max_subjects,
                       vocab_size=_new_vs, n_bins=n_bins, method=_new_method)
        a15 += run_a15(ds_name=_ds, max_subjects=max_subjects,
                       vocab_size=_new_vs, n_bins=n_bins, method=_new_method)

    all_results = (a1 + a2 + a3 + a4 + a5 + a7 + a8 + a9 + a10 + k7
                   + a11 + a12 + a13 + a14 + a15)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_dir = PLOTS_DIR / "ablations"
    multi = len(ablation_datasets) > 1

    if a1:
        plot_ablation_bar(a1, "A1_quant_method", "method", plot_dir)
        if multi:
            plot_ablation_grouped_datasets(a1, "A1_quant_method", "method", plot_dir)
    if a2:
        # Plot each condition separately to avoid averaging across confounded conditions
        for cond in ("proportional_V", "fixed_V"):
            a2_cond = [r for r in a2 if r.get("condition") == cond]
            if a2_cond:
                plot_ablation_bar(a2_cond, f"A2_n_bins_{cond}", "n_bins", plot_dir)
                if multi:
                    plot_ablation_grouped_datasets(
                        a2_cond, f"A2_n_bins_{cond}", "n_bins", plot_dir)
        # Also plot combined (original behaviour, for backwards compat)
        plot_ablation_bar(a2, "A2_n_bins", "n_bins", plot_dir)
    if a3:
        plot_ablation_bar(a3, "A3_freq_strategy", "condition", plot_dir)
    if a4:
        plot_ablation_bar(a4, "A4_channel_strategy", "condition", plot_dir)
        if multi:
            plot_ablation_grouped_datasets(a4, "A4_channel_strategy", "condition", plot_dir)
    if a5:
        plot_ablation_bar(a5, "A5_artifact_tokens", "condition", plot_dir)
        if multi:
            plot_ablation_grouped_datasets(a5, "A5_artifact_tokens", "condition", plot_dir)
    if a7:
        # Grouped chart so Hist_LogReg vs Seq_CNN are compared side-by-side
        plot_a7_grouped(a7, plot_dir)
        hist_only = [r for r in a7 if r.get("classifier", "Hist_LogReg") == "Hist_LogReg"]
        if hist_only:
            plot_ablation_bar(hist_only, "A7_bpe_vs_nobpe", "condition", plot_dir)
            if multi:
                plot_ablation_grouped_datasets(
                    hist_only, "A7_bpe_vs_nobpe", "condition", plot_dir
                )
    if a8:
        save_csv(a8, LOGS_DIR / "ablation_A8_results.csv")
        plot_ablation_bar(a8, "A8_discriminative_bpe", "condition", plot_dir)
        if multi:
            plot_ablation_grouped_datasets(a8, "A8_discriminative_bpe", "condition", plot_dir)
    if a9:
        save_csv(a9, LOGS_DIR / "ablation_A9_results.csv")
        _plot_a9_heatmap(a9, plot_dir)

    # A11-A15 plots
    new_abl_results = a11 + a12 + a13 + a14 + a15
    if new_abl_results:
        _plot_a11_a15(new_abl_results, plot_dir)
        for abl_name, abl_data in [("A11", a11), ("A12", a12), ("A13", a13),
                                    ("A14", a14), ("A15", a15)]:
            if abl_data:
                save_csv(abl_data, LOGS_DIR / f"ablation_{abl_name}_results.csv")

    save_json(all_results, LOGS_DIR / "ablation_all_results.json")
    save_csv(all_results, LOGS_DIR / "ablation_all_results.csv")

    _plot_ablation_summary(plot_dir)

    logger.info(f"Ablations complete: {len(all_results)} results")
    return {
        "n_results":       len(all_results),
        "ablations_run":   ["A1", "A2", "A3", "A4", "A5", "A7", "A8", "A9", "A10", "K7",
                            "A11", "A12", "A13", "A14", "A15"],
        "datasets":        ablation_datasets,
    }


if __name__ == "__main__":
    run_all_ablations()
