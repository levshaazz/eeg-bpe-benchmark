"""
Experiment 9: Spatial BPE
==========================
Fixes the channel-independence bottleneck identified in Ablation A4.

Standard BPE treats each EEG channel as an independent 1-D sequence.
A4 showed that encoding the whole-brain spatial state at each timestep as
a single token (Spatial VQ) improves Sleep-EDF accuracy from 54.3% to 83.4%
(+29 pp).  Standard BPE misses this cross-channel spatial structure entirely.

Spatial BPE pipeline (per fold, no leakage):
  1. Downsample to target_sfreq (default 64 Hz) to manage sequence length.
  2. Quantise amplitude per channel  →  (n_trials, n_ch, n_times) bin codes.
  3. At each timestep t: spatial snapshot = (bin_ch1, bin_ch2, ..., bin_chN).
  4. Fit k-means codebook on TRAIN spatial snapshots  →  n_spatial_codes.
  5. Encode ALL timesteps as spatial codes  →  (n_trials, n_times) sequences.
  6. Train BPE on TRAIN spatial-code sequences  →  BPEVocab.
  7. Apply BPE  →  per-trial histogram (n_trials, V).
  8. Classify with LogReg and RF.

Key difference from Exp 7 (Fourier-BPE):
  • Input to codebook: quantised amplitude per channel, NOT STFT log-power.
  • No frequency-axis normalisation  →  absolute amplitude differences
    preserved  →  avoids the VQ normalisation paradox that hurt Exp 7.
  • One sequence per TRIAL (not per channel), because the spatial code
    already encodes the whole brain at each timestep.

Key difference from A4 ablation:
  • Strict per-fold codebook + vocab training (no data leakage).
  • Both LogReg and RF classifiers reported.
  • All 6 datasets.
  • Disk-cached histograms.

Outputs
-------
results/logs/exp9_spatial_bpe_results.csv
results/logs/exp9_per_fold.csv
results/plots/exp9/exp9_accuracy_bar.png
results/models/spatial_codebook_C{n_codes}_{ds}.pkl  (per-fold not cached)
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from joblib import Parallel, delayed

from .config import (
    DATASET_INFO, RANDOM_SEEDS, LOGS_DIR, PLOTS_DIR, MODELS_DIR, CACHE_DIR,
    N_JOBS,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
    DEFAULT_QUANT_METHOD,
)
from .data_loading import load_dataset
from .quantization import quantize
from .bpe_engine import train_bpe, apply_bpe_batch, BPEVocab
from .utils import (
    ExperimentLogger, save_csv, append_csv, get_logger, timed, bootstrap_ci,
)

logger = get_logger("exp9_spatial_bpe")

# ─── Constants ────────────────────────────────────────────────────────────────

_TARGET_SFREQ      = 64.0    # downsample all datasets to this before k-means
_N_SPATIAL_CODES   = 64      # k-means codebook size (analogous to n_bins)
_MAX_KM_SAMPLES    = 200_000 # max spatial snapshots used for k-means fitting
_MAX_HIST_FEATURES = 2048    # SVD reduction threshold (only affects RF)
_N_BOOTSTRAP       = 500
_ALL_DATASETS      = [
    "bci_iv_2a", "physionet_mi", "sleep_edf",
    "mental_arithmetic", "epfl_p300", "ssvep_nakanishi",
]


# ─── Downsampling helper ───────────────────────────────────────────────────────

def _maybe_downsample(epochs: np.ndarray, sfreq: float,
                      target_sfreq: float) -> np.ndarray:
    """Simple decimate (integer factor) to target_sfreq, or return as-is."""
    if sfreq <= target_sfreq:
        return epochs
    factor = max(1, round(sfreq / target_sfreq))
    return epochs[:, :, ::factor]


# ─── Spatial codebook fitting ─────────────────────────────────────────────────

def fit_spatial_codebook(
    epochs_ds: np.ndarray,          # (n_tr, n_ch, n_times)  downsampled
    method: str,
    n_bins: int,
    n_codes: int = _N_SPATIAL_CODES,
    random_state: int = 42,
    max_samples: int = _MAX_KM_SAMPLES,
) -> MiniBatchKMeans:
    """Fit k-means over spatial amplitude snapshots from TRAIN epochs.

    Each spatial snapshot is a vector (bin_ch1, bin_ch2, ..., bin_chN) of
    quantised amplitude values — one snapshot per timestep.  The codebook
    maps spatial brain-states to discrete codes.

    Parameters
    ----------
    epochs_ds   : (n_tr, n_ch, n_times) — already downsampled to target_sfreq
    method / n_bins : quantisation settings
    n_codes     : k-means cluster count (analogous to n_bins in channel BPE)

    Returns
    -------
    Fitted MiniBatchKMeans model.
    """
    n_tr, n_ch, n_times = epochs_ds.shape

    # Quantise
    flat = epochs_ds.reshape(-1, n_times)
    codes_flat, _ = quantize(flat, method=method, n_bins=n_bins, normalize=True)
    codes = codes_flat.reshape(n_tr, n_ch, n_times)   # (n_tr, n_ch, n_times)

    # Spatial snapshots: (n_tr * n_times, n_ch)
    snapshots = codes.transpose(0, 2, 1).reshape(-1, n_ch).astype(np.float32)

    # Sub-sample for k-means if very large
    if len(snapshots) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(snapshots), max_samples, replace=False)
        train_snaps = snapshots[idx]
    else:
        train_snaps = snapshots

    logger.debug(
        f"  Fitting spatial codebook: {train_snaps.shape[0]:,} snapshots,"
        f" n_ch={n_ch}, n_codes={n_codes}"
    )
    km = MiniBatchKMeans(
        n_clusters=n_codes,
        batch_size=min(10_000, max(n_codes * 10, len(train_snaps))),
        n_init=5, max_iter=200, tol=1e-4,
        random_state=random_state,
    )
    km.fit(train_snaps)
    logger.info(f"  Spatial codebook: inertia={km.inertia_:.1f}, "
                f"k={n_codes}, samples={len(train_snaps)}")
    return km


def encode_spatial_codes(
    epochs_ds: np.ndarray,       # (n_tr, n_ch, n_times)
    km: MiniBatchKMeans,
    method: str,
    n_bins: int,
) -> np.ndarray:
    """Encode epochs as spatial-code sequences.

    Returns
    -------
    spatial_codes : (n_tr, n_times) int32  in {0 … n_codes-1}
    """
    n_tr, n_ch, n_times = epochs_ds.shape
    flat = epochs_ds.reshape(-1, n_times)
    codes_flat, _ = quantize(flat, method=method, n_bins=n_bins, normalize=True)
    codes = codes_flat.reshape(n_tr, n_ch, n_times)
    snapshots = codes.transpose(0, 2, 1).reshape(-1, n_ch).astype(np.float32)
    spatial_codes = km.predict(snapshots).astype(np.int32)
    return spatial_codes.reshape(n_tr, n_times)


# ─── BPE on spatial codes ─────────────────────────────────────────────────────

def spatial_bpe_histograms(
    spatial_codes: np.ndarray,     # (n_tr, n_times)
    vocab: BPEVocab,
) -> np.ndarray:
    """Compute L1-normalised BPE histograms over spatial-code sequences.

    Returns
    -------
    X : (n_tr, vocab_size) float32
    """
    n_tr, _ = spatial_codes.shape
    V = vocab.vocab_size
    seqs = [spatial_codes[i].tolist() for i in range(n_tr)]
    tokenised = apply_bpe_batch(seqs, vocab, n_jobs=N_JOBS)

    X = np.zeros((n_tr, V), dtype=np.float32)
    for i, toks in enumerate(tokenised):
        for tok in toks:
            if 0 <= tok < V:
                X[i, tok] += 1.0
    row_sums = X.sum(axis=1, keepdims=True) + 1e-12
    X /= row_sums
    return X                       # (n_tr, V)


# ─── Disk-cached spatial BPE histograms ───────────────────────────────────────

_SPATIAL_CACHE_DIR = CACHE_DIR / "spatial_hist_cache"


def _spatial_cache_key(
    epochs: np.ndarray, km: MiniBatchKMeans, vocab: BPEVocab,
    method: str, n_bins: int, target_sfreq: float,
) -> str:
    h = hashlib.md5()
    h.update(epochs.shape.__repr__().encode())
    h.update(epochs.ravel()[:2000].tobytes())
    h.update(km.cluster_centers_.tobytes())
    h.update(str(vocab.merges[:50]).encode())
    h.update(f"{method}_{n_bins}_{target_sfreq}".encode())
    return h.hexdigest()[:20]


def cached_spatial_bpe_histograms(
    epochs: np.ndarray,
    km: MiniBatchKMeans,
    vocab: BPEVocab,
    sfreq: float,
    method: str,
    n_bins: int,
    target_sfreq: float = _TARGET_SFREQ,
) -> np.ndarray:
    """Spatial BPE histograms with disk cache."""
    _SPATIAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = _spatial_cache_key(epochs, km, vocab, method, n_bins, target_sfreq)
    cache_path = _SPATIAL_CACHE_DIR / f"{key}.npz"

    if cache_path.exists():
        logger.debug(f"Spatial BPE hist cache hit: {cache_path.name}")
        return np.load(str(cache_path))["X"]

    epochs_ds = _maybe_downsample(epochs, sfreq, target_sfreq)
    sc = encode_spatial_codes(epochs_ds, km, method, n_bins)
    X = spatial_bpe_histograms(sc, vocab)
    np.savez_compressed(str(cache_path), X=X)
    return X


# ─── Per-fold runner ──────────────────────────────────────────────────────────

def _run_fold(
    X_tr_raw: np.ndarray, y_tr: np.ndarray,
    X_te_raw: np.ndarray, y_te: np.ndarray,
    sfreq: float, vocab_size: int, n_bins: int, method: str,
    n_codes: int, seed: int,
) -> dict:
    """Run one CV fold: fit spatial codebook on train, encode both, classify."""
    # ── Downsample ────────────────────────────────────────────────────────
    X_tr_ds = _maybe_downsample(X_tr_raw, sfreq, _TARGET_SFREQ)
    X_te_ds = _maybe_downsample(X_te_raw, sfreq, _TARGET_SFREQ)

    # ── Spatial codebook (train only) ─────────────────────────────────────
    km = fit_spatial_codebook(X_tr_ds, method, n_bins, n_codes=n_codes,
                              random_state=seed)

    # ── Encode ────────────────────────────────────────────────────────────
    sc_tr = encode_spatial_codes(X_tr_ds, km, method, n_bins)  # (n_tr, n_times)
    sc_te = encode_spatial_codes(X_te_ds, km, method, n_bins)

    # ── BPE vocab (train only) ────────────────────────────────────────────
    seqs_tr = [sc_tr[i].tolist() for i in range(len(sc_tr))]
    vocab = train_bpe(seqs_tr, vocab_size=vocab_size,
                      base_vocab_size=n_codes, verbose=False)

    # ── Histograms ────────────────────────────────────────────────────────
    X_tr_hist = spatial_bpe_histograms(sc_tr, vocab)
    X_te_hist = spatial_bpe_histograms(sc_te, vocab)

    metrics = {}

    # ── LogReg ────────────────────────────────────────────────────────────
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_hist)
    X_te_sc = sc.transform(X_te_hist)
    clf_lr = LogisticRegression(
        C=LOGREG_C, max_iter=LOGREG_MAX_ITER, solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
        random_state=seed, n_jobs=1,
    )
    clf_lr.fit(X_tr_sc, y_tr)
    y_pred_lr = clf_lr.predict(X_te_sc)

    metrics["LogReg"] = {
        "accuracy":          float(accuracy_score(y_te, y_pred_lr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred_lr)),
        "macro_f1":          float(f1_score(y_te, y_pred_lr, average="macro",
                                            zero_division=0)),
        "kappa":             float(cohen_kappa_score(y_te, y_pred_lr)),
    }

    # ── RF ────────────────────────────────────────────────────────────────
    X_tr_rf = X_tr_hist
    X_te_rf = X_te_hist
    if X_tr_rf.shape[1] > _MAX_HIST_FEATURES:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=_MAX_HIST_FEATURES // 2, random_state=seed)
        X_tr_rf = svd.fit_transform(X_tr_rf)
        X_te_rf = svd.transform(X_te_rf)

    clf_rf = RandomForestClassifier(
        n_estimators=200, n_jobs=1,  # folds run in parallel; avoid CPU over-subscription
        class_weight=LOGREG_CLASS_WEIGHT, random_state=seed,
    )
    clf_rf.fit(X_tr_rf, y_tr)
    y_pred_rf = clf_rf.predict(X_te_rf)

    metrics["RF"] = {
        "accuracy":          float(accuracy_score(y_te, y_pred_rf)),
        "balanced_accuracy": float(balanced_accuracy_score(y_te, y_pred_rf)),
        "macro_f1":          float(f1_score(y_te, y_pred_rf, average="macro",
                                            zero_division=0)),
        "kappa":             float(cohen_kappa_score(y_te, y_pred_rf)),
    }

    return metrics


def _run_loso(
    all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
    sfreq: float, vocab_size: int, n_bins: int, method: str,
    n_codes: int, seed: int,
) -> list[dict]:
    logo = LeaveOneGroupOut()
    splits = list(logo.split(all_epochs, y, groups))

    def _one_fold(fold_idx: int, tr_idx, te_idx):
        fold_metrics = _run_fold(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            sfreq, vocab_size, n_bins, method, n_codes, seed,
        )
        return [
            {"fold": fold_idx, "test_subject": int(groups[te_idx[0]]),
             "classifier": clf_name, **mets}
            for clf_name, mets in fold_metrics.items()
        ]

    nested = Parallel(n_jobs=min(len(splits), N_JOBS), prefer="threads")(
        delayed(_one_fold)(fold_idx, tr_idx, te_idx)
        for fold_idx, (tr_idx, te_idx) in enumerate(splits)
    )
    return [r for fold_list in nested for r in fold_list]


def _run_kfold(
    all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
    sfreq: float, vocab_size: int, n_bins: int, method: str,
    n_codes: int, seed: int, n_splits: int = 5,
) -> list[dict]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(sgkf.split(all_epochs, y, groups))

    def _one_fold(fold_idx: int, tr_idx, te_idx):
        fold_metrics = _run_fold(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            sfreq, vocab_size, n_bins, method, n_codes, seed,
        )
        return [
            {"fold": fold_idx, "test_subject": -1,
             "classifier": clf_name, **mets}
            for clf_name, mets in fold_metrics.items()
        ]

    nested = Parallel(n_jobs=min(len(splits), N_JOBS), prefer="threads")(
        delayed(_one_fold)(fold_idx, tr_idx, te_idx)
        for fold_idx, (tr_idx, te_idx) in enumerate(splits)
    )
    return [r for fold_list in nested for r in fold_list]


def _assemble_data(data: dict):
    all_ep, all_lb, all_gr = [], [], []
    for subj_id, subj_data in data.items():
        all_ep.append(subj_data["epochs"])
        all_lb.extend(subj_data["labels"])
        all_gr.extend([subj_id] * len(subj_data["labels"]))
    min_t = min(ep.shape[2] for ep in all_ep)
    all_ep = [ep[:, :, :min_t] for ep in all_ep]
    all_epochs = np.concatenate(all_ep, axis=0)
    le = LabelEncoder()
    y = le.fit_transform(np.array(all_lb))
    groups = np.array(all_gr)
    return all_epochs, y, groups


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_exp9(results: list[dict], plot_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib not available, skipping Exp 9 plot")
        return

    df = pd.DataFrame(results)
    df_mean = (df.groupby(["dataset", "classifier"])["accuracy_mean"]
               .mean().reset_index())

    datasets   = df_mean["dataset"].unique()
    classifiers = df_mean["classifier"].unique()
    x = np.arange(len(datasets))
    width = 0.35

    # Chance levels per dataset
    chance = {
        "bci_iv_2a": 0.25, "physionet_mi": 0.25, "sleep_edf": 0.20,
        "mental_arithmetic": 0.50, "epfl_p300": 0.50, "ssvep_nakanishi": 1/12,
    }

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4472C4", "#ED7D31"]
    for i, clf in enumerate(classifiers):
        vals = []
        for ds in datasets:
            row = df_mean[(df_mean["dataset"] == ds) & (df_mean["classifier"] == clf)]
            vals.append(float(row["accuracy_mean"].iloc[0]) if len(row) > 0 else 0.0)
        offset = (i - (len(classifiers) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=f"Spatial BPE {clf}",
               color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 9: Spatial BPE (amplitude spatial codebook)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out = plot_dir / "exp9_accuracy_bar.png"
    plt.savefig(str(out), dpi=150)
    plt.close()
    logger.info(f"Exp 9 plot saved to {out}")


# ─── Main experiment ──────────────────────────────────────────────────────────

@timed("exp9")
def run_experiment_9(
    datasets: list[str] | None = None,
    vocab_size: int = 256,
    n_bins: int = 64,
    method: str | None = None,
    n_codes: int = _N_SPATIAL_CODES,
    max_subjects: int | None = None,
) -> dict:
    """
    Run Experiment 9: Spatial BPE on all datasets.

    Parameters
    ----------
    datasets : list of str or None
        Datasets to run.  Default: all 6.
    vocab_size : int
        BPE vocabulary size (default 256; sequences are shorter than channel-BPE).
    n_bins : int
        Quantisation bins (default 64).
    method : str or None
        Quantisation method (default DEFAULT_QUANT_METHOD).
    n_codes : int
        Spatial codebook size (default 64; analogous to n_bins in channel BPE).
    max_subjects : int or None
        Limit subjects for quick runs.

    Returns
    -------
    dict
        Summary of results.
    """
    if method is None:
        method = DEFAULT_QUANT_METHOD

    target_datasets = datasets or _ALL_DATASETS
    exp_log = ExperimentLogger("exp9_spatial_bpe")
    all_results: list[dict] = []

    plot_dir = PLOTS_DIR / "exp9"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in target_datasets:
        if ds_name not in DATASET_INFO:
            logger.warning(f"Unknown dataset {ds_name}, skipping")
            continue

        info = DATASET_INFO[ds_name]
        sfreq = float(info["sfreq"])
        cv_strategy = info["cv_strategy"]
        n_classes = int(info.get("n_classes", 2))
        chance = 1.0 / n_classes

        logger.info(
            f"=== Exp 9: {ds_name}  (sfreq={sfreq}, CV={cv_strategy},"
            f" n_classes={n_classes}) ==="
        )
        data = load_dataset(ds_name, max_subjects=max_subjects)
        if not data:
            logger.warning(f"Exp 9: no data for {ds_name}, skipping")
            continue

        all_epochs, y, groups = _assemble_data(data)
        logger.info(
            f"  {ds_name}: {all_epochs.shape[0]} trials,"
            f" {len(np.unique(groups))} subjects,"
            f" n_ch={all_epochs.shape[1]}, n_times={all_epochs.shape[2]}"
        )

        for seed in RANDOM_SEEDS:
            # Resume: skip if both LogReg and RF results already exist for this seed.
            if exp_log.csv_path.exists():
                try:
                    import pandas as pd
                    _df9 = pd.read_csv(exp_log.csv_path)
                    _base9 = (
                        (_df9["dataset"].astype(str) == str(ds_name)) &
                        (_df9["vocab_size"].astype(str) == str(vocab_size)) &
                        (_df9["n_bins"].astype(str) == str(n_bins)) &
                        (_df9["n_codes"].astype(str) == str(n_codes)) &
                        (_df9["seed"].astype(str) == str(seed))
                    )
                    _done_clfs = set(_df9.loc[_base9, "classifier"].unique())
                    if {"Spatial_BPE_LogReg", "Spatial_BPE_RF"}.issubset(_done_clfs):
                        logger.info(f"  Seed={seed} — already done, skipping")
                        continue
                except Exception:
                    pass
            logger.info(f"  Seed={seed} ...")
            if cv_strategy == "LOSO":
                fold_results = _run_loso(
                    all_epochs, y, groups, sfreq,
                    vocab_size, n_bins, method, n_codes, seed,
                )
            else:
                fold_results = _run_kfold(
                    all_epochs, y, groups, sfreq,
                    vocab_size, n_bins, method, n_codes, seed,
                )

            if not fold_results:
                continue

            # Aggregate per classifier
            for clf_name in ["LogReg", "RF"]:
                clf_folds = [f for f in fold_results if f["classifier"] == clf_name]
                if not clf_folds:
                    continue

                accs      = [f["accuracy"]          for f in clf_folds]
                bal_accs  = [f["balanced_accuracy"]  for f in clf_folds]
                f1s       = [f["macro_f1"]           for f in clf_folds]
                kappas    = [f["kappa"]              for f in clf_folds]

                _, ci_low, ci_high = bootstrap_ci(np.array(accs), n_boot=_N_BOOTSTRAP)

                result = {
                    "dataset":               ds_name,
                    "classifier":            f"Spatial_BPE_{clf_name}",
                    "vocab_size":            vocab_size,
                    "n_bins":                n_bins,
                    "n_codes":               n_codes,
                    "method":                method,
                    "seed":                  seed,
                    "cv_strategy":           cv_strategy,
                    "n_folds":               len(clf_folds),
                    "chance_level":          chance,
                    "accuracy_mean":         float(np.mean(accs)),
                    "accuracy_std":          float(np.std(accs)),
                    "accuracy_ci_low":       ci_low,
                    "accuracy_ci_high":      ci_high,
                    "balanced_accuracy_mean": float(np.mean(bal_accs)),
                    "macro_f1_mean":         float(np.mean(f1s)),
                    "kappa_mean":            float(np.mean(kappas)),
                }

                exp_log.log_result(result)
                all_results.append(result)

                # Per-fold CSV
                for f in clf_folds:
                    append_csv({
                        "dataset":           ds_name,
                        "classifier":        f"Spatial_BPE_{clf_name}",
                        "vocab_size":        vocab_size,
                        "n_codes":           n_codes,
                        "seed":              seed,
                        "cv_strategy":       cv_strategy,
                        "fold":              f["fold"],
                        "accuracy":          f["accuracy"],
                        "balanced_accuracy": f["balanced_accuracy"],
                        "macro_f1":          f["macro_f1"],
                        "kappa":             f["kappa"],
                    }, LOGS_DIR / "exp9_per_fold.csv")

                logger.info(
                    f"  [Spatial_BPE_{clf_name}] seed={seed}: "
                    f"acc={np.mean(accs):.3f}+/-{np.std(accs):.3f}, "
                    f"kappa={np.mean(kappas):.3f}"
                )

    save_csv(all_results, LOGS_DIR / "exp9_spatial_bpe_results.csv")

    if all_results:
        _plot_exp9(all_results, plot_dir)

    logger.info(f"Exp 9 complete: {len(all_results)} result rows")
    return {
        "n_results": len(all_results),
        "datasets":  target_datasets,
    }
