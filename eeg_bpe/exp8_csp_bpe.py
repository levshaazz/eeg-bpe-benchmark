"""
Experiment 8: CSP + BPE
=======================
Applies Common Spatial Pattern (CSP) spatial filtering *before* BPE tokenisation.

Hypothesis: BPE fails on Motor Imagery (MI) not because BPE is fundamentally
incapable, but because the raw EEG channels are spatially mixed (volume
conduction).  CSP maximises class-discriminative variance across channels, so
CSP-filtered components should expose clear ERD/ERS amplitude modulation that
BPE can tokenise effectively.

Two variants are tested per dataset:
    CSP_BPE_LogReg        : raw CSP components → quantise → BPE → histogram → LogReg
    CSP_Envelope_BPE_LogReg: CSP components → Hilbert envelope (8-30 Hz) → BPE → LogReg

CSP is always fitted on the training fold only (no leakage).
The BPE vocabulary is also trained on training-fold CSP data per fold to match
the distribution.  This is more expensive but correct: test-fold CSP outputs
have very different statistics from raw EEG.

Datasets: bci_iv_2a (LOSO) and physionet_mi (5-fold subject-stratified).

Output
------
results/logs/exp8_csp_bpe_results.csv
results/logs/exp8_per_fold.csv
results/plots/exp8/exp8_accuracy_bar.png
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from .config import (
    DATASET_INFO, RANDOM_SEEDS, LOGS_DIR, PLOTS_DIR,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
    DEFAULT_QUANT_METHOD, DEVICE,
)
from .data_loading import load_dataset
from .bpe_engine import train_bpe, apply_bpe, BPEVocab
from .quantization import quantize
from .exp2_downstream import (
    epochs_to_bpe_histograms,
    compute_classification_metrics,
)
from .tokenization import envelope_epochs
from .utils import (
    ExperimentLogger, save_csv, append_csv, get_logger, timed, bootstrap_ci,
    pca_reduce,
)

try:
    from .classifiers import classify_seq_cnn as _classify_seq_cnn
    _HAS_SEQ_CNN = True
except Exception:
    _HAS_SEQ_CNN = False

logger = get_logger("exp8_csp_bpe")

_MI_DATASETS = ["bci_iv_2a", "physionet_mi"]
_CSP_COMPONENTS = 4  # n_comp sweep showed 4 > 6 (+1.7pp on BCI-IV-2a)
_ENV_FMIN = 8.0
_ENV_FMAX = 30.0
_ENV_TARGET_SFREQ = 64.0
_MAX_HIST_FEATURES = 2048


def _import_csp():
    try:
        from mne.decoding import CSP
        return CSP
    except ImportError:
        raise ImportError(
            "MNE-Python is required for Exp 8 (CSP).  "
            "Install with: pip install mne"
        )


def _bpe_histogram_for_fold(X_csp: np.ndarray, y_tr: np.ndarray,
                             vocab_size: int, n_bins: int, method: str
                             ) -> tuple[BPEVocab, np.ndarray]:
    """Train BPE on *X_csp* (CSP-transformed train epochs) and return histogram."""
    # Flatten channels into sequences for BPE training
    n_trials, n_comp, n_times = X_csp.shape
    flat = X_csp.reshape(n_trials * n_comp, n_times)
    codes_flat, _ = quantize(flat, method=method, n_bins=n_bins, normalize=True)
    seqs = [codes_flat[i].tolist() for i in range(codes_flat.shape[0])]
    vocab = train_bpe(seqs, vocab_size=vocab_size, base_vocab_size=n_bins, verbose=False,
                      max_train_tokens=5_000_000)
    X_hist = epochs_to_bpe_histograms(X_csp, vocab, method=method, n_bins=n_bins)
    return vocab, X_hist


def _bpe_sequences_for_fold(X_csp: np.ndarray, vocab: BPEVocab,
                             n_bins: int, method: str,
                             max_len: int = 512) -> np.ndarray:
    """Apply trained BPE vocab to CSP epochs → flat integer sequences.

    Returns (n_trials, n_comp * max_len) int32 padded token array.
    """
    n_trials, n_comp, n_times = X_csp.shape
    flat = X_csp.reshape(n_trials * n_comp, n_times)
    codes_flat, _ = quantize(flat, method=method, n_bins=n_bins, normalize=True)

    SEP = vocab.vocab_size
    PAD = vocab.vocab_size + 1

    all_rows = []
    for trial_i in range(n_trials):
        row_tokens = []
        for comp_i in range(n_comp):
            seq = codes_flat[trial_i * n_comp + comp_i].tolist()
            merged = apply_bpe(seq, vocab)
            # truncate per component
            merged = merged[:max_len]
            row_tokens.extend(merged + [SEP])
        all_rows.append(row_tokens)

    # Pad to uniform length
    max_row = max(len(r) for r in all_rows)
    out = np.full((n_trials, max_row), PAD, dtype=np.int32)
    for i, row in enumerate(all_rows):
        out[i, :len(row)] = row
    return out


def _run_one_fold_seq_cnn(X_tr_raw: np.ndarray, y_tr: np.ndarray,
                          X_te_raw: np.ndarray, y_te: np.ndarray,
                          vocab_size: int, n_bins: int,
                          method: str, seed: int) -> dict:
    """One CV fold for CSP+BPE+Seq_CNN (Conv1D over BPE token sequences)."""
    if not _HAS_SEQ_CNN:
        raise RuntimeError("classify_seq_cnn unavailable")

    CSP = _import_csp()

    csp = CSP(n_components=_CSP_COMPONENTS, reg=None, log=False, norm_trace=False)
    csp.fit(X_tr_raw, y_tr)
    filters = csp.filters_[:_CSP_COMPONENTS]
    X_tr_csp = np.einsum("ec,nct->net", filters, X_tr_raw).astype(np.float32)
    X_te_csp = np.einsum("ec,nct->net", filters, X_te_raw).astype(np.float32)

    # Train BPE vocab on CSP train data, then get sequences
    vocab, _ = _bpe_histogram_for_fold(X_tr_csp, y_tr, vocab_size, n_bins, method)
    max_len_per_comp = 128
    X_tr_seq = _bpe_sequences_for_fold(X_tr_csp, vocab, n_bins, method, max_len_per_comp)
    X_te_seq = _bpe_sequences_for_fold(X_te_csp, vocab, n_bins, method, max_len_per_comp)

    y_pred, y_proba = _classify_seq_cnn(
        X_tr_seq, y_tr, X_te_seq, y_te,
        vocab_size=vocab_size, device=DEVICE,
    )
    return compute_classification_metrics(y_te, y_pred, y_proba)


def _run_one_fold(X_tr_raw: np.ndarray, y_tr: np.ndarray,
                  X_te_raw: np.ndarray, y_te: np.ndarray,
                  sfreq: float, vocab_size: int, n_bins: int,
                  method: str, with_envelope: bool,
                  seed: int, n_comp: int | None = None) -> dict:
    """Run one CV fold for CSP+BPE and return metrics."""
    CSP = _import_csp()

    if n_comp is None:
        n_comp = _CSP_COMPONENTS
    n_ch = X_tr_raw.shape[1]
    n_comp = min(n_comp, n_ch)

    csp = CSP(n_components=n_comp, reg=None, log=False, norm_trace=False)
    csp.fit(X_tr_raw, y_tr)
    # csp.transform() returns 2D (n_trials, n_components) log-power features.
    # We need the 3D filtered time series (n_trials, n_comp, n_times) for BPE.
    # Apply spatial filters manually: filters_ shape is (n_components, n_channels).
    filters = csp.filters_[:n_comp]               # (n_comp, n_ch)
    X_tr_csp = np.einsum("ec,nct->net", filters, X_tr_raw).astype(np.float32)
    X_te_csp = np.einsum("ec,nct->net", filters, X_te_raw).astype(np.float32)

    if with_envelope:
        X_tr_csp = envelope_epochs(X_tr_csp, sfreq,
                                   fmin=_ENV_FMIN, fmax=_ENV_FMAX,
                                   target_sfreq=_ENV_TARGET_SFREQ)
        X_te_csp = envelope_epochs(X_te_csp, sfreq,
                                   fmin=_ENV_FMIN, fmax=_ENV_FMAX,
                                   target_sfreq=_ENV_TARGET_SFREQ)
        fold_sfreq = _ENV_TARGET_SFREQ
    else:
        fold_sfreq = sfreq

    vocab, X_tr_hist = _bpe_histogram_for_fold(X_tr_csp, y_tr, vocab_size, n_bins, method)
    X_te_hist = epochs_to_bpe_histograms(X_te_csp, vocab, method=method, n_bins=n_bins)

    if X_tr_hist.shape[1] > _MAX_HIST_FEATURES:
        from sklearn.decomposition import TruncatedSVD
        n_comp = min(_MAX_HIST_FEATURES // 4, X_tr_hist.shape[0] - 1, X_tr_hist.shape[1])
        svd = TruncatedSVD(n_components=n_comp, random_state=seed)
        X_tr_hist = svd.fit_transform(X_tr_hist)
        X_te_hist = svd.transform(X_te_hist)

    clf = LogisticRegression(
        C=LOGREG_C, max_iter=LOGREG_MAX_ITER, solver=LOGREG_SOLVER,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
    )
    clf.fit(X_tr_hist, y_tr)
    y_pred = clf.predict(X_te_hist)
    y_proba = clf.predict_proba(X_te_hist)

    return compute_classification_metrics(y_te, y_pred, y_proba)


def _run_loso(all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
              sfreq: float, vocab_size: int, n_bins: int, method: str,
              with_envelope: bool, seed: int,
              n_comp: int | None = None) -> list[dict]:
    from .config import N_JOBS
    logo = LeaveOneGroupOut()
    splits = list(logo.split(all_epochs, y, groups))

    def _one(fold_idx, tr_idx, te_idx):
        m = _run_one_fold(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            sfreq, vocab_size, n_bins, method, with_envelope, seed,
            n_comp=n_comp,
        )
        m["fold"] = fold_idx
        m["test_subject"] = int(groups[te_idx[0]])
        return m

    # n_jobs=1: scipy bandpass + CSP causes MKL thread-pool deadlock in parallel threads
    return Parallel(n_jobs=1)(
        delayed(_one)(fi, tr, te) for fi, (tr, te) in enumerate(splits)
    )


def _run_kfold(all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
               sfreq: float, vocab_size: int, n_bins: int, method: str,
               with_envelope: bool, seed: int, n_splits: int = 5,
               n_comp: int | None = None) -> list[dict]:
    from .config import N_JOBS
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(sgkf.split(all_epochs, y, groups))

    def _one(fold_idx, tr_idx, te_idx):
        m = _run_one_fold(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            sfreq, vocab_size, n_bins, method, with_envelope, seed,
            n_comp=n_comp,
        )
        m["fold"] = fold_idx
        m["test_subject"] = -1
        return m

    # n_jobs=1: same MKL deadlock fix as _run_loso
    return Parallel(n_jobs=1)(
        delayed(_one)(fi, tr, te) for fi, (tr, te) in enumerate(splits)
    )


def _run_loso_seq_cnn(all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      vocab_size: int, n_bins: int, method: str,
                      seed: int) -> list[dict]:
    from .config import N_JOBS
    logo = LeaveOneGroupOut()
    splits = list(logo.split(all_epochs, y, groups))

    def _one(fold_idx, tr_idx, te_idx):
        m = _run_one_fold_seq_cnn(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            vocab_size, n_bins, method, seed,
        )
        m["fold"] = fold_idx
        m["test_subject"] = int(groups[te_idx[0]])
        return m

    # n_jobs=1: torch.compile CUDA graphs conflict when multiple threads share GPU
    return Parallel(n_jobs=1)(
        delayed(_one)(fi, tr, te) for fi, (tr, te) in enumerate(splits)
    )


def _run_kfold_seq_cnn(all_epochs: np.ndarray, y: np.ndarray, groups: np.ndarray,
                       vocab_size: int, n_bins: int, method: str,
                       seed: int, n_splits: int = 5) -> list[dict]:
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    splits = list(sgkf.split(all_epochs, y, groups))

    def _one(fold_idx, tr_idx, te_idx):
        m = _run_one_fold_seq_cnn(
            all_epochs[tr_idx], y[tr_idx],
            all_epochs[te_idx], y[te_idx],
            vocab_size, n_bins, method, seed,
        )
        m["fold"] = fold_idx
        m["test_subject"] = -1
        return m

    # n_jobs=1: torch.compile CUDA graphs conflict when multiple threads share GPU
    return Parallel(n_jobs=1)(
        delayed(_one)(fi, tr, te) for fi, (tr, te) in enumerate(splits)
    )


def _assemble_data(data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_ep, all_lb, all_gr = [], [], []
    for subj_id, subj_data in data.items():
        epochs = subj_data["epochs"]
        labels = subj_data["labels"]
        all_ep.append(epochs)
        all_lb.extend(labels)
        all_gr.extend([subj_id] * len(labels))
    # Truncate to minimum time length across subjects (physionet_mi has variable-length epochs)
    min_t = min(ep.shape[2] for ep in all_ep)
    all_ep = [ep[:, :, :min_t] for ep in all_ep]
    all_epochs = np.concatenate(all_ep, axis=0)
    le = LabelEncoder()
    y = le.fit_transform(np.array(all_lb))
    groups = np.array(all_gr)
    return all_epochs, y, groups


@timed("exp8")
def run_experiment_8(
    datasets: list[str] | None = None,
    vocab_size: int = 512,
    n_bins: int = 64,
    method: str | None = None,
    max_subjects: int | None = None,
    sweep_ncomp: bool = True,
) -> dict:
    """
    Run Experiment 8: CSP + BPE on MI datasets.

    Parameters
    ----------
    datasets : list of str or None
        Datasets to run.  Default: ``["bci_iv_2a", "physionet_mi"]``.
    vocab_size : int
        BPE vocabulary size (default 512; small because CSP has only 6 components).
    n_bins : int
        Quantisation bins (default 64).
    method : str or None
        Quantisation method.  Default: ``DEFAULT_QUANT_METHOD``.
    max_subjects : int or None
        Max subjects per dataset (None = all).

    Returns
    -------
    dict
        Summary of results.
    """
    if method is None:
        method = DEFAULT_QUANT_METHOD
    target_datasets = datasets or _MI_DATASETS
    # Filter to MI only if not overridden
    target_datasets = [d for d in target_datasets if d in _MI_DATASETS]
    if not target_datasets:
        logger.warning("Exp 8: no MI datasets selected — skipping")
        return {"n_results": 0}

    exp_log = ExperimentLogger("exp8_csp_bpe")
    all_results: list[dict] = []
    per_fold_accs: dict = {}

    plot_dir = PLOTS_DIR / "exp8"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in target_datasets:
        info = DATASET_INFO[ds_name]
        sfreq = float(info["sfreq"])
        cv_strategy = info["cv_strategy"]

        logger.info(f"=== Exp 8: {ds_name} (sfreq={sfreq}, CV={cv_strategy}) ===")
        data = load_dataset(ds_name, max_subjects=max_subjects)
        if not data:
            logger.warning(f"Exp 8: no data for {ds_name}, skipping")
            continue

        all_epochs, y, groups = _assemble_data(data)
        logger.info(f"  {ds_name}: {all_epochs.shape[0]} trials, "
                    f"{len(np.unique(groups))} subjects, {len(np.unique(y))} classes")

        hist_classifiers = [
            ("CSP_BPE_LogReg",          False),
            ("CSP_Envelope_BPE_LogReg",  True),
        ]

        for clf_name, with_envelope in hist_classifiers:
            for seed in RANDOM_SEEDS:
                # Resume: skip if this (dataset, classifier, vocab_size, seed) is done.
                if exp_log.csv_path.exists():
                    try:
                        import pandas as pd
                        _df8 = pd.read_csv(exp_log.csv_path)
                        _done8 = (
                            (_df8["dataset"].astype(str) == str(ds_name)) &
                            (_df8["classifier"].astype(str) == str(clf_name)) &
                            (_df8["vocab_size"].astype(str) == str(vocab_size)) &
                            (_df8["seed"].astype(str) == str(seed))
                        ).any()
                        if _done8:
                            logger.info(f"  [{clf_name}] seed={seed} — already done, skipping")
                            continue
                    except Exception:
                        pass
                logger.info(f"  [{clf_name}] seed={seed} ...")
                if cv_strategy == "LOSO":
                    fold_results = _run_loso(
                        all_epochs, y, groups, sfreq,
                        vocab_size, n_bins, method, with_envelope, seed,
                    )
                else:
                    fold_results = _run_kfold(
                        all_epochs, y, groups, sfreq,
                        vocab_size, n_bins, method, with_envelope, seed,
                    )

                if not fold_results:
                    continue

                accs     = [f["accuracy"]                             for f in fold_results]
                bal_accs = [f.get("balanced_accuracy", f["accuracy"]) for f in fold_results]
                f1s      = [f["macro_f1"]                             for f in fold_results]
                kappas   = [f["kappa"]                                for f in fold_results]

                _, ci_low, ci_high = bootstrap_ci(np.array(accs))

                result = {
                    "dataset":          ds_name,
                    "classifier":       clf_name,
                    "vocab_size":       vocab_size,
                    "n_bins":           n_bins,
                    "method":           method,
                    "with_envelope":    with_envelope,
                    "seed":             seed,
                    "cv_strategy":      cv_strategy,
                    "n_folds":          len(fold_results),
                    "accuracy_mean":    float(np.mean(accs)),
                    "accuracy_std":     float(np.std(accs)),
                    "accuracy_ci_low":  ci_low,
                    "accuracy_ci_high": ci_high,
                    "balanced_accuracy_mean": float(np.mean(bal_accs)),
                    "macro_f1_mean":    float(np.mean(f1s)),
                    "kappa_mean":       float(np.mean(kappas)),
                }

                exp_log.log_result(result)
                all_results.append(result)
                per_fold_accs[(ds_name, clf_name, seed)] = accs

                # Per-fold CSV
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
                    }, LOGS_DIR / "exp8_per_fold.csv")

                logger.info(
                    f"  [{clf_name}] seed={seed}: "
                    f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
                    f"kappa={np.mean(kappas):.3f}"
                )

        # ── CSP + Seq_CNN variant ──────────────────────────────────────────
        if _HAS_SEQ_CNN:
            clf_name = "CSP_BPE_Seq_CNN"
            for seed in RANDOM_SEEDS:
                # Resume: skip if already done
                if exp_log.csv_path.exists():
                    try:
                        import pandas as pd
                        _df8s = pd.read_csv(exp_log.csv_path)
                        _done8s = (
                            (_df8s["dataset"].astype(str) == str(ds_name)) &
                            (_df8s["classifier"].astype(str) == str(clf_name)) &
                            (_df8s["vocab_size"].astype(str) == str(vocab_size)) &
                            (_df8s["seed"].astype(str) == str(seed))
                        ).any()
                        if _done8s:
                            logger.info(f"  [{clf_name}] seed={seed} — already done, skipping")
                            continue
                    except Exception:
                        pass
                logger.info(f"  [{clf_name}] seed={seed} ...")
                try:
                    if cv_strategy == "LOSO":
                        fold_results = _run_loso_seq_cnn(
                            all_epochs, y, groups,
                            vocab_size, n_bins, method, seed,
                        )
                    else:
                        fold_results = _run_kfold_seq_cnn(
                            all_epochs, y, groups,
                            vocab_size, n_bins, method, seed,
                        )
                except Exception as exc:
                    logger.warning(f"  [{clf_name}] failed: {exc}")
                    continue

                if not fold_results:
                    continue

                accs     = [f["accuracy"]                             for f in fold_results]
                bal_accs = [f.get("balanced_accuracy", f["accuracy"]) for f in fold_results]
                f1s      = [f["macro_f1"]                             for f in fold_results]
                kappas   = [f["kappa"]                                for f in fold_results]
                _, ci_low, ci_high = bootstrap_ci(np.array(accs))

                result = {
                    "dataset":          ds_name,
                    "classifier":       clf_name,
                    "vocab_size":       vocab_size,
                    "n_bins":           n_bins,
                    "method":           method,
                    "with_envelope":    False,
                    "seed":             seed,
                    "cv_strategy":      cv_strategy,
                    "n_folds":          len(fold_results),
                    "accuracy_mean":    float(np.mean(accs)),
                    "accuracy_std":     float(np.std(accs)),
                    "accuracy_ci_low":  ci_low,
                    "accuracy_ci_high": ci_high,
                    "balanced_accuracy_mean": float(np.mean(bal_accs)),
                    "macro_f1_mean":    float(np.mean(f1s)),
                    "kappa_mean":       float(np.mean(kappas)),
                }
                exp_log.log_result(result)
                all_results.append(result)
                per_fold_accs[(ds_name, clf_name, seed)] = accs

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
                    }, LOGS_DIR / "exp8_per_fold.csv")

                logger.info(
                    f"  [{clf_name}] seed={seed}: "
                    f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
                    f"kappa={np.mean(kappas):.3f}"
                )

    # Merge with existing CSV so skipped (resumed) rows are not lost
    csv_path = LOGS_DIR / "exp8_csp_bpe_results.csv"
    if csv_path.exists() and all_results:
        import pandas as _pd
        existing_df = _pd.read_csv(csv_path)
        new_df = _pd.DataFrame(all_results)
        combined = _pd.concat([existing_df, new_df]).drop_duplicates(
            subset=["dataset", "classifier", "seed"], keep="last"
        )
        all_results = combined.to_dict("records")
        combined.to_csv(csv_path, index=False)
    else:
        save_csv(all_results, csv_path)

    # ── Plot ──────────────────────────────────────────────────────────────
    if all_results:
        _plot_exp8(all_results, plot_dir)
        _plot_exp8_heatmap(all_results, plot_dir)

    # ── Optional n_comp sweep ────────────────────────────────────────────
    if sweep_ncomp:
        run_exp8_ncomp_sweep(
            datasets=target_datasets,
            vocab_size=vocab_size,
            n_bins=n_bins,
            method=method,
            max_subjects=max_subjects,
        )

    logger.info(f"Exp 8 complete: {len(all_results)} result rows")
    return {
        "n_results": len(all_results),
        "datasets":  target_datasets,
    }


_NCOMP_SWEEP_VALUES = [2, 4, 6, 8]


@timed("exp8_ncomp_sweep")
def run_exp8_ncomp_sweep(
    datasets: list[str] | None = None,
    vocab_size: int = 512,
    n_bins: int = 64,
    method: str | None = None,
    max_subjects: int | None = None,
) -> dict:
    """Sweep CSP n_components for CSP_BPE_LogReg and plot accuracy vs n_comp.

    Parameters
    ----------
    datasets : list of str or None
        MI datasets to evaluate. Default: ``["bci_iv_2a", "physionet_mi"]``.
    vocab_size, n_bins, method : BPE / quantisation settings.
    max_subjects : int or None
        Cap subjects per dataset (None = all).

    Returns
    -------
    dict  with ``n_results`` count.
    """
    import pandas as pd

    if method is None:
        method = DEFAULT_QUANT_METHOD
    target_datasets = datasets or _MI_DATASETS
    target_datasets = [d for d in target_datasets if d in _MI_DATASETS]
    if not target_datasets:
        logger.warning("Exp 8 n_comp sweep: no MI datasets — skipping")
        return {"n_results": 0}

    csv_path = LOGS_DIR / "exp8_ncomp_sweep.csv"
    plot_dir = PLOTS_DIR / "exp8"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    for ds_name in target_datasets:
        info = DATASET_INFO[ds_name]
        sfreq = float(info["sfreq"])
        cv_strategy = info["cv_strategy"]

        logger.info(f"=== Exp 8 n_comp sweep: {ds_name} ===")
        data = load_dataset(ds_name, max_subjects=max_subjects)
        if not data:
            logger.warning(f"Exp 8 n_comp sweep: no data for {ds_name}, skipping")
            continue

        all_epochs, y, groups = _assemble_data(data)
        n_ch = all_epochs.shape[1]
        logger.info(f"  {ds_name}: {all_epochs.shape[0]} trials, "
                    f"{n_ch} channels, {len(np.unique(groups))} subjects")

        for nc in _NCOMP_SWEEP_VALUES:
            effective_nc = min(nc, n_ch)
            for seed in RANDOM_SEEDS:
                # ── Resume: check CSV for completed seeds ──
                if csv_path.exists():
                    try:
                        _df = pd.read_csv(csv_path)
                        _done = (
                            (_df["dataset"].astype(str) == str(ds_name)) &
                            (_df["n_comp"].astype(str) == str(effective_nc)) &
                            (_df["vocab_size"].astype(str) == str(vocab_size)) &
                            (_df["seed"].astype(str) == str(seed))
                        ).any()
                        if _done:
                            logger.info(f"  [n_comp={effective_nc}] seed={seed} — already done, skipping")
                            continue
                    except Exception:
                        pass

                logger.info(f"  [n_comp={effective_nc}] seed={seed} ...")
                if cv_strategy == "LOSO":
                    fold_results = _run_loso(
                        all_epochs, y, groups, sfreq,
                        vocab_size, n_bins, method,
                        with_envelope=False, seed=seed,
                        n_comp=effective_nc,
                    )
                else:
                    fold_results = _run_kfold(
                        all_epochs, y, groups, sfreq,
                        vocab_size, n_bins, method,
                        with_envelope=False, seed=seed,
                        n_comp=effective_nc,
                    )

                if not fold_results:
                    continue

                accs   = [f["accuracy"]  for f in fold_results]
                kappas = [f["kappa"]     for f in fold_results]
                _, ci_low, ci_high = bootstrap_ci(np.array(accs))

                result = {
                    "dataset":          ds_name,
                    "n_comp":           effective_nc,
                    "vocab_size":       vocab_size,
                    "n_bins":           n_bins,
                    "method":           method,
                    "seed":             seed,
                    "cv_strategy":      cv_strategy,
                    "n_folds":          len(fold_results),
                    "accuracy_mean":    float(np.mean(accs)),
                    "accuracy_std":     float(np.std(accs)),
                    "accuracy_ci_low":  ci_low,
                    "accuracy_ci_high": ci_high,
                    "kappa_mean":       float(np.mean(kappas)),
                }

                append_csv(result, csv_path)
                all_results.append(result)

                logger.info(
                    f"  [n_comp={effective_nc}] seed={seed}: "
                    f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}, "
                    f"kappa={np.mean(kappas):.3f}"
                )

    # ── Plot: line plot with ±std shading ─────────────────────────────────
    if csv_path.exists():
        _plot_ncomp_sweep(csv_path, plot_dir)

    logger.info(f"Exp 8 n_comp sweep complete: {len(all_results)} new result rows")
    return {"n_results": len(all_results)}


def _plot_ncomp_sweep(csv_path: Path, plot_dir: Path) -> None:
    """Line plot: x=n_comp, y=accuracy, one line per dataset with ±std shading."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        logger.warning("matplotlib/pandas not available — skipping n_comp sweep plot")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Aggregate across seeds: mean and std of accuracy_mean per (dataset, n_comp)
    agg = df.groupby(["dataset", "n_comp"])["accuracy_mean"].agg(["mean", "std"]).reset_index()
    agg.columns = ["dataset", "n_comp", "acc_mean", "acc_std"]
    agg["acc_std"] = agg["acc_std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    datasets = sorted(agg["dataset"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, ds in enumerate(datasets):
        sub = agg[agg["dataset"] == ds].sort_values("n_comp")
        x = sub["n_comp"].values
        y = sub["acc_mean"].values
        yerr = sub["acc_std"].values
        color = colors[i % len(colors)]
        ax.plot(x, y, "o-", label=ds, color=color, linewidth=2, markersize=6)
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, color=color)

    ax.set_xlabel("CSP n_components")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 8: CSP_BPE_LogReg — Accuracy vs CSP Components")
    ax.set_xticks(_NCOMP_SWEEP_VALUES)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(0.6, agg["acc_mean"].max() + 0.1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = plot_dir / "exp8_ncomp_sweep.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    logger.info(f"Exp 8 n_comp sweep plot saved to {out_path}")


def _plot_exp8_heatmap(results: list[dict], plot_dir: Path) -> None:
    """Heatmap of mean accuracy: rows = datasets, columns = classifiers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping Exp 8 heatmap")
        return

    import pandas as pd
    df = pd.DataFrame(results)
    pivot = df.groupby(["dataset", "classifier"])["accuracy_mean"].mean().reset_index()
    pivot = pivot.pivot(index="dataset", columns="classifier", values="accuracy_mean")

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(3, len(pivot.index) * 1.2)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold")

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Exp 8: CSP + BPE — Mean Accuracy (dataset x classifier)")
    fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
    plt.tight_layout()
    out_path = plot_dir / "exp8_csp_bpe_heatmap.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    logger.info(f"Exp 8 heatmap saved to {out_path}")


def _plot_exp8(results: list[dict], plot_dir: Path) -> None:
    """Bar chart of CSP+BPE accuracy vs dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping Exp 8 plot")
        return

    import pandas as pd
    df = pd.DataFrame(results)
    df_mean = df.groupby(["dataset", "classifier"])["accuracy_mean"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = df_mean["dataset"].unique()
    classifiers = df_mean["classifier"].unique()
    x = np.arange(len(datasets))
    width = 0.35

    for i, clf in enumerate(classifiers):
        vals = []
        for ds in datasets:
            row = df_mean[(df_mean["dataset"] == ds) & (df_mean["classifier"] == clf)]
            vals.append(float(row["accuracy_mean"].iloc[0]) if len(row) > 0 else 0.0)
        ax.bar(x + i * width - width / 2, vals, width, label=clf)

    ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8, label="Chance (4-class)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 8: CSP + BPE on Motor Imagery")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    out_path = plot_dir / "exp8_accuracy_bar.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    logger.info(f"Exp 8 plot saved to {out_path}")
