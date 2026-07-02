"""
Experiment 7: Fourier-Guided BPE Tokenisation  (Paper 1a)
==========================================================
Tests the hypothesis that applying BPE to *spectral code sequences*
(STFT log-power → k-means codebook → BPE) outperforms standard amplitude
BPE on frequency-coded paradigms (Motor Imagery, SSVEP) while remaining
competitive on amplitude-coded paradigms (Sleep, P300).

Pipeline (per dataset):
    1. Compute STFT log-power: (n_trials, n_ch, n_windows, n_freq)
    2. Fit k-means spectral codebook on training data  →  n_codes clusters
    3. Encode all data: (n_trials, n_ch, n_windows)  int codes
    4. Train BPE on code sequences  →  BPEVocab (base_vocab_size = n_codes)
    5. Apply BPE  →  per-channel histogram  →  LogReg / RF

Key comparisons:
    • Fourier-BPE (ours)  vs  raw BPE_Hist_LogReg (Exp 2)
      → measures gain from spectral representation
    • Fourier-BPE         vs  spectral_500ms (Exp 6)
      → measures gain from joint VQ vs. independent freq-bin quantisation
    • Fourier-BPE         vs  PSD_LogReg
      → measures whether BPE motifs add value over direct spectral features
    • Fourier-BPE         vs  SSVEP_FFT_LogReg (SSVEP only)

Outputs:
    results/logs/exp7_fourier_bpe_results.csv
    results/logs/exp7_per_fold.csv
    results/logs/exp7_fourier_bpe_log.json
    results/plots/exp7/exp7_accuracy_heatmap.png
    results/plots/exp7/exp7_vs_raw_bpe.png
    results/models/fourier_codebook_C{n_codes}_W{win_ms}ms_{ds_name}.pkl
    results/models/fourier_bpe_vocab_V{V}_C{n_codes}_W{win_ms}ms_{ds_name}.json
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import (
    DATASET_INFO, LOGS_DIR, MODELS_DIR, PLOTS_DIR, RANDOM_SEEDS,
    N_JOBS, DEVICE,
    LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL,
)
from .data_loading import load_dataset
from .bpe_engine import BPEVocab
from .fourier_bpe import (
    compute_stft_log_power, fit_spectral_codebook, encode_with_codebook,
    train_fourier_bpe, fourier_bpe_histograms, fourier_bpe_compression,
    cached_fourier_bpe_histograms, save_codebook, load_codebook,
)
from .utils import (
    get_logger, append_csv, save_json, ExperimentLogger, bootstrap_ci,
)

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

logger = get_logger("exp7_fourier_bpe")

# ─── Constants ────────────────────────────────────────────────────────────────

_N_BOOTSTRAP     = 500
_FMIN            = 1.0
_FMAX            = 45.0
_NORMALIZE_SPEC  = True   # z-score each spectrum by frequency axis
_MAX_HIST_FEATS  = 2048   # SVD reduction threshold


# ─── Model paths ──────────────────────────────────────────────────────────────

def _codebook_path(ds_name: str, n_codes: int, win_ms: int) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"fourier_codebook_C{n_codes}_W{win_ms}ms_{ds_name}.pkl"


def _vocab_path(ds_name: str, vocab_size: int, n_codes: int, win_ms: int) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"fourier_bpe_vocab_V{vocab_size}_C{n_codes}_W{win_ms}ms_{ds_name}.json"


# ─── Spectral codebook: train or load ─────────────────────────────────────────

def get_or_train_codebook(
    ds_name: str,
    epochs_train: np.ndarray,
    sfreq: float,
    n_codes: int,
    win_sec: float,
    step_sec: float | None = None,
) -> object:
    """Load cached spectral codebook or train a new one."""
    win_ms = int(win_sec * 1000)
    cp = _codebook_path(ds_name, n_codes, win_ms)

    if cp.exists():
        logger.info(f"  Codebook cache hit: {cp.name}")
        return load_codebook(cp)

    logger.info(f"  Training spectral codebook ({ds_name}, C={n_codes}, W={win_ms}ms)…")
    spectral, freqs = compute_stft_log_power(
        epochs_train, sfreq,
        win_sec=win_sec, step_sec=step_sec,
        fmin=_FMIN, fmax=_FMAX, normalize=_NORMALIZE_SPEC,
    )
    codebook = fit_spectral_codebook(spectral, n_codes=n_codes)
    logger.info(f"  Codebook: inertia={codebook.inertia_:.1f}, "
                f"n_empty={sum(1 for c in np.bincount(codebook.labels_) if c == 0)}")
    save_codebook(codebook, cp)
    logger.info(f"  Codebook saved → {cp.name}  (n_freq={freqs.size}, freqs={freqs[[0,-1]].round(1)} Hz)")
    return codebook


# ─── BPE vocab: train or load ─────────────────────────────────────────────────

def get_or_train_vocab(
    ds_name: str,
    codes_train: np.ndarray,
    vocab_size: int,
    n_codes: int,
    win_sec: float,
    max_train_tokens: int = 5_000_000,
) -> BPEVocab:
    """Load cached Fourier-BPE vocab or train a new one."""
    win_ms = int(win_sec * 1000)
    vp = _vocab_path(ds_name, vocab_size, n_codes, win_ms)

    if vp.exists():
        logger.info(f"  Vocab cache hit: {vp.name}")
        vocab = BPEVocab.load(str(vp))
        if vocab.vocab_size >= vocab_size:
            return vocab
        logger.info(f"  Cached vocab size {vocab.vocab_size} < {vocab_size}; retraining.")

    logger.info(f"  Training Fourier-BPE vocab ({ds_name}, V={vocab_size}, C={n_codes})…")
    vocab = train_fourier_bpe(
        codes_train, vocab_size=vocab_size, n_codes=n_codes,
        max_train_tokens=max_train_tokens,
    )
    vocab.save(str(vp))
    logger.info(f"  Vocab saved → {vp.name}")
    return vocab


# ─── LogReg helper ────────────────────────────────────────────────────────────

def _logreg(seed: int = 42) -> LogisticRegression:
    return LogisticRegression(
        solver=LOGREG_SOLVER, max_iter=LOGREG_MAX_ITER, C=LOGREG_C,
        class_weight=LOGREG_CLASS_WEIGHT, tol=LOGREG_TOL,
        random_state=seed, n_jobs=1,
    )


# ─── CV helpers ───────────────────────────────────────────────────────────────

def _run_fold(X_tr, y_tr, X_te, y_te, fold_id: int, seed: int) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    clf = _logreg(seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return {
        "fold":     fold_id,
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "macro_f1": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "kappa":    float(cohen_kappa_score(y_te, y_pred)),
    }


def _run_cv(X, y, groups, cv_strategy: str, seed: int) -> list[dict]:
    from sklearn.model_selection import LeaveOneGroupOut
    results = []
    if cv_strategy == "LOSO":
        for fold_id, (tr, te) in enumerate(
                LeaveOneGroupOut().split(X, y, groups)):
            results.append(_run_fold(X[tr], y[tr], X[te], y[te], fold_id, seed))
    else:
        unique_subj = np.unique(groups)
        rng = np.random.default_rng(seed)
        unique_subj = rng.permutation(unique_subj)
        n_splits = min(5, len(unique_subj))
        fold_size = max(1, len(unique_subj) // n_splits)
        for fold_id in range(n_splits):
            start = fold_id * fold_size
            test_subj = (unique_subj[start:]
                         if fold_id == n_splits - 1
                         else unique_subj[start:start + fold_size])
            mask = np.isin(groups, test_subj)
            tr, te = np.where(~mask)[0], np.where(mask)[0]
            if len(tr) == 0 or len(te) == 0:
                continue
            results.append(_run_fold(X[tr], y[tr], X[te], y[te], fold_id, seed))
    return results


# ─── Dimensionality reduction ─────────────────────────────────────────────────

def _maybe_reduce(X: np.ndarray, label: str) -> np.ndarray:
    if X.shape[1] <= _MAX_HIST_FEATS:
        return X
    n_comp = min(_MAX_HIST_FEATS, X.shape[0] - 1)
    logger.info(f"    SVD: {X.shape[1]} → {n_comp} features ({label})")
    try:
        from .utils import pca_reduce
        return pca_reduce(X, n_comp, device=DEVICE)
    except Exception:
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        return svd.fit_transform(X)


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_vs_baseline(results: list[dict], output_dir: Path) -> None:
    """Grouped bar chart: best Fourier-BPE vs best BPE_Hist_LogReg from Exp 2."""
    exp2_csv = LOGS_DIR / "exp2_downstream_results.csv"
    if not exp2_csv.exists():
        logger.warning("exp2_downstream_results.csv not found — skipping vs-baseline plot")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df_exp2 = pd.read_csv(exp2_csv)
        df_exp7 = pd.DataFrame(results)
        if df_exp7.empty:
            return

        # Best Fourier-BPE accuracy per dataset (across all win_secs / seeds)
        best_exp7 = (df_exp7.groupby("dataset")["accuracy_mean"]
                     .max().to_dict())

        # Best BPE_Hist_LogReg accuracy per dataset from exp2
        mask_bpe = df_exp2["classifier"].astype(str).str.contains("BPE_Hist_LogReg")
        best_exp2 = {}
        if mask_bpe.any():
            best_exp2 = (df_exp2.loc[mask_bpe]
                         .groupby("dataset")["accuracy_mean"]
                         .max().to_dict())

        datasets = sorted(best_exp7.keys())
        if not datasets:
            return

        fourier_vals = [best_exp7.get(ds, 0.0) for ds in datasets]
        bpe_vals = [best_exp2.get(ds, 0.0) for ds in datasets]

        x = np.arange(len(datasets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(datasets) * 1.5), 5))
        ax.bar(x - width / 2, fourier_vals, width, label="Fourier-BPE (Exp 7)",
               color="#4472C4", alpha=0.85)
        ax.bar(x + width / 2, bpe_vals, width, label="BPE_Hist_LogReg (Exp 2)",
               color="#ED7D31", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Exp 7: Fourier-BPE vs Standard BPE")
        ax.legend(fontsize=8)
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / "exp7_vs_raw_bpe.png", dpi=150)
        plt.close(fig)
        logger.info("Saved exp7_vs_raw_bpe.png")

    except Exception as exc:
        logger.warning(f"vs-baseline plot failed: {exc}")


def _plot_results(all_results: list[dict]) -> None:
    """Generate accuracy heatmap and comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        plots_dir = PLOTS_DIR / "exp7"
        plots_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(all_results)
        if df.empty:
            return

        # Pivot: dataset × win_sec
        df["win_label"] = (df["win_sec"] * 1000).astype(int).astype(str) + " ms"
        pivot = df.pivot_table(
            index="dataset", columns="win_label",
            values="accuracy_mean", aggfunc="mean"
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                       vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1%}", ha="center", va="center",
                            fontsize=8, color="black")
        plt.colorbar(im, ax=ax, label="Accuracy")
        ax.set_title("Exp 7: Fourier-BPE Accuracy by Dataset and Window Size")
        plt.tight_layout()
        fig.savefig(plots_dir / "exp7_accuracy_heatmap.png", dpi=150)
        plt.close(fig)
        logger.info("Saved exp7_accuracy_heatmap.png")

        # Bar chart comparing win sizes per dataset
        datasets = df["dataset"].unique()
        win_labels = df["win_label"].unique()
        fig, axes = plt.subplots(1, len(datasets), figsize=(3 * len(datasets), 4),
                                 sharey=True)
        if len(datasets) == 1:
            axes = [axes]
        colors = plt.cm.tab10(np.linspace(0, 1, len(win_labels)))
        for ax, ds in zip(axes, datasets):
            sub = df[df["dataset"] == ds]
            for j, wl in enumerate(win_labels):
                row = sub[sub["win_label"] == wl]
                if row.empty:
                    continue
                acc = float(row["accuracy_mean"].mean())
                ci = float(row["accuracy_std"].mean())
                ax.bar(j, acc, color=colors[j], label=wl, alpha=0.85)
                ax.errorbar(j, acc, yerr=ci, fmt="none", color="black",
                            capsize=3, linewidth=1.2)
            chance = float(sub["chance_level"].mean())
            ax.axhline(chance, color="red", linestyle="--", linewidth=0.8,
                       label=f"chance {chance:.0%}")
            ax.set_title(ds, fontsize=9)
            ax.set_xticks(range(len(win_labels)))
            ax.set_xticklabels(win_labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("Accuracy")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", fontsize=7)
        fig.suptitle("Exp 7: Fourier-BPE per Dataset", fontsize=11)
        plt.tight_layout()
        fig.savefig(plots_dir / "exp7_vs_win_size.png", dpi=150)
        plt.close(fig)
        logger.info("Saved exp7_vs_win_size.png")

    except Exception as exc:
        logger.warning(f"Plotting failed: {exc}")


# ─── Main experiment ───────────────────────────────────────────────────────────

def run_experiment_7(
    datasets: list[str] | None = None,
    vocab_size: int = 1024,
    n_codes: int = 64,
    win_secs: list[float] | None = None,
    max_subjects: int | None = None,
    max_subjects_vocab_train: int = 5,
    max_train_tokens: int = 5_000_000,
    step_sec: float | None = None,
) -> list[dict]:
    """
    Run Experiment 7: Fourier-Guided BPE Tokenisation.

    Parameters
    ----------
    datasets : list[str] or None
        Datasets to evaluate.  Default: all 6.
    vocab_size : int
        Total BPE vocabulary size (n_codes base + n_merges merged).
    n_codes : int
        Spectral codebook size (k-means clusters, analogous to n_bins).
    win_secs : list[float] or None
        STFT window sizes in seconds to test.
        Default: [0.25, 0.5] — tests 4 Hz and 2 Hz spectral resolution.
    max_subjects : int or None
        Max subjects per dataset for classification.
    max_subjects_vocab_train : int
        Subjects used to train spectral codebook + BPE vocab.
    max_train_tokens : int
        Max tokens fed to BPE training (subsampled if exceeded).
    step_sec : float or None
        STFT step size.  Default = win_sec / 2 (50 %% overlap).

    Returns
    -------
    list of dict — per-seed aggregate results.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    if win_secs is None:
        win_secs = [0.25, 0.5]

    (PLOTS_DIR / "exp7").mkdir(parents=True, exist_ok=True)

    exp_log = ExperimentLogger("exp7_fourier_bpe")
    all_results: list[dict] = []

    exp_log.info(
        f"Exp 7 — Fourier-guided BPE: "
        f"datasets={datasets}, V={vocab_size}, n_codes={n_codes}, "
        f"win_secs={win_secs}"
    )

    ds_iter = (
        _tqdm(datasets, desc="Exp7 datasets", unit="ds")
        if _HAS_TQDM else datasets
    )

    for ds_name in ds_iter:
        info      = DATASET_INFO[ds_name]
        sfreq     = float(info["sfreq"])
        cv_strat  = info["cv_strategy"]
        paradigm  = info.get("paradigm", "")
        exp_log.info(f"\n{'='*55}")
        exp_log.info(f"Dataset: {ds_name}  (sfreq={sfreq} Hz, paradigm={paradigm})")

        # ── 1. Load data ──────────────────────────────────────────────────
        try:
            data = load_dataset(ds_name, max_subjects=max_subjects)
        except Exception as exc:
            exp_log.error(f"  Load failed: {exc}")
            continue
        if not data:
            exp_log.warning(f"  No data for {ds_name}")
            continue

        all_epochs, all_labels, all_groups = [], [], []
        for subj_id, subj_data in sorted(data.items()):
            all_epochs.append(subj_data["epochs"])
            all_labels.extend(subj_data["labels"])
            all_groups.extend([subj_id] * len(subj_data["labels"]))

        if not all_epochs:
            continue

        # Pad channels/time to common shape across subjects
        n_ch_max   = max(e.shape[1] for e in all_epochs)
        n_time_max = max(e.shape[2] for e in all_epochs)
        padded = []
        for e in all_epochs:
            p = np.zeros((e.shape[0], n_ch_max, n_time_max), dtype=np.float32)
            p[:, :e.shape[1], :e.shape[2]] = e
            padded.append(p)
        X_all = np.concatenate(padded, axis=0)          # (N_total, n_ch, n_time)

        y_raw    = np.array(all_labels)
        groups   = np.array(all_groups)
        le       = LabelEncoder()
        y_enc    = le.fit_transform(y_raw)
        n_classes = len(le.classes_)
        chance   = 1.0 / n_classes

        # Epochs used for codebook + vocab training (cap to max_subjects_vocab_train)
        unique_subj = np.unique(groups)
        train_subj  = unique_subj[:max_subjects_vocab_train]
        mask_train  = np.isin(groups, train_subj)
        X_train_cb  = X_all[mask_train]
        exp_log.info(
            f"  X_all: {X_all.shape} | X_train_cb: {X_train_cb.shape} "
            f"({len(train_subj)} subj for codebook)"
        )

        # ── 2. Per window size ────────────────────────────────────────────
        for win_sec in win_secs:
            win_ms = int(win_sec * 1000)
            exp_log.info(f"  ── Window: {win_ms} ms ──")

            # Resume: skip if all seeds already completed for this config.
            _e7_pending = list(RANDOM_SEEDS)
            if exp_log.csv_path.exists():
                try:
                    import pandas as pd
                    _df7 = pd.read_csv(exp_log.csv_path)
                    _mask7 = (
                        (_df7["dataset"].astype(str) == str(ds_name)) &
                        (_df7["win_ms"].astype(str) == str(win_ms)) &
                        (_df7["n_codes"].astype(str) == str(n_codes)) &
                        (_df7["vocab_size"].astype(str) == str(vocab_size))
                    )
                    _e7_done = set(_df7.loc[_mask7, "seed"].astype(str).unique())
                    _e7_pending = [s for s in RANDOM_SEEDS if str(s) not in _e7_done]
                except Exception:
                    pass
            if not _e7_pending:
                exp_log.info(f"    [resume] All seeds done — skipping")
                continue

            # ── 2a. Spectral codebook ──────────────────────────────────────
            try:
                codebook = get_or_train_codebook(
                    ds_name, X_train_cb, sfreq, n_codes,
                    win_sec=win_sec, step_sec=step_sec,
                )
            except Exception as exc:
                exp_log.error(f"    Codebook training failed: {exc}")
                continue

            # ── 2b. Encode all data with codebook ──────────────────────────
            try:
                t0 = time.perf_counter()
                spectral_all, freqs = compute_stft_log_power(
                    X_all, sfreq, win_sec=win_sec, step_sec=step_sec,
                    fmin=_FMIN, fmax=_FMAX, normalize=_NORMALIZE_SPEC,
                )
                codes_all = encode_with_codebook(spectral_all, codebook)
                exp_log.info(
                    f"    STFT+encode: {X_all.shape} → codes {codes_all.shape} "
                    f"({time.perf_counter()-t0:.1f}s)  "
                    f"n_windows={codes_all.shape[2]}, n_freq={freqs.size}"
                )
            except Exception as exc:
                exp_log.error(f"    STFT/encode failed: {exc}")
                continue

            # ── 2c. BPE vocab ──────────────────────────────────────────────
            codes_train = codes_all[mask_train]
            try:
                vocab = get_or_train_vocab(
                    ds_name, codes_train, vocab_size, n_codes,
                    win_sec=win_sec, max_train_tokens=max_train_tokens,
                )
            except Exception as exc:
                exp_log.error(f"    Vocab training failed: {exc}")
                continue

            # ── 2d. BPE histogram features ─────────────────────────────────
            try:
                t0 = time.perf_counter()
                X_hist = cached_fourier_bpe_histograms(
                    X_all, sfreq, vocab, codebook,
                    win_sec=win_sec, step_sec=step_sec,
                    fmin=_FMIN, fmax=_FMAX, normalize=_NORMALIZE_SPEC,
                    precomputed_codes=codes_all,
                )
                exp_log.info(
                    f"    Histograms: {X_hist.shape} ({time.perf_counter()-t0:.1f}s)"
                )
            except Exception as exc:
                exp_log.error(f"    Histogram extraction failed: {exc}")
                continue

            X_hist = _maybe_reduce(X_hist, f"{ds_name}/W{win_ms}ms")

            # ── 2e. Compression metrics ────────────────────────────────────
            try:
                comp = fourier_bpe_compression(codes_all, vocab, n_jobs=N_JOBS)
                n_windows = codes_all.shape[2]
                mean_tok_len_windows = n_windows / max(comp, 1e-6)
                mean_tok_len_ms = mean_tok_len_windows * (win_sec * 1000 / 2)  # step = win/2
                exp_log.info(
                    f"    Compression: {comp:.2f}×, "
                    f"mean_token_dur ≈ {mean_tok_len_ms:.0f} ms"
                )
            except Exception as exc:
                exp_log.warning(f"    Compression metric failed: {exc}")
                comp, mean_tok_len_ms = 1.0, 0.0

            # ── 2f. Classification (parallel seeds) ────────────────────────
            def _one_seed(seed, _X=X_hist, _y=y_enc, _g=groups,
                          _cv=cv_strat):
                return _run_cv(_X, _y, _g, _cv, seed)

            n_sjobs = min(len(_e7_pending), max(1, N_JOBS))
            seed_results = Parallel(n_jobs=n_sjobs, prefer="threads")(
                delayed(_one_seed)(s) for s in _e7_pending
            )

            # ── 2g. Log results ────────────────────────────────────────────
            for seed, fold_results in zip(_e7_pending, seed_results):
                if not fold_results:
                    continue
                accs   = [f["accuracy"] for f in fold_results]
                f1s    = [f["macro_f1"] for f in fold_results]
                kappas = [f["kappa"]    for f in fold_results]
                _, ci_lo, ci_hi = bootstrap_ci(
                    np.array(accs), n_boot=_N_BOOTSTRAP,
                )

                result = {
                    "dataset":           ds_name,
                    "paradigm":          paradigm,
                    "win_sec":           win_sec,
                    "win_ms":            win_ms,
                    "n_codes":           n_codes,
                    "vocab_size":        vocab_size,
                    "seed":              seed,
                    "n_classes":         n_classes,
                    "chance_level":      round(chance, 4),
                    "n_trials":          len(y_enc),
                    "n_subjects":        int(unique_subj.size),
                    "n_windows":         int(codes_all.shape[2]),
                    "n_freq":            int(freqs.size),
                    "freq_resolution_hz": round(float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0, 2),
                    "compression_ratio": round(comp, 3),
                    "mean_token_ms":     round(mean_tok_len_ms, 1),
                    "accuracy_mean":     float(np.mean(accs)),
                    "accuracy_std":      float(np.std(accs)),
                    "accuracy_ci_low":   float(ci_lo),
                    "accuracy_ci_high":  float(ci_hi),
                    "macro_f1_mean":     float(np.mean(f1s)),
                    "kappa_mean":        float(np.mean(kappas)),
                    "n_folds":           len(fold_results),
                    "n_hist_features":   int(X_hist.shape[1]),
                    "cv_strategy":       cv_strat,
                }

                exp_log.log_result(result)
                all_results.append(result)

                exp_log.info(
                    f"    [seed={seed}]  acc={np.mean(accs):.1%} ± {np.std(accs):.1%}  "
                    f"κ={np.mean(kappas):.3f}  chance={chance:.1%}"
                )

                # Per-fold CSV (for post-hoc analysis)
                for f in fold_results:
                    append_csv({
                        "dataset":           ds_name,
                        "paradigm":          paradigm,
                        "win_ms":            win_ms,
                        "n_codes":           n_codes,
                        "vocab_size":        vocab_size,
                        "seed":              seed,
                        "fold":              f["fold"],
                        "accuracy":          f["accuracy"],
                        "macro_f1":          f["macro_f1"],
                        "kappa":             f["kappa"],
                        "compression_ratio": comp,
                        "mean_token_ms":     mean_tok_len_ms,
                    }, LOGS_DIR / "exp7_per_fold.csv")

    # ── Summary & plots ───────────────────────────────────────────────────
    exp_log.finalize()
    save_json({"results": all_results}, LOGS_DIR / "exp7_fourier_bpe_log.json")

    _plot_data = all_results
    if not _plot_data and exp_log.csv_path.exists():
        try:
            import pandas as pd
            _plot_data = pd.read_csv(exp_log.csv_path).to_dict("records")
            logger.info(f"Loaded {len(_plot_data)} results from CSV for plotting")
        except Exception:
            pass
    if _plot_data:
        _plot_results(_plot_data)
        _plot_vs_baseline(_plot_data, PLOTS_DIR / "exp7")
        _print_summary(_plot_data)

    return all_results


def _print_summary(results: list[dict]) -> None:
    """Print a formatted summary table to the log."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("EXP 7 — FOURIER-BPE SUMMARY")
    logger.info(f"{'Dataset':<20} {'Win(ms)':<10} {'Acc':>8} {'κ':>8} {'Comp':>8} {'Chance':>8}")
    logger.info("-" * 70)

    seen = set()
    for r in results:
        key = (r["dataset"], r["win_ms"])
        if key in seen:
            continue
        seen.add(key)
        logger.info(
            f"{r['dataset']:<20} {r['win_ms']:<10} "
            f"{r['accuracy_mean']:>8.1%} {r['kappa_mean']:>8.3f} "
            f"{r['compression_ratio']:>8.2f}× {r['chance_level']:>8.1%}"
        )
    logger.info("=" * 70)
