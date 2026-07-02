"""
Experiment 10: Cross-Dataset Vocabulary Transfer
=================================================

Train BPE vocabulary on source dataset, apply to target dataset,
measure downstream classification accuracy.

Tests whether BPE-learned EEG tokens are universal or dataset-specific.

Pipeline (for each source→target pair):
  1. Load source dataset, train BPE vocabulary (or load from cache).
  2. Load target dataset, apply source vocabulary → histogram features.
  3. Classify target dataset with LogReg (LOSO or KFold per config).
  4. Repeat for all seeds; compare native vocab vs transferred vocab accuracy.

Outputs
-------
results/logs/exp10_cross_dataset_results.csv
results/logs/exp10_cross_dataset_results.json
results/plots/exp10/exp10_transfer_matrix.png
results/plots/exp10/exp10_native_vs_transfer.png
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

from .config import (
    DATASET_INFO, DEVICE, RANDOM_SEEDS, N_JOBS, LOGS_DIR, PLOTS_DIR, MODELS_DIR,
    DEFAULT_QUANT_METHOD,
)
from .quantization import quantize
from .bpe_engine import train_bpe, apply_bpe_batch, BPEVocab
from .data_loading import load_dataset
from .exp2_downstream import (
    epochs_to_bpe_histograms, classify_histogram_logreg,
    run_loso_cv, run_kfold_cv,
)
from .utils import (
    ExperimentLogger, save_json, save_csv, timed, get_logger, bootstrap_ci,
    pca_reduce,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = get_logger("exp10_cross_dataset")

_PLOT_DIR = PLOTS_DIR / "exp10"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_and_prepare(ds_name: str, max_subjects: int | None = None):
    """Load dataset and return (epochs, labels, groups, info)."""
    info = DATASET_INFO[ds_name]
    subjects = None
    if max_subjects is not None:
        subjects = list(range(1, max_subjects + 1))
    data = load_dataset(ds_name, subjects=subjects)
    if not data:
        return None, None, None, info

    all_epochs, all_labels, all_groups = [], [], []
    for sid, sd in sorted(data.items()):
        all_epochs.append(sd["epochs"])
        all_labels.append(sd["labels"])
        all_groups.extend([sid] * len(sd["labels"]))

    # Pad channels/time to max across subjects
    max_ch = max(e.shape[1] for e in all_epochs)
    max_t = max(e.shape[2] for e in all_epochs)
    padded = []
    for e in all_epochs:
        if e.shape[1] < max_ch or e.shape[2] < max_t:
            p = np.zeros((e.shape[0], max_ch, max_t), dtype=e.dtype)
            p[:, :e.shape[1], :e.shape[2]] = e
            padded.append(p)
        else:
            padded.append(e)

    epochs = np.concatenate(padded)
    labels = np.concatenate(all_labels)
    groups = np.array(all_groups)
    return epochs, labels, groups, info


def _train_or_load_vocab(ds_name: str, epochs: np.ndarray,
                          method: str, n_bins: int, vocab_size: int) -> BPEVocab:
    """Train BPE vocab on a dataset or load from cache."""
    vocab_path = MODELS_DIR / f"transfer_vocab_{ds_name}_{method}_B{n_bins}_V{vocab_size}.json"
    if vocab_path.exists():
        logger.info(f"  [vocab hit] {vocab_path.name}")
        return BPEVocab.load(str(vocab_path))

    logger.info(f"  Training BPE vocab on {ds_name} (V={vocab_size}, B={n_bins}, {method})")
    flat = epochs.reshape(-1, epochs.shape[-1])
    codes, _ = quantize(flat, method, n_bins, normalize=True)
    seqs = [codes[i].tolist() for i in range(min(5000, len(codes)))]
    vocab = train_bpe(seqs, vocab_size=vocab_size, base_vocab_size=n_bins,
                      verbose=False, max_train_tokens=5_000_000)
    vocab.save(str(vocab_path))
    logger.info(f"  [vocab saved] {vocab_path.name}")
    return vocab


def _reduce_if_large(X: np.ndarray, max_features: int = 2048) -> np.ndarray:
    """Apply PCA if feature dimensionality exceeds *max_features*."""
    if X.shape[1] <= max_features:
        return X
    n_comp = min(max_features, X.shape[0] - 1)
    return pca_reduce(X, n_comp, device=DEVICE)


def _make_result(base: dict, fold_results: list[dict]) -> dict:
    """Aggregate per-fold results into a single summary row."""
    accs = [f["accuracy"] for f in fold_results]
    bal_accs = [f.get("balanced_accuracy", f["accuracy"]) for f in fold_results]
    kappas = [f.get("kappa", 0.0) for f in fold_results]
    _mean, ci_low, ci_high = bootstrap_ci(np.array(accs))
    return {
        **base,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "accuracy_ci_low": float(ci_low),
        "accuracy_ci_high": float(ci_high),
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "kappa_mean": float(np.mean(kappas)),
    }


def _load_csv_done_keys(csv_path, base_dict: dict) -> set:
    """Return set of seed strings already completed for a given config."""
    if not Path(csv_path).exists():
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
    """Check whether all RANDOM_SEEDS are present in CSV for a given config."""
    done = _load_csv_done_keys(csv_path, base_dict)
    return all(str(s) in done for s in RANDOM_SEEDS)


# ─── Plots ───────────────────────────────────────────────────────────────────

def _plot_transfer_matrix(results: list[dict], plot_dir: Path) -> None:
    """Heatmap: source dataset (rows) x target dataset (cols), cell = accuracy."""
    if not results:
        return

    import pandas as pd
    df = pd.DataFrame(results)

    # Average across seeds
    pivot = df.groupby(["source", "target"])["accuracy_mean"].mean()

    sources = sorted(df["source"].unique())
    targets = sorted(df["target"].unique())

    matrix = np.full((len(sources), len(targets)), np.nan)
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            if (src, tgt) in pivot.index:
                matrix[i, j] = pivot[(src, tgt)]

    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.5),
                                     max(6, len(sources) * 1.2)))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8)
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels([s.replace("_", "\n") for s in sources], fontsize=8)
    ax.set_xlabel("Target dataset")
    ax.set_ylabel("Source vocab (trained on)")
    ax.set_title("Cross-Dataset BPE Transfer: Accuracy")

    # Annotate cells
    for i in range(len(sources)):
        for j in range(len(targets)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val > np.nanmean(matrix) else "black"
                weight = "bold" if sources[i] == targets[j] else "normal"
                ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Accuracy")
    fig.tight_layout()
    fig.savefig(plot_dir / "exp10_transfer_matrix.png", dpi=200)
    fig.savefig(plot_dir / "exp10_transfer_matrix.pdf")
    plt.close(fig)
    logger.info("  Saved transfer matrix plot")


def _plot_native_vs_transfer(results: list[dict], plot_dir: Path) -> None:
    """Bar chart: native vocab accuracy vs best transfer accuracy per target."""
    if not results:
        return

    import pandas as pd
    df = pd.DataFrame(results)

    targets = sorted(df["target"].unique())
    native_accs, best_transfer_accs, best_sources = [], [], []

    for tgt in targets:
        tgt_df = df[df["target"] == tgt]
        # Native = source == target
        native = tgt_df[tgt_df["source"] == tgt]["accuracy_mean"].mean()
        # Best transfer = source != target, max accuracy
        transfers = tgt_df[tgt_df["source"] != tgt]
        if transfers.empty:
            best_transfer_accs.append(np.nan)
            best_sources.append("")
        else:
            grouped = transfers.groupby("source")["accuracy_mean"].mean()
            best_src = grouped.idxmax()
            best_transfer_accs.append(grouped.max())
            best_sources.append(best_src)
        native_accs.append(native)

    x = np.arange(len(targets))
    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 1.5), 5))
    w = 0.35
    ax.bar(x - w / 2, native_accs, w, label="Native vocab",
           color="#2196F3", edgecolor="white")
    ax.bar(x + w / 2, best_transfer_accs, w, label="Best transfer",
           color="#FF9800", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in targets], fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Exp 10: Native vs Best-Transfer Vocabulary")
    ax.legend()

    valid_transfer = [v for v in best_transfer_accs if not np.isnan(v)]
    y_max = max(max(native_accs), max(valid_transfer)) if valid_transfer else max(native_accs)
    ax.set_ylim(0, y_max * 1.15)

    # Annotate best source name
    for i, src in enumerate(best_sources):
        if src:
            ax.text(x[i] + w / 2, best_transfer_accs[i] + 0.01,
                    src.replace("_", "\n"),
                    ha="center", va="bottom", fontsize=6, style="italic")

    fig.tight_layout()
    fig.savefig(plot_dir / "exp10_native_vs_transfer.png", dpi=200)
    fig.savefig(plot_dir / "exp10_native_vs_transfer.pdf")
    plt.close(fig)
    logger.info("  Saved native vs transfer plot")


# ─── Main experiment ─────────────────────────────────────────────────────────

@timed("exp10")
def run_experiment_10(
    datasets: list[str] | None = None,
    vocab_size: int = 1024,
    n_bins: int = 64,
    method: str | None = None,
    max_subjects: int | None = None,
) -> dict:
    """
    Cross-dataset vocabulary transfer experiment.

    For each pair (source, target) of datasets:
    1. Train BPE vocab on source dataset
    2. Apply that vocab to target dataset -> histogram features
    3. Classify target dataset with LogReg (LOSO/KFold)
    4. Compare native vocab vs transferred vocab accuracy

    Parameters
    ----------
    datasets : list of str, optional
        Dataset names to include. Defaults to five amplitude/frequency-coded
        datasets (sleep_edf, mental_arithmetic, epfl_p300, bci_iv_2a,
        physionet_mi).
    vocab_size : int
        BPE vocabulary size (default 1024).
    n_bins : int
        Number of quantization bins (default 64).
    method : str, optional
        Quantization method (default from config: ``DEFAULT_QUANT_METHOD``).
    max_subjects : int, optional
        Cap on subjects per dataset (useful for quick runs).

    Returns
    -------
    dict
        Summary with ``n_results`` and ``datasets``.
    """
    if method is None:
        method = DEFAULT_QUANT_METHOD

    if datasets is None:
        # Use datasets where BPE has been evaluated in Exp 2
        datasets = ["sleep_edf", "mental_arithmetic", "epfl_p300",
                     "bci_iv_2a", "physionet_mi"]

    exp_log = ExperimentLogger("exp10_cross_dataset")
    all_results: list[dict] = []

    # ── Pre-load all datasets and train vocabs ──────────────────────────
    dataset_cache: dict[str, tuple] = {}
    vocab_cache: dict[str, BPEVocab] = {}

    for ds_name in datasets:
        logger.info(f"Loading {ds_name}...")
        epochs, labels, groups, info = _load_and_prepare(ds_name, max_subjects)
        if epochs is None:
            logger.warning(f"  {ds_name}: no data loaded — skipping")
            continue
        y = LabelEncoder().fit_transform(labels)
        dataset_cache[ds_name] = (epochs, y, groups, info)

        # Train/load vocab for this dataset
        vocab_cache[ds_name] = _train_or_load_vocab(
            ds_name, epochs, method, n_bins, vocab_size)

    # ── Run all (source, target) pairs ──────────────────────────────────
    for source_ds in datasets:
        if source_ds not in vocab_cache:
            continue
        source_vocab = vocab_cache[source_ds]

        for target_ds in datasets:
            if target_ds not in dataset_cache:
                continue

            epochs_tgt, y_tgt, groups_tgt, info_tgt = dataset_cache[target_ds]

            base_dict = {
                "experiment": "exp10",
                "source": source_ds,
                "target": target_ds,
                "vocab_size": vocab_size,
                "n_bins": n_bins,
                "method": method,
                "classifier": "BPE_Hist_LogReg",
            }

            # Resume check — skip if all seeds already in CSV
            if _all_seeds_done(exp_log.csv_path, base_dict):
                logger.info(f"  {source_ds} -> {target_ds}: all seeds done — skipping")
                # Load existing results for plot generation
                try:
                    import pandas as pd
                    _df = pd.read_csv(exp_log.csv_path,
                                      on_bad_lines="skip", engine="python")
                    mask = pd.Series([True] * len(_df), index=_df.index)
                    for k, v in base_dict.items():
                        if k in _df.columns:
                            mask &= (_df[k].astype(str) == str(v))
                    for _, row in _df[mask].iterrows():
                        all_results.append(row.to_dict())
                except Exception:
                    pass
                continue

            logger.info(f"  Transfer: {source_ds} -> {target_ds}")

            # Apply source vocab to target data
            X_hist = epochs_to_bpe_histograms(epochs_tgt, source_vocab,
                                               method, n_bins)

            # PCA if feature dimensionality is very high
            X_hist = _reduce_if_large(X_hist)

            # Run classification with multiple seeds
            done_seeds = _load_csv_done_keys(exp_log.csv_path, base_dict)
            seeds_to_run = [s for s in RANDOM_SEEDS if str(s) not in done_seeds]

            for seed in seeds_to_run:
                if info_tgt["cv_strategy"] == "LOSO":
                    folds = run_loso_cv(X_hist, y_tgt, groups_tgt,
                                        classify_histogram_logreg, seed=seed)
                else:
                    folds = run_kfold_cv(X_hist, y_tgt, groups_tgt,
                                         classify_histogram_logreg, seed=seed)

                result = _make_result({**base_dict, "seed": seed}, folds)
                exp_log.log_result(result)
                all_results.append(result)

                is_native = "NATIVE" if source_ds == target_ds else "TRANSFER"
                logger.info(
                    f"    [{is_native}] {source_ds}->{target_ds} seed={seed}: "
                    f"acc={result['accuracy_mean']:.1%} "
                    f"kappa={result['kappa_mean']:.3f}"
                )

    # ── Plots ───────────────────────────────────────────────────────────
    _plot_results = all_results
    if not _plot_results and exp_log.csv_path.exists():
        try:
            import pandas as pd
            _plot_results = pd.read_csv(exp_log.csv_path).to_dict("records")
        except Exception:
            pass

    _plot_transfer_matrix(_plot_results, _PLOT_DIR)
    _plot_native_vs_transfer(_plot_results, _PLOT_DIR)

    exp_log.finalize()

    n_new = len([r for r in all_results if "seed" in r])
    logger.info(f"Exp 10 complete: {n_new} results")
    return {"n_results": n_new, "datasets": datasets}
