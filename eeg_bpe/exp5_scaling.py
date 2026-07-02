"""
Experiment 5: Scaling Laws & Efficiency (H4)
=============================================
Determines optimal operating point for BPE vocabulary size.
Measures: compression ratio, downstream accuracy, throughput, memory.

Produces Pareto frontiers and elbow plots.
"""
from __future__ import annotations

import numpy as np
import time
from pathlib import Path

from .config import (
    SCALING_VOCAB_SIZES, DATASET_INFO, N_JOBS, RANDOM_SEEDS,
    LOGS_DIR, PLOTS_DIR, MODELS_DIR,
)
from .bpe_engine import BPEVocab, apply_bpe_batch, compute_token_stats, train_bpe
from .exp1_vocab_analysis import dataset_to_sequences
from .exp2_downstream import (
    epochs_to_bpe_histograms, classify_histogram_logreg,
    run_loso_cv, run_kfold_cv, compute_classification_metrics,
)
from .data_loading import load_dataset
from .utils import ExperimentLogger, save_json, save_csv, timed, get_logger

from sklearn.preprocessing import LabelEncoder

logger = get_logger("exp5")


# ─── Throughput measurement ──────────────────────────────────────────────────

def measure_throughput(epochs: np.ndarray, vocab: BPEVocab,
                       method: str = "mu_law",
                       n_bins: int = 256) -> dict:
    """
    Measure tokenization throughput: samples/second and trials/second.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.
    vocab : BPEVocab
        Trained BPE vocabulary.
    method : str, optional
        Quantization method (default ``"mu_law"``).
    n_bins : int, optional
        Number of quantization bins (default 256).

    Returns
    -------
    dict
        ``samples_per_second``, ``trials_per_second``, ``wall_time_50_trials``.
    """
    n_trials = epochs.shape[0]

    t0 = time.perf_counter()
    _ = epochs_to_bpe_histograms(epochs[:min(50, n_trials)], vocab,
                                  method, n_bins)
    elapsed = time.perf_counter() - t0

    n_processed = min(50, n_trials)
    samples_per_trial = epochs.shape[1] * epochs.shape[2]

    return {
        "samples_per_second": float(n_processed * samples_per_trial / elapsed),
        "trials_per_second": float(n_processed / elapsed),
        "wall_time_50_trials": float(elapsed),
    }


# ─── Changepoint detection (simple elbow) ────────────────────────────────────

def find_elbow(x: np.ndarray, y: np.ndarray) -> int:
    """
    Simple elbow detection via maximum curvature.

    Parameters
    ----------
    x : np.ndarray
        Horizontal axis values.
    y : np.ndarray
        Vertical axis values.

    Returns
    -------
    int
        Index of the elbow point in the input arrays.
    """
    if len(x) < 3:
        return 0

    # Normalize
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Line from first to last point
    dx = x_norm[-1] - x_norm[0]
    dy = y_norm[-1] - y_norm[0]
    line_len = np.sqrt(dx ** 2 + dy ** 2)

    # Distance from each point to the line
    distances = np.abs(dy * x_norm - dx * y_norm +
                       x_norm[-1] * y_norm[0] - y_norm[-1] * x_norm[0]) / (line_len + 1e-10)

    return int(np.argmax(distances))


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pareto_frontier(results: list[dict], save_dir: Path) -> None:
    """
    Plot compression ratio vs. accuracy (Pareto frontier).

    Parameters
    ----------
    results : list of dict
        Experiment 5 result dicts.
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in results))

    for ds in datasets:
        subset = sorted(
            [r for r in results if r["dataset"] == ds],
            key=lambda r: r["vocab_size"]
        )
        if not subset:
            continue

        comp = [r["compression_ratio"] for r in subset]
        acc = [r["accuracy_mean"] for r in subset]
        vs = [r["vocab_size"] for r in subset]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(comp, acc, "o-", color="#1f77b4")
        for i, v in enumerate(vs):
            ax.annotate(f"V={v}", (comp[i], acc[i]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

        # Mark elbow
        comp_arr = np.array(comp)
        acc_arr = np.array(acc)
        elbow_idx = find_elbow(comp_arr, acc_arr)
        ax.plot(comp[elbow_idx], acc[elbow_idx], "r*", markersize=15,
                label=f"Optimal V*={vs[elbow_idx]}")

        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"Pareto Frontier — {ds}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"pareto_{ds}.png", dpi=150)
        plt.close(fig)


def plot_elbow(results: list[dict], save_dir: Path) -> None:
    """
    Plot accuracy vs. vocabulary size (elbow plot).

    Parameters
    ----------
    results : list of dict
        Experiment 5 result dicts.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in results))

    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets:
        subset = sorted(
            [r for r in results if r["dataset"] == ds],
            key=lambda r: r["vocab_size"]
        )
        vs = [r["vocab_size"] for r in subset]
        acc = [r["accuracy_mean"] for r in subset]
        ax.plot(vs, acc, "o-", label=ds)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Vocabulary Size (V)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs. Vocabulary Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "elbow_accuracy_vs_V.png", dpi=150)
    plt.close(fig)


def plot_throughput(results: list[dict], save_dir: Path) -> None:
    """
    Plot throughput vs. vocabulary size.

    Parameters
    ----------
    results : list of dict
        Experiment 5 result dicts with throughput data.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter results with throughput data
    with_tp = [r for r in results if "samples_per_second" in r]
    if not with_tp:
        return

    datasets = sorted(set(r["dataset"] for r in with_tp))

    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets:
        subset = sorted(
            [r for r in with_tp if r["dataset"] == ds],
            key=lambda r: r["vocab_size"]
        )
        vs = [r["vocab_size"] for r in subset]
        tp = [r["samples_per_second"] for r in subset]
        ax.plot(vs, tp, "o-", label=ds)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Vocabulary Size (V)")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Tokenization Throughput vs. Vocabulary Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_dir / "throughput_vs_V.png", dpi=150)
    plt.close(fig)


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp5")
def run_experiment_5(datasets: list[str] | None = None,
                     vocab_sizes: list[int] | None = None,
                     max_subjects: int = 5,
                     n_bins: int = 64,
                     method: str = "uniform") -> list[dict]:
    """
    Run Experiment 5: Scaling laws.

    Requires BPE vocabs from Experiment 1 to be already trained.

    Parameters
    ----------
    datasets : list of str or None, optional
        Datasets to evaluate (default: all).
    vocab_sizes : list of int or None, optional
        Vocabulary sizes to test (default from config).
    max_subjects : int, optional
        Max subjects per dataset (default 5).
    n_bins : int, optional
        Number of quantization bins (default 64, per A2 ablation).
    method : str, optional
        Quantization method (default ``"uniform"``, per A1 ablation).

    Returns
    -------
    list of dict
        Scaling result dicts with accuracy, compression, and throughput.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    if vocab_sizes is None:
        vocab_sizes = SCALING_VOCAB_SIZES

    exp_log = ExperimentLogger("exp5_scaling")
    all_results = []

    # Resume: load previously computed (dataset, vocab_size) results.
    csv_path = LOGS_DIR / "exp5_scaling_results.csv"
    _done5: set = set()
    if csv_path.exists():
        try:
            import pandas as pd
            _df5 = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
            for _, _row5 in _df5.iterrows():
                _done5.add((str(_row5["dataset"]), str(int(_row5["vocab_size"]))))
                all_results.append({k: v for k, v in _row5.items()
                                    if not (isinstance(v, float)
                                            and __import__("math").isnan(v))})
        except Exception:
            pass

    for ds_name in datasets:
        info = DATASET_INFO[ds_name]
        exp_log.info(f"=== Dataset: {ds_name} ===")

        # Skip entire dataset if all vocab sizes already computed.
        if all((str(ds_name), str(V)) in _done5 for V in vocab_sizes):
            exp_log.info(f"  All vocab sizes done for {ds_name}, skipping")
            continue

        try:
            data = load_dataset(ds_name, max_subjects=max_subjects)
        except Exception as e:
            exp_log.error(f"Failed: {ds_name}: {e}")
            continue

        if not data:
            continue

        # Assemble epochs
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
        X_epochs = np.concatenate(padded, axis=0)
        y = LabelEncoder().fit_transform(np.array(all_labels))
        groups = np.array(all_groups)

        # Prepare quantized sequences for compression measurement.
        # Use dataset_to_sequences() (same as exp1) so that the sequence
        # format matches what the BPE vocab was trained on: long per-channel
        # sequences (all trials concatenated), not short per-trial windows.
        seqs = dataset_to_sequences(ds_name, max_subjects=max_subjects,
                                    n_bins=n_bins, method=method)

        for V in vocab_sizes:
            if (str(ds_name), str(V)) in _done5:
                exp_log.info(f"  V={V} on {ds_name} — already done, skipping")
                continue
            vocab_path = MODELS_DIR / f"bpe_vocab_V{V}_B{n_bins}_{method}.json"
            if not vocab_path.exists():
                exp_log.info(f"Vocab V={V} not found — training on {ds_name}...")
                vocab = train_bpe(seqs, vocab_size=V, base_vocab_size=n_bins,
                                  verbose=False)
                vocab.save(str(vocab_path))
                exp_log.info(f"Vocab V={V} trained and saved to {vocab_path.name}")
            else:
                vocab = BPEVocab.load(str(vocab_path))

            # Compression
            bpe_seqs = apply_bpe_batch(seqs[:200], vocab, n_jobs=N_JOBS)
            stats = compute_token_stats(bpe_seqs, vocab)

            # Throughput
            tp = measure_throughput(X_epochs[:50], vocab, method, n_bins)

            # Downstream accuracy (quick: 1 seed, histogram+logreg)
            # Memory guard: estimate (n_trials, n_ch, V) float32 size.
            _mem_bytes = int(X_epochs.shape[0]) * int(X_epochs.shape[1]) * int(V) * 4
            _MEM_LIMIT = 8_000_000_000  # 8 GB
            _X_epochs_cv = X_epochs
            _y_cv = y
            _groups_cv = groups
            if _mem_bytes > _MEM_LIMIT:
                # Stratified subsample to fit within memory limit
                _max_trials = max(500, int(_MEM_LIMIT / (int(X_epochs.shape[1]) * int(V) * 4)))
                _max_trials = min(_max_trials, X_epochs.shape[0])
                exp_log.info(
                    f"  V={V} histogram would need {_mem_bytes/1e9:.1f} GB — "
                    f"subsampling to {_max_trials} trials for accuracy estimate"
                )
                from sklearn.model_selection import StratifiedShuffleSplit
                _sss = StratifiedShuffleSplit(n_splits=1, train_size=_max_trials,
                                              random_state=42)
                _idx, _ = next(_sss.split(X_epochs, y))
                _X_epochs_cv = X_epochs[_idx]
                _y_cv = y[_idx]
                _groups_cv = groups[_idx]
            X_hist = epochs_to_bpe_histograms(_X_epochs_cv, vocab, method, n_bins)
            seed = RANDOM_SEEDS[0]

            cv_strategy = info["cv_strategy"]
            if cv_strategy == "LOSO":
                folds = run_loso_cv(X_hist, _y_cv, _groups_cv,
                                     classify_histogram_logreg, seed=seed)
            else:
                folds = run_kfold_cv(X_hist, _y_cv, _groups_cv,
                                      classify_histogram_logreg, seed=seed)

            accs = [f["accuracy"] for f in folds]

            result = {
                "dataset": ds_name,
                "vocab_size": V,
                "compression_ratio": stats["compression_ratio"],
                "accuracy_mean": float(np.mean(accs)),
                "accuracy_std": float(np.std(accs)),
                "n_folds": len(folds),
                **tp,
            }
            exp_log.log_result(result)
            all_results.append(result)

    # Plots
    plot_dir = PLOTS_DIR / "exp5"
    plot_pareto_frontier(all_results, plot_dir)
    plot_elbow(all_results, plot_dir)
    plot_throughput(all_results, plot_dir)

    # Find optimal V* per dataset
    optimal_v_star = {}
    for ds_name in datasets:
        subset = sorted(
            [r for r in all_results if r["dataset"] == ds_name],
            key=lambda r: r["vocab_size"]
        )
        if not subset:
            continue
        vs = np.array([r["vocab_size"] for r in subset])
        accs = np.array([r["accuracy_mean"] for r in subset])
        elbow_idx = find_elbow(np.log2(vs), accs)
        v_star = int(subset[elbow_idx]["vocab_size"])
        optimal_v_star[ds_name] = {
            "V_star": v_star,
            "accuracy": float(accs[elbow_idx]),
            "compression_ratio": float(subset[elbow_idx].get("compression_ratio", 0)),
        }
        exp_log.info(f"Optimal V* for {ds_name}: {v_star} "
                     f"(acc={accs[elbow_idx]:.3f})")

    # L2: Save optimal V* to JSON
    save_json(optimal_v_star, LOGS_DIR / "exp5_optimal_v_star.json")
    exp_log.info(f"Saved optimal V* mapping to exp5_optimal_v_star.json")

    exp_log.finalize()
    logger.info(f"Experiment 5 complete: {len(all_results)} data points")
    return all_results


if __name__ == "__main__":
    run_experiment_5()
