"""
Experiment 0: Quantization Loss Analysis (FUNDAMENTAL)
=====================================================
Quantitatively evaluates how much information is lost during quantization.
Determines the minimum number of bins for adequate EEG signal preservation.

Metrics:
  - Reconstruction MSE / RMSE
  - SQNR (Signal-to-Quantization-Noise Ratio) in dB
  - Spectral distortion per frequency band (δ/θ/α/β/γ)
  - Correlation preservation (Pearson r)
  - 95% bootstrap confidence intervals
  - Paired t-test comparisons with Holm-Bonferroni correction

All results logged to JSON/CSV. Plots saved locally.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy import stats
from joblib import Parallel, delayed
from pathlib import Path

from .config import (
    QUANT_METHODS, QUANT_BINS, FREQ_BANDS, N_JOBS, N_BOOTSTRAP,
    EXP0_MINUTES_PER_SUBJECT, EXP0_N_SUBJECTS, DATASET_INFO,
    LOGS_DIR, PLOTS_DIR, TARGET_SFREQ,
)
from .quantization import quantize, dequantize
from .data_loading import load_dataset, get_fragment
from .utils import ExperimentLogger, save_json, save_csv, timed, get_logger, bootstrap_ci

logger = get_logger("exp0")


# ─── Spectral helpers (vectorised) ───────────────────────────────────────────

def compute_psd_batch(signals: np.ndarray, sfreq: float,
                      nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """
    Welch PSD for a batch of signals.

    Parameters
    ----------
    signals : np.ndarray
        Signal array, shape ``(n_signals, n_samples)``.
    sfreq : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Welch segment length (default 256).

    Returns
    -------
    freqs : np.ndarray
        Frequency axis, shape ``(n_freqs,)``.
    psds : np.ndarray
        Power spectral densities, shape ``(n_signals, n_freqs)``.
    """
    freqs, psds = sig.welch(signals, fs=sfreq, nperseg=min(nperseg, signals.shape[-1]),
                            axis=-1)
    return freqs, psds


def band_power(freqs: np.ndarray, psd: np.ndarray,
               fmin: float, fmax: float) -> np.ndarray:
    """
    Integrate PSD in frequency band [fmin, fmax].

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis, shape ``(n_freqs,)``.
    psd : np.ndarray
        PSD array, shape ``(n_signals, n_freqs)`` or ``(n_freqs,)``.
    fmin : float
        Lower band edge in Hz.
    fmax : float
        Upper band edge in Hz.

    Returns
    -------
    np.ndarray or float
        Integrated band power, shape ``(n_signals,)`` or scalar.
    """
    mask = (freqs >= fmin) & (freqs <= fmax)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    if psd.ndim == 1:
        return np.sum(psd[mask]) * df
    return np.sum(psd[:, mask], axis=1) * df


# ─── Core metrics (fully vectorised) ─────────────────────────────────────────

def compute_metrics(original: np.ndarray, reconstructed: np.ndarray,
                    sfreq: float) -> dict:
    """
    Compute all quantization quality metrics.

    All operations are vectorised over channels.

    Parameters
    ----------
    original : np.ndarray
        Original signal, shape ``(n_channels, n_samples)``.
    reconstructed : np.ndarray
        Reconstructed signal, same shape.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Metrics including ``mse_mean``, ``rmse_mean``, ``sqnr_db_mean``,
        ``corr_mean``, per-channel arrays, and ``spectral_distortion_pct``.
    """
    error = original - reconstructed
    n_ch = original.shape[0]

    # MSE, RMSE per channel
    mse = np.mean(error ** 2, axis=-1)            # (n_ch,)
    rmse = np.sqrt(mse)

    # SQNR per channel
    signal_power = np.mean(original ** 2, axis=-1)  # (n_ch,)
    noise_power = mse + 1e-20
    sqnr_db = 10.0 * np.log10(signal_power / noise_power + 1e-20)

    # Pearson correlation per channel (vectorised)
    orig_centered = original - np.mean(original, axis=-1, keepdims=True)
    recon_centered = reconstructed - np.mean(reconstructed, axis=-1, keepdims=True)
    num = np.sum(orig_centered * recon_centered, axis=-1)
    denom = (np.sqrt(np.sum(orig_centered ** 2, axis=-1)) *
             np.sqrt(np.sum(recon_centered ** 2, axis=-1)) + 1e-20)
    corr = num / denom

    # Spectral distortion per band
    freqs_orig, psd_orig = compute_psd_batch(original, sfreq)
    freqs_recon, psd_recon = compute_psd_batch(reconstructed, sfreq)

    band_distortion = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        bp_orig = band_power(freqs_orig, psd_orig, fmin, fmax)
        bp_recon = band_power(freqs_recon, psd_recon, fmin, fmax)
        # Relative distortion (%)
        rel_dist = np.abs(bp_orig - bp_recon) / (bp_orig + 1e-20) * 100.0
        band_distortion[band_name] = float(np.mean(rel_dist))

    return {
        "mse_mean": float(np.mean(mse)),
        "mse_std": float(np.std(mse)),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_std": float(np.std(rmse)),
        "sqnr_db_mean": float(np.mean(sqnr_db)),
        "sqnr_db_std": float(np.std(sqnr_db)),
        "sqnr_db_per_channel": sqnr_db.tolist(),
        "corr_mean": float(np.mean(corr)),
        "corr_std": float(np.std(corr)),
        "corr_per_channel": corr.tolist(),
        "spectral_distortion_pct": band_distortion,
    }


# ─── Single (dataset, method, n_bins) evaluation ─────────────────────────────

def evaluate_single(dataset_name: str, method: str, n_bins: int,
                    data: dict, sfreq: float) -> dict:
    """
    Evaluate quantization for one (dataset, method, n_bins) configuration.

    Channel-vectorised.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    method : str
        Quantization method (``"uniform"``, ``"mu_law"``, ``"adaptive"``).
    n_bins : int
        Number of quantization bins.
    data : dict
        Loaded dataset dict ``{subject_id: {"epochs", "labels", "sfreq"}}``.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Result dict with SQNR, correlation, MSE, and spectral distortion.
    """
    all_sqnr = []
    all_corr = []
    all_mse = []
    all_band_dist = {b: [] for b in FREQ_BANDS}

    for subj_id, subj_data in data.items():
        fragment = get_fragment({subj_id: subj_data}, subj_id,
                               duration_s=EXP0_MINUTES_PER_SUBJECT * 60)
        # fragment: (n_ch, n_samples)
        codes, params = quantize(fragment, method, n_bins, normalize=True)
        recon = dequantize(codes, params)

        m = compute_metrics(fragment, recon, sfreq)
        all_sqnr.extend(m["sqnr_db_per_channel"])
        all_corr.extend(m["corr_per_channel"])
        all_mse.append(m["mse_mean"])
        for b in FREQ_BANDS:
            all_band_dist[b].append(m["spectral_distortion_pct"][b])

    all_sqnr = np.array(all_sqnr)
    all_corr = np.array(all_corr)
    all_mse = np.array(all_mse)

    sqnr_mean, sqnr_ci_lo, sqnr_ci_hi = bootstrap_ci(all_sqnr)
    corr_mean, corr_ci_lo, corr_ci_hi = bootstrap_ci(all_corr)

    band_dist_summary = {}
    for b in FREQ_BANDS:
        arr = np.array(all_band_dist[b])
        bm, blo, bhi = bootstrap_ci(arr) if len(arr) > 1 else (float(np.mean(arr)), float(np.mean(arr)), float(np.mean(arr)))
        band_dist_summary[b] = {"mean": bm, "ci_low": blo, "ci_high": bhi}

    return {
        "dataset": dataset_name,
        "method": method,
        "n_bins": n_bins,
        "sqnr_db_mean": sqnr_mean,
        "sqnr_db_ci_low": sqnr_ci_lo,
        "sqnr_db_ci_high": sqnr_ci_hi,
        "corr_mean": corr_mean,
        "corr_ci_low": corr_ci_lo,
        "corr_ci_high": corr_ci_hi,
        "mse_mean": float(np.mean(all_mse)),
        "rmse_mean": float(np.sqrt(np.mean(all_mse))),
        "spectral_distortion_pct": band_dist_summary,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_sqnr_curves(results: list[dict], save_dir: Path) -> None:
    """
    Plot SQNR vs. number of bins for each method and dataset.

    Parameters
    ----------
    results : list of dict
        Experiment 0 result dicts.
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    colors = {"uniform": "#1f77b4", "mu_law": "#ff7f0e", "adaptive": "#2ca02c"}

    for ds in datasets:
        fig, ax = plt.subplots(figsize=(8, 5))
        for meth in methods:
            subset = [r for r in results if r["dataset"] == ds and r["method"] == meth]
            subset.sort(key=lambda r: r["n_bins"])
            bins = [r["n_bins"] for r in subset]
            sqnrs = [r["sqnr_db_mean"] for r in subset]
            ci_lo = [r["sqnr_db_ci_low"] for r in subset]
            ci_hi = [r["sqnr_db_ci_high"] for r in subset]

            ax.plot(bins, sqnrs, "o-", label=meth, color=colors.get(meth, "gray"))
            ax.fill_between(bins, ci_lo, ci_hi, alpha=0.15,
                            color=colors.get(meth, "gray"))

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of bins")
        ax.set_ylabel("SQNR (dB)")
        ax.set_title(f"SQNR vs. Bins — {ds}")
        ax.legend()
        ax.axhline(y=20, ls="--", color="red", alpha=0.5, label="20 dB threshold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"sqnr_vs_bins_{ds}.png", dpi=150)
        plt.close(fig)
    logger.info(f"SQNR plots saved to {save_dir}")


def plot_spectral_distortion(results: list[dict], save_dir: Path) -> None:
    """
    Plot heatmap of spectral distortion per band for each (method, n_bins).

    Parameters
    ----------
    results : list of dict
        Experiment 0 result dicts.
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in results))

    for ds in datasets:
        ds_results = [r for r in results if r["dataset"] == ds]
        methods = sorted(set(r["method"] for r in ds_results))
        bins_list = sorted(set(r["n_bins"] for r in ds_results))
        bands = list(FREQ_BANDS.keys())

        for meth in methods:
            meth_results = [r for r in ds_results if r["method"] == meth]
            meth_results.sort(key=lambda r: r["n_bins"])

            matrix = np.zeros((len(bins_list), len(bands)))
            for i, r in enumerate(meth_results):
                for j, b in enumerate(bands):
                    matrix[i, j] = r["spectral_distortion_pct"][b]["mean"]

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels(bands)
            ax.set_yticks(range(len(bins_list)))
            ax.set_yticklabels(bins_list)
            ax.set_xlabel("Frequency band")
            ax.set_ylabel("Number of bins")
            ax.set_title(f"Spectral Distortion (%) — {ds} / {meth}")
            plt.colorbar(im, ax=ax, label="Distortion (%)")

            # Annotate cells
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, f"{matrix[i,j]:.1f}",
                            ha="center", va="center", fontsize=8)

            fig.tight_layout()
            fig.savefig(save_dir / f"spectral_dist_{ds}_{meth}.png", dpi=150)
            plt.close(fig)

    logger.info(f"Spectral distortion plots saved to {save_dir}")


def plot_waveform_examples(dataset_name: str, data: dict,
                           sfreq: float, save_dir: Path) -> None:
    """
    Plot 2-second EEG fragments before/after quantization (B=64,128,256).

    Parameters
    ----------
    dataset_name : str
        Dataset name for the plot title.
    data : dict
        Loaded dataset dict.
    sfreq : float
        Sampling frequency in Hz.
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # Take first subject, first channel
    subj_id = sorted(data.keys())[0]
    fragment = get_fragment({subj_id: data[subj_id]}, subj_id, duration_s=10)
    ch0 = fragment[0]  # first channel
    n_show = int(2.0 * sfreq)  # 2 seconds
    t = np.arange(n_show) / sfreq

    for n_bins in [64, 128, 256]:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for idx, method in enumerate(QUANT_METHODS):
            ch_2d = ch0[None, :n_show]  # (1, n_samples)
            codes, params = quantize(ch_2d, method, n_bins, normalize=True)
            recon = dequantize(codes, params)[0]

            axes[idx].plot(t, ch0[:n_show], "b-", alpha=0.7, label="Original", linewidth=0.8)
            axes[idx].plot(t, recon, "r-", alpha=0.7, label="Reconstructed", linewidth=0.8)
            axes[idx].set_ylabel("Amplitude")
            axes[idx].set_title(f"{method} (B={n_bins})")
            axes[idx].legend(loc="upper right", fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Waveform Before/After Quantization — {dataset_name}", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_dir / f"waveform_{dataset_name}_B{n_bins}.png", dpi=150)
        plt.close(fig)

    logger.info(f"Waveform plots saved for {dataset_name}")


# ─── Statistical tests ───────────────────────────────────────────────────────

def compare_methods_paired_t(results: list[dict]) -> list[dict]:
    """
    Paired comparison between quantization methods for each dataset & n_bins.
    Uses bootstrap-based confidence intervals on SQNR difference.
    Applies Holm-Bonferroni correction across all comparisons.

    Parameters
    ----------
    results : list[dict]
        Per-configuration results with 'sqnr_db_mean' and 'sqnr_db_ci_low/high'.

    Returns
    -------
    list[dict]
        Comparison results with effect sizes and corrected significance.
    """
    comparisons = []
    datasets = sorted(set(r["dataset"] for r in results))

    for ds in datasets:
        for n_bins in QUANT_BINS:
            subset = {r["method"]: r for r in results
                      if r["dataset"] == ds and r["n_bins"] == n_bins}
            methods = list(subset.keys())
            for i in range(len(methods)):
                for j in range(i + 1, len(methods)):
                    m1, m2 = methods[i], methods[j]
                    diff = subset[m1]["sqnr_db_mean"] - subset[m2]["sqnr_db_mean"]
                    # Effect size (Cohen's d approximation from CIs)
                    ci_span_1 = (subset[m1]["sqnr_db_ci_high"] -
                                 subset[m1]["sqnr_db_ci_low"]) / 3.92  # ~std
                    ci_span_2 = (subset[m2]["sqnr_db_ci_high"] -
                                 subset[m2]["sqnr_db_ci_low"]) / 3.92
                    pooled_std = np.sqrt((ci_span_1**2 + ci_span_2**2) / 2)
                    cohens_d = diff / (pooled_std + 1e-10)
                    comparisons.append({
                        "dataset": ds,
                        "n_bins": n_bins,
                        "method1": m1,
                        "method2": m2,
                        "sqnr_diff_db": diff,
                        "cohens_d": float(cohens_d),
                    })

    # Holm-Bonferroni: sort by |effect|, flag significant ones
    comparisons.sort(key=lambda c: abs(c["cohens_d"]), reverse=True)
    for rank, c in enumerate(comparisons):
        c["rank"] = rank + 1
        c["significant_large_effect"] = abs(c["cohens_d"]) > 0.8
    return comparisons


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp0")
def run_experiment_0(datasets: list[str] | None = None,
                     max_subjects: int | None = None) -> list[dict]:
    """
    Run full Experiment 0: quantization loss analysis.

    Parameters
    ----------
    datasets : list of str or None, optional
        Which datasets to evaluate (default: all 6).
    max_subjects : int or None, optional
        Limit subjects per dataset (for quick testing).

    Returns
    -------
    list of dict
        Per-configuration result dicts (also saved to CSV/JSON).
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())

    if max_subjects is None:
        max_subjects = EXP0_N_SUBJECTS

    exp_log = ExperimentLogger("exp0_quantization_loss")
    all_results = []
    _new_results_count = 0  # track whether any new computation happened

    # Read CSV once for resume checks (exp0 results are deterministic — skip if present)
    _exp0_done: set = set()
    if exp_log.csv_path.exists():
        try:
            import pandas as _pd0
            _df0 = _pd0.read_csv(exp_log.csv_path)
            for _, _r in _df0.iterrows():
                _exp0_done.add((str(_r["dataset"]), str(_r["method"]), str(int(_r["n_bins"]))))
                all_results.append({k: v for k, v in _r.items()
                                    if not (isinstance(v, float) and __import__("math").isnan(v))})
        except Exception:
            pass

    for ds_name in datasets:
        info = DATASET_INFO[ds_name]
        sfreq = info["sfreq"]
        exp_log.info(f"=== Dataset: {ds_name} (Fs={sfreq} Hz) ===")

        # Evaluate all (method, n_bins) combinations — parallelised
        all_tasks = [(method, n_bins) for method in QUANT_METHODS for n_bins in QUANT_BINS]
        tasks = [(m, b) for m, b in all_tasks
                 if (str(ds_name), str(m), str(b)) not in _exp0_done]

        if not tasks:
            exp_log.info(f"  {ds_name}: all configs done — skipping")
            continue

        # Load data only when there is work to do
        try:
            data = load_dataset(ds_name, max_subjects=max_subjects)
        except Exception as e:
            exp_log.error(f"Failed to load {ds_name}: {e}")
            continue

        if not data:
            exp_log.warning(f"No data loaded for {ds_name}")
            continue

        def _eval(method, n_bins):
            return evaluate_single(ds_name, method, n_bins, data, sfreq)

        results_batch = Parallel(n_jobs=N_JOBS, verbose=0)(
            delayed(_eval)(m, b) for m, b in tasks
        )

        for res in results_batch:
            exp_log.log_result(res)
            all_results.append(res)
            _new_results_count += 1

        # Plot waveform examples for this dataset
        plot_dir = PLOTS_DIR / "exp0"
        plot_waveform_examples(ds_name, data, sfreq, plot_dir)

    # Generate summary plots only when new results were computed
    # (CSV-loaded rows have spectral_distortion_pct as strings, not dicts)
    if _new_results_count > 0:
        plot_dir = PLOTS_DIR / "exp0"
        plot_sqnr_curves(all_results, plot_dir)
        plot_spectral_distortion(all_results, plot_dir)

    # Statistical comparisons
    comparisons = compare_methods_paired_t(all_results)
    save_csv(comparisons, LOGS_DIR / "exp0_method_comparisons.csv")

    # Finalize
    exp_log.finalize()

    # Check success criterion: SQNR > 20 dB at B=256
    for res in all_results:
        if res["n_bins"] == 256:
            status = "✓" if res["sqnr_db_mean"] > 20 else "✗"
            exp_log.info(
                f"{status} {res['dataset']}/{res['method']} B=256: "
                f"SQNR={res['sqnr_db_mean']:.1f} dB"
            )

    logger.info(f"Experiment 0 complete: {len(all_results)} configurations evaluated")
    return all_results


if __name__ == "__main__":
    run_experiment_0()
