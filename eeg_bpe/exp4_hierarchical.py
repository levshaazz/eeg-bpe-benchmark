"""
Experiment 4: Hierarchical Analysis of BPE Tokens (H3)
======================================================
Investigates the structure of the BPE merge hierarchy.
For each token: duration, dominant frequency, spectral entropy,
band powers, Hjorth complexity vs. merge level.

Statistics: Spearman ρ with Bonferroni correction, permutation tests.
Visualisations: UMAP, dendrograms, waveform gallery.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy import stats as scipy_stats
from collections import Counter
from joblib import Parallel, delayed
from pathlib import Path

from .config import (
    FREQ_BANDS, TARGET_SFREQ, N_JOBS, LOGS_DIR, PLOTS_DIR, MODELS_DIR,
)
from .bpe_engine import BPEVocab, apply_bpe_batch
from .quantization import quantize
from .data_loading import load_dataset
from .utils import ExperimentLogger, save_json, save_csv, timed, get_logger

logger = get_logger("exp4")


# ─── Hjorth parameters (vectorised) ──────────────────────────────────────────

def hjorth_params(x: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Hjorth parameters (activity, mobility, complexity).

    Parameters
    ----------
    x : np.ndarray
        1-D signal array.

    Returns
    -------
    activity : float
        Signal variance.
    mobility : float
        Ratio of std of first derivative to std of signal.
    complexity : float
        Ratio of mobility of first derivative to mobility of signal.
    """
    dx = np.diff(x)
    ddx = np.diff(dx)

    activity = float(np.var(x))
    mobility = float(np.sqrt(np.var(dx) / (activity + 1e-20)))
    complexity = float(
        np.sqrt(np.var(ddx) / (np.var(dx) + 1e-20)) / (mobility + 1e-20)
    )
    return activity, mobility, complexity


# ─── Token characterisation ──────────────────────────────────────────────────

def characterise_token(token_id: int, vocab: BPEVocab,
                       sfreq: float,
                       bin_centers: np.ndarray | None = None) -> dict | None:
    """
    Compute neurophysiological characteristics for a single BPE token.

    Parameters
    ----------
    token_id : int
        BPE token ID to characterise.
    vocab : BPEVocab
        Trained BPE vocabulary.
    sfreq : float
        Sampling frequency in Hz.
    bin_centers : np.ndarray or None
        Quantization bin centers for dequantizing codes to signal values.
        If None, assumes uniform quantization on [-4, 4].

    Returns
    -------
    dict or None
        Token characteristics, or None if token too short.
    """
    # Default bin_centers: uniform quantization on [-4, 4]
    if bin_centers is None:
        n_bins = vocab.base_vocab_size
        edges = np.linspace(-4.0, 4.0, n_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

    base_seq = vocab.decode_token(token_id)
    if len(base_seq) < 4:
        return None

    # Dequantize: convert bin indices to actual signal amplitudes
    codes_arr = np.array(base_seq, dtype=int)
    codes_arr = np.clip(codes_arr, 0, len(bin_centers) - 1)
    waveform = bin_centers[codes_arr]
    n = len(waveform)
    duration_ms = n / sfreq * 1000

    # Hjorth parameters (computed first; mobility used for dominant frequency)
    activity, mobility, complexity = hjorth_params(waveform)

    # Dominant frequency via Hjorth mobility: robust for short signals.
    # For a sinusoid x(t)=A·sin(2πft), mobility = 2πf/f_s, so f = mobility·f_s/(2π).
    # This avoids FFT aliasing artefacts (e.g. only 0 Hz and 32 Hz visible for
    # 4-sample tokens at 128 Hz where the FFT resolution is 32 Hz/bin).
    dominant_freq = float(mobility * sfreq / (2.0 * np.pi))

    # PSD for spectral entropy and band powers
    nperseg = min(n, max(4, n // 2))
    try:
        freqs, psd = sig.welch(waveform, fs=sfreq, nperseg=nperseg)
    except Exception:
        return None

    if len(psd) == 0:
        return None

    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-20)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-20)))

    # Band powers (relative)
    band_powers = {}
    total_power = np.sum(psd)
    for bname, (fmin, fmax) in FREQ_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        bp = float(np.sum(psd[mask]) / (total_power + 1e-20))
        band_powers[bname] = bp

    # Merge level
    merge_level = token_id - vocab.base_vocab_size + 1

    return {
        "token_id": token_id,
        "merge_level": merge_level,
        "length_base": n,
        "duration_ms": duration_ms,
        "dominant_freq_hz": dominant_freq,
        "spectral_entropy": spectral_entropy,
        "hjorth_activity": activity,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        **{f"band_{b}": v for b, v in band_powers.items()},
    }


# ─── Statistical tests ───────────────────────────────────────────────────────

def spearman_with_bonferroni(token_data: list[dict],
                              features: list[str]) -> list[dict]:
    """
    Spearman rank correlation: merge_level vs. each feature.

    Applies Bonferroni correction on p-values.

    Parameters
    ----------
    token_data : list of dict
        Each dict must contain ``merge_level`` and all *features*.
    features : list of str
        Feature names to correlate with merge level.

    Returns
    -------
    list of dict
        Per-feature results with ``spearman_rho``, ``p_value``,
        ``p_bonferroni``, ``significant_001``, ``significant_005``.
    """
    merge_levels = np.array([t["merge_level"] for t in token_data])
    n_tests = len(features)
    results = []

    for feat in features:
        values = np.array([t[feat] for t in token_data])
        # Skip zero-variance features (e.g. slow bands in short tokens)
        if np.std(values) < 1e-12:
            results.append({
                "feature": feat,
                "spearman_rho": float("nan"),
                "p_value": float("nan"),
                "p_bonferroni": float("nan"),
                "significant_001": False,
                "significant_005": False,
                "note": "zero_variance_skipped",
            })
            continue
        rho, p_value = scipy_stats.spearmanr(merge_levels, values)
        results.append({
            "feature": feat,
            "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
            "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
            "p_bonferroni": float(min(p_value * n_tests, 1.0)) if not np.isnan(p_value) else float("nan"),
            "significant_001": bool(p_value * n_tests < 0.01) if not np.isnan(p_value) else False,
            "significant_005": bool(p_value * n_tests < 0.05) if not np.isnan(p_value) else False,
        })

    return results


def permutation_test(merge_levels: np.ndarray, values: np.ndarray,
                     n_perms: int = 1000) -> float:
    """
    Permutation test for Spearman correlation significance.

    Fully vectorised: batch-shuffles and computes rank correlations
    via matrix operations.

    Parameters
    ----------
    merge_levels : np.ndarray
        1-D array of merge levels.
    values : np.ndarray
        1-D array of feature values.
    n_perms : int, optional
        Number of permutations (default 1000).

    Returns
    -------
    float
        Two-sided permutation p-value.
    """
    observed_rho, _ = scipy_stats.spearmanr(merge_levels, values)
    n = len(merge_levels)

    # Vectorised batch permutations: generate all permutation indices at once
    perm_idx = np.argsort(
        np.random.rand(n_perms, n), axis=1
    )  # (n_perms, n) — each row is a random permutation

    # Precompute ranks once (avoid repeated ranking inside spearmanr)
    from scipy.stats import rankdata
    ranks_ml = rankdata(merge_levels)
    ranks_v = rankdata(values)

    # Vectorised Spearman via Pearson on ranks
    # For each permutation, correlate permuted ranks_ml with ranks_v
    perm_ranks_ml = ranks_ml[perm_idx]  # (n_perms, n)
    # Demean
    perm_mean = perm_ranks_ml.mean(axis=1, keepdims=True)
    perm_centered = perm_ranks_ml - perm_mean
    v_centered = ranks_v - ranks_v.mean()

    # Batch Pearson correlation: dot product / (norm * norm)
    num = perm_centered @ v_centered  # (n_perms,)
    denom = (np.sqrt(np.sum(perm_centered ** 2, axis=1)) *
             np.sqrt(np.sum(v_centered ** 2)) + 1e-20)
    perm_rhos = num / denom

    p_value = float(np.mean(np.abs(perm_rhos) >= np.abs(observed_rho)))
    return p_value


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_scatter_merge_vs_features(token_data: list[dict],
                                    save_dir: Path) -> None:
    """
    Scatter plots: merge level vs. each neurophysiological feature.

    Parameters
    ----------
    token_data : list of dict
        Token characteristic dicts (from ``characterise_token``).
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    features = ["duration_ms", "dominant_freq_hz", "spectral_entropy",
                "hjorth_complexity", "band_delta", "band_alpha", "band_beta"]

    merge_levels = [t["merge_level"] for t in token_data]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    for i, feat in enumerate(features):
        if i >= len(axes):
            break
        values = [t.get(feat, 0) for t in token_data]
        axes[i].scatter(merge_levels, values, alpha=0.3, s=5)
        axes[i].set_xlabel("Merge Level")
        axes[i].set_ylabel(feat)
        axes[i].set_title(feat)
        axes[i].grid(True, alpha=0.3)

    # Hide unused subplot
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Merge Level vs. Token Properties", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "merge_vs_features.png", dpi=150)
    plt.close(fig)


def plot_waveform_gallery(vocab: BPEVocab, save_dir: Path,
                          sfreq: float = TARGET_SFREQ) -> None:
    """
    Waveform gallery: 5 examples each for low, medium, high merge levels.

    Parameters
    ----------
    vocab : BPEVocab
        Trained BPE vocabulary.
    save_dir : Path
        Directory to save the plot PNG.
    sfreq : float, optional
        Sampling frequency in Hz (default from config).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    n_merges = len(vocab.merges)
    if n_merges < 10:
        return

    categories = {
        "low (1-3)": list(range(vocab.base_vocab_size,
                                vocab.base_vocab_size + min(3, n_merges))),
        "medium": list(range(vocab.base_vocab_size + n_merges // 3,
                             vocab.base_vocab_size + n_merges // 3 + 5)),
        "high": list(range(vocab.base_vocab_size + n_merges - 5,
                           vocab.base_vocab_size + n_merges)),
    }

    fig, axes = plt.subplots(3, 5, figsize=(20, 10))

    for row, (cat_name, tok_ids) in enumerate(categories.items()):
        for col, tok_id in enumerate(tok_ids[:5]):
            seq = vocab.decode_token(tok_id)
            t = np.arange(len(seq)) / sfreq * 1000  # ms
            axes[row, col].plot(t, seq, "b-", linewidth=0.8)
            axes[row, col].set_title(f"Token {tok_id}\n({len(seq)} samples)",
                                      fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(cat_name)
            axes[row, col].grid(True, alpha=0.3)

    fig.suptitle("Waveform Gallery: Low / Medium / High Merge Level", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_dir / "waveform_gallery.png", dpi=150)
    plt.close(fig)


def plot_umap(token_data: list[dict], save_dir: Path) -> None:
    """
    UMAP of token features, colored by merge level.

    Parameters
    ----------
    token_data : list of dict
        Token characteristic dicts.
    save_dir : Path
        Directory to save the plot PNG.
    """
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed, skipping UMAP plot")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    features = ["duration_ms", "dominant_freq_hz", "spectral_entropy",
                "band_delta", "band_theta", "band_alpha", "band_beta", "band_gamma"]

    X = np.array([[t.get(f, 0) for f in features] for t in token_data])
    merge_levels = np.array([t["merge_level"] for t in token_data])

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=merge_levels,
                    cmap="viridis", alpha=0.5, s=10)
    plt.colorbar(sc, ax=ax, label="Merge Level")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of BPE Token Features (color=merge level)")
    fig.tight_layout()
    fig.savefig(save_dir / "umap_merge_level.png", dpi=150)
    plt.close(fig)


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp4")
def run_experiment_4(vocab_size: int = 16384,
                     n_bins: int = 64,
                     sfreq: float = TARGET_SFREQ,
                     method: str = "adaptive") -> dict:
    """
    Run Experiment 4: Hierarchical analysis of BPE tokens.

    Parameters
    ----------
    vocab_size : int, optional
        BPE vocabulary size to analyse (default 16384).
    n_bins : int, optional
        Number of quantization bins (default 256).
    sfreq : float, optional
        Sampling frequency in Hz (default from config).

    Returns
    -------
    dict
        Summary with correlation results and significance flags.
    """
    exp_log = ExperimentLogger("exp4_hierarchical")

    # Resume: skip if characterization JSON already exists for this exact vocab config.
    # Use versioned filename only — never skip based on a different vocab's JSON.
    _exp4_json = LOGS_DIR / f"exp4_token_characteristics_V{vocab_size}_B{n_bins}_{method}.json"
    if _exp4_json.exists():
        try:
            import json as _json4
            with open(_exp4_json) as _f4:
                _cached4 = _json4.load(_f4)
            exp_log.info(f"exp4: token characteristics JSON exists ({len(_cached4)} tokens) — skipping")
            exp_log.finalize()
            _sum_path = LOGS_DIR / "exp4_hierarchical_results.json"
            if _sum_path.exists():
                with open(_sum_path) as _fs:
                    _sum = _json4.load(_fs)
                return _sum.get("results", [{}])[0] if _sum.get("results") else {}
            return {}
        except Exception:
            pass

    # Load vocab
    vocab_path = MODELS_DIR / f"bpe_vocab_V{vocab_size}_B{n_bins}_{method}.json"
    if not vocab_path.exists():
        exp_log.error(f"Vocab not found: {vocab_path}. Run exp1 first.")
        return {}

    vocab = BPEVocab.load(str(vocab_path))
    exp_log.info(f"Loaded vocab V={vocab.vocab_size}")

    # Characterise all merged tokens (parallel)
    merged_ids = list(range(vocab.base_vocab_size, vocab.vocab_size))
    exp_log.info(f"Characterising {len(merged_ids)} merged tokens...")

    token_data = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(characterise_token)(tok_id, vocab, sfreq)
        for tok_id in merged_ids
    )
    token_data = [t for t in token_data if t is not None]
    exp_log.info(f"Successfully characterised {len(token_data)} tokens")

    # Save raw data (versioned by vocab config for correct resume logic)
    _exp4_json_save = LOGS_DIR / f"exp4_token_characteristics_V{vocab_size}_B{n_bins}_{method}.json"
    save_json(token_data, _exp4_json_save)
    save_json(token_data, LOGS_DIR / "exp4_token_characteristics.json")  # legacy name for reports
    save_csv(token_data, LOGS_DIR / "exp4_token_characteristics.csv")

    # Statistical tests
    features = ["duration_ms", "dominant_freq_hz", "spectral_entropy",
                "hjorth_complexity", "band_delta", "band_theta",
                "band_alpha", "band_beta", "band_gamma"]

    correlations = spearman_with_bonferroni(token_data, features)
    save_csv(correlations, LOGS_DIR / "exp4_spearman_correlations.csv")
    save_json(correlations, LOGS_DIR / "exp4_spearman_correlations.json")

    for c in correlations:
        sig_str = "***" if c["significant_001"] else ("*" if c["significant_005"] else "n.s.")
        exp_log.info(
            f"  {c['feature']:25s}: ρ={c['spearman_rho']:.3f}, "
            f"p={c['p_bonferroni']:.4f} {sig_str}"
        )

    # Permutation tests for significant results
    for c in correlations:
        if c["significant_005"]:
            merge_levels = np.array([t["merge_level"] for t in token_data])
            values = np.array([t[c["feature"]] for t in token_data])
            perm_p = permutation_test(merge_levels, values, n_perms=1000)
            c["perm_p_value"] = perm_p
            exp_log.info(f"  Permutation test {c['feature']}: p={perm_p:.4f}")

    save_json(correlations, LOGS_DIR / "exp4_correlations_with_perm.json")

    # Plots
    plot_dir = PLOTS_DIR / "exp4"
    plot_scatter_merge_vs_features(token_data, plot_dir)
    plot_waveform_gallery(vocab, plot_dir, sfreq)
    plot_umap(token_data, plot_dir)

    # Summary
    summary = {
        "n_tokens_analysed": len(token_data),
        "correlations": correlations,
        "any_significant_001": any(c["significant_001"] for c in correlations),
        "any_significant_005": any(c["significant_005"] for c in correlations),
    }
    save_json(summary, LOGS_DIR / "exp4_summary.json")

    # Log summary as a single result so exp_log.finalize() produces a non-empty file
    exp_log.log_result(summary)
    exp_log.finalize()
    return summary


if __name__ == "__main__":
    run_experiment_4()
