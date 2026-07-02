"""
Experiment 1: BPE Vocabulary Training & Property Analysis
=========================================================
Train BPE vocabulary on real heterogeneous EEG data.
Analyse: compression, Zipf's law, Heaps' law, neurophysiological properties,
dataset-specificity.

All results logged to JSON/CSV. Plots saved locally.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sig
from scipy import stats as scipy_stats
from collections import Counter
from joblib import Parallel, delayed
from pathlib import Path

from .config import (
    DATASET_INFO, BPE_VOCAB_SIZES, BPE_BALANCED_HOURS,
    FREQ_BANDS, TARGET_SFREQ, N_JOBS, QUANT_BINS,
    LOGS_DIR, PLOTS_DIR, MODELS_DIR,
)
from .quantization import quantize
from .bpe_engine import (
    train_bpe, apply_bpe, apply_bpe_batch, BPEVocab,
    compute_token_stats,
)
from .data_loading import load_dataset, get_fragment
from .utils import (
    ExperimentLogger, save_json, save_csv, timed, get_logger,
)

logger = get_logger("exp1")


# ─── Preprocessing: quantize dataset to sequences ────────────────────────────

def dataset_to_sequences(ds_name: str, max_subjects: int = 10,
                         max_hours: float = BPE_BALANCED_HOURS,
                         n_bins: int = 256,
                         method: str = "mu_law") -> list[list[int]]:
    """
    Load dataset, preprocess, quantize to list of integer sequences.

    Each sequence corresponds to one channel of one subject (trials
    concatenated). Balanced: cap at *max_hours* per dataset.

    Parameters
    ----------
    ds_name : str
        Dataset name.
    max_subjects : int, optional
        Maximum number of subjects to load (default 10).
    max_hours : float, optional
        Maximum hours of data per dataset.
    n_bins : int, optional
        Number of quantization bins (default 256).
    method : str, optional
        Quantization method (default ``"mu_law"``).

    Returns
    -------
    list of list of int
        Integer token sequences.
    """
    info = DATASET_INFO[ds_name]
    sfreq = info["sfreq"]
    max_samples = int(max_hours * 3600 * sfreq)

    data = load_dataset(ds_name, max_subjects=max_subjects)
    sequences = []
    total_samples = 0

    for subj_id, subj_data in sorted(data.items()):
        if total_samples >= max_samples:
            break

        epochs = subj_data["epochs"]  # (n_trials, n_ch, n_time)
        # Concatenate trials per channel
        n_trials, n_ch, n_time = epochs.shape

        # Batch quantize all channels of this subject at once
        # Reshape to (n_ch, n_trials * n_time)
        channel_data_all = epochs.transpose(1, 0, 2).reshape(n_ch, -1)
        remaining = max_samples - total_samples
        if remaining <= 0:
            break
        # Cap per subject
        cap = min(channel_data_all.shape[1], remaining // max(n_ch, 1))
        channel_data_all = channel_data_all[:, :cap]
        if channel_data_all.shape[1] == 0:
            break

        codes_batch, _ = quantize(channel_data_all, method, n_bins,
                                  normalize=True)
        for ch in range(n_ch):
            sequences.append(codes_batch[ch].tolist())
            total_samples += channel_data_all.shape[1]
            if total_samples >= max_samples:
                break

    logger.info(f"{ds_name}: {len(sequences)} sequences, "
                f"{total_samples / sfreq / 3600:.2f}h")
    return sequences


# ─── 1a: Basic vocabulary statistics ─────────────────────────────────────────

def analyse_zipf(token_freqs: Counter) -> dict:
    """
    Test Zipf's law: log(freq) vs log(rank) should be linear.

    Performs both linear regression on log-log and a Kolmogorov-Smirnov
    test against a power-law distribution.

    Parameters
    ----------
    token_freqs : Counter
        Token frequency counts.

    Returns
    -------
    dict
        Zipf slope, R², KS statistic, KS p-value.
    """
    sorted_freqs = sorted(token_freqs.values(), reverse=True)
    freqs_arr = np.array(sorted_freqs, dtype=float)
    ranks = np.arange(1, len(sorted_freqs) + 1, dtype=float)
    log_ranks = np.log10(ranks)
    log_freqs = np.log10(freqs_arr + 1e-10)

    # Linear regression on log-log
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        log_ranks, log_freqs
    )

    # Kolmogorov-Smirnov test: compare empirical freq distribution
    # against fitted power-law (Zipf) CDF
    # Normalize frequencies to a probability distribution
    probs = freqs_arr / freqs_arr.sum()
    # Theoretical Zipf probabilities with fitted exponent
    zipf_exponent = abs(slope)
    theoretical = ranks ** (-zipf_exponent)
    theoretical = theoretical / theoretical.sum()
    # KS test between empirical and theoretical CDFs
    ks_stat, ks_p = scipy_stats.ks_2samp(
        np.repeat(ranks.astype(int), (probs * 10000).astype(int)),
        np.repeat(ranks.astype(int), (theoretical * 10000).astype(int)),
    )

    return {
        "zipf_slope": float(slope),
        "zipf_intercept": float(intercept),
        "zipf_r_squared": float(r_value ** 2),
        "zipf_p_value": float(p_value),
        "zipf_ks_statistic": float(ks_stat),
        "zipf_ks_p_value": float(ks_p),
        "n_unique_tokens": len(sorted_freqs),
    }


def analyse_heaps(sequences: list[list[int]]) -> dict:
    """
    Heaps' law: growth of unique tokens with corpus size.

    Fully vectorised: uses ``np.unique(return_index=True)`` to find the
    first-occurrence position of every token, then ``np.searchsorted``
    to compute cumulative unique counts at sampled checkpoints.

    Parameters
    ----------
    sequences : list of list of int
        BPE-tokenized sequences.

    Returns
    -------
    dict
        Contains ``heaps_points`` (list of ``{corpus_size, n_unique}``),
        ``final_unique``, and ``final_size``.
    """
    # Concatenate all tokens into a single numpy array
    all_tokens = np.concatenate([np.asarray(seq, dtype=np.int32) for seq in sequences])
    total = len(all_tokens)
    if total == 0:
        return {"heaps_points": [], "final_unique": 0, "final_size": 0}

    # Find first occurrence position of each unique token (vectorised)
    _, first_occ = np.unique(all_tokens, return_index=True)
    first_occ.sort()  # sorted positions where new tokens first appear
    final_unique = len(first_occ)

    # Sample ~200 checkpoint positions (at sequence boundaries)
    seq_ends = np.cumsum([len(s) for s in sequences])
    n_points = min(200, len(seq_ends))
    cp_indices = np.linspace(0, len(seq_ends) - 1, n_points, dtype=int)
    cp_positions = seq_ends[cp_indices]  # 1-based token positions

    # At each checkpoint position p, the number of unique tokens seen so far
    # = number of first-occurrence indices < p  (searchsorted gives this)
    unique_at_cp = np.searchsorted(first_occ, cp_positions, side="left")

    points = [
        {"corpus_size": int(cp_positions[i]), "n_unique": int(unique_at_cp[i])}
        for i in range(len(cp_positions))
    ]

    # Ensure exact final point
    if points[-1]["corpus_size"] != total:
        points.append({"corpus_size": total, "n_unique": final_unique})
    else:
        points[-1]["n_unique"] = final_unique

    return {"heaps_points": points, "final_unique": final_unique, "final_size": total}


# ─── 1b: Neurophysiological interpretation ───────────────────────────────────

def analyse_token_spectral(vocab: BPEVocab, sfreq: float,
                           top_n: int = 100,
                           bin_centers: np.ndarray | None = None) -> list[dict]:
    """
    For top-N most frequent tokens, compute spectral properties
    of the decoded waveform.

    Parameters
    ----------
    vocab : BPEVocab
        Trained BPE vocabulary.
    sfreq : float
        Sampling frequency in Hz.
    top_n : int
        Number of merged tokens to analyse.
    bin_centers : np.ndarray or None
        Quantization bin centers for dequantizing codes to signal values.
        If None, uses linear mapping assuming uniform quantization with
        Z-normalized input clipped to [-4, 4].

    Returns
    -------
    list[dict]
        Spectral properties per token.
    """
    # Default bin_centers: assume uniform quantization on [-4, 4] with 256 bins
    if bin_centers is None:
        n_bins = vocab.base_vocab_size
        edges = np.linspace(-4.0, 4.0, n_bins + 1)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

    results = []
    for tok_id in range(vocab.base_vocab_size,
                        min(vocab.vocab_size, vocab.base_vocab_size + top_n)):
        base_seq = vocab.decode_token(tok_id)
        if len(base_seq) < 4:
            continue

        # Dequantize: convert bin indices to actual signal amplitudes
        codes_arr = np.array(base_seq, dtype=int)
        codes_arr = np.clip(codes_arr, 0, len(bin_centers) - 1)
        waveform = bin_centers[codes_arr]
        duration_ms = len(base_seq) / sfreq * 1000

        # PSD (if enough samples for Welch)
        nperseg = min(len(waveform), 32)
        if nperseg >= 4:
            freqs, psd = sig.welch(waveform, fs=sfreq, nperseg=nperseg)
            dominant_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0.0

            # Spectral entropy
            psd_norm = psd / (np.sum(psd) + 1e-20)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))

            # Band powers
            band_powers = {}
            for bname, (fmin, fmax) in FREQ_BANDS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                bp = np.sum(psd[mask]) / (np.sum(psd) + 1e-20)
                band_powers[bname] = float(bp)
        else:
            dominant_freq = 0.0
            spectral_entropy = 0.0
            band_powers = {b: 0.0 for b in FREQ_BANDS}

        results.append({
            "token_id": tok_id,
            "merge_level": tok_id - vocab.base_vocab_size + 1,
            "length_base": len(base_seq),
            "duration_ms": duration_ms,
            "dominant_freq_hz": float(dominant_freq),
            "spectral_entropy": float(spectral_entropy),
            **{f"band_{b}": bp for b, bp in band_powers.items()},
        })

    return results


# ─── 1c: Dataset-specificity ─────────────────────────────────────────────────

def analyse_dataset_specificity(per_dataset_tokens: dict[str, set[int]]) -> dict:
    """
    Compute Jaccard similarity between token sets of different datasets.

    Also identifies dataset-specific vs. shared tokens.

    Parameters
    ----------
    per_dataset_tokens : dict of {str: set of int}
        Mapping from dataset name to the set of active BPE token IDs.

    Returns
    -------
    dict
        Contains ``jaccard_matrix``, ``dataset_names``,
        ``n_shared_tokens``, and ``n_specific_tokens``.
    """
    ds_names = sorted(per_dataset_tokens.keys())
    n = len(ds_names)
    jaccard = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = per_dataset_tokens[ds_names[i]]
            b = per_dataset_tokens[ds_names[j]]
            if len(a | b) > 0:
                jaccard[i, j] = len(a & b) / len(a | b)

    # Shared tokens (present in all datasets)
    all_sets = list(per_dataset_tokens.values())
    shared = set.intersection(*all_sets) if all_sets else set()

    # Dataset-specific (present in only one dataset)
    specific = {}
    for ds in ds_names:
        other_sets = [per_dataset_tokens[d] for d in ds_names if d != ds]
        others = set.union(*other_sets) if other_sets else set()
        specific[ds] = len(per_dataset_tokens[ds] - others)

    return {
        "jaccard_matrix": jaccard.tolist(),
        "dataset_names": ds_names,
        "n_shared_tokens": len(shared),
        "n_specific_tokens": specific,
    }


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_zipf(token_freqs: Counter, label: str, save_dir: Path) -> None:
    """
    Log-log plot of token frequencies (Zipf's law).

    Parameters
    ----------
    token_freqs : Counter
        Token frequency counts.
    label : str
        Label for the plot title and filename.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    sorted_freqs = sorted(token_freqs.values(), reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ranks, sorted_freqs, "b-", alpha=0.7, linewidth=0.8)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Zipf's Law — {label}")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(save_dir / f"zipf_{label}.png", dpi=150)
    plt.close(fig)


def plot_heaps(heaps_data: dict, label: str, save_dir: Path) -> None:
    """
    Plot Heaps' law: unique tokens vs. corpus size.

    Parameters
    ----------
    heaps_data : dict
        Output from ``analyse_heaps``.
    label : str
        Label for the plot title and filename.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    points = heaps_data["heaps_points"]
    sizes = [p["corpus_size"] for p in points]
    uniques = [p["n_unique"] for p in points]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sizes, uniques, "b-", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Corpus Size (tokens)")
    ax.set_ylabel("Unique Tokens")
    ax.set_title(f"Heaps' Law — {label}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / f"heaps_{label}.png", dpi=150)
    plt.close(fig)


def plot_compression(results: list[dict], save_dir: Path) -> None:
    """
    Plot compression ratio vs. vocabulary size per dataset.

    Parameters
    ----------
    results : list of dict
        Experiment 1 result dicts.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = sorted(set(r["dataset"] for r in results))

    for ds in datasets:
        subset = sorted([r for r in results if r["dataset"] == ds],
                        key=lambda r: r["vocab_size"])
        vs = [r["vocab_size"] for r in subset]
        comps = [r["ds_compression"] for r in subset]  # per-dataset ratio, not global
        ax.plot(vs, comps, "o-", label=ds)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Vocabulary Size")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Compression vs. Vocabulary Size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "compression_vs_vocab.png", dpi=150)
    plt.close(fig)


def plot_jaccard_heatmap(specificity: dict, save_dir: Path) -> None:
    """
    Plot heatmap of Jaccard similarity between datasets.

    Parameters
    ----------
    specificity : dict
        Output from ``analyse_dataset_specificity``.
    save_dir : Path
        Directory to save the plot PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.array(specificity["jaccard_matrix"])
    names = specificity["dataset_names"]

    # Short display labels for readability
    _label_map = {
        "bci_iv_2a": "BCI-IV-2a",
        "epfl_p300": "P300",
        "mental_arithmetic": "Mental Arith.",
        "physionet_mi": "PhysioNet-MI",
        "sleep_edf": "Sleep-EDF",
        "ssvep_nakanishi": "SSVEP",
    }
    labels = [_label_map.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title("Jaccard Similarity: Active BPE Tokens ($V=1024$)")
    plt.colorbar(im, ax=ax)

    # Adaptive text color: white on dark cells (val > 0.72), black on light cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            color = "white" if val > 0.72 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)

    fig.tight_layout()
    fig.savefig(save_dir / "jaccard_heatmap.png", dpi=150)
    plt.close(fig)


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp1")
def run_experiment_1(datasets: list[str] | None = None,
                     vocab_sizes: list[int] | None = None,
                     max_subjects: int = 10,
                     n_bins: int = 64,
                     method: str = "uniform",
                     max_hours: float | None = None,
                     max_train_tokens: int | None = None) -> list[dict]:
    """
    Run Experiment 1: BPE vocabulary training and analysis.

    Parameters
    ----------
    datasets : list of str or None, optional
        Datasets to include (default: all).
    vocab_sizes : list of int or None, optional
        BPE vocabulary sizes to train (default from config).
    max_subjects : int, optional
        Max subjects per dataset (default 10).
    n_bins : int, optional
        Number of quantization bins (default 256).
    method : str, optional
        Quantization method (default ``"mu_law"``).
    max_hours : float or None, optional
        Max hours per dataset for BPE training (default from config).
    max_train_tokens : int or None, optional
        Max tokens for BPE training; subsample if exceeded.

    Returns
    -------
    list of dict
        Per-configuration result dicts.
    """
    if datasets is None:
        datasets = list(DATASET_INFO.keys())
    if vocab_sizes is None:
        vocab_sizes = BPE_VOCAB_SIZES

    exp_log = ExperimentLogger("exp1_vocab_analysis")
    all_results = []

    # Resume: if all (vocab_size, dataset) combos already done, skip entirely.
    _exp1_done: set = set()
    if exp_log.csv_path.exists():
        try:
            import pandas as _pd1
            _df1 = _pd1.read_csv(exp_log.csv_path)
            for _, _r in _df1.iterrows():
                _exp1_done.add((str(int(_r["vocab_size"])), str(_r["dataset"])))
                all_results.append({k: v for k, v in _r.items()
                                    if not (isinstance(v, float) and __import__("math").isnan(v))})
        except Exception:
            pass

    _exp1_all_done = all(
        (str(V), str(ds)) in _exp1_done
        for V in (vocab_sizes or []) for ds in (datasets or list(DATASET_INFO.keys()))
    )
    if _exp1_all_done and all_results:
        exp_log.info("All (vocab_size, dataset) combos done — skipping exp1 analysis")
        exp_log.finalize()
        return all_results

    # Step 1: Gather sequences from all datasets (balanced)
    exp_log.info("Step 1: Loading and quantizing data")
    per_dataset_seqs: dict[str, list[list[int]]] = {}

    _max_hours = max_hours if max_hours is not None else BPE_BALANCED_HOURS
    for ds_name in datasets:
        try:
            seqs = dataset_to_sequences(ds_name,
                                         max_subjects=max_subjects,
                                         max_hours=_max_hours,
                                         n_bins=n_bins, method=method)
            per_dataset_seqs[ds_name] = seqs
            exp_log.info(f"  {ds_name}: {len(seqs)} sequences")
        except Exception as e:
            exp_log.error(f"  Failed {ds_name}: {e}")

    # Combined corpus — track per-dataset index ranges for slicing later
    all_seqs = []
    ds_index_ranges: dict[str, tuple[int, int]] = {}
    for ds_name, seqs in per_dataset_seqs.items():
        start = len(all_seqs)
        all_seqs.extend(seqs)
        ds_index_ranges[ds_name] = (start, len(all_seqs))
    exp_log.info(f"Total: {len(all_seqs)} sequences for BPE training")

    # Step 2: Train BPE once at the largest vocab size, then extract
    #         sub-vocabularies (merges are cumulative — V=1024 is a prefix
    #         of V=65536 when trained on the same corpus).
    V_max = max(vocab_sizes)
    vocab_max_path = MODELS_DIR / f"bpe_vocab_V{V_max}_B{n_bins}_{method}.json"
    if vocab_max_path.exists():
        exp_log.info(f"=== Loading cached V={V_max} (sub-vocabs extracted from prefix) ===")
        vocab_max = BPEVocab.load(str(vocab_max_path))
    else:
        exp_log.info(f"=== Training BPE V={V_max} (largest) — sub-vocabs extracted from prefix ===")
        vocab_max = train_bpe(all_seqs, vocab_size=V_max, base_vocab_size=n_bins,
                              max_train_tokens=max_train_tokens)
        vocab_max.save(str(vocab_max_path))

    for V in vocab_sizes:
        # Per-V resume: skip if all datasets already done for this V
        if all((str(V), str(ds)) in _exp1_done for ds in per_dataset_seqs):
            exp_log.info(f"=== V={V}: all datasets done — skipping ===")
            continue

        if V == V_max:
            vocab = vocab_max
        else:
            vocab_path = MODELS_DIR / f"bpe_vocab_V{V}_B{n_bins}_{method}.json"
            if vocab_path.exists():
                vocab = BPEVocab.load(str(vocab_path))
            else:
                # Extract sub-vocabulary: first (V - n_bins) merges
                n_sub = V - n_bins
                vocab = BPEVocab(base_vocab_size=n_bins)
                for pair in vocab_max.merges[:n_sub]:
                    vocab.add_merge(pair)
                vocab.save(str(vocab_path))
        exp_log.info(f"=== Analysing V={V} ({len(vocab.merges)} merges) ===")

        # Tokenize all sequences
        bpe_all = apply_bpe_batch(all_seqs, vocab, n_jobs=N_JOBS)

        # 1a: Basic statistics
        stats = compute_token_stats(bpe_all, vocab)

        # Token frequencies (vectorised)
        all_tokens_flat = np.concatenate(
            [np.asarray(seq, dtype=np.int32) for seq in bpe_all]
        )
        token_freqs = Counter(all_tokens_flat.tolist())

        zipf = analyse_zipf(token_freqs)
        heaps = analyse_heaps(bpe_all)

        # Save Heaps' law data (D12 fix)
        save_json(heaps, LOGS_DIR / f"exp1_heaps_V{V}.json")

        base_result = {
            "vocab_size": V,
            "n_bins": n_bins,
            "method": method,
            "total_bpe_tokens": stats["total_bpe_tokens"],
            "total_base_tokens": stats["total_base_tokens"],
            "compression_ratio": stats["compression_ratio"],
            "vocab_utilization": stats["vocab_utilization"],
            "mean_token_length": stats["mean_token_length"],
            "zipf_slope": zipf["zipf_slope"],
            "zipf_r_squared": zipf["zipf_r_squared"],
            "zipf_ks_statistic": zipf.get("zipf_ks_statistic"),
            "zipf_ks_p_value": zipf.get("zipf_ks_p_value"),
            "heaps_final_unique": heaps["final_unique"],
            "heaps_final_size": heaps["final_size"],
        }

        # Per-dataset compression — slice from bpe_all (no re-tokenization)
        per_dataset_tokens = {}
        for ds_name in per_dataset_seqs:
            ds_start, ds_end = ds_index_ranges[ds_name]
            bpe_ds = bpe_all[ds_start:ds_end]
            ds_stats = compute_token_stats(bpe_ds, vocab)

            ds_tok_set = set()
            for seq in bpe_ds:
                ds_tok_set.update(seq)
            per_dataset_tokens[ds_name] = ds_tok_set

            ds_result = {
                **base_result,
                "dataset": ds_name,
                "ds_compression": ds_stats["compression_ratio"],
                "ds_vocab_utilization": ds_stats["vocab_utilization"],
                "ds_mean_token_length": ds_stats["mean_token_length"],
            }
            exp_log.log_result(ds_result)
            all_results.append(ds_result)

        # 1b: Spectral analysis of tokens
        sfreq_ref = TARGET_SFREQ
        spectral_data = analyse_token_spectral(vocab, sfreq_ref, top_n=100)
        save_json(spectral_data, LOGS_DIR / f"exp1_spectral_V{V}.json")

        # 1c: Dataset specificity
        specificity = analyse_dataset_specificity(per_dataset_tokens)
        save_json(specificity, LOGS_DIR / f"exp1_specificity_V{V}.json")

        # Plots
        plot_dir = PLOTS_DIR / "exp1"
        plot_zipf(token_freqs, f"V={V}", plot_dir)
        plot_heaps(heaps, f"V={V}", plot_dir)
        plot_jaccard_heatmap(specificity, plot_dir)

    # Summary plot
    plot_compression(all_results, PLOTS_DIR / "exp1")

    exp_log.finalize()
    logger.info(f"Experiment 1 complete: {len(all_results)} configs")
    return all_results


if __name__ == "__main__":
    run_experiment_1()
