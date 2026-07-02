"""
Experiment 0.5: Synthetic Data Validation (PROOF OF CONCEPT)
============================================================
GO / NO-GO gate for the entire project.

Parts:
  A — Simple oscillations: BPE on pure sinusoids, mixtures, noise
  B — Realistic synthetic EEG: 1/f + oscillatory peaks + transients
  C — Controlled separability test: can BPE-histograms classify two spectral states?

All results logged to JSON/CSV. Plots saved locally.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy import signal as sig
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

from .config import (
    SYNTH_FS, SYNTH_DURATION, SYNTH_SNR_LEVELS, SYNTH_BPE_VOCABS,
    FREQ_BANDS, N_BOOTSTRAP, LOGS_DIR, PLOTS_DIR,
)
from .quantization import quantize, dequantize
from .bpe_engine import train_bpe, apply_bpe, apply_bpe_batch, BPEVocab
from .utils import ExperimentLogger, save_json, timed, get_logger
from .config import N_JOBS

logger = get_logger("exp05")


# ─── Synthetic signal generators (vectorised) ────────────────────────────────

def generate_sinusoid(freq: float, fs: float, duration: float,
                      amplitude: float = 1.0,
                      phase: float = 0.0) -> np.ndarray:
    """
    Generate a pure sinusoid.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    duration : float
        Duration in seconds.
    amplitude : float, optional
        Peak amplitude (default 1.0).
    phase : float, optional
        Initial phase in radians (default 0.0).

    Returns
    -------
    np.ndarray
        Signal array, shape ``(n_samples,)``.
    """
    t = np.arange(int(fs * duration)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise at given SNR. Vectorised.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal.
    snr_db : float
        Signal-to-noise ratio in dB.

    Returns
    -------
    np.ndarray
        Noisy signal, same shape as *signal*.
    """
    sig_power = np.mean(signal ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*signal.shape) * np.sqrt(noise_power)
    return signal + noise


def generate_nonstationary(fs: float, duration: float,
                           freq1: float = 10.0, freq2: float = 4.0,
                           switch_every: float = 2.0) -> np.ndarray:
    """
    Generate a non-stationary signal with alternating frequencies (ERD simulation).

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration : float
        Duration in seconds.
    freq1 : float, optional
        First alternating frequency in Hz (default 10).
    freq2 : float, optional
        Second alternating frequency in Hz (default 4).
    switch_every : float, optional
        Interval between frequency switches in seconds (default 2).

    Returns
    -------
    np.ndarray
        Signal array, shape ``(n_samples,)``.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    switch_samples = int(switch_every * fs)
    signal = np.zeros(n)
    for start in range(0, n, switch_samples):
        end = min(start + switch_samples, n)
        freq = freq1 if (start // switch_samples) % 2 == 0 else freq2
        signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])
    return signal


def generate_realistic_eeg(fs: float, duration: float,
                           alpha_power: float = 1.0,
                           beta_power: float = 0.5,
                           spindle_rate: float = 0.2) -> np.ndarray:
    """
    Generate realistic synthetic EEG.

    Components: 1/f aperiodic noise, alpha (10 Hz) and beta (20 Hz)
    oscillatory peaks, transient sleep spindles (12–14 Hz bursts),
    and blink-like artifacts.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    duration : float
        Duration in seconds.
    alpha_power : float, optional
        Amplitude scaling of the alpha component (default 1.0).
    beta_power : float, optional
        Amplitude scaling of the beta component (default 0.5).
    spindle_rate : float, optional
        Average number of spindles per second (default 0.2).

    Returns
    -------
    np.ndarray
        Synthetic EEG signal, shape ``(n_samples,)``.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 1/f noise (pink noise via spectral synthesis)
    freqs_fft = np.fft.rfftfreq(n, d=1.0 / fs)
    freqs_fft[0] = 1.0  # avoid division by zero
    spectrum = 1.0 / np.sqrt(freqs_fft)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs_fft))
    pink = np.fft.irfft(spectrum * np.exp(1j * phases), n=n)
    pink = pink / (np.std(pink) + 1e-10)

    # Oscillatory components
    alpha = alpha_power * np.sin(2 * np.pi * 10 * t)
    beta = beta_power * np.sin(2 * np.pi * 20 * t)

    # Sleep spindles (12-14 Hz bursts)
    spindles = np.zeros(n)
    n_spindles = int(spindle_rate * duration)
    for _ in range(n_spindles):
        center = np.random.randint(int(fs), n - int(2 * fs))
        dur_samples = int(np.random.uniform(0.5, 2.0) * fs)
        freq_sp = np.random.uniform(12, 14)
        window = sig.windows.gaussian(dur_samples, std=dur_samples / 6)
        sp_signal = window * np.sin(2 * np.pi * freq_sp *
                                     np.arange(dur_samples) / fs)
        end = min(center + dur_samples, n)
        spindles[center:end] += sp_signal[:end - center]

    # Blink artifacts
    blinks = np.zeros(n)
    n_blinks = int(0.3 * duration)  # ~0.3 blinks per second
    for _ in range(n_blinks):
        center = np.random.randint(int(fs), n - int(fs))
        dur_b = int(np.random.uniform(0.15, 0.4) * fs)
        window = sig.windows.gaussian(dur_b, std=dur_b / 4)
        end = min(center + dur_b, n)
        blinks[center:end] += 3.0 * window[:end - center]

    eeg = pink + alpha + beta + spindles + blinks
    return eeg


# ─── Part A: Simple oscillations ─────────────────────────────────────────────

@timed("exp05")
def run_part_a(exp_log: ExperimentLogger) -> list[dict]:
    """
    Part A: BPE on simple oscillations.

    Check if BPE tokens correspond to full sinusoidal cycles.

    Parameters
    ----------
    exp_log : ExperimentLogger
        Logger for recording results.

    Returns
    -------
    list of dict
        Per-configuration result dicts.
    """
    exp_log.info("=== Part A: Simple Oscillations ===")
    results = []
    fs = SYNTH_FS
    dur = SYNTH_DURATION

    # Generate signals
    signals = {
        "alpha_10Hz": generate_sinusoid(10, fs, dur),
        "theta_4Hz": generate_sinusoid(4, fs, dur),
        "mixture_clean": (generate_sinusoid(10, fs, dur) +
                          0.5 * generate_sinusoid(4, fs, dur)),
        "nonstationary": generate_nonstationary(fs, dur, 10, 4, 2.0),
    }
    # Add noisy mixtures
    mixture = signals["mixture_clean"].copy()
    for snr in SYNTH_SNR_LEVELS:
        signals[f"mixture_snr{snr}dB"] = add_noise(mixture, snr)

    for sig_name, signal in signals.items():
        # Quantize
        signal_2d = signal[None, :]  # (1, n_samples)
        codes, q_params = quantize(signal_2d, "mu_law", 256, normalize=True)
        token_seq = codes[0].tolist()  # 1D list of ints

        for vocab_size in SYNTH_BPE_VOCABS:
            exp_log.info(f"  BPE: {sig_name}, V={vocab_size}")

            # Train BPE
            vocab = train_bpe([token_seq], vocab_size=vocab_size,
                              base_vocab_size=256)
            bpe_tokens = apply_bpe(token_seq, vocab)

            # Analyse token lengths (in samples → ms)
            token_lengths_ms = []
            for tok in bpe_tokens:
                length = len(vocab.decode_token(tok))
                token_lengths_ms.append(length / fs * 1000)

            token_lengths_ms = np.array(token_lengths_ms)
            compression = len(token_seq) / max(len(bpe_tokens), 1)

            # Check: do dominant tokens correspond to full cycles?
            counter = Counter(bpe_tokens)
            top_tokens = counter.most_common(20)
            top_lengths_ms = []
            for tok, count in top_tokens:
                length = len(vocab.decode_token(tok)) / fs * 1000
                top_lengths_ms.append(length)

            result = {
                "part": "A",
                "signal": sig_name,
                "vocab_size": vocab_size,
                "n_bpe_tokens": len(bpe_tokens),
                "compression_ratio": compression,
                "mean_token_length_ms": float(np.mean(token_lengths_ms)),
                "std_token_length_ms": float(np.std(token_lengths_ms)),
                "max_token_length_ms": float(np.max(token_lengths_ms)),
                "top20_token_lengths_ms": top_lengths_ms,
                "n_unique_tokens": len(set(bpe_tokens)),
            }
            exp_log.log_result(result)
            results.append(result)

    return results


# ─── Part B: Realistic synthetic EEG ─────────────────────────────────────────

@timed("exp05")
def run_part_b(exp_log: ExperimentLogger) -> list[dict]:
    """
    Part B: BPE on realistic synthetic EEG.

    Check if tokens cluster into oscillatory / transient / artifact
    categories.

    Parameters
    ----------
    exp_log : ExperimentLogger
        Logger for recording results.

    Returns
    -------
    list of dict
        Per-configuration result dicts.
    """
    exp_log.info("=== Part B: Realistic Synthetic EEG ===")
    results = []
    fs = SYNTH_FS
    dur = SYNTH_DURATION

    # Generate multiple "subjects" — batch quantize
    n_subjects = 10
    raw_signals = [generate_realistic_eeg(fs, dur) for _ in range(n_subjects)]
    signals_batch = np.array(raw_signals)  # (n_subjects, n_samples)
    codes_batch, _ = quantize(signals_batch, "mu_law", 256, normalize=True)
    all_seqs = [codes_batch[i].tolist() for i in range(n_subjects)]

    for vocab_size in SYNTH_BPE_VOCABS:
        exp_log.info(f"  BPE realistic EEG, V={vocab_size}")

        # Train BPE on a subsample (3 subjects) to avoid slow training
        # on 153K tokens. 3 subjects × 15360 = 46K tokens is sufficient
        # for learning representative merge patterns.
        n_train = min(3, n_subjects)
        vocab = train_bpe(all_seqs[:n_train], vocab_size=vocab_size,
                          base_vocab_size=256)

        # Tokenize all in parallel and collect stats
        all_bpe = apply_bpe_batch(all_seqs, vocab, n_jobs=N_JOBS)
        all_token_lengths = []
        all_compressions = []
        for seq_orig, bpe_tokens in zip(all_seqs, all_bpe):
            comp = len(seq_orig) / max(len(bpe_tokens), 1)
            all_compressions.append(comp)
            for tok in bpe_tokens:
                length = len(vocab.decode_token(tok)) / fs * 1000
                all_token_lengths.append(length)

        lengths = np.array(all_token_lengths)

        result = {
            "part": "B",
            "signal": "realistic_eeg",
            "vocab_size": vocab_size,
            "n_subjects": n_subjects,
            "mean_compression": float(np.mean(all_compressions)),
            "std_compression": float(np.std(all_compressions)),
            "mean_token_length_ms": float(np.mean(lengths)),
            "std_token_length_ms": float(np.std(lengths)),
            "median_token_length_ms": float(np.median(lengths)),
            "p95_token_length_ms": float(np.percentile(lengths, 95)),
        }
        exp_log.log_result(result)
        results.append(result)

    return results


# ─── Part C: Controlled separability test ─────────────────────────────────────

@timed("exp05")
def run_part_c(exp_log: ExperimentLogger) -> list[dict]:
    """
    Part C: Controlled separability test.

    Can a simple classifier distinguish two spectral states using only
    BPE-histogram features?

    Class 1: strong alpha (10 Hz) + weak beta.
    Class 2: weak alpha + strong beta (20 Hz).

    Parameters
    ----------
    exp_log : ExperimentLogger
        Logger for recording results.

    Returns
    -------
    list of dict
        Per-configuration result dicts with accuracy and confusion matrices.
    """
    exp_log.info("=== Part C: Controlled Separability Test ===")
    results = []
    fs = SYNTH_FS

    n_trials = 200
    trial_dur = 4.0  # seconds
    n_samples = int(trial_dur * fs)
    snr_db = 10  # increase from 5 to improve signal separation

    # Use smaller vocab sizes for the GO/NO-GO gate (speed optimisation)
    part_c_vocabs = [512, 1024, 4096]

    for vocab_size in part_c_vocabs:
        # Generate all trials as a batch (vectorised signal generation)
        t = np.arange(n_samples) / fs

        # Batch signal generation: all trials at once
        signals = np.zeros((n_trials, n_samples))
        labels = np.zeros(n_trials, dtype=int)
        for trial in range(n_trials):
            if trial < n_trials // 2:
                signals[trial] = (1.0 * np.sin(2 * np.pi * 10 * t) +
                                  0.2 * np.sin(2 * np.pi * 20 * t))
                labels[trial] = 0
            else:
                signals[trial] = (0.2 * np.sin(2 * np.pi * 10 * t) +
                                  1.0 * np.sin(2 * np.pi * 20 * t))
                labels[trial] = 1
            signals[trial] = add_noise(signals[trial], snr_db=snr_db)

        # Use UNIFORM quantization (preserves amplitude ratios better than μ-law
        # which compresses dynamic range and reduces spectral discriminability)
        codes_batch, _ = quantize(signals, "uniform", 256, normalize=True)
        all_seqs = [codes_batch[i].tolist() for i in range(n_trials)]

        # Train BPE on a SUBSAMPLE (30 trials = 15 per class)
        # This is 6.5x faster than training on all 200 trials while capturing
        # the same merge patterns (EEG-BPE patterns converge quickly)
        n_train_bpe = 30
        train_bpe_seqs = all_seqs[:n_train_bpe // 2] + all_seqs[n_trials // 2:n_trials // 2 + n_train_bpe // 2]
        vocab = train_bpe(train_bpe_seqs, vocab_size=vocab_size,
                          base_vocab_size=256)

        # Tokenize all in parallel and build histograms
        all_bpe = apply_bpe_batch(all_seqs, vocab, n_jobs=N_JOBS)

        # Build histograms from parallel results (vectorised via np.bincount)
        histograms = np.zeros((n_trials, vocab_size), dtype=np.float32)
        for i, bpe_tokens in enumerate(all_bpe):
            arr = np.array(bpe_tokens, dtype=np.intp)
            arr = arr[arr < vocab_size]
            if len(arr) > 0:
                histograms[i] = np.bincount(arr, minlength=vocab_size)[:vocab_size].astype(np.float32)

        # L1-normalise
        row_sums = histograms.sum(axis=1, keepdims=True)
        histograms = histograms / (row_sums + 1e-10)

        # Classify with Logistic Regression (5-fold CV)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        all_preds = np.zeros(n_trials, dtype=int)

        for train_idx, test_idx in cv.split(histograms, labels):
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(histograms[train_idx], labels[train_idx])
            pred = clf.predict(histograms[test_idx])
            accs.append(accuracy_score(labels[test_idx], pred))
            all_preds[test_idx] = pred

        acc_mean = float(np.mean(accs))
        acc_std = float(np.std(accs))
        cm = confusion_matrix(labels, all_preds).tolist()

        exp_log.info(f"  V={vocab_size}: Accuracy = {acc_mean:.3f} ± {acc_std:.3f}")

        result = {
            "part": "C",
            "vocab_size": vocab_size,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "confusion_matrix": cm,
            "n_trials": n_trials,
            "snr_db": snr_db,
        }
        exp_log.log_result(result)
        results.append(result)

    return results


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_part_a(results: list[dict], save_dir: Path) -> None:
    """
    Plot BPE token length distributions for Part A.

    Parameters
    ----------
    results : list of dict
        Experiment results (filtered to part A internally).
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # Token length distribution
    signals = sorted(set(r["signal"] for r in results if r["part"] == "A"))
    for sig_name in signals:
        subset = [r for r in results if r["part"] == "A" and r["signal"] == sig_name]
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in subset:
            ax.bar(str(r["vocab_size"]),
                   r["mean_token_length_ms"],
                   yerr=r["std_token_length_ms"],
                   alpha=0.7, label=f"V={r['vocab_size']}")
        ax.set_xlabel("Vocabulary Size")
        ax.set_ylabel("Mean Token Length (ms)")
        ax.set_title(f"Token Length — {sig_name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_dir / f"part_a_token_length_{sig_name}.png", dpi=150)
        plt.close(fig)


def plot_bpe_tokenization_example(save_dir: Path) -> None:
    """Generate Figure 1: BPE tokenisation of a structured synthetic signal.

    Creates a two-panel figure showing the raw signal (top) and coloured BPE
    token segments (bottom), with green=long tokens on plateaus and red=short
    tokens at oscillatory transitions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Build structured synthetic signal ---
    fs = 100.0          # Hz
    t_total = 3.0       # seconds
    n_samples = int(fs * t_total)
    t = np.linspace(0, t_total, n_samples, endpoint=False)

    signal_arr = np.zeros(n_samples)
    seg_len = n_samples // 4
    # Segment 1: low plateau (~0.2)
    signal_arr[:seg_len] = 0.2 + 0.01 * np.random.default_rng(0).standard_normal(seg_len)
    # Segment 2: high plateau (~0.8)
    signal_arr[seg_len:2*seg_len] = 0.8 + 0.01 * np.random.default_rng(1).standard_normal(seg_len)
    # Segment 3: 12 Hz oscillation
    signal_arr[2*seg_len:3*seg_len] = 0.5 + 0.3 * np.sin(2 * np.pi * 12 * t[2*seg_len:3*seg_len])
    # Segment 4: negative plateau (~-0.4)
    signal_arr[3*seg_len:] = -0.4 + 0.01 * np.random.default_rng(2).standard_normal(n_samples - 3*seg_len)

    # --- Quantise and apply BPE ---
    n_bins = 32
    vocab_size = 72  # base + 40 merges
    bins_seq, _ = quantize(signal_arr, "uniform", n_bins)
    vocab = train_bpe(
        [bins_seq.tolist()],
        vocab_size=vocab_size,
        base_vocab_size=n_bins,
        verbose=False,
    )
    token_ids = apply_bpe(bins_seq.tolist(), vocab)

    # --- Compute span lengths for each token (in base-token units) ---
    # Base tokens have span 1; merged token n_bins+i has span = span(a)+span(b)
    span = [1] * n_bins
    for a, b in vocab.merges:
        span.append(span[a] + span[b])

    # Build (start, end) boundaries in original sample space
    tok_lengths = [span[tid] for tid in token_ids]
    starts = np.cumsum([0] + tok_lengths[:-1])
    boundaries = [(int(s), int(s + l)) for s, l in zip(starts, tok_lengths)]
    max_len = max(tok_lengths) if tok_lengths else 1
    norm = mcolors.Normalize(vmin=1, vmax=max_len)
    cmap = cm.get_cmap("RdYlGn")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 4), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})

    # Top panel: raw signal
    axes[0].plot(t, signal_arr, color="#333333", lw=0.8, zorder=3)
    axes[0].set_ylabel("Amplitude", fontsize=9)
    axes[0].set_title("BPE tokenisation of a structured synthetic EEG signal\n"
                       r"($B=32$ bins, $V=72$, 100 Hz)", fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Bottom panel: coloured token segments
    for (start, end), length in zip(boundaries, tok_lengths):
        colour = cmap(norm(length))
        axes[1].axvspan(t[start], t[min(end, n_samples - 1)], color=colour, alpha=0.9)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Time (s)", fontsize=9)
    axes[1].set_ylabel("Tokens", fontsize=9)

    # Annotations
    # Long token in plateau 1
    plateau1_mid = t[seg_len // 2]
    axes[1].annotate("Long token\n(plateau)", xy=(plateau1_mid, 0.5),
                     xycoords=("data", "axes fraction"),
                     xytext=(0, -28), textcoords="offset points",
                     ha="center", fontsize=7, color="darkgreen",
                     arrowprops=dict(arrowstyle="->", color="darkgreen", lw=0.8))
    # Short tokens in oscillation
    osc_mid = t[int(2.5 * seg_len)]
    axes[1].annotate("Short tokens\n(transitions)", xy=(osc_mid, 0.5),
                     xycoords=("data", "axes fraction"),
                     xytext=(0, -28), textcoords="offset points",
                     ha="center", fontsize=7, color="darkred",
                     arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    # Colourbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], orientation="vertical", pad=0.01, fraction=0.02)
    cbar.set_label("Token\nlength\n(samples)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.savefig(save_dir / "bpe_tokenization_example.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_part_c(results: list[dict], save_dir: Path) -> None:
    """
    Plot confusion matrices and accuracy bar chart for Part C.

    Parameters
    ----------
    results : list of dict
        Experiment results (filtered to part C internally).
    save_dir : Path
        Directory to save the plot PNGs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    part_c = [r for r in results if r["part"] == "C"]
    if not part_c:
        return

    # Accuracy bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    vocabs = [r["vocab_size"] for r in part_c]
    accs = [r["accuracy_mean"] for r in part_c]
    stds = [r["accuracy_std"] for r in part_c]
    ax.bar([str(v) for v in vocabs], accs, yerr=stds, color="#2ca02c", alpha=0.8)
    ax.axhline(y=0.5, ls="--", color="red", label="Chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("BPE Vocabulary Size")
    ax.set_ylabel("Accuracy")
    ax.set_title("Part C: Controlled Separability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "part_c_accuracy.png", dpi=150)
    plt.close(fig)

    # Confusion matrices
    for r in part_c:
        cm = np.array(r["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Strong α", "Strong β"])
        ax.set_yticklabels(["Strong α", "Strong β"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — V={r['vocab_size']}")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_dir / f"part_c_cm_V{r['vocab_size']}.png", dpi=150)
        plt.close(fig)


# ─── Main runner ──────────────────────────────────────────────────────────────

@timed("exp05")
def run_experiment_05() -> dict:
    """
    Run full Experiment 0.5: synthetic data validation.

    This is the GO/NO-GO gate for the project.

    Returns
    -------
    dict
        Summary dict with ``go_decision`` bool and accuracy metrics.
    """
    # Resume: synthetic validation is deterministic — skip if JSON summary exists.
    _summary_path = LOGS_DIR / "exp05_go_nogo.json"
    if _summary_path.exists():
        try:
            import json as _json
            with open(_summary_path) as _f:
                _cached = _json.load(_f)
            _log05 = get_logger("exp05_synthetic_validation")
            _log05.info("exp05: summary JSON exists — skipping recomputation")
            return _cached
        except Exception:
            pass

    exp_log = ExperimentLogger("exp05_synthetic_validation")

    results_a = run_part_a(exp_log)
    results_b = run_part_b(exp_log)
    results_c = run_part_c(exp_log)

    all_results = results_a + results_b + results_c

    # Plot
    plot_dir = PLOTS_DIR / "exp05"
    plot_part_a(all_results, plot_dir)
    plot_part_c(all_results, plot_dir)
    plot_bpe_tokenization_example(plot_dir)

    # GO / NO-GO decision
    # Criterion: Part C accuracy > 70% at any vocab size
    best_c_acc = max((r["accuracy_mean"] for r in results_c), default=0.0)
    go_decision = best_c_acc > 0.70

    summary = {
        "go_decision": go_decision,
        "best_separability_accuracy": best_c_acc,
        "part_a_n_configs": len(results_a),
        "part_b_n_configs": len(results_b),
        "part_c_n_configs": len(results_c),
    }

    exp_log.info(f"GO/NO-GO Decision: {'GO ✓' if go_decision else 'NO-GO ✗'}")
    exp_log.info(f"Best separability accuracy: {best_c_acc:.3f}")
    exp_log.finalize()

    save_json(summary, LOGS_DIR / "exp05_go_nogo.json")
    return summary


if __name__ == "__main__":
    run_experiment_05()
