"""
Fourier-Guided BPE Tokenisation  (Paper 1a core)
=================================================
Novel gradient-free spectral tokenisation pipeline:

    raw EEG  (n_trials, n_ch, n_times)
        ─► STFT log-power  (n_trials, n_ch, n_windows, n_freq)
        ─► k-means spectral codebook  →  discrete spectral codes  {0…n_codes-1}
        ─► BPE on code sequences  →  spectral-motif vocabulary
        ─► per-channel histogram  →  LogReg / RF classifier

Key difference from Exp 6 `spectral_epochs`:
    • Exp 6: each frequency bin treated as an independent amplitude value;
      the time×freq matrix is *flattened* before BPE.
    • Fourier-BPE: each time window's power spectrum is treated as a UNIT
      (vectorially) and mapped to a single discrete code via k-means.
      BPE then operates on 1-D sequences of spectral codes, learning
      *temporal patterns of spectral states* (e.g., mu-power→beta-rebound).

Scientific novelty vs LaBraM VQ:
    • LaBraM VQ: trained end-to-end via Fourier reconstruction loss
      (requires backpropagation, large corpus, GPU).
    • Fourier-BPE: entirely gradient-free (k-means + BPE pair counting);
      each token maps to an interpretable sequence of spectral states.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .bpe_engine import BPEVocab, train_bpe, apply_bpe_batch
from .utils import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("fourier_bpe")


# ─── STFT log-power ───────────────────────────────────────────────────────────

def compute_stft_log_power(
    epochs: np.ndarray,
    sfreq: float,
    win_sec: float = 0.5,
    step_sec: float | None = None,
    fmin: float = 1.0,
    fmax: float = 45.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute STFT log-power spectra for every trial and channel.

    Parameters
    ----------
    epochs   : (n_trials, n_ch, n_times)
    sfreq    : sampling frequency in Hz
    win_sec  : STFT window length in seconds (adaptive cap applied; see below)
    step_sec : step between windows (default = win_sec / 2, i.e. 50 % overlap)
    fmin, fmax : frequency range to retain (Hz)
    normalize : if True, z-score each spectral snapshot across the frequency
                axis so that the k-means codebook focuses on spectral *shape*
                rather than absolute power level.

    Returns
    -------
    spectral : (n_trials, n_ch, n_windows, n_freq)  float32
    freqs    : (n_freq,)  Hz values of retained bins

    Notes
    -----
    Adaptive window cap: win_samples is clamped to ``max(4, n_times // 4)``
    to guarantee ≥ 4 complete windows per trial regardless of sfreq
    (prevents degenerate behaviour at P300 2 048 Hz epochs).

    Window size choice for common paradigms
    ───────────────────────────────────────
    0.5 s  → 2 Hz resolution  adequate for MI (4–8 Hz alpha band width)
             and SSVEP (Nakanishi stimuli spaced 2 Hz apart)
    0.25 s → 4 Hz resolution  faster, fine for Sleep / P300
    """
    n_trials, n_ch, n_times = epochs.shape

    # ── window geometry ──────────────────────────────────────────────────────
    win_samples = int(win_sec * sfreq)
    win_samples = min(win_samples, max(4, n_times // 4))  # adaptive cap
    if step_sec is None:
        step_sec = win_sec / 2.0
    step_samples = max(1, int(step_sec * sfreq))
    step_samples = min(step_samples, max(1, win_samples // 2))

    # ── frequency axis ───────────────────────────────────────────────────────
    nyq = sfreq / 2.0
    fmax_eff = min(fmax, nyq - 0.5)
    freqs_all = np.fft.rfftfreq(win_samples, d=1.0 / sfreq)
    freq_mask = (freqs_all >= fmin) & (freqs_all <= fmax_eff)
    n_freq = int(freq_mask.sum())
    freqs = freqs_all[freq_mask]

    if n_freq == 0:
        raise ValueError(
            f"No frequency bins in [{fmin}, {fmax_eff}] Hz "
            f"with win_samples={win_samples} at sfreq={sfreq} Hz"
        )

    n_windows = max(1, (n_times - win_samples) // step_samples + 1)
    hann = np.hanning(win_samples).astype(np.float32)

    logger.debug(
        f"STFT: win={win_samples/sfreq*1000:.0f} ms, "
        f"step={step_samples/sfreq*1000:.0f} ms, "
        f"n_windows={n_windows}, n_freq={n_freq} "
        f"({fmin:.0f}–{fmax_eff:.0f} Hz)"
    )

    # ── vectorised STFT (flatten trials × channels) ──────────────────────────
    flat = epochs.reshape(n_trials * n_ch, n_times).astype(np.float32)
    N = n_trials * n_ch

    spectral = np.empty((N, n_windows, n_freq), dtype=np.float32)
    starts = np.arange(n_windows) * step_samples
    idx = (starts[:, None] + np.arange(win_samples)[None, :]).clip(0, n_times - 1)

    # Extract all windows at once: (N, n_windows, win_samples)
    segments = flat[:, idx]           # fancy-index over time axis
    segments *= hann[None, None, :]   # apply window

    fft_out = np.fft.rfft(segments, axis=-1)                          # (N, n_windows, n_rfft)
    power   = (np.abs(fft_out) ** 2) / max(win_samples, 1)           # normalised power
    log_psd = np.log1p(power[:, :, freq_mask])                        # (N, n_windows, n_freq)
    spectral = log_psd.astype(np.float32)

    if normalize:
        mu = spectral.mean(axis=-1, keepdims=True)
        sd = spectral.std(axis=-1, keepdims=True) + 1e-8
        spectral = ((spectral - mu) / sd).astype(np.float32)

    return spectral.reshape(n_trials, n_ch, n_windows, n_freq), freqs


# ─── Spectral codebook (k-means) ─────────────────────────────────────────────

def fit_spectral_codebook(
    spectral: np.ndarray,
    n_codes: int = 64,
    random_state: int = 42,
) -> MiniBatchKMeans:
    """Fit a k-means spectral codebook on STFT log-power snapshots.

    This is the Fourier-BPE analog of amplitude quantisation:
        amplitude BPE  : single sample  x  →  bin  ∈ {0…B-1}
        Fourier-BPE    : spectrum vector S  →  code ∈ {0…n_codes-1}

    Parameters
    ----------
    spectral : (n_trials, n_ch, n_windows, n_freq)  — training data
    n_codes  : codebook size  (analogous to n_bins in amplitude BPE)

    Returns
    -------
    Fitted MiniBatchKMeans model (sklearn)
    """
    n_trials, n_ch, n_windows, n_freq = spectral.shape
    flat = spectral.reshape(-1, n_freq)  # (N, n_freq)

    logger.info(
        f"Fitting spectral codebook: {flat.shape[0]:,} snapshots, "
        f"n_freq={n_freq}, n_codes={n_codes}"
    )
    km = MiniBatchKMeans(
        n_clusters=n_codes,
        random_state=random_state,
        batch_size=min(10_000, max(n_codes * 10, flat.shape[0])),
        n_init=5,
        max_iter=300,
        tol=1e-4,
    )
    km.fit(flat)
    inertia_per_sample = km.inertia_ / max(flat.shape[0], 1)
    logger.info(f"Codebook fitted: inertia/sample = {inertia_per_sample:.4f}")
    return km


def encode_with_codebook(
    spectral: np.ndarray,
    codebook: MiniBatchKMeans,
) -> np.ndarray:
    """Map spectral snapshots to codebook indices.

    Parameters
    ----------
    spectral : (n_trials, n_ch, n_windows, n_freq)
    codebook : fitted MiniBatchKMeans

    Returns
    -------
    codes : (n_trials, n_ch, n_windows)  int32  ∈ {0…n_codes-1}
    """
    n_trials, n_ch, n_windows, n_freq = spectral.shape
    flat = spectral.reshape(-1, n_freq)
    codes = codebook.predict(flat).astype(np.int32)
    return codes.reshape(n_trials, n_ch, n_windows)


# ─── BPE training and application on spectral codes ──────────────────────────

def codes_to_sequences(codes: np.ndarray) -> list[list[int]]:
    """Flatten (n_trials, n_ch, n_windows) codes → list of per-channel sequences."""
    n_trials, n_ch, _ = codes.shape
    return [codes[t, c].tolist() for t in range(n_trials) for c in range(n_ch)]


def train_fourier_bpe(
    codes: np.ndarray,
    vocab_size: int,
    n_codes: int,
    max_train_tokens: int | None = None,
) -> BPEVocab:
    """Train BPE vocabulary on spectral code sequences.

    Parameters
    ----------
    codes      : (n_trials, n_ch, n_windows)  — all subjects, all trials
    vocab_size : total vocabulary size (n_codes base + n_merges merged)
    n_codes    : spectral codebook size (= base_vocab_size for BPE)

    Returns
    -------
    BPEVocab with base_vocab_size = n_codes
    """
    seqs = codes_to_sequences(codes)
    logger.info(
        f"Training Fourier-BPE: {len(seqs):,} sequences, "
        f"base_vocab={n_codes}, target_vocab={vocab_size}"
    )
    return train_bpe(
        seqs,
        vocab_size=vocab_size,
        base_vocab_size=n_codes,
        max_train_tokens=max_train_tokens,
    )


def fourier_bpe_histograms(
    codes: np.ndarray,
    vocab: BPEVocab,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Compute L1-normalised BPE histograms from spectral codes.

    Parameters
    ----------
    codes  : (n_trials, n_ch, n_windows)  int32
    vocab  : trained BPEVocab (Fourier-BPE)

    Returns
    -------
    X : (n_trials, n_ch * vocab_size)  float32
    """
    from .config import N_JOBS
    if n_jobs is None:
        n_jobs = N_JOBS

    n_trials, n_ch, _ = codes.shape
    V = vocab.base_vocab_size + len(vocab.merges)

    seqs = codes_to_sequences(codes)
    tokenised = apply_bpe_batch(seqs, vocab, n_jobs=n_jobs)

    X = np.zeros((n_trials * n_ch, V), dtype=np.float32)
    for i, toks in enumerate(tokenised):
        for tok in toks:
            if tok < V:
                X[i, tok] += 1.0

    row_sums = X.sum(axis=1, keepdims=True) + 1e-12
    X /= row_sums
    return X.reshape(n_trials, n_ch * V)


def fourier_bpe_compression(
    codes: np.ndarray,
    vocab: BPEVocab,
    n_jobs: int | None = None,
) -> float:
    """Mean compression ratio: n_windows / mean_bpe_token_length."""
    from .config import N_JOBS
    if n_jobs is None:
        n_jobs = N_JOBS
    n_windows = codes.shape[2]
    seqs = codes_to_sequences(codes)
    tokenised = apply_bpe_batch(seqs, vocab, n_jobs=n_jobs)
    mean_len = float(np.mean([len(t) for t in tokenised])) if tokenised else 1.0
    return n_windows / max(mean_len, 1.0)


# ─── Caching helpers ──────────────────────────────────────────────────────────

def _spectral_cache_key(
    epochs: np.ndarray,
    sfreq: float,
    win_sec: float,
    step_sec: float | None,
    fmin: float,
    fmax: float,
) -> str:
    """Stable hash key for spectral code cache."""
    h = hashlib.md5()
    h.update(epochs.shape.__repr__().encode())
    h.update(epochs.ravel()[:1000].tobytes())           # sample first 1k values
    h.update(f"{sfreq}_{win_sec}_{step_sec}_{fmin}_{fmax}".encode())
    return h.hexdigest()[:16]


def _codebook_key(codebook: MiniBatchKMeans) -> str:
    h = hashlib.md5()
    h.update(codebook.cluster_centers_.tobytes())
    return h.hexdigest()[:16]


def cached_fourier_bpe_histograms(
    epochs: np.ndarray,
    sfreq: float,
    vocab: BPEVocab,
    codebook: MiniBatchKMeans,
    win_sec: float = 0.5,
    step_sec: float | None = None,
    fmin: float = 1.0,
    fmax: float = 45.0,
    normalize: bool = True,
    cache_dir: Path | None = None,
    n_jobs: int | None = None,
    precomputed_codes: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Fourier-BPE histograms with disk cache.

    Cache key = hash(epochs) + hash(codebook) + hash(vocab merges) + params.
    On cache hit: ~2 ms instead of ~1–30 s.

    Parameters
    ----------
    precomputed_codes : np.ndarray or None
        If provided, skip STFT + codebook encoding (already done by caller).
        Shape: (n_trials, n_ch, n_windows) int32.
    """
    from .config import CACHE_DIR as DEFAULT_CACHE
    _cache_dir = Path(cache_dir or DEFAULT_CACHE) / "fourier_hist_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)

    sk = _spectral_cache_key(epochs, sfreq, win_sec, step_sec, fmin, fmax)
    ck = _codebook_key(codebook)
    vk = hashlib.md5(str(vocab.merges[:50]).encode()).hexdigest()[:8]
    key = f"{sk}_{ck}_{vk}"
    cache_path = _cache_dir / f"{key}.npz"

    if cache_path.exists():
        logger.debug(f"Fourier-BPE hist cache hit: {cache_path.name}")
        return np.load(str(cache_path))["X"]

    if precomputed_codes is not None:
        codes = precomputed_codes
    else:
        spectral, _ = compute_stft_log_power(
            epochs, sfreq, win_sec=win_sec, step_sec=step_sec,
            fmin=fmin, fmax=fmax, normalize=normalize,
        )
        codes = encode_with_codebook(spectral, codebook)
    X = fourier_bpe_histograms(codes, vocab, n_jobs=n_jobs)

    np.savez_compressed(str(cache_path), X=X)
    logger.debug(f"Fourier-BPE hist cached: {cache_path.name}")
    return X


# ─── Codebook / vocab persistence ─────────────────────────────────────────────

def save_codebook(codebook: MiniBatchKMeans, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(codebook, f)
    logger.info(f"Spectral codebook saved → {path}")


def load_codebook(path: Path) -> MiniBatchKMeans:
    with open(path, "rb") as f:
        cb = pickle.load(f)
    logger.info(f"Spectral codebook loaded ← {path}")
    return cb
