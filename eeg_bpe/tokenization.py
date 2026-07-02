"""
Alternative EEG tokenization strategies for BPE.

Three preprocessing approaches to improve BPE compression and downstream
classification on frequency-coded paradigms (Motor Imagery, SSVEP):

    A. Temporal downsampling  — decimate signal before quantization so BPE
       tokens span physiologically meaningful durations (~15–30 ms at 64 Hz
       vs ~4 ms at 250 Hz).

    B. Amplitude envelope     — Hilbert transform of band-passed signal (mu/beta
       8–30 Hz) captures ERD/ERS power modulation while being phase-invariant.

    C. Short-time spectral    — STFT log-power sequences tokenise spectral
       *shape* patterns (alpha peak, beta suppression) rather than raw amplitude.

All functions accept and return NumPy arrays with shape
    ``(n_trials, n_channels, n_times)``
so they slot in transparently wherever raw epochs are expected.

Dependencies: scipy (signal), numpy.
"""
from __future__ import annotations

import numpy as np
from math import gcd

from .utils import get_logger

logger = get_logger("tokenization")


# ─── A: Temporal downsampling ─────────────────────────────────────────────────

def downsample_epochs(
    epochs: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float = 64.0,
) -> np.ndarray:
    """
    Anti-aliased downsampling of EEG epochs.

    Uses ``scipy.signal.resample_poly`` which automatically applies an
    FIR anti-aliasing low-pass filter before decimation.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.
    orig_sfreq : float
        Original sampling frequency (Hz).
    target_sfreq : float, optional
        Target sampling frequency (Hz).  Default 64 Hz.

    Returns
    -------
    np.ndarray
        Downsampled epochs, shape ``(n_trials, n_channels, n_times_new)``
        where ``n_times_new = round(n_times * target_sfreq / orig_sfreq)``.
    """
    if abs(orig_sfreq - target_sfreq) < 1e-3:
        return epochs.astype(np.float32)

    from scipy.signal import resample_poly

    # Rational up/down to avoid floating-point ratio issues
    g    = gcd(int(orig_sfreq), int(target_sfreq))
    up   = int(target_sfreq) // g
    down = int(orig_sfreq) // g

    n_trials, n_ch, n_time = epochs.shape
    flat = epochs.reshape(n_trials * n_ch, n_time)

    resampled = resample_poly(flat, up, down, axis=-1)
    n_time_new = resampled.shape[-1]
    logger.debug(
        f"downsample_epochs: {orig_sfreq:.0f}→{target_sfreq:.0f} Hz, "
        f"n_times {n_time}→{n_time_new}"
    )
    return resampled.reshape(n_trials, n_ch, n_time_new).astype(np.float32)


# ─── B: Amplitude envelope ────────────────────────────────────────────────────

def envelope_epochs(
    epochs: np.ndarray,
    orig_sfreq: float,
    fmin: float = 8.0,
    fmax: float = 30.0,
    target_sfreq: float = 64.0,
    butter_order: int = 4,
) -> np.ndarray:
    """
    Amplitude envelope via Hilbert transform of band-passed EEG.

    Pipeline per channel:
        1. Butterworth bandpass filter (zero-phase ``filtfilt``).
        2. Hilbert analytic signal → take absolute value (instantaneous amplitude).
        3. Anti-aliased downsample to ``target_sfreq``.

    Captures ERD/ERS power modulation in a form visible to amplitude
    quantisation.  Phase-invariant: same power change always yields the
    same envelope regardless of carrier phase.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.
    orig_sfreq : float
        Original sampling frequency (Hz).
    fmin : float, optional
        Low-pass bandpass cutoff (Hz).  Default 8 Hz (low-mu).
    fmax : float, optional
        High-pass bandpass cutoff (Hz).  Default 30 Hz (high-beta).
    target_sfreq : float, optional
        Envelope output sampling frequency (Hz).  Default 64 Hz.
    butter_order : int, optional
        Butterworth filter order.  Default 4.

    Returns
    -------
    np.ndarray
        Envelope epochs, shape ``(n_trials, n_channels, n_times_new)``.
    """
    from scipy.signal import butter, filtfilt, hilbert
    from scipy.signal import resample_poly

    n_trials, n_ch, n_time = epochs.shape
    nyq = orig_sfreq / 2.0

    # Clamp cutoffs to valid Butterworth range
    lo = max(fmin / nyq, 1e-4)
    hi = min(fmax / nyq, 0.999)
    if lo >= hi:
        raise ValueError(
            f"Invalid bandpass [{fmin}, {fmax}] Hz at sfreq={orig_sfreq} Hz: "
            f"lo={lo:.4f} >= hi={hi:.4f}"
        )

    b, a = butter(butter_order, [lo, hi], btype="bandpass")
    flat = epochs.reshape(n_trials * n_ch, n_time)

    # Zero-phase bandpass filter
    filtered = filtfilt(b, a, flat, axis=-1)

    # Instantaneous amplitude envelope
    envelope = np.abs(hilbert(filtered, axis=-1))

    # Downsample
    g    = gcd(int(orig_sfreq), int(target_sfreq))
    up   = int(target_sfreq) // g
    down = int(orig_sfreq) // g
    resampled = resample_poly(envelope, up, down, axis=-1)
    n_time_new = resampled.shape[-1]

    logger.debug(
        f"envelope_epochs: [{fmin},{fmax}] Hz, "
        f"{orig_sfreq:.0f}→{target_sfreq:.0f} Hz, "
        f"n_times {n_time}→{n_time_new}"
    )
    return resampled.reshape(n_trials, n_ch, n_time_new).astype(np.float32)


def multiband_envelope_epochs(
    epochs: np.ndarray,
    orig_sfreq: float,
    bands: list | None = None,
    target_sfreq: float = 64.0,
    butter_order: int = 4,
) -> np.ndarray:
    """
    Per-band Hilbert envelope concatenated as virtual channels.

    Computes the amplitude envelope (via Hilbert transform) in each
    frequency band independently, then concatenates the results along
    the channel axis.  This preserves band-specific ERD/ERS dynamics
    that a single broadband envelope would mix together.

    Parameters
    ----------
    epochs : np.ndarray
        Shape ``(n_trials, n_channels, n_times)``.
    orig_sfreq : float
        Original sampling frequency (Hz).
    bands : list of (float, float) or None
        List of ``(fmin, fmax)`` pairs.  Default: theta (4–8 Hz),
        alpha (8–13 Hz), beta (13–30 Hz).
    target_sfreq : float
        Envelope output sampling frequency (Hz).  Default 64 Hz.
    butter_order : int
        Butterworth filter order.  Default 4.

    Returns
    -------
    np.ndarray
        Shape ``(n_trials, n_channels * len(bands), n_times_new)``.
    """
    if bands is None:
        bands = [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0)]
    parts = [
        envelope_epochs(epochs, orig_sfreq,
                        fmin=lo, fmax=hi,
                        target_sfreq=target_sfreq,
                        butter_order=butter_order)
        for lo, hi in bands
    ]
    return np.concatenate(parts, axis=1)


# ─── C: Short-time spectral ────────────────────────────────────────────────────

def spectral_epochs(
    epochs: np.ndarray,
    orig_sfreq: float,
    win_sec: float = 0.5,
    step_sec: float = 0.25,
    fmin: float = 1.0,
    fmax: float = 45.0,
) -> np.ndarray:
    """
    Convert epochs to short-time log-power sequences.

    For each channel, sliding windows are computed via FFT, yielding
    a sequence of log-power spectra.  Adjacent values in the output
    sequence correspond to adjacent frequency bins in the same time window;
    adjacent windows follow each other.  BPE learns *spectral shape patterns*
    (e.g., simultaneous alpha peak + beta suppression) that co-occur across
    frequency and time.

    Output shape: ``(n_trials, n_channels, n_windows * n_freq_bins)``
    where the ordering is ``[f0_t0, f1_t0, …, fN_t0, f0_t1, f1_t1, …]``.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.
    orig_sfreq : float
        Sampling frequency (Hz).
    win_sec : float, optional
        Sliding window duration (s).  Default 0.5 s.
    step_sec : float, optional
        Step between windows (s).  Default 0.25 s (50 %% overlap).
    fmin : float, optional
        Minimum frequency to include (Hz).  Default 1 Hz.
    fmax : float, optional
        Maximum frequency to include (Hz).  Default 45 Hz.

    Returns
    -------
    np.ndarray
        Spectral sequences, shape ``(n_trials, n_channels, n_windows * n_freq_bins)``.
        Values are ``log1p(power)``.
    """
    n_trials, n_ch, n_time = epochs.shape
    nyq = orig_sfreq / 2.0
    fmax_eff = min(fmax, nyq - 0.5)

    win_samples  = max(4, int(win_sec  * orig_sfreq))
    # Adaptive cap: ensure at least 4 complete STFT windows per epoch.
    # Prevents pathological behaviour at high sfreq (e.g. P300 at 2048 Hz
    # where a 1000 ms window fills the entire epoch, leaving only 1–2 windows).
    win_samples  = min(win_samples, max(4, n_time // 4))
    step_samples = max(1, int(step_sec * orig_sfreq))
    step_samples = min(step_samples, max(1, win_samples // 2))

    # Frequency bin indices
    freqs     = np.fft.rfftfreq(win_samples, d=1.0 / orig_sfreq)
    freq_mask = (freqs >= fmin) & (freqs <= fmax_eff)
    n_freq    = int(freq_mask.sum())
    if n_freq == 0:
        raise ValueError(
            f"No frequency bins in [{fmin}, {fmax_eff}] Hz "
            f"with win_sec={win_sec}, sfreq={orig_sfreq}"
        )

    # Number of complete windows
    n_windows = max(1, (n_time - win_samples) // step_samples + 1)

    # Build index matrix for all windows: (n_windows, win_samples)
    starts = np.arange(n_windows) * step_samples
    # Clip to valid range (last window may go slightly over)
    ends = np.minimum(starts + win_samples, n_time)
    win_idx = (starts[:, None] + np.arange(win_samples)[None, :])
    win_idx = np.clip(win_idx, 0, n_time - 1)  # (n_windows, win_samples)

    # Hanning window for spectral leakage suppression
    hann = np.hanning(win_samples).astype(np.float32)

    # Flatten trials × channels for batch processing
    flat = epochs.reshape(n_trials * n_ch, n_time).astype(np.float32)

    # Extract windowed segments: (N, n_windows, win_samples)
    windowed = flat[:, win_idx] * hann[None, None, :]  # broadcast

    # FFT along last axis → log-power, select freq range
    fft_vals = np.fft.rfft(windowed, axis=-1)                    # (N, n_windows, n_rfft)
    power    = (np.abs(fft_vals) ** 2) / win_samples             # (N, n_windows, n_rfft)
    log_psd  = np.log1p(power[:, :, freq_mask])                  # (N, n_windows, n_freq)

    # Flatten windows × freq → single time dimension
    log_psd_flat = log_psd.reshape(n_trials * n_ch, n_windows * n_freq)

    logger.debug(
        f"spectral_epochs: win={win_sec}s, step={step_sec}s, "
        f"freq=[{fmin},{fmax_eff}]Hz, "
        f"n_windows={n_windows}, n_freq={n_freq}, "
        f"output_len={n_windows * n_freq}"
    )
    return log_psd_flat.reshape(n_trials, n_ch, n_windows * n_freq)


# ─── D: Amplitude delta (first-order differences) ─────────────────────────────

def delta_epochs(epochs: np.ndarray) -> np.ndarray:
    """
    First-order amplitude differences: ``delta[t] = x[t] - x[t-1]``.

    Converts absolute amplitude to instantaneous amplitude *changes*.
    BPE trained on delta sequences learns **transition patterns** (rises,
    falls, plateaus) rather than absolute amplitude levels.

    Motivation
    ----------
    ERD/ERS (Motor Imagery): the sustained mu/beta power decrease appears as
    a sustained period of near-zero deltas after an initial negative
    transition at ERD onset, followed by a positive transition at ERS
    recovery.  BPE may learn these onset/recovery patterns even though raw
    amplitude is quasi-random during ERD.

    SSVEP: a periodic sinusoidal oscillation at the stimulus frequency (e.g.
    12 Hz) produces alternating ± deltas at a fixed interval.  BPE merges
    may capture this periodic structure as compound tokens.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.

    Returns
    -------
    np.ndarray
        Delta epochs, same shape ``(n_trials, n_channels, n_times)``.
        The first sample of each trial is set to 0 (no predecessor).
    """
    # np.diff shortens the last axis by 1; prepend a zero column to restore shape
    deltas = np.diff(epochs.astype(np.float32), axis=-1)
    zeros  = np.zeros((*epochs.shape[:-1], 1), dtype=np.float32)
    return np.concatenate([zeros, deltas], axis=-1)


# ─── Helpers: build sequence corpus for BPE training ──────────────────────────

def epochs_to_sequences_from_preproc(
    epochs_preproc: np.ndarray,
    method: str = "uniform",
    n_bins: int = 64,
) -> list[list[int]]:
    """
    Quantize preprocessed epoch array and return list of integer sequences.

    Intended to be used after one of the three preprocessing functions above.
    Produces one sequence per (trial, channel) combination.

    Parameters
    ----------
    epochs_preproc : np.ndarray
        Preprocessed epochs, shape ``(n_trials, n_channels, n_times)``.
    method : str, optional
        Quantization method (``"uniform"``, ``"mu_law"``, ``"adaptive"``).
    n_bins : int, optional
        Number of quantization bins.

    Returns
    -------
    list of list of int
        One integer token sequence per (trial × channel) pair.
    """
    from .quantization import quantize

    n_trials, n_ch, n_time = epochs_preproc.shape
    flat = epochs_preproc.reshape(n_trials * n_ch, n_time)
    codes, _ = quantize(flat, method, n_bins, normalize=True)
    return [codes[i].tolist() for i in range(codes.shape[0])]


def describe_preprocessing(
    epochs_raw: np.ndarray,
    epochs_preproc: np.ndarray,
    orig_sfreq: float,
    target_sfreq: float | None,
    approach_name: str,
) -> dict:
    """
    Compute descriptive statistics about a preprocessing transformation.

    Returns dict with keys:
        ``approach``, ``orig_sfreq``, ``target_sfreq``,
        ``n_trials``, ``orig_n_times``, ``new_n_times``,
        ``orig_duration_sec``, ``new_duration_sec``,
        ``amplitude_range_orig``, ``amplitude_range_preproc``
    """
    n_trials, n_ch, n_time_orig = epochs_raw.shape
    n_time_new = epochs_preproc.shape[-1]

    return {
        "approach":             approach_name,
        "orig_sfreq":           orig_sfreq,
        "target_sfreq":         target_sfreq or orig_sfreq,
        "n_trials":             n_trials,
        "n_channels":           n_ch,
        "orig_n_times":         n_time_orig,
        "new_n_times":          n_time_new,
        "orig_duration_sec":    n_time_orig / orig_sfreq,
        "new_duration_sec":     n_time_new / (target_sfreq or orig_sfreq),
        "amplitude_range_orig":   float(np.ptp(epochs_raw)),
        "amplitude_range_preproc": float(np.ptp(epochs_preproc)),
    }
