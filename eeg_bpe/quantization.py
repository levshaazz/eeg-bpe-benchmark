"""
Amplitude quantization methods: Uniform, μ-law, Adaptive.
All operations are fully vectorized (no Python loops over samples).
Batch processing supported.
"""
from __future__ import annotations

import numpy as np
from typing import Literal

from .config import MU_LAW_MU
from .utils import get_logger

logger = get_logger("quantization")

QuantMethod = Literal["uniform", "mu_law", "adaptive"]


# ─── Z-normalization (vectorised) ────────────────────────────────────────────

def z_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-normalize along given axis.

    Works on any shape — fully vectorised.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    axis : int, optional
        Axis along which to normalize (default -1).
    eps : float, optional
        Epsilon to avoid division by zero.

    Returns
    -------
    x_normed : np.ndarray
        Z-normalized signal, same shape as *x*.
    mean : np.ndarray
        Per-slice means (axis squeezed).
    std : np.ndarray
        Per-slice standard deviations (axis squeezed).
    """
    mu = np.mean(x, axis=axis, keepdims=True)
    sigma = np.std(x, axis=axis, keepdims=True)
    sigma = np.where(sigma < eps, eps, sigma)
    return (x - mu) / sigma, mu.squeeze(axis), sigma.squeeze(axis)


# ─── μ-law companding ────────────────────────────────────────────────────────

# Precomputed constants for default μ (avoids recomputation per call)
_LOG1P_MU = float(np.log1p(MU_LAW_MU))      # plain float avoids float64 promotion
_INV_MU = float(1.0 / MU_LAW_MU)             # keeps downstream arrays in float32
_ONE_PLUS_MU = float(1.0 + MU_LAW_MU)


def mu_law_compress(x: np.ndarray, mu: float = MU_LAW_MU) -> np.ndarray:
    """
    μ-law compression: F(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ).

    Input should be in [-1, 1] range. Fully vectorised.
    Uses precomputed ln(1+μ) constant for default μ=255.

    Parameters
    ----------
    x : np.ndarray
        Input signal in [-1, 1].
    mu : float, optional
        Compression parameter (default 255).

    Returns
    -------
    np.ndarray
        Compressed signal in [-1, 1], same shape as *x*.
    """
    log1p_mu = _LOG1P_MU if mu == MU_LAW_MU else np.log1p(mu)
    return np.sign(x) * np.log1p(mu * np.abs(x)) / log1p_mu


def mu_law_expand(y: np.ndarray, mu: float = MU_LAW_MU) -> np.ndarray:
    """
    Inverse μ-law: x = sign(y) * (1/μ) * ((1 + μ)^|y| - 1).

    Uses precomputed constants for default μ=255.

    Parameters
    ----------
    y : np.ndarray
        Compressed signal.
    mu : float, optional
        Compression parameter (default 255).

    Returns
    -------
    np.ndarray
        Expanded signal, same shape as *y*.
    """
    inv_mu = _INV_MU if mu == MU_LAW_MU else 1.0 / mu
    one_plus_mu = _ONE_PLUS_MU if mu == MU_LAW_MU else 1.0 + mu
    return np.sign(y) * inv_mu * (np.power(one_plus_mu, np.abs(y)) - 1.0)


# ─── Uniform quantization ────────────────────────────────────────────────────

def quantize_uniform(x: np.ndarray, n_bins: int,
                     vmin: float | None = None,
                     vmax: float | None = None) -> tuple[np.ndarray, dict]:
    """
    Uniform quantization of continuous signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal (any shape).
    n_bins : int
        Number of quantization bins.
    vmin : float or None, optional
        Lower clip range (default: data min).
    vmax : float or None, optional
        Upper clip range (default: data max).

    Returns
    -------
    codes : np.ndarray
        Integer codes in [0, n_bins-1], same shape as *x*.
    params : dict
        Parameters for dequantization (bin_edges, centers, method, etc.).
    """
    if vmin is None:
        vmin = float(np.min(x))
    if vmax is None:
        vmax = float(np.max(x))

    # Clip & scale to [0, 1]
    x_clipped = np.clip(x, vmin, vmax)
    x_scaled = (x_clipped - vmin) / (vmax - vmin + 1e-10)

    # Quantize: map [0,1] -> {0, 1, ..., n_bins-1}
    codes = np.floor(x_scaled * n_bins).astype(np.int32)
    codes = np.clip(codes, 0, n_bins - 1)

    # Bin centers for dequantization
    bin_edges = np.linspace(vmin, vmax, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    params = {"method": "uniform", "n_bins": n_bins,
              "vmin": vmin, "vmax": vmax,
              "bin_edges": bin_edges, "bin_centers": bin_centers}
    return codes, params


def dequantize_uniform(codes: np.ndarray, params: dict) -> np.ndarray:
    """
    Reconstruct signal from uniform quantization codes.

    Parameters
    ----------
    codes : np.ndarray
        Integer quantization codes.
    params : dict
        Parameters from ``quantize_uniform``.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    return params["bin_centers"][codes]


# ─── μ-law quantization ──────────────────────────────────────────────────────

def quantize_mu_law(x: np.ndarray, n_bins: int,
                    mu: float = MU_LAW_MU,
                    clip_range: float = 4.0) -> tuple[np.ndarray, dict]:
    """
    μ-law quantization: clip → scale to [-1,1] → compress → uniform quantize.

    Input is Z-normalized (roughly in [-3, 3]).

    Parameters
    ----------
    x : np.ndarray
        Input signal (any shape, typically Z-normalized).
    n_bins : int
        Number of quantization bins.
    mu : float, optional
        μ-law compression parameter (default 255).
    clip_range : float, optional
        Symmetric clip boundary (default ±4σ covers 99.99%% of Gaussian).

    Returns
    -------
    codes : np.ndarray
        Integer codes in [0, n_bins-1].
    params : dict
        Parameters for dequantization.
    """
    # 1. Clip to [-clip_range, clip_range]
    x_clipped = np.clip(x, -clip_range, clip_range)
    # 2. Scale to [-1, 1]
    x_scaled = x_clipped / clip_range
    # 3. Apply μ-law compression (input & output in [-1, 1])
    x_compressed = mu_law_compress(x_scaled, mu=mu)

    # 4. Uniform quantize on [-1, 1]
    codes, uni_params = quantize_uniform(x_compressed, n_bins, vmin=-1.0, vmax=1.0)

    params = {"method": "mu_law", "n_bins": n_bins, "mu": mu,
              "clip_range": clip_range, "uni_params": uni_params}
    return codes, params


def dequantize_mu_law(codes: np.ndarray, params: dict) -> np.ndarray:
    """
    Reconstruct signal from μ-law quantization codes.

    Parameters
    ----------
    codes : np.ndarray
        Integer quantization codes.
    params : dict
        Parameters from ``quantize_mu_law``.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    x_compressed = dequantize_uniform(codes, params["uni_params"])
    x_scaled = mu_law_expand(x_compressed, mu=params["mu"])
    return x_scaled * params["clip_range"]


# ─── Adaptive (percentile-based) quantization ────────────────────────────────

def quantize_adaptive(x: np.ndarray, n_bins: int) -> tuple[np.ndarray, dict]:
    """
    Adaptive quantization based on percentiles with Lloyd-Max centroids.

    Each bin covers equal probability mass. Fully vectorised via
    ``np.searchsorted``. Uses conditional means (centroids) instead
    of midpoints for optimal MSE.

    Parameters
    ----------
    x : np.ndarray
        Input signal (any shape).
    n_bins : int
        Number of quantization bins.

    Returns
    -------
    codes : np.ndarray
        Integer codes in [0, actual_bins-1], same shape as *x*.
    params : dict
        Parameters for dequantization (bin_edges, bin_centers, etc.).
    """
    # Compute percentile boundaries
    percentiles = np.linspace(0, 100, n_bins + 1)
    flat = x.ravel()
    if flat.size == 0:
        flat = np.zeros(1, dtype=np.float32)
    bin_edges = np.percentile(flat, percentiles)
    # Ensure monotonically increasing (handle ties)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        bin_edges = np.array([flat.min() - 1e-6, flat.max() + 1e-6])
    actual_bins = len(bin_edges) - 1

    # Vectorised digitize
    codes_flat = np.digitize(flat, bin_edges[1:-1]).astype(np.int32)
    codes_flat = np.clip(codes_flat, 0, actual_bins - 1)

    # Compute centroids (conditional means) — vectorised via bincount
    sums = np.bincount(codes_flat, weights=flat, minlength=actual_bins)
    counts = np.bincount(codes_flat, minlength=actual_bins)
    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers = np.where(counts > 0, sums / np.maximum(counts, 1), midpoints)

    codes = codes_flat.reshape(x.shape)

    params = {"method": "adaptive", "n_bins": actual_bins,
              "bin_edges": bin_edges, "bin_centers": bin_centers}
    return codes, params


def dequantize_adaptive(codes: np.ndarray, params: dict) -> np.ndarray:
    """
    Reconstruct signal from adaptive quantization codes.

    Parameters
    ----------
    codes : np.ndarray
        Integer quantization codes.
    params : dict
        Parameters from ``quantize_adaptive``.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    return params["bin_centers"][np.clip(codes, 0, len(params["bin_centers"]) - 1)]


# ─── Universal dispatcher ────────────────────────────────────────────────────

def quantize(x: np.ndarray, method: QuantMethod, n_bins: int,
             normalize: bool = True, **kwargs) -> tuple[np.ndarray, dict]:
    """
    Quantize signal with given method.

    Parameters
    ----------
    x : np.ndarray
        Signal array (any shape, e.g. ``(n_channels, n_samples)``).
    method : {"uniform", "mu_law", "adaptive"}
        Quantization method.
    n_bins : int
        Number of quantization levels.
    normalize : bool, optional
        If True, Z-normalize per channel (axis=-1).
    **kwargs
        Extra keyword arguments forwarded to the chosen quantizer.

    Returns
    -------
    codes : np.ndarray
        Integer codes in [0, n_bins-1].
    params : dict
        Parameters for dequantization.
    """
    norm_params = None
    if normalize:
        x, mu, sigma = z_normalize(x, axis=-1)
        norm_params = {"mean": mu, "std": sigma}

    if method == "uniform":
        # After Z-norm, clip to [-4, 4] σ
        codes, q_params = quantize_uniform(x, n_bins, vmin=-4.0, vmax=4.0)
    elif method == "mu_law":
        codes, q_params = quantize_mu_law(x, n_bins, **kwargs)
    elif method == "adaptive":
        codes, q_params = quantize_adaptive(x, n_bins)
    else:
        raise ValueError(f"Unknown method: {method}")

    q_params["normalize"] = normalize
    q_params["norm_params"] = norm_params
    return codes, q_params


def dequantize(codes: np.ndarray, params: dict) -> np.ndarray:
    """
    Dequantize codes back to continuous signal.

    Parameters
    ----------
    codes : np.ndarray
        Integer quantization codes.
    params : dict
        Parameters dict from a ``quantize*`` call.

    Returns
    -------
    np.ndarray
        Reconstructed continuous signal.
    """
    method = params["method"]
    if method == "uniform":
        x_hat = dequantize_uniform(codes, params)
    elif method == "mu_law":
        x_hat = dequantize_mu_law(codes, params)
    elif method == "adaptive":
        x_hat = dequantize_adaptive(codes, params)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Undo Z-normalization
    if params.get("normalize") and params.get("norm_params") is not None:
        np_ = params["norm_params"]
        mu = np_["mean"]
        sigma = np_["std"]
        # Restore shape for broadcasting
        if x_hat.ndim == 2 and mu.ndim == 1:
            mu = mu[:, None]
            sigma = sigma[:, None]
        x_hat = x_hat * sigma + mu

    return x_hat


# ─── Batch quantization (multi-channel, multi-trial) ─────────────────────────

def quantize_batch(epochs: np.ndarray, method: QuantMethod, n_bins: int,
                   normalize: bool = True) -> tuple[np.ndarray, list[dict]]:
    """
    Quantize a batch of epochs.

    Each channel is Z-normalized independently. Vectorised: processes
    all trials via reshape.

    Parameters
    ----------
    epochs : np.ndarray
        EEG epochs, shape ``(n_trials, n_channels, n_times)``.
    method : {"uniform", "mu_law", "adaptive"}
        Quantization method.
    n_bins : int
        Number of quantization levels.
    normalize : bool, optional
        If True, Z-normalize per channel.

    Returns
    -------
    codes : np.ndarray
        Integer codes array, same shape as *epochs*.
    params : list of dict
        Quantization parameters (one entry, broadcast from vectorised call).
    """
    n_trials, n_ch, n_times = epochs.shape
    # Reshape to (n_trials * n_ch, n_times) for batch processing
    flat = epochs.reshape(-1, n_times)

    codes, params = quantize(flat, method, n_bins, normalize=normalize)
    codes = codes.reshape(n_trials, n_ch, n_times)

    return codes, params
