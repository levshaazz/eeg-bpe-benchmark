"""
Baselines for Experiment 2: Downstream Classification
======================================================
Implements comparison methods that do NOT use BPE tokenization:

1. CSP + LDA (Motor Imagery only)
2. EEGNet-lite (raw signal → Conv2D)
3. Fixed-size patching + LogReg / Transformer
4. VQ-VAE tokenization + Transformer (approximation via k-means VQ)
5. Chronos-style binning + Transformer
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from pathlib import Path

from .config import (
    DATASET_INFO, DEVICE, N_JOBS, TARGET_SFREQ,
)
from .quantization import quantize
from .utils import get_logger, pca_reduce

logger = get_logger("baselines")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _reduce_dense_features(X_train: np.ndarray,
                            X_test: np.ndarray,
                            max_features: int = 512):
    """
    Per-fold PCA reduction for dense feature matrices (GPU-accelerated).

    Applied when n_features > max_features.  Fit on train only (no leakage).
    Uses torch.pca_lowrank on GPU; falls back to sklearn on CPU.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices, shape (n_samples, n_features).
    max_features : int
        Target dimensionality.

    Returns
    -------
    X_train_r, X_test_r : np.ndarray
        Possibly reduced matrices.
    """
    if X_train.shape[1] <= max_features:
        return X_train, X_test
    n_comp = min(max_features, X_train.shape[0] - 1, X_train.shape[1])

    # Fit on train, derive V, project both splits — GPU path avoids re-running
    # sklearn PCA which doesn't expose the right singular vectors easily.
    try:
        import torch
        if DEVICE == "cuda" and torch.cuda.is_available():
            free_vram = torch.cuda.mem_get_info()[0]
            bytes_fp16 = X_train.shape[0] * X_train.shape[1] * 2
            if bytes_fp16 < free_vram * 0.70:
                X_t = torch.tensor(
                    np.asarray(X_train, dtype=np.float32),
                    dtype=torch.float16, device="cuda",
                )
                _, _, V = torch.pca_lowrank(X_t, q=n_comp, niter=1)
                X_tr_r = (X_t @ V).cpu().float().numpy()
                X_te_t = torch.tensor(
                    np.asarray(X_test, dtype=np.float32),
                    dtype=torch.float16, device="cuda",
                )
                # centre test with train mean (stored implicitly in V via pca_lowrank)
                X_te_r = (X_te_t @ V).cpu().float().numpy()
                del X_t, X_te_t, V
                torch.cuda.empty_cache()
                logger.debug("PCA: using GPU torch.pca_lowrank")
                return X_tr_r, X_te_r
    except Exception:
        pass

    # CPU fallback
    logger.debug("PCA: falling back to CPU sklearn TruncatedSVD")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=42)
    return pca.fit_transform(X_train), pca.transform(X_test)


_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    pass

_HAS_MNE = False
try:
    import mne
    from mne.decoding import CSP
    _HAS_MNE = True
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# 1) CSP + LDA  (MI datasets only)
# ──────────────────────────────────────────────────────────────────────────────

def classify_csp_lda(X_train, y_train, X_test, y_test,
                     n_components: int = 6, **kwargs):
    """
    Common Spatial Patterns + Linear Discriminant Analysis.

    Parameters
    ----------
    X_train : np.ndarray
        Training data, shape (n_trials, n_channels, n_times).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test data, same shape convention.
    y_test : np.ndarray
        Test labels (unused during fitting).
    n_components : int
        Number of CSP components (default 6).

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities if available.
    """
    if not _HAS_MNE:
        logger.warning("MNE not available, falling back to LDA on flattened features")
        X_tr_flat = X_train.reshape(len(X_train), -1)
        X_te_flat = X_test.reshape(len(X_test), -1)
        # PCA reduce if too many features
        from sklearn.decomposition import PCA
        n_feat = X_tr_flat.shape[1]
        if n_feat > 500:
            pca = PCA(n_components=min(200, len(X_tr_flat) - 1))
            X_tr_flat = pca.fit_transform(X_tr_flat)
            X_te_flat = pca.transform(X_te_flat)
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_tr_flat, y_train)
        y_pred = lda.predict(X_te_flat)
        y_proba = lda.predict_proba(X_te_flat) if hasattr(lda, "predict_proba") else None
        return y_pred, y_proba

    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    X_tr_csp = csp.fit_transform(X_train, y_train)
    X_te_csp = csp.transform(X_test)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr_csp, y_train)
    y_pred = lda.predict(X_te_csp)
    y_proba = lda.predict_proba(X_te_csp) if hasattr(lda, "predict_proba") else None
    return y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────────
# 2) EEGNet-lite (simplified, PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

if _HAS_TORCH:
    class EEGNetLite(nn.Module):
        """
        Simplified EEGNet (Lawhern et al., 2018) for raw EEG classification.

        Architecture:
            Conv2D(temporal) → BatchNorm → DepthwiseConv2D(spatial)
            → BatchNorm → ELU → AvgPool → Dropout → SeparableConv2D
            → BatchNorm → ELU → AvgPool → Dropout → FC

        Parameters
        ----------
        n_channels : int
            Number of EEG channels.
        n_times : int
            Number of time samples.
        n_classes : int
            Number of output classes.
        F1 : int
            Number of temporal filters (default 8).
        D : int
            Depth multiplier for depthwise conv (default 2).
        F2 : int
            Number of separable conv filters (default 16).
        dropout : float
            Dropout rate (default 0.25).
        """

        def __init__(self, n_channels: int, n_times: int, n_classes: int,
                     F1: int = 8, D: int = 2, F2: int = 16,
                     dropout: float = 0.25):
            super().__init__()
            # Temporal convolution
            self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
            self.bn1 = nn.BatchNorm2d(F1)

            # Depthwise spatial convolution
            self.dw_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                     groups=F1, bias=False)
            self.bn2 = nn.BatchNorm2d(F1 * D)
            self.elu = nn.ELU()
            self.pool1 = nn.AvgPool2d((1, 4))
            self.drop1 = nn.Dropout(dropout)

            # Separable convolution
            self.sep_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8),
                                      bias=False)
            self.bn3 = nn.BatchNorm2d(F2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.drop2 = nn.Dropout(dropout)

            # Compute feature size
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                x = self.pool1(self.elu(self.bn2(
                    self.dw_conv(self.bn1(self.conv1(dummy))))))
                x = self.pool2(self.elu(self.bn3(self.sep_conv(x))))
                feat_size = x.view(1, -1).shape[1]

            self.fc = nn.Linear(feat_size, n_classes)

        def forward(self, x):
            """Forward pass. x: (batch, 1, n_channels, n_times)."""
            x = self.bn1(self.conv1(x))
            x = self.elu(self.bn2(self.dw_conv(x)))
            x = self.drop1(self.pool1(x))
            x = self.elu(self.bn3(self.sep_conv(x)))
            x = self.drop2(self.pool2(x))
            x = x.view(x.size(0), -1)
            return self.fc(x)


def classify_eegnet(X_train, y_train, X_test, y_test,
                    n_epochs: int = 50, lr: float = 1e-3,
                    batch_size: int = 64, device: str = DEVICE, **kwargs):
    """
    EEGNet-lite classifier on raw EEG epochs.

    Includes:
      - Cosine-annealing LR schedule
      - Early stopping (patience=5, 10% val split)
      - Automatic Mixed Precision on CUDA (fp16 forward/backward)
      - ``set_to_none=True`` for gradient zeroing (memory-efficient)

    Parameters
    ----------
    X_train : np.ndarray
        Training epochs, shape (n_trials, n_ch, n_times).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test epochs.
    y_test : np.ndarray
        Test labels (unused for training).
    n_epochs : int
        Maximum training epochs (default 50).
    lr : float
        Initial learning rate (default 1e-3).
    batch_size : int
        Batch size (default 64).
    device : str
        Compute device (auto-detected from config if not overridden).

    Returns
    -------
    y_pred : np.ndarray
    y_proba : np.ndarray or None
    """
    if not _HAS_TORCH:
        logger.warning("PyTorch unavailable, falling back to LogReg on flattened features")
        from sklearn.decomposition import PCA
        X_tr = X_train.reshape(len(X_train), -1)
        X_te = X_test.reshape(len(X_test), -1)
        n_feat = X_tr.shape[1]
        if n_feat > 500:
            pca = PCA(n_components=min(200, len(X_tr) - 1))
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="saga")
        clf.fit(X_tr, y_train)
        return clf.predict(X_te), clf.predict_proba(X_te)

    n_ch, n_times = X_train.shape[1], X_train.shape[2]
    n_classes = len(np.unique(y_train))
    dev = torch.device(device)
    logger.debug(f"EEGNet training on {device}")

    # ── Validation split (10%) ────────────────────────────────────────────
    n = len(X_train)
    n_val = max(1, int(0.10 * n))
    rng = np.random.RandomState(42)
    val_idx   = rng.choice(n, size=n_val, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False
    X_tr, y_tr = X_train[train_mask], y_train[train_mask]
    X_val, y_val = X_train[val_idx],  y_train[val_idx]

    # (N,1,C,T) tensors for Conv2D
    X_tr_t  = torch.FloatTensor(X_tr ).unsqueeze(1).to(dev)
    y_tr_t  = torch.LongTensor (y_tr ).to(dev)
    X_val_t = torch.FloatTensor(X_val).unsqueeze(1).to(dev)
    y_val_t = torch.LongTensor (y_val).to(dev)
    X_te_t  = torch.FloatTensor(X_test).unsqueeze(1).to(dev)

    model     = EEGNetLite(n_ch, n_times, n_classes).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=n_epochs)
    criterion = nn.CrossEntropyLoss()

    # AMP scaler — only activated for CUDA (fp16 → ~2× throughput)
    use_amp = (device == "cuda") and torch.cuda.is_available()
    if use_amp:
        try:
            scaler = torch.amp.GradScaler("cuda")    # PyTorch >= 2.0
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()      # PyTorch < 2.0 fallback
    else:
        scaler = None

    best_val_loss = float("inf")
    best_state    = None
    patience      = 5
    no_improve    = 0
    n_tr          = len(X_tr_t)

    _epoch_iter = (
        _tqdm(range(n_epochs), desc="EEGNet", unit="ep",
              leave=False, dynamic_ncols=True)
        if _HAS_TQDM else range(n_epochs)
    )
    for epoch in _epoch_iter:
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        for i in range(0, n_tr, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                try:
                    _amp_ctx = torch.amp.autocast("cuda")      # PyTorch >= 2.0
                except (AttributeError, TypeError):
                    _amp_ctx = torch.cuda.amp.autocast()        # PyTorch < 2.0
                with _amp_ctx:
                    loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(X_tr_t[idx]), y_tr_t[idx])
                loss.backward()
                optimizer.step()
        scheduler.step()

        # ── Validation / early stopping ───────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone()
                             for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if _HAS_TQDM and hasattr(_epoch_iter, "set_postfix"):
            _epoch_iter.set_postfix(
                val=f"{val_loss:.4f}",
                best=f"{best_val_loss:.4f}",
                pat=f"{no_improve}/{patience}",
                refresh=False,
            )
        if no_improve >= patience:
            logger.debug(f"EEGNet early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        proba  = torch.softmax(model(X_te_t), dim=1).cpu().numpy()
        y_pred = np.argmax(proba, axis=1)

    return y_pred, proba


# ──────────────────────────────────────────────────────────────────────────────
# 3) Fixed-size patching + LogReg
# ──────────────────────────────────────────────────────────────────────────────

def _patch_features(X: np.ndarray, patch_size_ms: int = 100,
                    sfreq: float = 256.0) -> np.ndarray:
    """
    Extract fixed-size patch features from EEG epochs.

    Parameters
    ----------
    X : np.ndarray
        Epochs, shape (n_trials, n_ch, n_times).
    patch_size_ms : int
        Patch duration in milliseconds.
    sfreq : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Features, shape (n_trials, n_patches * n_ch * patch_len).
    """
    patch_len = max(1, int(patch_size_ms * sfreq / 1000))
    n_trials, n_ch, n_times = X.shape
    n_patches = n_times // patch_len

    # Reshape into patches: (n_trials, n_ch, n_patches, patch_len)
    trimmed = X[:, :, :n_patches * patch_len]
    patches = trimmed.reshape(n_trials, n_ch, n_patches, patch_len)

    # Features: mean + std per patch per channel
    means = patches.mean(axis=-1)   # (n_trials, n_ch, n_patches)
    stds = patches.std(axis=-1)
    features = np.concatenate([means, stds], axis=-1)  # (n_trials, n_ch, 2*n_patches)
    return features.reshape(n_trials, -1)


def classify_patching_logreg(X_train, y_train, X_test, y_test,
                             patch_size_ms: int = 100, sfreq: float = 256.0, **kwargs):
    """
    Fixed-size patching + Logistic Regression baseline.

    Parameters
    ----------
    X_train : np.ndarray
        Training epochs (n_trials, n_ch, n_times).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test epochs.
    y_test : np.ndarray
        Test labels (unused for training).
    patch_size_ms : int
        Patch size in milliseconds (default 100).
    sfreq : float
        Sampling frequency in Hz (default 256).

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities.
    """
    X_tr_feat = _patch_features(X_train, patch_size_ms, sfreq)
    X_te_feat = _patch_features(X_test, patch_size_ms, sfreq)
    # Reduce high-dimensional patch features (e.g. physionet_mi: 64ch×50pts = 3200)
    X_tr_feat, X_te_feat = _reduce_dense_features(X_tr_feat, X_te_feat)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="saga")
    clf.fit(X_tr_feat, y_train)
    y_pred = clf.predict(X_te_feat)
    y_proba = clf.predict_proba(X_te_feat)
    return y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────────
# 4) VQ tokenization + LogReg (approximation of VQ-VAE + Transformer)
# ──────────────────────────────────────────────────────────────────────────────

def _vq_tokenize(X: np.ndarray, n_codes: int = 256,
                 patch_size: int = 8) -> np.ndarray:
    """
    Simple VQ tokenization: patches → k-means codes → histogram.

    Parameters
    ----------
    X : np.ndarray
        Epochs (n_trials, n_ch, n_times).
    n_codes : int
        Size of VQ codebook (default 256).
    patch_size : int
        Number of samples per patch (default 8).

    Returns
    -------
    np.ndarray
        Histogram features (n_trials, n_codes).
    """
    from sklearn.cluster import MiniBatchKMeans

    n_trials, n_ch, n_times = X.shape
    n_patches = n_times // patch_size

    # Extract patches
    trimmed = X[:, :, :n_patches * patch_size]
    patches = trimmed.reshape(n_trials, n_ch, n_patches, patch_size)
    # Flatten channels and patches for k-means: each sample = one patch across channels
    all_patches = patches.transpose(0, 2, 1, 3).reshape(-1, n_ch * patch_size)

    # Sub-sample for k-means training (max 50k patches)
    rng = np.random.RandomState(42)
    n_total = len(all_patches)
    if n_total > 50000:
        idx = rng.choice(n_total, 50000, replace=False)
        train_patches = all_patches[idx]
    else:
        train_patches = all_patches

    km = MiniBatchKMeans(n_clusters=n_codes, batch_size=1024,
                         n_init=3, random_state=42)
    km.fit(train_patches)
    codes = km.predict(all_patches).reshape(n_trials, n_patches)

    # Histogram features — batch bincount (vectorised)
    hist = np.zeros((n_trials, n_codes), dtype=np.float32)
    # Use offset trick: shift each trial's codes by trial_idx * n_codes
    offsets = np.arange(n_trials)[:, None] * n_codes
    shifted = (codes + offsets).ravel()
    counts = np.bincount(shifted, minlength=n_trials * n_codes)
    hist = counts[:n_trials * n_codes].reshape(n_trials, n_codes).astype(np.float32)
    # Normalize
    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-10)
    return hist


def classify_vq_logreg(X_train, y_train, X_test, y_test,
                       n_codes: int = 256, patch_size: int = 8, **kwargs):
    """
    VQ tokenization + Logistic Regression baseline.

    Parameters
    ----------
    X_train : np.ndarray
        Training epochs.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test epochs.
    y_test : np.ndarray
        Test labels.
    n_codes : int
        VQ codebook size (default 256).
    patch_size : int
        Patch size in samples (default 8).

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities.
    """
    from sklearn.cluster import MiniBatchKMeans

    n_ch, n_times = X_train.shape[1], X_train.shape[2]
    n_patches_tr = n_times // patch_size
    n_patches_te = n_times // patch_size

    # Extract patches
    all_data = np.concatenate([X_train, X_test], axis=0)
    trimmed = all_data[:, :, :n_patches_tr * patch_size]
    patches = trimmed.reshape(len(all_data), n_ch,
                              n_patches_tr, patch_size)
    flat_patches = patches.transpose(0, 2, 1, 3).reshape(-1, n_ch * patch_size)

    # Train k-means on training patches only
    n_tr_patches = len(X_train) * n_patches_tr
    train_p = flat_patches[:n_tr_patches]
    rng = np.random.RandomState(42)
    if len(train_p) > 50000:
        idx = rng.choice(len(train_p), 50000, replace=False)
        train_p_sub = train_p[idx]
    else:
        train_p_sub = train_p

    km = MiniBatchKMeans(n_clusters=n_codes, batch_size=1024,
                         n_init=3, random_state=42)
    km.fit(train_p_sub)

    all_codes = km.predict(flat_patches).reshape(len(all_data), n_patches_tr)

    # Histograms — batch bincount (vectorised)
    n_total_data = len(all_data)
    offsets = np.arange(n_total_data)[:, None] * n_codes
    shifted = (all_codes + offsets).ravel()
    counts = np.bincount(shifted, minlength=n_total_data * n_codes)
    hist = counts[:n_total_data * n_codes].reshape(n_total_data, n_codes).astype(np.float32)
    hist = hist / (hist.sum(axis=1, keepdims=True) + 1e-10)

    X_tr_h = hist[:len(X_train)]
    X_te_h = hist[len(X_train):]

    clf = LogisticRegression(max_iter=500, C=1.0, solver="saga")
    clf.fit(X_tr_h, y_train)
    y_pred = clf.predict(X_te_h)
    y_proba = clf.predict_proba(X_te_h)
    return y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────────
# 5) Chronos-style binning + LogReg
# ──────────────────────────────────────────────────────────────────────────────

def classify_chronos_binning(X_train, y_train, X_test, y_test,
                             n_bins: int = 256, method: str = "mu_law", **kwargs):
    """
    Chronos-style binning baseline: quantize → histogram (no BPE).

    Parameters
    ----------
    X_train : np.ndarray
        Training epochs (n_trials, n_ch, n_times).
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test epochs.
    y_test : np.ndarray
        Test labels.
    n_bins : int
        Number of quantization bins (default 256).
    method : str
        Quantization method ('mu_law', 'uniform', 'adaptive').

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels.
    y_proba : np.ndarray or None
        Predicted probabilities.
    """
    def _make_hist(X, n_bins, method, chunk_trials: int = 256):
        """Chunked histogram to avoid multi-GB 'shifted' arrays on large datasets."""
        n_trials, n_ch, n_times = X.shape
        out = np.empty((n_trials, n_ch * n_bins), dtype=np.float32)
        for t0 in range(0, n_trials, chunk_trials):
            t1 = min(t0 + chunk_trials, n_trials)
            chunk = X[t0:t1]                            # (C, n_ch, n_times)
            n_c = t1 - t0
            flat = chunk.reshape(-1, n_times)            # (C*n_ch, n_times)
            codes_flat, _ = quantize(flat, method, n_bins, normalize=True)
            # offset-trick bincount
            n_rows = codes_flat.shape[0]
            offsets = np.arange(n_rows, dtype=np.int64)[:, None] * n_bins
            shifted = (codes_flat.astype(np.int64) + offsets).ravel()
            counts = np.bincount(shifted, minlength=n_rows * n_bins)
            h = counts[:n_rows * n_bins].reshape(n_c, n_ch, n_bins).astype(np.float32)
            h /= h.sum(axis=-1, keepdims=True) + 1e-10
            out[t0:t1] = h.reshape(n_c, -1)
        return out

    X_tr_feat = _make_hist(X_train, n_bins, method)
    X_te_feat = _make_hist(X_test, n_bins, method)
    # Reduce if many channels × bins (e.g. physionet_mi: 64ch×256bins = 16 384)
    X_tr_feat, X_te_feat = _reduce_dense_features(X_tr_feat, X_te_feat)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="saga")
    clf.fit(X_tr_feat, y_train)
    y_pred = clf.predict(X_te_feat)
    y_proba = clf.predict_proba(X_te_feat)
    return y_pred, y_proba


# ──────────────────────────────────────────────────────────────────────────────
# PSD Band-Power Baseline
# ──────────────────────────────────────────────────────────────────────────────

def classify_psd_logreg(X_train, y_train, X_test, y_test,
                        sfreq: float = 256.0, **kwargs):
    """Welch PSD in 5 canonical EEG bands (δ/θ/α/β/γ) per channel → LogReg.

    Vectorised: reshapes (n_trials, n_ch, n_times) → (n_trials*n_ch, n_times),
    calls scipy.signal.welch once, then extracts band-mean log-power.
    Features: (n_trials, n_ch × 5), fully within-fold normalisation.

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (n_trials, n_ch, n_times)
        Raw EEG epochs.
    y_train, y_test : np.ndarray
        Integer class labels.
    sfreq : float
        Sampling frequency in Hz (passed from exp2 closure).
    """
    from scipy.signal import welch as _welch
    from sklearn.preprocessing import StandardScaler
    from .config import LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL

    BANDS = [("delta", 1.0, 4.0),
             ("theta", 4.0, 8.0),
             ("alpha", 8.0, 13.0),
             ("beta",  13.0, 30.0),
             ("gamma", 30.0, 45.0)]

    def _extract(X):
        n_tr, n_ch, n_t = X.shape
        Xf = X.reshape(n_tr * n_ch, n_t)                          # (N, T)
        nperseg = min(256, n_t)
        freqs, pxx = _welch(Xf, fs=sfreq, nperseg=nperseg)        # (N, F)
        pxx_log = np.log1p(pxx).reshape(n_tr, n_ch, -1)           # (n_tr, n_ch, F)
        band_feats = []
        for _, lo, hi in BANDS:
            mask = (freqs >= lo) & (freqs < hi)
            if mask.sum() == 0:
                # band not resolvable at this sfreq — use zeros
                band_feats.append(np.zeros((n_tr, n_ch), dtype=np.float32))
            else:
                band_feats.append(pxx_log[:, :, mask].mean(axis=-1))
        return np.concatenate(band_feats, axis=-1)                 # (n_tr, n_ch*5)

    Xtr = _extract(X_train)
    Xte = _extract(X_test)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER,
                             solver=LOGREG_SOLVER, class_weight=LOGREG_CLASS_WEIGHT,
                             tol=LOGREG_TOL)
    clf.fit(Xtr, y_train)
    return clf.predict(Xte), clf.predict_proba(Xte)


# ──────────────────────────────────────────────────────────────────────────────
# SSVEP FFT Baseline
# ──────────────────────────────────────────────────────────────────────────────

def classify_ssvep_fft(X_train, y_train, X_test, y_test,
                       sfreq: float = 256.0, **kwargs):
    """FFT power spectrum (1–45 Hz) per channel → PCA → LogReg.

    SSVEP responses appear as sharp peaks at the stimulus frequency and its
    harmonics.  Computing the full FFT magnitude spectrum gives the classifier
    access to all frequency content; PCA compresses before LogReg.

    Vectorised: applies Hann window and rfft over (n_trials, n_ch, n_times)
    simultaneously.

    Parameters
    ----------
    X_train, X_test : np.ndarray, shape (n_trials, n_ch, n_times)
        Raw EEG epochs.
    y_train, y_test : np.ndarray
        Integer class labels.
    sfreq : float
        Sampling frequency in Hz.
    """
    from sklearn.preprocessing import StandardScaler
    from .config import LOGREG_C, LOGREG_MAX_ITER, LOGREG_SOLVER, LOGREG_CLASS_WEIGHT, LOGREG_TOL

    def _extract(X):
        n_tr, n_ch, n_t = X.shape
        window = np.hanning(n_t)                                    # (n_t,)
        Xw = X * window                                             # broadcast over (n_tr, n_ch)
        ps = np.abs(np.fft.rfft(Xw, axis=-1)) ** 2                 # (n_tr, n_ch, n_freq)
        freqs = np.fft.rfftfreq(n_t, d=1.0 / sfreq)
        mask = (freqs >= 1.0) & (freqs <= 45.0)
        ps_band = np.log1p(ps[:, :, mask])                         # (n_tr, n_ch, n_f_band)
        return ps_band.reshape(n_tr, -1)                            # (n_tr, n_ch * n_f_band)

    Xtr = _extract(X_train)
    Xte = _extract(X_test)

    # PCA: for SSVEP (8ch × ~175 freq bins = 1400 features) this is fine;
    # for larger datasets _reduce_dense_features applies TruncatedSVD.
    Xtr, Xte = _reduce_dense_features(Xtr, Xte, max_features=512)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(C=LOGREG_C, max_iter=LOGREG_MAX_ITER,
                             solver=LOGREG_SOLVER, class_weight=LOGREG_CLASS_WEIGHT,
                             tol=LOGREG_TOL)
    clf.fit(Xtr, y_train)
    return clf.predict(Xte), clf.predict_proba(Xte)


# ──────────────────────────────────────────────────────────────────────────────
# Registry: name → (callable, requires_raw_epochs)
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_CLASSIFIERS = {
    "CSP_LDA": {
        "fn": classify_csp_lda,
        "needs_raw": True,
        "mi_only": True,
        "max_parallel_seeds": 1,  # BLAS/LAPACK segfault on large datasets (physionet 109 subj) with threaded parallelism
        "description": "Common Spatial Patterns + LDA (classical MI baseline)",
    },
    "EEGNet": {
        "fn": classify_eegnet,
        "needs_raw": True,
        "mi_only": False,
        "no_parallel_seeds": True,   # GPU: only one CUDA context at a time
        "description": "EEGNet-lite on raw EEG",
    },
    "Patching_LogReg": {
        "fn": classify_patching_logreg,
        "needs_raw": True,
        "mi_only": False,
        "description": "Fixed-size patching (100ms) + LogReg",
    },
    "VQ_LogReg": {
        "fn": classify_vq_logreg,
        "needs_raw": True,
        "mi_only": False,
        "description": "VQ tokenization + LogReg (VQ-VAE approx)",
    },
    "Chronos_Binning": {
        "fn": classify_chronos_binning,
        "needs_raw": True,
        "mi_only": False,
        "max_parallel_seeds": 3,     # chunked _make_hist limits peak RAM → safe at 3
        "description": "Chronos-style μ-law binning + LogReg (no BPE)",
    },
    "PSD_LogReg": {
        "fn": classify_psd_logreg,
        "needs_raw": True,
        "mi_only": False,
        "description": "Welch PSD 5 bands × channels → LogReg (all paradigms)",
    },
    "SSVEP_FFT_LogReg": {
        "fn": classify_ssvep_fft,
        "needs_raw": True,
        "mi_only": False,
        "ssvep_only": True,          # FFT baseline is specifically designed for SSVEP
        "description": "Full FFT power spectrum (1–45 Hz) → PCA → LogReg (SSVEP only)",
    },
}
