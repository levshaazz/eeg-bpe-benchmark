"""
Neural network classifiers for BPE token sequences.
====================================================
Implements:
  - 1D-CNN on BPE token sequences
  - Small Transformer (2 layers, d=128) on BPE token sequences
  - ChannelwiseTransformerClassifier: per-channel Transformer with
    channel aggregation — avoids the max_len truncation problem that
    arises when all channels are naively concatenated into one sequence.

All classifiers use:
  - PAD_TOKEN = vocab_size + 1  (token 0 is a valid amplitude bin, not padding)
  - SEP_TOKEN = vocab_size       (channel separator)
  - Cosine-annealing LR scheduler
  - Early stopping with 10% validation split (patience=5)

Falls back to scikit-learn MLP when PyTorch is not installed.
"""
from __future__ import annotations

import numpy as np
from typing import Optional
from .config import DEVICE
from .utils import get_logger

logger = get_logger("classifiers")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─── Check for PyTorch availability ──────────────────────────────────────────
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not installed. CNN/Transformer classifiers will use "
                   "sklearn MLP fallback. Install pytorch for full functionality.")


# ══════════════════════════════════════════════════════════════════════════════
# Token ID conventions
# ══════════════════════════════════════════════════════════════════════════════
# base tokens : 0 … n_bins-1        (valid amplitude bins — NOT padding)
# merged tokens: n_bins … vocab_size-1
# SEP_TOKEN   : vocab_size           (channel separator in flat sequences)
# PAD_TOKEN   : vocab_size + 1       (padding — outside real token range)
#
# Embedding table size: vocab_size + 2
# padding_idx         : vocab_size + 1   (PAD_TOKEN only, never token 0)

def get_pad_token(vocab_size: int) -> int:
    """Return the PAD token id for a given vocabulary size."""
    return vocab_size + 1


def get_sep_token(vocab_size: int) -> int:
    """Return the SEP token id for a given vocabulary size."""
    return vocab_size


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch models
# ══════════════════════════════════════════════════════════════════════════════

if _HAS_TORCH:
    from contextlib import nullcontext as _nullctx  # no-op context manager

    class Conv1DClassifier(nn.Module):
        """
        1D-CNN classifier for BPE token sequences.

        Architecture:
            Embedding(vocab_size+2, d_model, padding_idx=PAD) →
            Conv1D(d_model, 64, k=3) → ReLU → MaxPool →
            Conv1D(64, 128, k=3) → ReLU → AdaptiveMaxPool →
            Dropout → Linear(128, n_classes)

        Parameters
        ----------
        vocab_size : int
            Total vocabulary size (base + BPE merged tokens).
        n_classes : int
            Number of output classes.
        d_model : int
            Embedding dimension.
        """

        def __init__(self, vocab_size: int, n_classes: int, d_model: int = 64):
            super().__init__()
            pad_token = get_pad_token(vocab_size)
            # +2 covers SEP (vocab_size) and PAD (vocab_size+1)
            self.embedding = nn.Embedding(vocab_size + 2, d_model,
                                          padding_idx=pad_token)
            self.conv1 = nn.Conv1d(d_model, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(128, n_classes)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            self._pad_token = pad_token

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Integer token IDs, shape (batch, seq_len).
                PAD positions are masked to zero after embedding.

            Returns
            -------
            torch.Tensor
                Logits, shape (batch, n_classes).
            """
            emb = self.embedding(x)                  # (B, L, D)
            # Zero out PAD positions so pooling ignores them
            pad_mask = (x == self._pad_token).unsqueeze(-1)  # (B, L, 1)
            emb = emb.masked_fill(pad_mask, 0.0)
            emb = emb.transpose(1, 2)                # (B, D, L)
            h = self.relu(self.conv1(emb))           # (B, 64, L)
            h = self.relu(self.conv2(h))             # (B, 128, L)
            h = self.pool(h).squeeze(-1)             # (B, 128)
            h = self.dropout(h)
            return self.fc(h)                        # (B, n_classes)

    # ─────────────────────────────────────────────────────────────────────────

    class TransformerClassifier(nn.Module):
        """
        Small Transformer classifier for BPE token sequences.

        Processes a flat (all-channels-concatenated) sequence.  For high-
        channel-count EEG, prefer ChannelwiseTransformerClassifier which
        avoids the max_len truncation problem.

        Architecture:
            Embedding(vocab_size+2, d_model=128, padding_idx=PAD)
            + positional encoding →
            TransformerEncoder(n_layers=2, n_heads=4) →
            masked mean-pool → Linear(d_model, n_classes)

        Parameters
        ----------
        vocab_size : int
            Total vocabulary size.
        n_classes : int
            Number of output classes.
        d_model : int
            Model dimension (default 128).
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer encoder layers.
        max_len : int
            Maximum sequence length for positional encoding.
        """

        def __init__(self, vocab_size: int, n_classes: int,
                     d_model: int = 128, n_heads: int = 4,
                     n_layers: int = 2, max_len: int = 512):
            super().__init__()
            pad_token = get_pad_token(vocab_size)
            self._pad_token = pad_token
            self.embedding = nn.Embedding(vocab_size + 2, d_model,
                                          padding_idx=pad_token)
            self.pos_enc = nn.Embedding(max_len, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model, n_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, L = x.shape
            positions = torch.arange(L, device=x.device).clamp(max=self.pos_enc.num_embeddings - 1).unsqueeze(0).expand(B, L)
            emb = self.embedding(x) + self.pos_enc(positions)
            pad_mask = (x == self._pad_token)
            h = self.encoder(emb, src_key_padding_mask=pad_mask)
            # Mean-pool over non-PAD positions only
            mask_exp = (~pad_mask).unsqueeze(-1).float()
            h = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
            h = self.dropout(h)
            return self.fc(h)

    # ─────────────────────────────────────────────────────────────────────────

    class ChannelwiseTransformerClassifier(nn.Module):
        """
        Per-channel Transformer classifier for BPE token sequences.

        Each EEG channel's BPE sequence is processed independently through
        a shared Transformer encoder, then per-channel vectors are
        aggregated (mean-pooled) before final classification.

        This avoids the max_len truncation problem: instead of concatenating
        all channels (e.g., 22 × 830 ≈ 18 000 tokens, truncated to 512),
        each channel gets up to *max_len_per_ch* tokens.

        Input shape: ``(batch, n_ch, max_len_per_ch)``

        Architecture:
            Shared Embedding(vocab_size+2, d_model, padding_idx=PAD) →
            Shared TransformerEncoder(n_layers, n_heads) →
            per-channel masked mean-pool →
            mean over channels → Dropout → Linear(d_model, n_classes)

        Parameters
        ----------
        vocab_size : int
            Total BPE vocabulary size.
        n_classes : int
            Number of output classes.
        n_ch : int
            Number of EEG channels (needed only for type checking).
        d_model : int
            Model dimension (default 128).
        n_heads : int
            Number of attention heads.
        n_layers : int
            Number of transformer encoder layers.
        max_len_per_ch : int
            Maximum token length per channel for positional encoding.
        """

        def __init__(self, vocab_size: int, n_classes: int, n_ch: int,
                     d_model: int = 128, n_heads: int = 4,
                     n_layers: int = 2, max_len_per_ch: int = 128):
            super().__init__()
            pad_token = get_pad_token(vocab_size)
            self._pad_token = pad_token
            self.embedding = nn.Embedding(vocab_size + 2, d_model,
                                          padding_idx=pad_token)
            self.pos_enc = nn.Embedding(max_len_per_ch, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.fc = nn.Linear(d_model, n_classes)
            self.dropout = nn.Dropout(0.1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : torch.Tensor
                Shape ``(batch, n_ch, max_len_per_ch)``.

            Returns
            -------
            torch.Tensor
                Logits, shape ``(batch, n_classes)``.
            """
            B, C, L = x.shape
            # Flatten batch and channels → process all channel-sequences at once
            x_flat = x.reshape(B * C, L)                        # (B*C, L)
            positions = torch.arange(L, device=x.device)\
                             .clamp(max=self.pos_enc.num_embeddings - 1)\
                             .unsqueeze(0).expand(B * C, L)
            emb = self.embedding(x_flat) + self.pos_enc(positions)  # (B*C, L, D)
            pad_mask = (x_flat == self._pad_token)
            h = self.encoder(emb, src_key_padding_mask=pad_mask)    # (B*C, L, D)
            # Per-channel masked mean-pool
            mask_exp = (~pad_mask).unsqueeze(-1).float()             # (B*C, L, 1)
            h_ch = (h * mask_exp).sum(dim=1) / (mask_exp.sum(dim=1) + 1e-8)
            # (B*C, D) → (B, C, D) → mean over channels → (B, D)
            h_agg = h_ch.reshape(B, C, -1).mean(dim=1)
            h_agg = self.dropout(h_agg)
            return self.fc(h_agg)                                    # (B, n_classes)

    # ─────────────────────────────────────────────────────────────────────────

    def _train_torch_model(model: nn.Module, X_train: np.ndarray,
                           y_train: np.ndarray, epochs: int = 30,
                           batch_size: int = None, lr: float = 1e-3,
                           device: str = DEVICE,
                           patience: int = 5) -> nn.Module:
        """
        Train a PyTorch model on integer sequences.

        Includes:
          - 10% validation split for early stopping
          - Cosine-annealing learning-rate schedule
          - Automatic device placement

        Parameters
        ----------
        model : nn.Module
        X_train : np.ndarray
            Integer token sequences, shape (n_samples, ...).
        y_train : np.ndarray
            Integer labels.
        epochs : int
            Maximum training epochs.
        batch_size : int
        lr : float
            Initial learning rate.
        device : str
            ``'cpu'``, ``'cuda'``, or ``'mps'``.
        patience : int
            Early-stopping patience (epochs without val-loss improvement).

        Returns
        -------
        nn.Module
            Best model (restored from best val-loss checkpoint).
        """
        dev = torch.device(device)

        # ── Resolve default batch size from config ────────────────────────
        if batch_size is None:
            from .config import BATCH_SIZE as _BATCH_SIZE
            batch_size = _BATCH_SIZE  # 512 by default

        # NOTE: torch.cuda.empty_cache() removed — it is a synchronous GPU
        # barrier (~1-10 ms each) and was called ~500× per full run.  PyTorch's
        # caching allocator reuses freed blocks automatically.

        # ── For 3D input (n_trials, n_ch, seq_len), scale batch_size down ──────
        # CW_Transformer flattens to (B*n_ch, seq_len) on the GPU.
        # Profiled optimal batch sizes (RTX 5070 Ti, 17 GB):
        #   n_ch= 2: bs=256 → 3.5 GB, 3319 samp/s
        #   n_ch=22: bs= 16 → 2.4 GB,  239 samp/s (bs=32 → 40 samp/s!)
        #   n_ch=64: bs=  4 → 1.8 GB,   40 samp/s (bs=8  →  3 samp/s!)
        # The VRAM cliff happens at ~3.5 GB effective; beyond that,
        # CUDA paging drops throughput 6-100x.
        if X_train.ndim == 3:
            n_ch_eff = X_train.shape[1]
            try:
                free_vram_mb = torch.cuda.mem_get_info()[0] / 1e6
            except Exception:
                free_vram_mb = 12000  # conservative default
            # Profiled: ~7 MB per effective element (B*n_ch) at AMP fp16.
            # Target ~3 GB peak to stay below the throughput cliff.
            vram_budget_mb = min(free_vram_mb * 0.40, 3500)
            max_eff = max(8, int(vram_budget_mb / 7))
            batch_size = max(4, min(batch_size, max_eff // max(n_ch_eff, 1)))
            logger.debug(f"3D input: auto-scaled batch_size={batch_size} "
                         f"(n_ch={n_ch_eff}, effective={batch_size * n_ch_eff}, "
                         f"vram_budget={vram_budget_mb:.0f} MB)")

        model = model.to(dev)

        # ── torch.compile — CUDA only, PyTorch ≥ 2.0 ─────────────────────
        # Kernel fusion via Triton gives ~20–40 % extra throughput on Ampere+.
        # NOTE: 'reduce-overhead' and 'inductor' backends require Triton.
        # The TritonMissing error fires on the first forward pass (not at
        # compile time), so we check for Triton import *before* compiling.
        if device == "cuda" and hasattr(torch, "compile"):
            try:
                import triton  # noqa: F401 — raises ImportError if not installed
                model = torch.compile(model, mode="reduce-overhead")
                logger.debug("torch.compile(reduce-overhead) enabled")
            except ImportError:
                logger.debug("torch.compile skipped: triton not installed")
            except Exception as _ce:
                logger.debug(f"torch.compile skipped: {_ce}")

        # ── AMP (Automatic Mixed Precision) — CUDA only ───────────────────
        # Provides ~2× throughput on modern GPUs with no accuracy loss.
        # Not used on MPS (bfloat16 support experimental) or CPU.
        use_amp = (device == "cuda") and hasattr(torch, "cuda") and torch.cuda.is_available()
        try:
            # PyTorch ≥ 2.0 new API (device-agnostic)
            scaler = torch.amp.GradScaler("cuda") if use_amp else None
            def _autocast():
                return torch.amp.autocast("cuda") if use_amp else _nullctx()
        except (AttributeError, TypeError):
            # PyTorch < 2.0 legacy API
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
            def _autocast():
                return torch.cuda.amp.autocast() if use_amp else _nullctx()

        # ── Validation split (10%) ────────────────────────────────────────
        n = len(X_train)
        n_val = max(1, int(0.10 * n))
        rng = np.random.RandomState(42)
        val_idx = rng.choice(n, size=n_val, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]

        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        X_tr_t  = torch.tensor(X_tr,  dtype=torch.long, device=dev)
        y_tr_t  = torch.tensor(y_tr,  dtype=torch.long, device=dev)
        X_val_t = torch.tensor(X_val, dtype=torch.long, device=dev)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=dev)

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             pin_memory=False)  # tensors already on device

        optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion  = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state    = None
        no_improve    = 0

        _desc = model.__class__.__name__
        _epoch_iter = (
            _tqdm(range(epochs), desc=_desc, unit="ep",
                  leave=False, dynamic_ncols=True)
            if _HAS_TQDM else range(epochs)
        )
        for epoch in _epoch_iter:
            model.train()
            running_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)  # more GPU-memory-efficient
                if use_amp:
                    with _autocast():
                        loss = criterion(model(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * len(xb)
            epoch_train_loss = running_loss / len(X_tr)
            scheduler.step()

            # ── Validation ────────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss   = criterion(val_logits, y_val_t).item()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.clone()
                                 for k, v in model.state_dict().items()}
                no_improve    = 0
            else:
                no_improve += 1

            if _HAS_TQDM and hasattr(_epoch_iter, "set_postfix"):
                _epoch_iter.set_postfix(
                    train=f"{epoch_train_loss:.4f}",
                    val=f"{val_loss:.4f}",
                    best=f"{best_val_loss:.4f}",
                    pat=f"{no_improve}/{patience}",
                    refresh=False,
                )
            if no_improve >= patience:
                logger.debug(f"Early stop at epoch {epoch+1} "
                             f"(val_loss={best_val_loss:.4f})")
                break

        # Restore best weights (already on correct device — no transfer needed)
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        return model

    def _predict_torch_model(model: nn.Module, X_test: np.ndarray,
                             device: str = DEVICE,
                             pred_batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict with a trained PyTorch model.

        Uses chunked batches to avoid VRAM spikes on large test sets
        (critical for 3-D CW_Transformer inputs with many channels).

        Returns
        -------
        y_pred : np.ndarray
        y_proba : np.ndarray
        """
        dev = torch.device(device)
        model.eval()
        # For 3D inputs, scale pred_batch_size using same VRAM-aware logic
        if X_test.ndim == 3:
            n_ch_eff = X_test.shape[1]
            try:
                free_vram_mb = torch.cuda.mem_get_info()[0] / 1e6
            except Exception:
                free_vram_mb = 12000
            # Inference uses ~50% less VRAM than training (no gradients)
            max_eff = max(8, int(free_vram_mb * 0.60 / 7))
            pred_batch_size = max(4, min(pred_batch_size, max_eff // max(n_ch_eff, 1)))

        all_proba = []
        with torch.no_grad():
            for start in range(0, len(X_test), pred_batch_size):
                chunk = X_test[start:start + pred_batch_size]
                X_chunk = torch.tensor(chunk, dtype=torch.long, device=dev)
                logits = model(X_chunk)
                all_proba.append(torch.softmax(logits, dim=-1).cpu().numpy())
        proba = np.concatenate(all_proba, axis=0)
        return np.argmax(proba, axis=1), proba


# ══════════════════════════════════════════════════════════════════════════════
# Public classifier functions
# ══════════════════════════════════════════════════════════════════════════════

def classify_seq_cnn(X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     vocab_size: int = 4096,
                     device: str = DEVICE) -> tuple[np.ndarray, np.ndarray | None]:
    """
    1D-CNN classifier on flat BPE token sequences.

    Parameters
    ----------
    X_train : np.ndarray
        Training sequences, shape (n_train, max_len), dtype int.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test sequences.
    y_test : np.ndarray
        Test labels (unused; for API consistency).
    vocab_size : int
        Total BPE vocabulary size.
    device : str
        Compute device.

    Returns
    -------
    y_pred : np.ndarray
    y_proba : np.ndarray or None
    """
    if not _HAS_TORCH:
        return _fallback_mlp(X_train, y_train, X_test, vocab_size)

    n_classes = len(np.unique(y_train))
    model = Conv1DClassifier(vocab_size, n_classes, d_model=64)
    model = _train_torch_model(model, X_train, y_train, epochs=30, device=device)
    return _predict_torch_model(model, X_test, device=device)


def classify_seq_transformer(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              vocab_size: int = 4096,
                              device: str = DEVICE) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Flat-sequence Transformer classifier.

    Input shape: ``(n_trials, max_len)`` — all channels concatenated.
    Prefer ``classify_seq_transformer_channelwise`` for datasets with
    many channels (> 8) to avoid max_len truncation loss.
    """
    if not _HAS_TORCH:
        return _fallback_mlp(X_train, y_train, X_test, vocab_size)

    n_classes = len(np.unique(y_train))
    max_len   = X_train.shape[1]
    model = TransformerClassifier(vocab_size, n_classes,
                                   d_model=128, n_heads=4, n_layers=2,
                                   max_len=max_len)
    model = _train_torch_model(model, X_train, y_train, epochs=30, device=device)
    return _predict_torch_model(model, X_test, device=device)


def classify_seq_transformer_channelwise(
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        vocab_size: int = 4096,
        device: str = DEVICE) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Per-channel Transformer classifier.

    Input shape: ``(n_trials, n_ch, max_len_per_ch)`` — each channel
    tokenized independently.  Avoids the max_len truncation problem
    of the flat-sequence Transformer.

    Parameters
    ----------
    X_train : np.ndarray
        Shape ``(n_train, n_ch, max_len_per_ch)``, integer token ids.
    y_train : np.ndarray
        Integer labels.
    X_test : np.ndarray
        Shape ``(n_test, n_ch, max_len_per_ch)``.
    y_test : np.ndarray
        Test labels (unused).
    vocab_size : int
        Total BPE vocabulary size.
    device : str
        Compute device.

    Returns
    -------
    y_pred : np.ndarray
    y_proba : np.ndarray
    """
    if not _HAS_TORCH:
        # Flatten channels for MLP fallback
        X_tr_flat = X_train.reshape(len(X_train), -1)
        X_te_flat = X_test.reshape(len(X_test), -1)
        return _fallback_mlp(X_tr_flat, y_train, X_te_flat, vocab_size)

    n_classes      = len(np.unique(y_train))
    n_ch           = X_train.shape[1]
    max_len_per_ch = X_train.shape[2]
    model = ChannelwiseTransformerClassifier(
        vocab_size, n_classes, n_ch,
        d_model=128, n_heads=4, n_layers=2,
        max_len_per_ch=max_len_per_ch,
    )
    model = _train_torch_model(model, X_train, y_train, epochs=30, device=device)
    return _predict_torch_model(model, X_test, device=device)


def classify_windowed_hist_cnn(
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        n_windows: int = 5,
        device: str = DEVICE) -> tuple[np.ndarray, np.ndarray | None]:
    """
    1D-CNN over BPE windowed histograms (float features, NOT token IDs).

    Takes pre-computed windowed BPE histograms of shape
    ``(n_trials, n_windows * feat_dim)`` and applies a Conv1D over the
    temporal window dimension, preserving window order information that
    flat histogram classifiers (LogReg) discard.

    Architecture:
        Reshape (B, W*F) → (B, W, F)
        Linear(F, 128) → ReLU  [applied identically per window]
        Conv1d(128, 64, k=3, p=1) → ReLU
        AdaptiveMaxPool1d(1)
        Dropout(0.3) → Linear(64, n_classes)

    Parameters
    ----------
    X_train / X_test : (n_trials, n_windows * feat_dim)  float32
        Pre-computed windowed BPE histogram features.
    n_windows : int
        Number of temporal windows (must match histogram construction).
    device : str
        Compute device (default: from config).

    Returns
    -------
    y_pred  : np.ndarray
    y_proba : np.ndarray or None
    """
    if not _HAS_TORCH:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced",
                                solver="saga")
        lr.fit(sc.fit_transform(X_train), y_train)
        y_pred = lr.predict(sc.transform(X_test))
        return y_pred, lr.predict_proba(sc.transform(X_test))

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    total_feats = X_train.shape[1]
    feat_dim    = total_feats // n_windows
    if total_feats % n_windows != 0:
        usable = feat_dim * n_windows
        logger.warning(f"total_feats={total_feats} not divisible by n_windows={n_windows}, "
                       f"truncating {total_feats - usable} features")
        X_train = X_train[:, :usable]
        X_test  = X_test[:, :usable]
    n_classes   = len(np.unique(y_train))
    hidden      = 128

    class _WindowedCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj   = nn.Linear(feat_dim, hidden)
            self.conv   = nn.Conv1d(hidden, 64, kernel_size=3, padding=1)
            self.pool   = nn.AdaptiveMaxPool1d(1)
            self.fc     = nn.Linear(64, n_classes)
            self.relu   = nn.ReLU()
            self.drop   = nn.Dropout(0.3)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            B = x.shape[0]
            x = x.view(B, n_windows, feat_dim)         # (B, W, F)
            x = self.relu(self.proj(x))                 # (B, W, H)
            x = x.transpose(1, 2)                       # (B, H, W)
            h = self.relu(self.conv(x))                 # (B, 64, W)
            h = self.pool(h).squeeze(-1)                # (B, 64)
            h = self.drop(h)
            return self.fc(h)                           # (B, n_classes)

    dev      = torch.device(device)
    model    = _WindowedCNN().to(dev)

    # ── Validation split ────────────────────────────────────────────────
    n_tr     = len(X_train)
    val_idx  = np.random.RandomState(42).choice(n_tr, max(2, max(1, int(0.10 * n_tr))), replace=False)
    tr_mask  = np.ones(n_tr, dtype=bool)
    tr_mask[val_idx] = False
    X_tr, y_tr   = X_train[tr_mask], y_train[tr_mask]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    X_tr_t  = torch.tensor(X_tr,  dtype=torch.float32, device=dev)
    y_tr_t  = torch.tensor(y_tr,  dtype=torch.long,    device=dev)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.tensor(y_val, dtype=torch.long,    device=dev)

    loader    = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    crit      = nn.CrossEntropyLoss()

    # ── AMP setup (matches _train_torch_model pattern) ───────────────
    use_amp = (device == "cuda") and torch.cuda.is_available()
    try:
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        def _autocast():
            return torch.amp.autocast("cuda") if use_amp else _nullctx()
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        def _autocast():
            return torch.cuda.amp.autocast() if use_amp else _nullctx()

    best_val_loss = float("inf")
    best_state    = None
    patience      = 10

    for epoch in range(40):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with _autocast():
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = crit(model(xb), yb)
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * len(xb)
        epoch_train_loss = running_loss / len(X_tr)
        logger.debug(f"WindowedCNN epoch {epoch+1}: train_loss={epoch_train_loss:.4f}")
        scheduler.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(X_val_t), y_val_t).item()
        if vl < best_val_loss - 1e-6:
            best_val_loss = vl
            # Keep best_state on GPU — avoids PCIe round-trip per improvement
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience      = 10
        else:
            patience -= 1
            if patience <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=dev)
    with torch.no_grad():
        logits = model(X_te_t)
        proba  = torch.softmax(logits, dim=-1).cpu().numpy()
    return np.argmax(proba, axis=1), proba


def _fallback_mlp(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray,
                  vocab_size: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Fallback when PyTorch is unavailable: bag-of-tokens + MLP."""
    from sklearn.neural_network import MLPClassifier
    logger.info("Using sklearn MLP fallback (no PyTorch)")

    pad_token = vocab_size + 1  # matches get_pad_token()
    max_tok   = int(max(X_train.max(), X_test.max())) + 1

    def _to_hist(X: np.ndarray) -> np.ndarray:
        flat = X.reshape(len(X), -1)
        n_samples = len(flat)
        # Vectorised: mask invalid tokens, build COO indices, single bincount
        valid = (flat > 0) & (flat != pad_token)
        row_idx, col_pos = np.where(valid)
        tok_vals = flat[row_idx, col_pos]
        flat_idx = row_idx.astype(np.int64) * max_tok + tok_vals.astype(np.int64)
        counts = np.bincount(flat_idx, minlength=n_samples * max_tok)
        h = counts[:n_samples * max_tok].reshape(n_samples, max_tok).astype(np.float32)
        sums = h.sum(axis=1, keepdims=True)
        return h / (sums + 1e-10)

    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                        random_state=42, early_stopping=True)
    clf.fit(_to_hist(X_train), y_train)
    y_pred  = clf.predict(_to_hist(X_test))
    y_proba = clf.predict_proba(_to_hist(X_test))
    return y_pred, y_proba
