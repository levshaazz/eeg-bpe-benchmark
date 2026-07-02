"""
BPE (Byte Pair Encoding) engine for EEG token sequences.
=========================================================
Optimized implementation:
  - Pair counting with NumPy vectorization where possible
  - Batch tokenization with joblib parallelization
  - Efficient merge operations using hash-based lookups

References the standard BPE algorithm (Sennrich et al. 2016),
adapted for integer token sequences (not text).
"""
from __future__ import annotations

import numpy as np
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional
from joblib import Parallel, delayed
import json
import time

from .config import N_JOBS
from .utils import get_logger, save_json, load_json

logger = get_logger("bpe_engine")

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─── Optional Numba JIT acceleration ─────────────────────────────────────────
try:
    import numba as _numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @_numba.njit(cache=True, fastmath=False)
    def _apply_bpe_numba_core(tokens_in: np.ndarray,
                               merge_a: np.ndarray,
                               merge_b: np.ndarray,
                               merge_ids: np.ndarray,
                               n_merges: int) -> np.ndarray:
        """
        Apply BPE merges sequentially in-place (O(N × M), JIT-compiled).

        Iterates over all merge rules in priority order (rank 0, 1, …).
        For each rule performs a single greedy left-to-right scan and
        replaces every non-overlapping occurrence of (a, b) with new_id.

        Parameters
        ----------
        tokens_in : int32 array
            Input token sequence.
        merge_a, merge_b : int32 arrays of length n_merges
            Left/right token IDs for each merge rule.
        merge_ids : int32 array of length n_merges
            New token ID produced by each merge.
        n_merges : int
            Number of merge rules to apply.

        Returns
        -------
        int32 array
            Tokenised output (shorter than or equal to tokens_in).
        """
        n = len(tokens_in)
        buf = tokens_in.copy()
        active = np.ones(n, dtype=np.bool_)

        for m in range(n_merges):
            a      = merge_a[m]
            b      = merge_b[m]
            new_id = merge_ids[m]

            i = 0
            while i < n:
                if not active[i]:
                    i += 1
                    continue
                # find the next active position after i
                j = i + 1
                while j < n and not active[j]:
                    j += 1
                if j >= n:
                    break
                if buf[i] == a and buf[j] == b:
                    buf[i]    = new_id
                    active[j] = False
                    i = j + 1          # skip j (now merged into i)
                else:
                    i = j

        # collect result
        result_size = 0
        for i in range(n):
            if active[i]:
                result_size += 1
        result = np.empty(result_size, dtype=np.int32)
        k = 0
        for i in range(n):
            if active[i]:
                result[k] = buf[i]
                k += 1
        return result


@dataclass
class BPEVocab:
    """
    BPE vocabulary storing merges and token→sequence mappings.
    """
    base_vocab_size: int                       # number of base (atomic) symbols
    merges: list[tuple[int, int]] = field(default_factory=list)
    token_to_seq: dict[int, list[int]] = field(default_factory=dict)
    merge_to_token: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        # Base tokens map to themselves
        for i in range(self.base_vocab_size):
            self.token_to_seq[i] = [i]
        # Cached pair→(rank, new_id) lookup for apply_bpe (invalidated on add_merge)
        self._pair_to_merge_cache: dict[tuple[int, int], tuple[int, int]] | None = None
        # Cached token_id→decode_length for vectorized stats
        self._decode_len_cache: dict[int, int] | None = None
        # Cached NumPy merge arrays for Numba fast path (invalidated on add_merge)
        self._numba_arrays: tuple | None = None

    @property
    def vocab_size(self) -> int:
        return self.base_vocab_size + len(self.merges)

    @property
    def pair_to_merge(self) -> dict[tuple[int, int], tuple[int, int]]:
        """Lazily built and cached pair→(rank, new_id) lookup."""
        if self._pair_to_merge_cache is None:
            d: dict[tuple[int, int], tuple[int, int]] = {}
            for rank, pair in enumerate(self.merges):
                new_id = self.base_vocab_size + rank
                d[pair] = (rank, new_id)
            self._pair_to_merge_cache = d
        return self._pair_to_merge_cache

    def decode_length(self, token_id: int) -> int:
        """Return length of decoded base sequence for token_id (cached)."""
        if self._decode_len_cache is None:
            self._decode_len_cache = {}
        if token_id not in self._decode_len_cache:
            self._decode_len_cache[token_id] = len(self.decode_token(token_id))
        return self._decode_len_cache[token_id]

    def decode_token(self, token_id: int) -> list[int]:
        """Recursively decode a BPE token to base symbols."""
        return self.token_to_seq.get(token_id, [token_id])

    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        data = {
            "base_vocab_size": self.base_vocab_size,
            "merges": self.merges,
            "token_to_seq": {str(k): v for k, v in self.token_to_seq.items()},
        }
        save_json(data, path)
        logger.info(f"BPE vocab saved: {path} ({self.vocab_size} tokens)")

    @classmethod
    def load(cls, path: str) -> "BPEVocab":
        """Load vocabulary from JSON."""
        data = load_json(path)
        vocab = cls(base_vocab_size=data["base_vocab_size"])
        vocab.merges = [tuple(m) for m in data["merges"]]
        vocab.token_to_seq = {int(k): v for k, v in data["token_to_seq"].items()}
        vocab.merge_to_token = {}
        for i, (a, b) in enumerate(vocab.merges):
            new_id = vocab.base_vocab_size + i
            vocab.merge_to_token[(a, b)] = new_id
        # Reset caches (will be lazily rebuilt)
        vocab._pair_to_merge_cache = None
        vocab._decode_len_cache = None
        vocab._numba_arrays = None
        return vocab

    def add_merge(self, pair: tuple[int, int]) -> int:
        """Add a merge rule, return new token ID."""
        new_id = self.base_vocab_size + len(self.merges)
        self.merges.append(pair)
        self.merge_to_token[pair] = new_id
        # Combine sequences
        self.token_to_seq[new_id] = (self.token_to_seq[pair[0]] +
                                      self.token_to_seq[pair[1]])
        # Invalidate caches
        self._pair_to_merge_cache = None
        self._decode_len_cache = None
        self._numba_arrays = None
        return new_id


# ─── Pair counting (optimised) ───────────────────────────────────────────────

def count_pairs(sequences: list[list[int]]) -> Counter:
    """
    Count all adjacent pairs across multiple sequences.

    Hybrid: uses NumPy vectorisation for long sequences, Python loop for short.

    Parameters
    ----------
    sequences : list of list of int
        Integer token sequences.

    Returns
    -------
    Counter
        Mapping from ``(left, right)`` pair to frequency count.
    """
    counts = Counter()
    for seq in sequences:
        n = len(seq)
        if n < 2:
            continue
        if n > 500:
            # NumPy-accelerated: convert to array, build pair keys via bit-shift
            arr = np.array(seq, dtype=np.int64)
            # Pack (left, right) into a single int64 key: left << 20 | right
            # Safe for token IDs < 2^20 = 1,048,576 (covers up to V=1M)
            keys = (arr[:-1] << 20) | arr[1:]
            unique, ucounts = np.unique(keys, return_counts=True)
            for k, c in zip(unique, ucounts):
                pair = (int(k >> 20), int(k & 0xFFFFF))
                counts[pair] += int(c)
        else:
            for i in range(n - 1):
                counts[(seq[i], seq[i + 1])] += 1
    return counts


def count_pairs_parallel(sequences: list[list[int]], n_jobs: int = N_JOBS) -> Counter:
    """
    Parallel pair counting: shard sequences across workers, merge.

    Parameters
    ----------
    sequences : list of list of int
        Integer token sequences.
    n_jobs : int, optional
        Number of parallel workers (default from config).

    Returns
    -------
    Counter
        Merged pair frequency counts from all workers.
    """
    if len(sequences) < n_jobs * 2 or n_jobs <= 1:
        return count_pairs(sequences)

    chunk_size = max(1, len(sequences) // n_jobs)
    chunks = [sequences[i:i + chunk_size]
              for i in range(0, len(sequences), chunk_size)]

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(count_pairs)(chunk) for chunk in chunks
    )

    merged = Counter()
    for r in results:
        merged.update(r)
    return merged


# ─── Apply single merge to a sequence ────────────────────────────────────────

def apply_merge_to_seq(seq: list[int], pair: tuple[int, int],
                       new_id: int) -> list[int]:
    """
    Replace all occurrences of ``pair`` in ``seq`` with ``new_id``.

    Single pass O(n).

    Parameters
    ----------
    seq : list of int
        Token sequence to modify.
    pair : tuple of (int, int)
        Adjacent token pair to merge.
    new_id : int
        Replacement token ID for the merged pair.

    Returns
    -------
    list of int
        New sequence with all matching pairs replaced.
    """
    if len(seq) < 2:
        return seq

    result = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(seq[i])
            i += 1
    return result


def apply_merge_to_batch(sequences: list[list[int]], pair: tuple[int, int],
                         new_id: int, n_jobs: int = 1) -> list[list[int]]:
    """
    Apply a single merge to all sequences. Optionally parallel.

    Parameters
    ----------
    sequences : list of list of int
        Token sequences to modify.
    pair : tuple of (int, int)
        Adjacent token pair to merge.
    new_id : int
        Replacement token ID.
    n_jobs : int, optional
        Number of parallel workers (default 1).

    Returns
    -------
    list of list of int
        Sequences with the merge applied.
    """
    if n_jobs > 1 and len(sequences) > 100:
        return Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(apply_merge_to_seq)(seq, pair, new_id)
            for seq in sequences
        )
    return [apply_merge_to_seq(seq, pair, new_id) for seq in sequences]


# ─── BPE training ────────────────────────────────────────────────────────────

def train_bpe(sequences: list[list[int]],
              vocab_size: int,
              base_vocab_size: int = 256,
              verbose: bool = True,
              max_train_tokens: int | None = None) -> BPEVocab:
    """
    Train a BPE vocabulary on integer sequences.

    Uses incremental pair counting with a doubly-linked list and max-heap
    for O(M) per merge step (M = occurrences of the best pair), instead of
    the naive O(N log N) full recount each step.

    Parameters
    ----------
    sequences : list of list of int
        Integer sequences (e.g., quantized EEG channels).
    vocab_size : int
        Target vocabulary size (base + merges).
    base_vocab_size : int, optional
        Size of the atomic alphabet (default 256 for 256-bin quantization).
    verbose : bool, optional
        Whether to log training progress.
    max_train_tokens : int or None, optional
        If set and total tokens exceed this, randomly subsample sequences
        to approximately this many tokens before training.

    Returns
    -------
    BPEVocab
        Trained vocabulary with merge rules.

    Determinism
    -----------
    ``train_bpe`` is fully deterministic.  The only stochastic step is
    optional sequence subsampling (when *max_train_tokens* is set), which
    uses a fixed internal ``numpy.random.RandomState(42)`` seed — results
    are reproducible across runs without any external seed.  Pair-count
    tie-breaking in the max-heap follows Python ``dict`` insertion order
    (guaranteed stable since Python 3.7).  No external random seed is
    required or accepted.
    """
    import heapq
    from collections import defaultdict

    vocab = BPEVocab(base_vocab_size=base_vocab_size)
    n_merges = vocab_size - base_vocab_size

    if n_merges <= 0:
        return vocab

    # ── Optional subsampling ──────────────────────────────────────────────
    work_sequences = sequences
    if max_train_tokens is not None:
        total = sum(len(s) for s in sequences)
        if total > max_train_tokens:
            ratio = max_train_tokens / total
            rng = np.random.RandomState(42)
            n_keep = max(1, int(len(sequences) * ratio))
            indices = rng.choice(len(sequences), size=n_keep, replace=False)
            work_sequences = [sequences[i] for i in sorted(indices)]
            actual = sum(len(s) for s in work_sequences)
            logger.info(
                f"BPE subsampled: {len(work_sequences)} seqs, "
                f"{actual:,} tokens (cap={max_train_tokens:,})"
            )

    # ── Build flat numpy array with sentinels ─────────────────────────────
    SENTINEL = np.int32(-1)
    total_len = sum(len(s) for s in work_sequences) + len(work_sequences)
    flat = np.empty(total_len, dtype=np.int32)
    pos = 0
    for seq in work_sequences:
        n = len(seq)
        flat[pos:pos + n] = seq
        pos += n
        flat[pos] = SENTINEL
        pos += 1

    # Sentinel cumsum for same-sequence adjacency check (immutable)
    is_sentinel = flat == SENTINEL
    sent_cumsum = np.cumsum(is_sentinel, dtype=np.int32)
    active = ~is_sentinel  # sentinels are never active
    total_tokens_initial = int(np.sum(active))
    total_active = total_tokens_initial

    # ── Build doubly-linked list of active positions ──────────────────────
    active_pos = np.where(active)[0]
    next_arr = np.full(total_len, -1, dtype=np.int32)
    prev_arr = np.full(total_len, -1, dtype=np.int32)
    if len(active_pos) > 1:
        next_arr[active_pos[:-1]] = active_pos[1:]
        prev_arr[active_pos[1:]] = active_pos[:-1]

    # ── Build initial pair counts and position index (vectorised) ─────────
    same_seq = sent_cumsum[active_pos[:-1]] == sent_cumsum[active_pos[1:]]
    lp = active_pos[:-1][same_seq]
    rp = active_pos[1:][same_seq]
    lv = flat[lp]
    rv = flat[rp]

    keys = lv.astype(np.int64) << 20 | rv.astype(np.int64)
    sorted_idx = np.argsort(keys, kind='mergesort')
    sorted_keys = keys[sorted_idx]
    sorted_lp = lp[sorted_idx]

    # Group boundaries for unique pairs
    diff_mask = np.empty(len(sorted_keys), dtype=bool)
    diff_mask[0] = True
    if len(sorted_keys) > 1:
        diff_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.where(diff_mask)[0]
    group_ends = np.append(group_starts[1:], len(sorted_keys))
    group_keys = sorted_keys[group_starts]

    pair_counts: dict[tuple[int, int], int] = {}
    pair_positions: dict[tuple[int, int], set[int]] = defaultdict(set)

    for i in range(len(group_starts)):
        k = int(group_keys[i])
        pair = (k >> 20, k & 0xFFFFF)
        count = int(group_ends[i] - group_starts[i])
        pair_counts[pair] = count
        s = int(group_starts[i])
        e = int(group_ends[i])
        pair_positions[pair] = set(sorted_lp[s:e].tolist())

    # ── Build max-heap (lazy deletion via negative counts) ────────────────
    heap: list[tuple[int, tuple[int, int]]] = [
        (-c, p) for p, c in pair_counts.items()
    ]
    heapq.heapify(heap)

    t0 = time.perf_counter()
    log_interval = max(1, n_merges // 20)
    training_log = []  # L4: track merge progress

    _merge_bar = (
        _tqdm(range(n_merges), desc=f"BPE V={vocab_size}", unit="merge",
              dynamic_ncols=True, leave=True)
        if _HAS_TQDM else range(n_merges)
    )
    for step in _merge_bar:
        # ── Find best pair via heap (skip stale entries) ──────────────────
        best_pair = None
        while heap:
            neg_c, candidate = heapq.heappop(heap)
            actual_count = pair_counts.get(candidate, 0)
            if actual_count > 0 and actual_count == -neg_c:
                best_pair = candidate
                best_count = actual_count
                break

        if best_pair is None:
            logger.warning(
                f"BPE stopped early at step {step}: no valid pairs"
            )
            break

        a, b = best_pair
        new_id = vocab.add_merge(best_pair)

        # ── Get sorted positions for greedy left-to-right overlap ─────────
        positions = sorted(pair_positions.get(best_pair, []))

        # Remove best_pair from tracking (cannot reappear since new_id
        # is a fresh token distinct from a and b)
        pair_counts.pop(best_pair, None)
        pair_positions.pop(best_pair, None)

        # ── Process each merge position ──────────────────────────────────
        last_right = -1
        for lpos in positions:
            # Validate: tokens unchanged and no overlap
            if int(flat[lpos]) != a:
                continue
            rpos = int(next_arr[lpos])
            if rpos == -1 or int(flat[rpos]) != b:
                continue
            if lpos <= last_right:
                continue
            last_right = rpos

            left_nb = int(prev_arr[lpos])
            right_nb = int(next_arr[rpos])

            # ── Remove old neighbouring pairs ─────────────────────────────
            if left_nb != -1 and sent_cumsum[left_nb] == sent_cumsum[lpos]:
                old_pair = (int(flat[left_nb]), a)
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= 1
                    ps = pair_positions.get(old_pair)
                    if ps is not None:
                        ps.discard(left_nb)
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_positions.pop(old_pair, None)

            if right_nb != -1 and sent_cumsum[rpos] == sent_cumsum[right_nb]:
                old_pair = (b, int(flat[right_nb]))
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= 1
                    ps = pair_positions.get(old_pair)
                    if ps is not None:
                        ps.discard(rpos)
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_positions.pop(old_pair, None)

            # ── Apply merge ──────────────────────────────────────────────
            flat[lpos] = np.int32(new_id)
            active[rpos] = False
            total_active -= 1

            # Update linked list: remove rpos
            next_arr[lpos] = np.int32(right_nb)
            if right_nb != -1:
                prev_arr[right_nb] = np.int32(lpos)

            # ── Add new neighbouring pairs ────────────────────────────────
            if left_nb != -1 and sent_cumsum[left_nb] == sent_cumsum[lpos]:
                np_ = (int(flat[left_nb]), new_id)
                pair_counts[np_] = pair_counts.get(np_, 0) + 1
                pair_positions[np_].add(left_nb)
                heapq.heappush(heap, (-pair_counts[np_], np_))

            if right_nb != -1 and sent_cumsum[lpos] == sent_cumsum[right_nb]:
                np_ = (new_id, int(flat[right_nb]))
                pair_counts[np_] = pair_counts.get(np_, 0) + 1
                pair_positions[np_].add(lpos)
                heapq.heappush(heap, (-pair_counts[np_], np_))

        # ── Logging ──────────────────────────────────────────────────────
        if step % log_interval == 0 or step == n_merges - 1:
            compression = total_tokens_initial / max(total_active, 1)
            elapsed = time.perf_counter() - t0
            training_log.append({
                "step": step + 1,
                "pair": list(best_pair),
                "new_id": new_id,
                "pair_count": best_count,
                "total_tokens": total_active,
                "compression": round(compression, 4),
                "elapsed_s": round(elapsed, 2),
            })
            if _HAS_TQDM and hasattr(_merge_bar, "set_postfix"):
                _merge_bar.set_postfix(
                    comp=f"{compression:.2f}x",
                    cnt=best_count,
                    refresh=False,
                )
            if verbose:
                logger.info(
                    f"BPE merge {step + 1}/{n_merges}: "
                    f"pair={best_pair} → {new_id} (count={best_count}), "
                    f"compression={compression:.2f}x, {elapsed:.1f}s"
                )

    elapsed = time.perf_counter() - t0

    # L4: Save training progress log
    if training_log:
        from .config import LOGS_DIR
        progress_path = LOGS_DIR / f"bpe_training_progress_V{vocab_size}.json"
        save_json(training_log, progress_path)
    logger.info(
        f"BPE training complete: {len(vocab.merges)} merges, "
        f"vocab={vocab.vocab_size}, "
        f"compression={total_tokens_initial / max(total_active, 1):.2f}x, "
        f"time={elapsed:.1f}s"
    )
    return vocab


def train_bpe_discriminative(
    sequences: list[list[int]],
    vocab_size: int,
    base_vocab_size: int = 256,
    y_labels: list[int] | None = None,
    n_classes: int = 2,
    alpha: float = 0.5,
    top_k: int = 10,
    verbose: bool = True,
    max_train_tokens: int | None = None,
) -> BPEVocab:
    """
    Discriminative BPE: blend frequency-based and Fisher-ratio-based merge selection.

    At each merge step, the top *top_k* candidates from the frequency heap
    are evaluated and scored by a composite score:

        score = count^(1-alpha) * class_std^alpha

    where ``class_std`` is the standard deviation of per-class pair frequencies
    (normalized by class size).  ``alpha=0`` recovers standard BPE;
    ``alpha=1`` is purely discriminative.

    Parameters
    ----------
    sequences : list of list of int
        Integer sequences — one list per trial.
    vocab_size : int
        Target vocabulary size.
    base_vocab_size : int
        Atomic alphabet size (default 256).
    y_labels : list of int or None
        Trial-level class labels, parallel to *sequences*.
        If None, falls back to standard BPE (ignores alpha).
    n_classes : int
        Number of classes.
    alpha : float
        Blend factor in [0, 1].  0 = standard BPE, 1 = pure discriminative.
    top_k : int
        Number of candidates to evaluate per merge step (default 10).
    verbose : bool
        Whether to log progress.
    max_train_tokens : int or None
        Optional sequence subsampling cap.

    Returns
    -------
    BPEVocab
        Trained vocabulary with merge rules.
    """
    if y_labels is None or alpha == 0.0:
        return train_bpe(sequences, vocab_size, base_vocab_size,
                         verbose=verbose, max_train_tokens=max_train_tokens)

    import heapq
    from collections import defaultdict

    vocab = BPEVocab(base_vocab_size=base_vocab_size)
    n_merges = vocab_size - base_vocab_size
    if n_merges <= 0:
        return vocab

    # ── Optional subsampling ──────────────────────────────────────────────
    work_sequences = sequences
    work_labels = list(y_labels)
    if max_train_tokens is not None:
        total = sum(len(s) for s in sequences)
        if total > max_train_tokens:
            ratio = max_train_tokens / total
            rng = np.random.RandomState(42)
            n_keep = max(1, int(len(sequences) * ratio))
            indices = sorted(rng.choice(len(sequences), size=n_keep, replace=False))
            work_sequences = [sequences[i] for i in indices]
            work_labels = [y_labels[i] for i in indices]

    # ── Build class counts ────────────────────────────────────────────────
    class_counts = np.zeros(n_classes, dtype=np.float64)
    for lbl in work_labels:
        class_counts[int(lbl)] += 1

    # ── Build flat array with sentinels + sent_cumsum ─────────────────────
    SENTINEL = np.int32(-1)
    total_len = sum(len(s) for s in work_sequences) + len(work_sequences)
    flat = np.empty(total_len, dtype=np.int32)
    pos = 0
    for seq in work_sequences:
        n = len(seq)
        flat[pos:pos + n] = seq
        pos += n
        flat[pos] = SENTINEL
        pos += 1

    is_sentinel = flat == SENTINEL
    sent_cumsum = np.cumsum(is_sentinel, dtype=np.int32)
    active = ~is_sentinel
    total_tokens_initial = int(np.sum(active))
    total_active = total_tokens_initial

    # seq_label[i] = label of the i-th sequence; sent_cumsum[lpos] gives seq idx
    seq_label_arr = np.array(work_labels, dtype=np.int32)

    # ── Build doubly-linked list ──────────────────────────────────────────
    active_pos = np.where(active)[0]
    next_arr = np.full(total_len, -1, dtype=np.int32)
    prev_arr = np.full(total_len, -1, dtype=np.int32)
    if len(active_pos) > 1:
        next_arr[active_pos[:-1]] = active_pos[1:]
        prev_arr[active_pos[1:]] = active_pos[:-1]

    # ── Build initial pair counts and position index ──────────────────────
    same_seq = sent_cumsum[active_pos[:-1]] == sent_cumsum[active_pos[1:]]
    lp = active_pos[:-1][same_seq]
    rp = active_pos[1:][same_seq]
    lv = flat[lp]
    rv = flat[rp]

    keys = lv.astype(np.int64) << 20 | rv.astype(np.int64)
    sorted_idx = np.argsort(keys, kind='mergesort')
    sorted_keys = keys[sorted_idx]
    sorted_lp = lp[sorted_idx]

    diff_mask = np.empty(len(sorted_keys), dtype=bool)
    diff_mask[0] = True
    if len(sorted_keys) > 1:
        diff_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    group_starts = np.where(diff_mask)[0]
    group_ends = np.append(group_starts[1:], len(sorted_keys))
    group_keys = sorted_keys[group_starts]

    pair_counts: dict[tuple[int, int], int] = {}
    pair_positions: dict[tuple[int, int], set[int]] = defaultdict(set)

    for i in range(len(group_starts)):
        k = int(group_keys[i])
        pair = (k >> 20, k & 0xFFFFF)
        count = int(group_ends[i] - group_starts[i])
        pair_counts[pair] = count
        s = int(group_starts[i])
        e = int(group_ends[i])
        pair_positions[pair] = set(sorted_lp[s:e].tolist())

    heap: list[tuple[int, tuple[int, int]]] = [
        (-c, p) for p, c in pair_counts.items()
    ]
    heapq.heapify(heap)

    def _disc_std(pair: tuple[int, int]) -> float:
        """Std of per-class pair frequency (normalized by class size)."""
        positions = pair_positions.get(pair)
        if not positions:
            return 0.0
        per_class = np.zeros(n_classes, dtype=np.float64)
        for lpos in positions:
            c = int(seq_label_arr[int(sent_cumsum[lpos])])
            per_class[c] += 1
        # normalize by class size to avoid bias
        safe_cc = np.where(class_counts > 0, class_counts, 1.0)
        per_class /= safe_cc
        return float(np.std(per_class))

    def _composite_score(pair: tuple[int, int], count: int) -> float:
        if alpha == 0.0:
            return float(count)
        disc = _disc_std(pair)
        return (float(count) ** (1.0 - alpha)) * (max(disc, 1e-12) ** alpha)

    t0 = time.perf_counter()
    log_interval = max(1, n_merges // 20)

    _merge_bar = (
        _tqdm(range(n_merges), desc=f"DiscBPE V={vocab_size}", unit="merge",
              dynamic_ncols=True, leave=True)
        if _HAS_TQDM else range(n_merges)
    )
    for step in _merge_bar:
        # ── Find best pair: pop top-K, score, select best ──────────────────
        candidates: list[tuple[float, int, tuple[int, int]]] = []
        pushed_back: list[tuple[int, tuple[int, int]]] = []

        while heap and len(candidates) < top_k:
            neg_c, candidate = heapq.heappop(heap)
            actual_count = pair_counts.get(candidate, 0)
            if actual_count > 0 and actual_count == -neg_c:
                score = _composite_score(candidate, actual_count)
                candidates.append((score, actual_count, candidate))
            # stale entries are simply discarded

        if not candidates:
            logger.warning(f"DiscBPE stopped early at step {step}: no valid pairs")
            break

        # Select the candidate with the highest composite score
        best_score, best_count, best_pair = max(candidates, key=lambda x: x[0])

        # Push non-winners back into heap with current counts
        for score, cnt, pair in candidates:
            if pair is not best_pair:
                heapq.heappush(heap, (-cnt, pair))

        a, b = best_pair
        new_id = vocab.add_merge(best_pair)

        positions = sorted(pair_positions.get(best_pair, []))
        pair_counts.pop(best_pair, None)
        pair_positions.pop(best_pair, None)

        last_right = -1
        for lpos in positions:
            if int(flat[lpos]) != a:
                continue
            rpos = int(next_arr[lpos])
            if rpos == -1 or int(flat[rpos]) != b:
                continue
            if lpos <= last_right:
                continue
            last_right = rpos

            left_nb = int(prev_arr[lpos])
            right_nb = int(next_arr[rpos])

            if left_nb != -1 and sent_cumsum[left_nb] == sent_cumsum[lpos]:
                old_pair = (int(flat[left_nb]), a)
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= 1
                    ps = pair_positions.get(old_pair)
                    if ps is not None:
                        ps.discard(left_nb)
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_positions.pop(old_pair, None)

            if right_nb != -1 and sent_cumsum[rpos] == sent_cumsum[right_nb]:
                old_pair = (b, int(flat[right_nb]))
                if old_pair in pair_counts:
                    pair_counts[old_pair] -= 1
                    ps = pair_positions.get(old_pair)
                    if ps is not None:
                        ps.discard(rpos)
                    if pair_counts[old_pair] <= 0:
                        pair_counts.pop(old_pair, None)
                        pair_positions.pop(old_pair, None)

            flat[lpos] = np.int32(new_id)
            active[rpos] = False
            total_active -= 1

            next_arr[lpos] = np.int32(right_nb)
            if right_nb != -1:
                prev_arr[right_nb] = np.int32(lpos)

            if left_nb != -1 and sent_cumsum[left_nb] == sent_cumsum[lpos]:
                np_ = (int(flat[left_nb]), new_id)
                pair_counts[np_] = pair_counts.get(np_, 0) + 1
                pair_positions[np_].add(left_nb)
                heapq.heappush(heap, (-pair_counts[np_], np_))

            if right_nb != -1 and sent_cumsum[lpos] == sent_cumsum[right_nb]:
                np_ = (new_id, int(flat[right_nb]))
                pair_counts[np_] = pair_counts.get(np_, 0) + 1
                pair_positions[np_].add(lpos)
                heapq.heappush(heap, (-pair_counts[np_], np_))

        if verbose and (step % log_interval == 0 or step == n_merges - 1):
            compression = total_tokens_initial / max(total_active, 1)
            elapsed = time.perf_counter() - t0
            logger.info(
                f"DiscBPE merge {step + 1}/{n_merges}: "
                f"pair={best_pair} → {new_id} (count={best_count}, "
                f"score={best_score:.4f}), compression={compression:.2f}x"
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"DiscBPE training complete: {len(vocab.merges)} merges, "
        f"vocab={vocab.vocab_size}, "
        f"compression={total_tokens_initial / max(total_active, 1):.2f}x, "
        f"alpha={alpha}, time={elapsed:.1f}s"
    )
    return vocab


# ─── Apply BPE (tokenize) ────────────────────────────────────────────────────

def _get_numba_arrays(vocab: BPEVocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (merge_a, merge_b, merge_ids) NumPy arrays, lazily built and cached."""
    if vocab._numba_arrays is None:
        n = len(vocab.merges)
        merge_a   = np.empty(n, dtype=np.int32)
        merge_b   = np.empty(n, dtype=np.int32)
        merge_ids = np.empty(n, dtype=np.int32)
        for i, (a, b) in enumerate(vocab.merges):
            merge_a[i]   = a
            merge_b[i]   = b
            merge_ids[i] = vocab.base_vocab_size + i
        vocab._numba_arrays = (merge_a, merge_b, merge_ids)
    return vocab._numba_arrays


def apply_bpe(sequence: list[int], vocab: BPEVocab) -> list[int]:
    """
    Apply trained BPE merges to a single sequence.

    Uses a priority-queue (heap) and doubly-linked list for O(N log N)
    complexity instead of O(N × M) where M is the number of merges.

    Algorithm:
      1. Build pair → (rank, new_id) lookup from vocab.merges.
      2. Represent the sequence as a doubly-linked list (arrays).
      3. Seed a min-heap with (rank, position) for every adjacent pair
         that has a merge rule.
      4. Pop the lowest-rank (highest-priority) entry.  If the entry
         is still valid (both positions active, pair still matches),
         apply the merge and enqueue any new pairs created with
         neighbours.
      5. Repeat until the heap is empty.

    Parameters
    ----------
    sequence : list of int
        Input token sequence (e.g., quantized EEG channel).
    vocab : BPEVocab
        Trained BPE vocabulary.

    Returns
    -------
    list of int
        BPE-tokenized sequence.
    """
    import heapq

    n = len(sequence)
    if n < 2 or not vocab.merges:
        return list(sequence)

    # NOTE: A Numba O(N×M) path (_apply_bpe_numba_core / _get_numba_arrays) is
    # defined above and available when _HAS_NUMBA=True.  Benchmarking on EEG
    # data (compression ≈1.1–1.3×) shows it is 4× *slower* than the heap
    # approach: the heap fires only ~180 merges per sequence while the Numba
    # scan always iterates M×N positions regardless of compression.  The Numba
    # code is retained for completeness; for high-compression corpora (ratio
    # >3×) it may become the faster choice.

    # ── use cached pair → (rank, new_id) lookup ────────────────────
    pair_to_merge = vocab.pair_to_merge

    # ── doubly-linked list on NumPy int32 arrays ────────────────────
    # Using numpy arrays instead of Python lists eliminates per-element
    # boxing overhead (~16.8M saved across a full run).
    tokens = np.array(sequence, dtype=np.int32)
    nxt = np.arange(1, n + 1, dtype=np.int32)     # nxt[i] → next active pos (n = end)
    prv = np.arange(-1, n - 1, dtype=np.int32)    # prv[i] → prev active pos (-1 = start)

    # ── seed heap with every matchable adjacent pair ────────────────
    heap: list[tuple[int, int]] = []   # (rank, left_position)
    for i in range(n - 1):
        pair = (int(tokens[i]), int(tokens[i + 1]))
        if pair in pair_to_merge:
            heapq.heappush(heap, (pair_to_merge[pair][0], i))

    # ── main loop ───────────────────────────────────────────────────
    while heap:
        rank, left = heapq.heappop(heap)

        # Validate: left must still be alive and have a right neighbour
        right = int(nxt[left])
        if right >= n:
            continue

        # The pair at (left, right) must still match the expected merge
        tl = int(tokens[left])
        tr = int(tokens[right])
        pair = (tl, tr)
        m = pair_to_merge.get(pair)
        if m is None or m[0] != rank:
            continue

        new_id = m[1]

        # Apply merge: update left token, remove right from list
        tokens[left] = new_id

        # Re-link: left.next = right.next;  right.next.prev = left
        right_next = int(nxt[right])
        nxt[left] = right_next
        if right_next < n:
            prv[right_next] = left

        # Enqueue new left-neighbour pair: (tokens[prev(left)], new_id)
        p = int(prv[left])
        if p >= 0:
            lp = (int(tokens[p]), new_id)
            lm = pair_to_merge.get(lp)
            if lm is not None:
                heapq.heappush(heap, (lm[0], p))

        # Enqueue new right-neighbour pair: (new_id, tokens[next(left)])
        rn = int(nxt[left])
        if rn < n:
            rp = (new_id, int(tokens[rn]))
            rm = pair_to_merge.get(rp)
            if rm is not None:
                heapq.heappush(heap, (rm[0], left))

    # ── collect result via linked-list traversal ────────────────────
    result: list[int] = []
    pos = 0
    while pos < n:
        result.append(int(tokens[pos]))
        pos = int(nxt[pos])
    return result


def _apply_bpe_batch_concat(sequences: list[list[int]],
                             vocab: BPEVocab) -> list[list[int]]:
    """Apply BPE to all sequences at once using concatenated array + sentinels.

    Faster than per-sequence apply_bpe when there are many short sequences
    (common in EEG: n_trials × n_ch sequences of ~250-1000 tokens).
    Uses the same linked-list + heap approach as training but applies
    pre-existing merge rules instead of discovering them.
    """
    import heapq

    if not vocab.merges:
        return [list(s) for s in sequences]

    # ── Concatenate with sentinels ──────────────────────────────────
    SENTINEL = np.int32(-1)
    lengths = [len(s) for s in sequences]
    total_len = sum(lengths) + len(sequences)
    flat = np.empty(total_len, dtype=np.int32)
    seq_starts: list[int] = []  # start position of each sequence in flat
    pos = 0
    for i, seq in enumerate(sequences):
        n = len(seq)
        seq_starts.append(pos)
        if n > 0:
            flat[pos:pos + n] = seq
        pos += n
        flat[pos] = SENTINEL
        pos += 1

    # ── Build linked list ───────────────────────────────────────────
    is_sentinel = flat == SENTINEL
    sent_cumsum = np.cumsum(is_sentinel, dtype=np.int32)
    active_mask = ~is_sentinel
    active_pos = np.where(active_mask)[0]

    if len(active_pos) < 2:
        return [list(s) for s in sequences]

    next_arr = np.full(total_len, -1, dtype=np.int32)
    prev_arr = np.full(total_len, -1, dtype=np.int32)
    next_arr[active_pos[:-1]] = active_pos[1:]
    prev_arr[active_pos[1:]] = active_pos[:-1]

    # Break links across sentinel boundaries
    same_seq = sent_cumsum[active_pos[:-1]] == sent_cumsum[active_pos[1:]]
    cross_boundary = ~same_seq
    if cross_boundary.any():
        cb_left = active_pos[:-1][cross_boundary]
        cb_right = active_pos[1:][cross_boundary]
        next_arr[cb_left] = -1
        prev_arr[cb_right] = -1

    # ── Build pair→(rank, new_id) lookup ────────────────────────────
    pair_to_merge = vocab.pair_to_merge

    # ── Seed heap with all matchable adjacent pairs ─────────────────
    valid_left = active_pos[:-1][same_seq]
    valid_right = active_pos[1:][same_seq]
    heap: list[tuple[int, int]] = []
    for k in range(len(valid_left)):
        lp = int(valid_left[k])
        rp = int(valid_right[k])
        pair = (int(flat[lp]), int(flat[rp]))
        m = pair_to_merge.get(pair)
        if m is not None:
            heapq.heappush(heap, (m[0], lp))

    # ── Main merge loop ─────────────────────────────────────────────
    while heap:
        rank, left = heapq.heappop(heap)
        # Skip if this position was deactivated (right side of a prior merge)
        if not active_mask[left]:
            continue
        right = int(next_arr[left])
        if right == -1:
            continue

        tl = int(flat[left])
        tr = int(flat[right])
        pair = (tl, tr)
        m = pair_to_merge.get(pair)
        if m is None or m[0] != rank:
            continue

        new_id = m[1]
        flat[left] = new_id

        # Remove right from linked list
        right_next = int(next_arr[right])
        next_arr[left] = right_next
        if right_next != -1:
            prev_arr[right_next] = left
        active_mask[right] = False

        # Enqueue left neighbour pair
        p = int(prev_arr[left])
        if p != -1:
            lp = (int(flat[p]), new_id)
            lm = pair_to_merge.get(lp)
            if lm is not None:
                heapq.heappush(heap, (lm[0], p))

        # Enqueue right neighbour pair
        rn = int(next_arr[left])
        if rn != -1:
            rp = (new_id, int(flat[rn]))
            rm = pair_to_merge.get(rp)
            if rm is not None:
                heapq.heappush(heap, (rm[0], left))

    # ── Split result back into sequences ────────────────────────────
    # Use linked-list traversal from first active position in each segment.
    # If start was merged away, find first active via prev_arr/next_arr chain
    # or fall back to linear scan.
    results: list[list[int]] = []
    for i in range(len(sequences)):
        if lengths[i] == 0:
            results.append([])
            continue
        start = seq_starts[i]
        end = start + lengths[i]  # exclusive (doesn't include sentinel)

        # Find first active position in this segment
        first_active = -1
        for j in range(start, end):
            if active_mask[j]:
                first_active = j
                break

        if first_active == -1:
            results.append([])
            continue

        # Traverse linked list from first_active
        result = []
        pos = first_active
        while pos != -1 and pos < end:
            result.append(int(flat[pos]))
            pos = int(next_arr[pos])

        results.append(result)

    return results


def apply_bpe_batch(sequences: list[list[int]], vocab: BPEVocab,
                    n_jobs: int = N_JOBS) -> list[list[int]]:
    """
    Apply BPE to multiple sequences in parallel.

    Uses concatenated single-pass for many short sequences (>50),
    or per-sequence parallelism via joblib for fewer long sequences.

    Parameters
    ----------
    sequences : list of list of int
        Input token sequences.
    vocab : BPEVocab
        Trained BPE vocabulary.
    n_jobs : int, optional
        Number of parallel workers (default from config).

    Returns
    -------
    list of list of int
        BPE-tokenized sequences.
    """
    if not sequences:
        return []
    # Benchmarking on EEG data (compression ~1.1-1.3×, 500 seqs × 1000 tokens):
    #   Sequential:   255 ms  ← fastest for typical EEG
    #   Joblib(8):   1903 ms  ← pickle overhead dominates
    #   Concat batch: 384 ms  ← larger single heap is slower for low compression
    # Concat batch wins only when many sequences (>2000) and higher compression (>2×).
    avg_len = sum(len(s) for s in sequences) / max(len(sequences), 1)
    if len(sequences) > 2000 and avg_len < 2000:
        return _apply_bpe_batch_concat(sequences, vocab)
    if n_jobs > 1 and len(sequences) > 500 and avg_len > 2000:
        return Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(apply_bpe)(seq, vocab) for seq in sequences
        )
    return [apply_bpe(seq, vocab) for seq in sequences]


# ─── Token statistics ─────────────────────────────────────────────────────────

def compute_token_stats(bpe_sequences: list[list[int]],
                        vocab: BPEVocab) -> dict:
    """
    Compute statistics over BPE-tokenized sequences.

    Parameters
    ----------
    bpe_sequences : list of list of int
        BPE-tokenized sequences.
    vocab : BPEVocab
        BPE vocabulary used for tokenization.

    Returns
    -------
    dict
        Frequency distribution, length distribution, compression stats
        including keys: ``total_bpe_tokens``, ``compression_ratio``,
        ``n_unique``, ``vocab_utilization``, ``mean_token_length``,
        ``top_20_tokens``.
    """
    # Filter out empty sequences
    non_empty = [seq for seq in bpe_sequences if len(seq) > 0]
    if not non_empty:
        return {
            "n_sequences": len(bpe_sequences),
            "n_tokens_total": 0,
            "n_unique_tokens": 0,
            "vocab_utilization": 0.0,
            "compression_ratio": 0.0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "max_length": 0,
            "min_length": 0,
            "token_frequencies": {},
            "entropy": 0.0,
        }

    all_tokens = np.concatenate([np.asarray(seq, dtype=np.int32) for seq in non_empty])
    total_bpe = len(all_tokens)

    unique_ids, unique_freqs = np.unique(all_tokens, return_counts=True)
    n_unique = len(unique_ids)
    decode_lens = np.array([vocab.decode_length(int(t)) for t in unique_ids],
                           dtype=np.int64)

    total_base_tokens = int(np.sum(unique_freqs * decode_lens))

    # Token length distribution (in base symbols) — already computed
    lengths = decode_lens
    freqs_arr = unique_freqs

    # Weighted mean length
    weighted_mean_len = np.average(lengths, weights=freqs_arr)

    return {
        "total_bpe_tokens": total_bpe,
        "total_base_tokens": total_base_tokens,
        "compression_ratio": total_base_tokens / max(total_bpe, 1),
        "n_unique": n_unique,
        "vocab_utilization": n_unique / vocab.vocab_size,
        "mean_token_length": float(weighted_mean_len),
        "max_token_length": int(np.max(lengths)),
        "top_20_tokens": [(int(unique_ids[i]), int(unique_freqs[i]))
                          for i in np.argsort(unique_freqs)[::-1][:20]],
    }
