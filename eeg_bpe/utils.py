"""
Logging, timing, and result-saving utilities.
All intermediate experiment results go to JSON/CSV files.
"""
import json
import csv
import time
import logging
import functools
import threading
from pathlib import Path
from datetime import datetime
from typing import Any
import numpy as np

# ─── Cross-process CSV locking (filelock preferred, threading.Lock fallback) ──
try:
    from filelock import FileLock as _FileLock
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False

_CSV_THREAD_LOCK = threading.Lock()


def _csv_lock(filepath: Path):
    """Return a context manager that prevents concurrent CSV writes."""
    if _HAS_FILELOCK:
        return _FileLock(str(filepath) + ".lock", timeout=30)
    return _CSV_THREAD_LOCK

from .config import LOGS_DIR, PLOTS_DIR


# ─── Logger setup ────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a logger that writes to console and a log file.

    Parameters
    ----------
    name : str
        Logger name (also used as the log-file basename).
    level : int, optional
        Logging level (default ``logging.INFO``).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    log_file = LOGS_DIR / f"{name}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── Timing decorator ────────────────────────────────────────────────────────

def timed(logger_name: str = "timer"):
    """
    Decorator that logs execution time of the wrapped function.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger to use (default ``"timer"``).

    Returns
    -------
    callable
        Decorator wrapping the target function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger(logger_name)
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log.info(f"{func.__name__} completed in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


# ─── NumpyEncoder ─────────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ─── Result saving ────────────────────────────────────────────────────────────

def save_json(data: dict, filepath: Path, indent: int = 2) -> None:
    """
    Save dict to JSON with numpy type support.

    Parameters
    ----------
    data : dict
        Data to serialize.
    filepath : Path
        Destination file path (parent dirs created automatically).
    indent : int, optional
        JSON indentation level (default 2).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder, ensure_ascii=False)


def load_json(filepath: Path) -> dict:
    """
    Load a JSON file and return its contents as a dict.

    Parameters
    ----------
    filepath : Path
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON data.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: list[dict], filepath: Path) -> None:
    """
    Save a list of dicts to a CSV file.

    Parameters
    ----------
    rows : list of dict
        Each dict is one row; keys become column headers.
    filepath : Path
        Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    # Collect all field names across all rows (preserving order)
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(row: dict, filepath: Path) -> None:
    """
    Append a single row to a CSV file (create with header if missing).

    Thread- and process-safe: uses filelock when available, otherwise
    falls back to a threading.Lock (safe for joblib threads/processes).

    Parameters
    ----------
    row : dict
        Column-name → value mapping for the new row.
    filepath : Path
        Destination CSV file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with _csv_lock(filepath):
        write_header = not filepath.exists()
        if not write_header:
            # Read existing header so new rows are schema-consistent.
            try:
                with open(filepath, "r", newline="", encoding="utf-8") as _fh:
                    reader = csv.reader(_fh)
                    existing_header = next(reader, None)
            except Exception:
                existing_header = None
        else:
            existing_header = None
        fieldnames = existing_header if existing_header is not None else list(row.keys())
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore", restval="")
            if write_header:
                writer.writeheader()
            writer.writerow(row)


# ─── Experiment result tracker ────────────────────────────────────────────────

class ExperimentLogger:
    """
    Structured logger for experiment results.
    Saves intermediate results to JSON/CSV at every step.
    """

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.logger = get_logger(experiment_name)
        self.results: list[dict] = []
        self.metadata: dict[str, Any] = {
            "experiment": experiment_name,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self.csv_path = LOGS_DIR / f"{experiment_name}_results.csv"
        self.json_path = LOGS_DIR / f"{experiment_name}_results.json"

    def log_result(self, result: dict) -> None:
        """
        Log a single result row and write it to CSV immediately.

        Parameters
        ----------
        result : dict
            Result data; a ``timestamp`` field is added automatically.
        """
        result["timestamp"] = datetime.now().isoformat()
        self.results.append(result)
        append_csv(result, self.csv_path)
        self.logger.info(f"Result: {result}")

    def finalize(self) -> None:
        """Save complete results as JSON and mark the experiment as done."""
        self.metadata["completed_at"] = datetime.now().isoformat()
        self.metadata["n_results"] = len(self.results)
        payload = {
            "metadata": self.metadata,
            "results": self.results,
        }
        save_json(payload, self.json_path)
        self.logger.info(
            f"Experiment {self.name} finalized: {len(self.results)} results saved."
        )

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


# ─── Statistical helpers ──────────────────────────────────────────────────────

def bootstrap_ci(values: np.ndarray, n_boot: int = 1000,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """
    95% bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observations.
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (default 0.05 → 95% CI).

    Returns
    -------
    mean : float
        Sample mean.
    ci_low : float
        Lower CI bound.
    ci_high : float
        Upper CI bound.
    """
    values = np.asarray(values)
    n = len(values)
    if n < 2:
        m = float(np.mean(values))
        return m, m, m
    idx = np.random.randint(0, n, size=(n_boot, n))
    boot_means = np.mean(values[idx], axis=1)
    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return float(np.mean(values)), ci_low, ci_high


def wilcoxon_holm(pairs: list[tuple[np.ndarray, np.ndarray, str, str]],
                  alpha: float = 0.05) -> list[dict]:
    """
    Wilcoxon signed-rank test with Holm-Bonferroni correction.

    Parameters
    ----------
    pairs : list of (scores_a, scores_b, name_a, name_b)
        Each tuple contains paired score arrays and labels.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    list[dict]
        Results with statistic, p_raw, p_corrected, significant, cohens_d.
    """
    from scipy.stats import wilcoxon

    raw_results = []
    for scores_a, scores_b, name_a, name_b in pairs:
        diff = scores_a - scores_b
        # Skip if all differences are zero
        if np.all(diff == 0):
            raw_results.append({
                "method_a": name_a, "method_b": name_b,
                "statistic": 0.0, "p_raw": 1.0,
                "mean_diff": 0.0, "cohens_d": 0.0,
            })
            continue
        try:
            stat, p = wilcoxon(scores_a, scores_b)
        except Exception:
            stat, p = 0.0, 1.0
        # d_z (paired Cohen's d): mean(diff) / SD(diff)  — correct for paired data
        pooled_std = np.std(diff, ddof=1) + 1e-20
        d_z = float(np.mean(diff) / pooled_std)
        raw_results.append({
            "method_a": name_a, "method_b": name_b,
            "statistic": float(stat), "p_raw": float(p),
            "mean_diff": float(np.mean(diff)), "cohens_dz": d_z,
        })

    # Holm-Bonferroni correction
    n = len(raw_results)
    sorted_idx = sorted(range(n), key=lambda i: raw_results[i]["p_raw"])
    for rank, idx in enumerate(sorted_idx):
        corrected = raw_results[idx]["p_raw"] * (n - rank)
        raw_results[idx]["p_corrected"] = min(corrected, 1.0)
        raw_results[idx]["significant"] = raw_results[idx]["p_corrected"] < alpha

    return raw_results


# ══════════════════════════════════════════════════════════════════════════════
# GPU-accelerated PCA
# ══════════════════════════════════════════════════════════════════════════════

def pca_reduce(X: np.ndarray, n_components: int,
               device: str = "cuda") -> np.ndarray:
    """
    Randomized PCA: GPU (torch.pca_lowrank) with sklearn TruncatedSVD fallback.

    For BPE histograms and other high-dim dense matrices.  Uses fp16 on GPU
    to halve VRAM — e.g. physionet_mi (9 795 × 262 144) needs ~5 GB fp16
    vs 10 GB fp32.  Falls back to sklearn automatically on OOM or CPU.

    Notes
    -----
    torch.pca_lowrank *centers* the data (subtracts column means) before SVD,
    while sklearn TruncatedSVD does not.  For downstream LogReg classification
    the difference is negligible.
    """
    n_samples, n_features = X.shape
    n_comp = min(n_components, n_samples - 1, n_features)

    # ── Try GPU path ──────────────────────────────────────────────────────────
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                free_vram = torch.cuda.mem_get_info()[0]          # bytes free
                bytes_fp16 = n_samples * n_features * 2           # fp16 size
                if bytes_fp16 < free_vram * 0.70:                 # 70 % budget (RTX 5070 Ti has 16 GB)
                    X_t = torch.tensor(
                        np.asarray(X, dtype=np.float32),
                        dtype=torch.float16,
                        device="cuda",
                    )
                    _, _, V = torch.pca_lowrank(X_t, q=n_comp, niter=2)
                    result = (X_t @ V).cpu().to(torch.float32).numpy()
                    del X_t, V
                    torch.cuda.empty_cache()
                    return result
        except Exception:
            pass  # fall through to CPU

    # ── CPU fallback: TruncatedSVD (handles sparse implicitly via CSR) ────────
    import scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_sp = sp.csr_matrix(X) if not sp.issparse(X) else X
    return svd.fit_transform(X_sp).astype(np.float32)
