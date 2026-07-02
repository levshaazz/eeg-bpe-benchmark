"""
Global configuration for EEG-BPE Paper 1 experiments.
All paths, hyperparameters, and constants in one place.
"""
from pathlib import Path
import multiprocessing as mp
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent          # project root
SRC_DIR = ROOT_DIR / "eeg_bpe"
RAW_DATA_DIR = ROOT_DIR / "datasets" / "raw"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
CACHE_DIR = RESULTS_DIR / "cache"

# Create all output dirs
for d in (RESULTS_DIR, LOGS_DIR, PLOTS_DIR, MODELS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ─── Dataset paths ───────────────────────────────────────────────────────────
DATASET_PATHS = {
    "bci_iv_2a":         RAW_DATA_DIR / "moabb_cache",
    "physionet_mi":      RAW_DATA_DIR / "physionet_mi" / "MNE-eegbci-data",
    "sleep_edf":         RAW_DATA_DIR / "sleep_edf" / "physionet-sleep-data",
    "mental_arithmetic": RAW_DATA_DIR / "mental_arithmetic",
    "epfl_p300":         RAW_DATA_DIR / "moabb_cache",
    "ssvep_nakanishi":   RAW_DATA_DIR / "ssvep_nakanishi",
}

# ─── Dataset metadata ────────────────────────────────────────────────────────
DATASET_INFO = {
    "bci_iv_2a": {
        "paradigm": "Motor Imagery",
        "n_classes": 4,
        "n_channels": 22,
        "sfreq": 250,
        "n_subjects": 9,
        "task": "MI-4class",
        "cv_strategy": "LOSO",
    },
    "physionet_mi": {
        "paradigm": "Motor Imagery",
        "n_classes": 4,
        "n_channels": 64,
        "sfreq": 160,
        "n_subjects": 109,
        "task": "MI-4class",
        # 109 subjects → subject-stratified 5-fold (LOSO would be 109 folds = too slow)
        "cv_strategy": "5fold-subject",
    },
    "sleep_edf": {
        "paradigm": "Sleep",
        "n_classes": 5,
        "n_channels": 2,
        "sfreq": 100,
        "n_subjects": 78,
        "task": "sleep-staging-5class",
        "cv_strategy": "5fold-subject",
    },
    "mental_arithmetic": {
        "paradigm": "Cognitive",
        "n_classes": 2,
        "n_channels": 19,
        "sfreq": 500,
        "n_subjects": 36,
        "task": "cognitive-load-2class",
        "cv_strategy": "5fold-subject",
    },
    "epfl_p300": {
        "paradigm": "P300",
        "n_classes": 2,
        "n_channels": 16,   # BNCI2014_009 (Hoffmann) — 16 ch after MOABB selection
        "sfreq": 2048,
        "n_subjects": 10,   # 10 subjects (not 8); loader uses subjects 1-10
        "task": "P300-detection",
        "cv_strategy": "LOSO",
    },
    "ssvep_nakanishi": {
        "paradigm": "SSVEP",
        "n_classes": 12,
        "n_channels": 8,
        "sfreq": 256,
        "n_subjects": 10,
        "task": "SSVEP-12class",
        "cv_strategy": "LOSO",
    },
}

# ─── Preprocessing ───────────────────────────────────────────────────────────
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 45.0 # Hz
NOTCH_FREQS = [50.0, 60.0]  # Hz (both power-line freqs)

# ─── Quantization ────────────────────────────────────────────────────────────
QUANT_METHODS = ["uniform", "mu_law", "adaptive"]
QUANT_BINS = [32, 64, 128, 256, 512, 1024]
MU_LAW_MU = 255  # μ-law compression parameter
# Ablation A1 result: adaptive gives +19pp on Sleep-EDF, negligible change on MI.
# Using adaptive as the global default; vocab filenames include method suffix so
# caches remain valid when method changes (bpe_vocab_V{V}_B{n_bins}_{method}.json).
DEFAULT_QUANT_METHOD = "adaptive"

# ─── BPE ─────────────────────────────────────────────────────────────────────
BPE_VOCAB_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BPE_BALANCED_HOURS = 10  # hours per dataset for balanced BPE training

# ─── Frequency-aware ─────────────────────────────────────────────────────────
TARGET_SFREQ = 128  # Hz (for Approach A: resample to common frequency)

# ─── Downstream Classifiers ──────────────────────────────────────────────────
RANDOM_SEEDS = [42, 123, 456, 789, 2024]  # 5 seeds for reproducible results
N_BOOTSTRAP = 1000

# ─── Logistic Regression Hyperparameters ─────────────────────────────────────
# Single source of truth — imported by exp2, ablations, exp6, baselines.
LOGREG_C            = 1.0          # inverse regularisation strength
LOGREG_MAX_ITER     = 500          # convergence cap (saga typically < 200)
LOGREG_SOLVER       = "saga"       # supports multiclass + sparse/dense features
LOGREG_CLASS_WEIGHT = "balanced"   # up-weights minority classes (critical for sleep_edf)
LOGREG_TOL          = 1e-3         # convergence tolerance (sklearn default 1e-4; 1e-3 is
                                   # sufficient for comparing relative ablation differences)

# ─── Primary metric per dataset ───────────────────────────────────────────────
# Cohen's κ for 5-class imbalanced sleep staging (accuracy is misleading there).
# Balanced accuracy for all other datasets.
PRIMARY_METRIC_PER_DATASET: dict = {"sleep_edf": "kappa"}
PRIMARY_METRIC_DEFAULT: str = "balanced_accuracy"

# ─── Vocab-size sweep (Exp 2 sensitivity analysis) ────────────────────────────
VOCAB_SWEEP_SIZES: list = [1024, 2048, 4096]

# ─── Compute ─────────────────────────────────────────────────────────────────
N_JOBS = mp.cpu_count()                  # use all cores (GPU handles model training separately)
N_IO_JOBS = min(mp.cpu_count(), 16)      # I/O-bound: all cores + thread overhead
BATCH_SIZE = 512                         # mini-batch size for PyTorch training


def _detect_device() -> str:
    """Auto-detect best available compute device.

    Priority: CUDA GPU → Apple Silicon (MPS) → CPU.
    Can be overridden via the ``BPE_EEG_DEVICE`` environment variable
    (accepted values: ``cuda``, ``mps``, ``cpu``).

    Returns
    -------
    str
        ``'cuda'``, ``'mps'``, or ``'cpu'``.
    """
    env = os.environ.get("BPE_EEG_DEVICE", "").strip().lower()
    if env in ("cuda", "mps", "cpu"):
        return env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


DEVICE: str = _detect_device()

# ─── GPU performance flags ────────────────────────────────────────────────────
if DEVICE == "cuda":
    try:
        import torch as _torch
        # TF32: ~10× faster matmul on Ampere/Blackwell with negligible accuracy loss.
        _torch.set_float32_matmul_precision("high")
        # cuDNN auto-tuner: benchmark conv algorithms on first run, then reuse.
        _torch.backends.cudnn.benchmark = True
    except Exception:
        pass

# ─── Experiment 0 ────────────────────────────────────────────────────────────
EXP0_MINUTES_PER_SUBJECT = 10
EXP0_N_SUBJECTS = 10

# ─── Experiment 0.5 ──────────────────────────────────────────────────────────
SYNTH_FS = 256       # Hz
SYNTH_DURATION = 60  # seconds
SYNTH_SNR_LEVELS = [10, 5, 0]  # dB
SYNTH_BPE_VOCABS = [1024, 4096, 8192]

# ─── Experiment 5 — Scaling ──────────────────────────────────────────────────
SCALING_VOCAB_SIZES = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

# ─── Frequency bands (Hz) ───────────────────────────────────────────────────
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
