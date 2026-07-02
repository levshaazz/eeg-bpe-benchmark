#!/usr/bin/env python3
"""
EEG-BPE Paper 1 — Master Experiment Runner
==========================================
Runs all experiments in order, respecting dependencies and GO/NO-GO gates.

Usage:
    python -m eeg_bpe.run_all                     # full pipeline
    python -m eeg_bpe.run_all --quick              # 3 subj, V=1024
    python -m eeg_bpe.run_all --medium             # 9 subj, V=2048
    python -m eeg_bpe.run_all --exp 1 2 ablations  # specific experiments
    python -m eeg_bpe.run_all --datasets bci_iv_2a sleep_edf

Modes:
    --quick   : 3 subjects, vocab_size=1024, ~15 min on GPU
    --medium  : 9 subjects, vocab_size=2048, ~60 min on GPU  (recommended for dev)
    full      : all subjects, vocab_size=4096, ~4-8h on GPU

All intermediate results → results/logs/
All plots            → results/plots/
All BPE models       → results/models/
"""
from __future__ import annotations

import sys
# Force UTF-8 output so Unicode chars in the docstring/help work on Windows cp1251 consoles.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

from .config import DEVICE, RESULTS_DIR, LOGS_DIR, DEFAULT_QUANT_METHOD
from .utils import get_logger, save_json

logger = get_logger("run_all")


# ─── Progress helpers ──────────────────────────────────────────────────────────

class _ExpTimer:
    """Simple per-experiment timer with pretty logging."""

    def __init__(self):
        self.records: list[dict] = []
        self._t_total = time.perf_counter()

    def run(self, name: str, fn: Callable, *args, **kwargs):
        """Run *fn* with timing and structured logging."""
        sep = "=" * 60
        logger.info(sep)
        logger.info(f"STARTING: {name}")
        logger.info(sep)
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        elapsed_str = _fmt_time(elapsed)
        logger.info(f"DONE: {name}  [{elapsed_str}]")
        self.records.append({"name": name, "elapsed_s": round(elapsed, 2),
                             "status": "ok"})
        return result

    def summary(self) -> dict:
        total = time.perf_counter() - self._t_total
        return {
            "experiments": self.records,
            "total_time_s": round(total, 2),
            "total_time_str": _fmt_time(total),
        }

    def log_summary(self):
        logger.info("=" * 60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        for r in self.records:
            logger.info(f"  {r['name']:40s}  {_fmt_time(r['elapsed_s'])}")
        s = self.summary()
        logger.info(f"  {'TOTAL':40s}  {s['total_time_str']}")
        logger.info("=" * 60)
        self._plot_timing()


    def _plot_timing(self):
        """Save a horizontal bar chart of per-experiment timing."""
        if not self.records:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.debug("matplotlib not available — skipping timing plot")
            return

        try:
            plot_dir = RESULTS_DIR / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)

            names = [r["name"] for r in self.records]
            minutes = [r["elapsed_s"] / 60.0 for r in self.records]

            fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.45)))
            y_pos = range(len(names))
            bars = ax.barh(y_pos, minutes, color="#4C72B0", edgecolor="white")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel("Time (minutes)")
            ax.set_title("Run Timing Breakdown")

            # Annotate bars with exact values
            for bar, val in zip(bars, minutes):
                label = f"{val:.1f}m" if val >= 1 else f"{val * 60:.0f}s"
                ax.text(bar.get_width() + max(minutes) * 0.01, bar.get_y() + bar.get_height() / 2,
                        label, va="center", fontsize=8)

            plt.tight_layout()
            out_path = plot_dir / "run_all_timing.png"
            plt.savefig(str(out_path), dpi=150)
            plt.close()
            logger.info(f"Timing plot saved to {out_path}")
        except Exception as exc:
            logger.debug(f"Could not save timing plot: {exc}")


def _fmt_time(seconds: float) -> str:
    """Format seconds as 'Xh Ym Zs' or 'Ym Zs' or 'Z.1fs'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{seconds:.1f}s"


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Parse CLI arguments and run selected experiments in dependency order."""
    parser = argparse.ArgumentParser(
        description="EEG-BPE Paper 1 Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--exp", nargs="*", default=None,
                        help="Experiments to run: 0, 0.5, 1, 2, 4, 5, 6, 7, 8, ablations")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 subjects, vocab_size=1024, ~15 min")
    parser.add_argument("--medium", action="store_true",
                        help="Medium mode: 9 subjects, vocab_size=2048, ~60 min")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to use")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Max subjects per dataset (overrides --quick/--medium)")
    args = parser.parse_args()

    if args.quick and args.medium:
        parser.error("Cannot use --quick and --medium together")

    # ─── Mode settings ────────────────────────────────────────────────────
    if args.quick:
        mode = "QUICK"
        max_subj          = args.max_subjects or 3
        vocab_size        = 1024
        vocab_sizes       = [128, 256, 512, 1024]
        max_hours         = 0.25
        max_train_tokens  = None
        ablation_subj     = min(max_subj, 3)
    elif args.medium:
        mode = "MEDIUM"
        max_subj          = args.max_subjects or 9
        vocab_size        = 2048
        vocab_sizes       = [512, 1024, 2048]
        max_hours         = 1.0
        max_train_tokens  = 10_000_000
        ablation_subj     = min(max_subj, 9)
    else:
        mode = "FULL"
        max_subj          = args.max_subjects
        vocab_size        = 4096
        vocab_sizes       = None   # use defaults from config
        max_hours         = None   # use BPE_BALANCED_HOURS from config
        max_train_tokens  = 20_000_000
        ablation_subj     = max_subj or 9

    exps_to_run = args.exp or ["0", "0.5", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11", "ablations"]

    logger.info("=" * 60)
    logger.info(f"EEG-BPE Paper 1 — Master Runner  [mode={mode}]")
    logger.info(f"Device      : {DEVICE}")
    logger.info(f"Max subjects: {max_subj or 'all'}")
    logger.info(f"Vocab size  : {vocab_size}")
    logger.info(f"Experiments : {exps_to_run}")
    logger.info(f"Results dir : {RESULTS_DIR}")
    logger.info("=" * 60)

    timer = _ExpTimer()

    # ─── Exp 0: Quantization Loss Analysis ────────────────────────────────
    if "0" in exps_to_run:
        from .exp0_quantization import run_experiment_0
        results = timer.run(
            "Exp 0: Quantization Loss",
            run_experiment_0,
            datasets=args.datasets,
            max_subjects=max_subj or 10,
        )
        for r in (results or []):
            if r["n_bins"] == 256 and r["sqnr_db_mean"] < 15:
                logger.error(
                    f"FAILURE: SQNR < 15 dB at B=256 for {r['dataset']}/{r['method']}. "
                    f"Quantization may be too destructive."
                )

    # ─── Exp 0.5: Synthetic Validation (GO/NO-GO) ────────────────────────
    if "0.5" in exps_to_run:
        from .exp05_synthetic import run_experiment_05
        summary = timer.run(
            "Exp 0.5: Synthetic Validation",
            run_experiment_05,
        )
        if not (summary or {}).get("go_decision", False):
            logger.error(
                "=" * 60 + "\n"
                "NO-GO: BPE cannot distinguish spectral states.\n"
                "   Separability accuracy too low.\n"
                "   Consider revising the discretization approach.\n"
                + "=" * 60
            )
            if "1" in exps_to_run or "2" in exps_to_run:
                logger.warning("Continuing despite NO-GO (for analysis purposes)")

    # ─── Exp 1: BPE Vocabulary Training & Analysis ────────────────────────
    if "1" in exps_to_run:
        from .exp1_vocab_analysis import run_experiment_1
        timer.run(
            "Exp 1: BPE Vocab Training",
            run_experiment_1,
            datasets=args.datasets,
            vocab_sizes=vocab_sizes,
            max_subjects=max_subj or 10,
            max_hours=max_hours,
            max_train_tokens=max_train_tokens,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
        )

    # ─── Exp 2: Downstream Classification ─────────────────────────────────
    if "2" in exps_to_run:
        from .exp2_downstream import run_experiment_2
        timer.run(
            "Exp 2: Downstream Classification",
            run_experiment_2,
            datasets=args.datasets,
            vocab_size=vocab_size,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            max_subjects=max_subj,
        )

    # ─── Exp 4: Hierarchical Analysis ─────────────────────────────────────
    if "4" in exps_to_run:
        from .exp4_hierarchical import run_experiment_4
        vs4 = vocab_size * 4 if not args.quick else vocab_size
        timer.run(
            "Exp 4: Hierarchical Analysis",
            run_experiment_4,
            vocab_size=vs4,
        )

    # ─── Exp 5: Scaling Laws ──────────────────────────────────────────────
    if "5" in exps_to_run:
        from .exp5_scaling import run_experiment_5
        timer.run(
            "Exp 5: Scaling Laws",
            run_experiment_5,
            datasets=args.datasets,
            vocab_sizes=vocab_sizes,
            max_subjects=max_subj or 5,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
        )

    # ─── Exp 6: Alternative Tokenisation Strategies ───────────────────────
    if "6" in exps_to_run:
        from .exp6_alt_tokenization import run_experiment_6
        # Always run exp6 on all 6 datasets: max_subjects cap already keeps it fast.
        exp6_datasets = args.datasets or None
        exp6_vocab = vocab_size if not args.quick else 512
        exp6_max_vt = 3 if args.quick else (5 if args.medium else 8)
        timer.run(
            "Exp 6: Alternative Tokenisation",
            run_experiment_6,
            datasets=exp6_datasets,
            vocab_size=exp6_vocab,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            max_subjects=max_subj,
            max_subjects_vocab_train=exp6_max_vt,
        )

    # ─── Exp 7: Fourier-guided BPE ────────────────────────────────────────
    if "7" in exps_to_run:
        from .exp7_fourier_bpe import run_experiment_7
        exp7_vocab   = vocab_size if not args.quick else 512
        exp7_n_codes = 64
        # quick: only 0.5 s window; medium/full: also test 0.25 s
        exp7_wins    = [0.5] if args.quick else [0.25, 0.5]
        exp7_max_vt  = 3 if args.quick else (5 if args.medium else 8)
        timer.run(
            "Exp 7: Fourier-guided BPE",
            run_experiment_7,
            datasets=args.datasets,
            vocab_size=exp7_vocab,
            n_codes=exp7_n_codes,
            win_secs=exp7_wins,
            max_subjects=max_subj,
            max_subjects_vocab_train=exp7_max_vt,
        )

    # ─── Exp 8: CSP + BPE ─────────────────────────────────────────────────
    if "8" in exps_to_run:
        from .exp8_csp_bpe import run_experiment_8
        exp8_vocab = 512 if args.quick else (512 if args.medium else 1024)
        timer.run(
            "Exp 8: CSP + BPE",
            run_experiment_8,
            datasets=args.datasets,
            vocab_size=exp8_vocab,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            max_subjects=max_subj,
        )

    # ─── Exp 9: Spatial BPE ───────────────────────────────────────────────
    if "9" in exps_to_run:
        from .exp9_spatial_bpe import run_experiment_9
        exp9_vocab  = 256 if args.quick else (512 if args.medium else 1024)
        exp9_codes  = 64  if args.quick else 128
        timer.run(
            "Exp 9: Spatial BPE",
            run_experiment_9,
            datasets=args.datasets,
            vocab_size=exp9_vocab,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            n_codes=exp9_codes,
            max_subjects=max_subj,
        )

    # ─── Exp 10: Cross-Dataset Vocabulary Transfer ────────────────────────
    if "10" in exps_to_run:
        from .exp10_cross_dataset import run_experiment_10
        exp10_vocab = 512 if args.quick else (1024 if args.medium else 1024)
        # In quick mode, use fewer datasets to keep runtime manageable
        exp10_ds = (["sleep_edf", "bci_iv_2a", "mental_arithmetic"]
                    if args.quick else None)
        timer.run(
            "Exp 10: Cross-Dataset Transfer",
            run_experiment_10,
            datasets=exp10_ds,
            vocab_size=exp10_vocab,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            max_subjects=max_subj,
        )

    # ─── Exp 11: Improvement Ideas (S1-S6) ─────────────────────────────
    if "11" in exps_to_run:
        from .exp11_improvements import run_experiment_11
        exp11_vocab = 256 if args.quick else (512 if args.medium else 1024)
        exp11_ds = (["sleep_edf", "bci_iv_2a"]
                    if args.quick else args.datasets)
        timer.run(
            "Exp 11: Improvement Ideas",
            run_experiment_11,
            datasets=exp11_ds,
            vocab_size=exp11_vocab,
            n_bins=64,
            method=DEFAULT_QUANT_METHOD,
            max_subjects=max_subj,
        )

    # ─── Ablations ────────────────────────────────────────────────────────
    if "ablations" in exps_to_run:
        from .ablations import run_all_ablations
        timer.run(
            "Ablations A1-A15, K7",
            run_all_ablations,
            max_subjects=ablation_subj,
            vocab_size=vocab_size,
            n_bins=64,
        )

    # ─── Summary ──────────────────────────────────────────────────────────
    timer.log_summary()
    run_summary = timer.summary()
    save_json(run_summary, LOGS_DIR / "run_all_log.json")
    logger.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
