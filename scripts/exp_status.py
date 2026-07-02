#!/usr/bin/env python3
"""
EEG-BPE Experiment Status Monitor
===================================
Shows which experiments are done, running, or pending.
Read-only — safe to run alongside an active experiment.

Usage:
    python scripts/exp_status.py
    python scripts/exp_status.py --watch        # refresh every 60s
    python scripts/exp_status.py --watch 30     # refresh every 30s
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "results" / "logs"
MODELS = ROOT / "results" / "models"

# ─── Expected completion criteria per experiment ────────────────────────────

DATASETS = ["bci_iv_2a", "physionet_mi", "sleep_edf",
            "mental_arithmetic", "epfl_p300", "ssvep_nakanishi"]
N_SEEDS = 5
N_DATASETS = 6

EXP_SPECS: dict[str, dict] = {
    "0":         {"csv": "exp0_quantization_loss_results.csv",
                  "label": "Exp 0  Quantization Loss",
                  "expected": 18,          # 6ds × 3 methods
                  "key_col": "dataset"},
    "0.5":       {"csv": "exp05_synthetic_validation_results.csv",
                  "label": "Exp 0.5 Synthetic Validation",
                  "expected": 10,
                  "key_col": "condition"},
    "1":         {"csv": "exp1_vocab_analysis_results.csv",
                  "label": "Exp 1  Vocab Training",
                  "expected": 48,          # 8 vocab × 6 ds
                  "key_col": "vocab_size"},
    "2":         {"csv": "exp2_downstream_results.csv",
                  "label": "Exp 2  Downstream Classification",
                  "expected": 405,         # 6ds × ~13.5clf × 5seeds (CSP/SSVEP_FFT paradigm-specific)
                  "key_col": "classifier"},
    "4":         {"csv": "exp4_hierarchical_results.csv",
                  "label": "Exp 4  Hierarchical Analysis",
                  "expected": 1,
                  "key_col": "dataset"},
    "5":         {"csv": "exp5_scaling_results.csv",
                  "label": "Exp 5  Scaling Laws",
                  "expected": 48,          # 6ds × 8 vocab sizes
                  "key_col": "vocab_size"},
    "6":         {"csv": "exp6_alt_tokenization_results.csv",
                  "label": "Exp 6  Alt Tokenisation",
                  "expected": 180,         # 6ds × 6 approaches × 5seeds
                  "key_col": "approach"},
    "7":         {"csv": "exp7_fourier_bpe_results.csv",
                  "label": "Exp 7  Fourier BPE",
                  "expected": 60,          # 6ds × 2 win × 5seeds
                  "key_col": "win_sec"},
    "8":         {"csv": "exp8_csp_bpe_results.csv",
                  "label": "Exp 8  CSP + BPE",
                  "expected": 30,          # 2ds × 3clf × 5seeds
                  "key_col": "classifier"},
    "9":         {"csv": "exp9_spatial_bpe_results.csv",
                  "label": "Exp 9  Spatial BPE",
                  "expected": 60,          # 6ds × 2clf × 5seeds
                  "key_col": "classifier"},
    "ablations": {"csv": "ablation_results.csv",
                  "label": "Ablations A1–A9",
                  "expected": 900,         # rough estimate
                  "key_col": "ablation"},
}

LOG_FILES = {
    "6":         "exp6_alt_tokenization.log",
    "7":         "exp7_fourier_bpe.log",
    "8":         "exp8_csp_bpe.log",
    "9":         "exp9_spatial_bpe.log",
    "ablations": "ablation_results.log",
    "run_all":   "run_all.log",
}


def _mtime_str(path: Path) -> str:
    if not path.exists():
        return "—"
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    delta = datetime.now() - dt
    if delta < timedelta(minutes=1):
        return f"{int(delta.total_seconds())}s ago"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() / 60)}m ago"
    if delta < timedelta(days=1):
        return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m ago"
    return dt.strftime("%m-%d %H:%M")


def _row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        count = 0
        with open(csv_path, encoding="utf-8") as f:
            next(f)  # skip header
            for _ in f:
                count += 1
        return count
    except Exception:
        return 0


def _last_log_line(log_path: Path, n: int = 1) -> str:
    """Return the last n non-empty lines of a log file."""
    if not log_path.exists():
        return ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = [l.rstrip() for l in f if l.strip()]
        return "\n  ".join(lines[-n:]) if lines else ""
    except Exception:
        return ""


def _parse_active_dataset(log_path: Path) -> str:
    """Try to extract which dataset/approach is currently being processed."""
    if not log_path.exists():
        return ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for line in reversed(lines[-200:]):
            if "Dataset:" in line or "Approach:" in line or "approach" in line.lower():
                # extract the relevant part
                if "|" in line:
                    msg = line.split("|")[-1].strip()
                    return msg[:80]
                return line.strip()[-80:]
    except Exception:
        pass
    return ""


def _status_icon(rows: int, expected: int, csv_exists: bool) -> str:
    if not csv_exists:
        return "○"
    if rows == 0:
        return "○"
    if rows >= expected * 0.97:
        return "✓"
    return "⟳"


def show_status(verbose: bool = False) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'─'*72}")
    print(f"  EEG-BPE Experiment Status   [{now_str}]")
    print(f"{'─'*72}")

    run_all_log = LOGS / "run_all.log"
    run_all_mtime = _mtime_str(run_all_log)
    print(f"  run_all.log last activity : {run_all_mtime}")

    # Check if run is still active (last log < 10 min ago)
    if run_all_log.exists():
        delta = datetime.now() - datetime.fromtimestamp(run_all_log.stat().st_mtime)
        if delta < timedelta(minutes=10):
            print(f"  Status: \033[32mACTIVE\033[0m (run_all.log updated {run_all_mtime})")
        else:
            print(f"  Status: \033[33mIDLE / COMPLETE?\033[0m (last update {run_all_mtime})")
    print()

    total_done = 0
    for exp_id, spec in EXP_SPECS.items():
        csv_path = LOGS / spec["csv"]
        rows = _row_count(csv_path)
        expected = spec["expected"]
        icon = _status_icon(rows, expected, csv_path.exists())

        if icon == "✓":
            total_done += 1

        pct = f"{100*rows/expected:.0f}%" if expected > 0 else "?"
        bar_len = 20
        filled = int(bar_len * min(rows / expected, 1.0)) if expected > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        print(f"  {icon}  {spec['label']:<32s}  {rows:>4}/{expected:<4}  [{bar}] {pct:>4}")

        if icon == "⟳" and verbose:
            log_key = exp_id if exp_id in LOG_FILES else None
            if log_key:
                log_path = LOGS / LOG_FILES[log_key]
                active = _parse_active_dataset(log_path)
                if active:
                    print(f"       └─ {active}")
                mtime = _mtime_str(log_path)
                print(f"       └─ log updated: {mtime}")

    print()
    print(f"  Experiments complete: {total_done}/{len(EXP_SPECS)}")

    # Show active experiment last log line
    for exp_id, logfile in LOG_FILES.items():
        if exp_id == "run_all":
            continue
        log_path = LOGS / logfile
        if not log_path.exists():
            continue
        delta = datetime.now() - datetime.fromtimestamp(log_path.stat().st_mtime)
        if delta < timedelta(minutes=30):
            last = _last_log_line(log_path, n=1)
            if last:
                label = EXP_SPECS.get(exp_id, {}).get("label", exp_id)
                # trim to 100 chars
                last_short = last[-100:] if len(last) > 100 else last
                print(f"  Active [{label}]:")
                print(f"    {last_short}")
    print(f"{'─'*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG-BPE experiment status monitor")
    parser.add_argument("--watch", nargs="?", const=60, type=int, metavar="SECS",
                        help="Refresh every N seconds (default 60)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show active dataset/approach for running experiments")
    args = parser.parse_args()

    if args.watch:
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                show_status(verbose=args.verbose)
                print(f"  Refreshing every {args.watch}s — Ctrl+C to stop")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        show_status(verbose=args.verbose)


if __name__ == "__main__":
    main()
