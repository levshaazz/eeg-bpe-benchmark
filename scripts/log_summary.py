#!/usr/bin/env python3
"""
EEG-BPE Log File Summarizer
=============================
Parses structured log lines (INFO | Result: {...}) to extract:
  - Timing per experiment
  - Cache hit/miss ratio
  - PCA reduction events
  - Errors and warnings
  - Active dataset/approach

Read-only — safe to run alongside an active experiment.

Usage:
    python scripts/log_summary.py                    # summarize all logs
    python scripts/log_summary.py --exp 6            # Exp 6 only
    python scripts/log_summary.py --exp 6 --tail 20  # last 20 result lines
    python scripts/log_summary.py --errors           # only errors/warnings
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "results" / "logs"

LOG_MAP = {
    "run_all":   "run_all.log",
    "0":         "exp0_quantization_loss.log",
    "0.5":       "exp05_synthetic_validation.log",
    "1":         "exp1_vocab_analysis.log",
    "2":         "exp2_downstream.log",
    "4":         "exp4_hierarchical.log",
    "5":         "exp5_scaling.log",
    "6":         "exp6.log",            # cache/pca lines; exp6_alt_tokenization.log has results
    "6r":        "exp6_alt_tokenization.log",  # result lines
    "7":         "exp7_fourier_bpe.log",
    "8":         "exp8_csp_bpe.log",
    "9":         "exp9_spatial_bpe.log",
    "ablations": "ablation_results.log",
}

# Regex patterns
RE_TIMESTAMP = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
RE_LEVEL     = re.compile(r"\|\s*(INFO|WARNING|ERROR|DEBUG)\s*\|")
RE_RESULT    = re.compile(r"Result: (\{.+\})")
RE_CACHE_HIT = re.compile(r"\[hist cache hit\]|\[cache hit\]|Cache hit")
RE_CACHE_MISS= re.compile(r"\[hist cache miss\]|\[cache miss\]")
RE_PCA       = re.compile(r"pca_reduce|TruncatedSVD|pca_reduce\((\d+)\)")
RE_DATASET   = re.compile(r"=== Dataset: (\w+)")
RE_APPROACH  = re.compile(r"Approach: (\w+)")
RE_MEM_GUARD = re.compile(r"subsampling to (\d+) trials|memory guard")
RE_DONE      = re.compile(r"DONE: (.+?) \[(.+?)\]")
RE_STARTING  = re.compile(r"STARTING: (.+)")


def _parse_log(path: Path, tail: int | None = None) -> dict:
    """Parse a log file and return structured summary."""
    if not path.exists():
        return {"exists": False}

    stats = {
        "exists": True,
        "size_kb": path.stat().st_size // 1024,
        "mtime": datetime.fromtimestamp(path.stat().st_mtime),
        "errors": [],
        "warnings": [],
        "cache_hits": 0,
        "cache_misses": 0,
        "pca_reductions": 0,
        "memory_guards": [],
        "results": [],
        "timing": {},
        "datasets_seen": [],
        "approaches_seen": [],
        "current_dataset": None,
        "current_approach": None,
        "first_ts": None,
        "last_ts": None,
        "total_lines": 0,
    }

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        stats["errors"].append(f"Could not read: {e}")
        return stats

    stats["total_lines"] = len(lines)
    if tail:
        lines = lines[-tail:]

    for line in lines:
        line = line.rstrip()
        if not line:
            continue

        # Timestamp
        m = RE_TIMESTAMP.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if stats["first_ts"] is None:
                    stats["first_ts"] = ts
                stats["last_ts"] = ts
            except ValueError:
                pass

        # Level
        level_m = RE_LEVEL.search(line)
        level = level_m.group(1) if level_m else "INFO"

        if level == "ERROR":
            stats["errors"].append(line[-120:])
        elif level == "WARNING":
            stats["warnings"].append(line[-120:])

        # Cache
        if RE_CACHE_HIT.search(line):
            stats["cache_hits"] += 1
        if RE_CACHE_MISS.search(line):
            stats["cache_misses"] += 1

        # PCA
        if RE_PCA.search(line):
            stats["pca_reductions"] += 1

        # Memory guard
        mg = RE_MEM_GUARD.search(line)
        if mg:
            stats["memory_guards"].append(line[-100:])

        # Dataset / approach tracking
        ds_m = RE_DATASET.search(line)
        if ds_m:
            ds = ds_m.group(1)
            stats["current_dataset"] = ds
            if ds not in stats["datasets_seen"]:
                stats["datasets_seen"].append(ds)

        ap_m = RE_APPROACH.search(line)
        if ap_m:
            ap = ap_m.group(1)
            stats["current_approach"] = ap
            if ap not in stats["approaches_seen"]:
                stats["approaches_seen"].append(ap)

        # Result lines
        res_m = RE_RESULT.search(line)
        if res_m:
            try:
                r = json.loads(res_m.group(1))
                stats["results"].append(r)
            except json.JSONDecodeError:
                pass

        # Done timing
        done_m = RE_DONE.search(line)
        if done_m:
            stats["timing"][done_m.group(1).strip()] = done_m.group(2).strip()

    # Estimate elapsed
    if stats["first_ts"] and stats["last_ts"]:
        elapsed = stats["last_ts"] - stats["first_ts"]
        stats["elapsed"] = str(elapsed).split(".")[0]
    else:
        stats["elapsed"] = "—"

    return stats


def _fmt_delta(dt: datetime) -> str:
    if dt is None:
        return "—"
    delta = datetime.now() - dt
    if delta < timedelta(minutes=1):
        return f"{int(delta.total_seconds())}s ago"
    if delta < timedelta(hours=1):
        return f"{int(delta.total_seconds() / 60)}m ago"
    return f"{delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m ago"


def print_summary(exp_id: str, stats: dict, errors_only: bool = False,
                  show_results: int = 0) -> None:
    log_name = LOG_MAP.get(exp_id, "?")
    print(f"\n{'─'*60}")
    print(f"  [{exp_id}] {log_name}")
    print(f"{'─'*60}")

    if not stats.get("exists"):
        print("  ○ Log file not found (experiment not yet started)")
        return

    mtime_str = _fmt_delta(stats.get("mtime"))
    print(f"  Size:         {stats['size_kb']} KB")
    print(f"  Last update:  {mtime_str}")
    print(f"  Total lines:  {stats['total_lines']}")
    print(f"  Elapsed:      {stats.get('elapsed', '—')}")

    if errors_only:
        if stats["errors"]:
            print(f"\n  ERRORS ({len(stats['errors'])}):")
            for e in stats["errors"][-5:]:
                print(f"    {e}")
        if stats["warnings"]:
            print(f"\n  WARNINGS ({len(stats['warnings'])}):")
            for w in stats["warnings"][-5:]:
                print(f"    {w}")
        return

    print(f"  Cache hits:   {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    if stats["cache_hits"] + stats["cache_misses"] > 0:
        total = stats["cache_hits"] + stats["cache_misses"]
        pct = 100 * stats["cache_hits"] / total
        print(f"  Cache rate:   {pct:.0f}%")

    if stats["pca_reductions"]:
        print(f"  PCA events:   {stats['pca_reductions']}")
    if stats["memory_guards"]:
        print(f"  Memory guards: {len(stats['memory_guards'])}")
        for mg in stats["memory_guards"][-2:]:
            print(f"    {mg}")
    if stats["errors"]:
        print(f"\n  ⚠ ERRORS ({len(stats['errors'])}):")
        for e in stats["errors"][-3:]:
            print(f"    {e}")
    if stats["warnings"]:
        print(f"\n  ! WARNINGS ({len(stats['warnings'])}):")
        for w in stats["warnings"][-3:]:
            print(f"    {w}")

    if stats["datasets_seen"]:
        print(f"\n  Datasets seen: {', '.join(stats['datasets_seen'])}")
    if stats["current_dataset"]:
        print(f"  Current:       dataset={stats['current_dataset']}"
              + (f", approach={stats['current_approach']}" if stats["current_approach"] else ""))

    if stats["timing"]:
        print(f"\n  Timing (DONE entries):")
        for name, t in list(stats["timing"].items())[-10:]:
            print(f"    {name[:50]:<50s}  {t}")

    if stats["results"] and show_results > 0:
        print(f"\n  Last {show_results} results:")
        for r in stats["results"][-show_results:]:
            ds   = r.get("dataset", "?")
            ap   = r.get("approach", r.get("classifier", "?"))
            acc  = r.get("accuracy_mean", r.get("accuracy", float("nan")))
            seed = r.get("seed", "?")
            print(f"    [{ds}] {ap}  seed={seed}  acc={acc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG-BPE log file summarizer")
    parser.add_argument("--exp", nargs="*", default=list(LOG_MAP.keys()),
                        help="Experiments to summarize (default: all)")
    parser.add_argument("--errors", action="store_true",
                        help="Show only errors and warnings")
    parser.add_argument("--tail", type=int, default=None,
                        help="Parse only last N lines of each log")
    parser.add_argument("--results", type=int, default=0,
                        help="Show last N result lines per experiment")
    args = parser.parse_args()

    for exp_id in args.exp:
        log_name = LOG_MAP.get(exp_id)
        if not log_name:
            print(f"Unknown experiment: {exp_id}", file=sys.stderr)
            continue
        log_path = LOGS / log_name
        stats = _parse_log(log_path, tail=args.tail)
        print_summary(exp_id, stats, errors_only=args.errors,
                      show_results=args.results)

    print()


if __name__ == "__main__":
    main()
