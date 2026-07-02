#!/usr/bin/env python3
"""
EEG-BPE Key Findings Extractor
================================
Reads all experiment CSV logs and produces a structured markdown summary
with the most important numbers, comparisons, and flags.

Read-only — safe to run alongside an active experiment (partial results shown as-is).

Usage:
    python scripts/key_findings.py
    python scripts/key_findings.py --output results/key_findings.md
    python scripts/key_findings.py --exp 2 5 ablations
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas required: pip install pandas")

ROOT = Path(__file__).parent.parent
LOGS = ROOT / "results" / "logs"

DATASETS = ["bci_iv_2a", "physionet_mi", "sleep_edf",
            "mental_arithmetic", "epfl_p300", "ssvep_nakanishi"]

DS_LABEL = {
    "bci_iv_2a":          "BCI-IV-2a (MI)",
    "physionet_mi":       "PhysioNet-MI",
    "sleep_edf":          "Sleep-EDF",
    "mental_arithmetic":  "Mental Arithmetic",
    "epfl_p300":          "EPFL P300",
    "ssvep_nakanishi":    "SSVEP Nakanishi",
}

CHANCE = {
    "bci_iv_2a": 0.25, "physionet_mi": 0.25, "sleep_edf": 0.20,
    "mental_arithmetic": 0.50, "epfl_p300": 0.50, "ssvep_nakanishi": 1/12,
}

PARADIGM = {
    "bci_iv_2a": "Motor Imagery", "physionet_mi": "Motor Imagery",
    "sleep_edf": "Sleep Staging", "mental_arithmetic": "Cognitive Load",
    "epfl_p300": "P300 ERP", "ssvep_nakanishi": "SSVEP",
}

BPE_CLASSIFIERS = [
    "BPE_Hist_LogReg", "BPE_Hist_RF", "BPE_Windowed_LogReg",
    "BPE_WindowedSeq_CNN", "BPE_Bigram_LogReg",
    "BPE_Seq_CNN", "BPE_Seq_Transformer", "BPE_Seq_CW_Transformer",
]

BASELINE_CLASSIFIERS = [
    "EEGNet", "CSP_LDA", "PSD_LogReg", "Patching_LogReg",
    "VQ_LogReg", "Chronos_Binning", "SSVEP_FFT_LogReg",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _pct(x: float) -> str:
    return f"{100*x:.1f}%"


def _pp(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{100*delta:.1f}pp"


def _load(filename: str) -> pd.DataFrame | None:
    path = LOGS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  [warn] Could not read {filename}: {e}", file=sys.stderr)
        return None


def _best(df: pd.DataFrame, ds: str, classifiers: list[str],
          acc_col: str = "accuracy_mean") -> tuple[str, float, float] | None:
    """Return (classifier, accuracy_mean, kappa_mean) for best clf on dataset."""
    sub = df[(df["dataset"] == ds) & (df["classifier"].isin(classifiers))]
    if sub.empty:
        return None
    # Average over seeds first, then pick best classifier
    agg = sub.groupby("classifier")[acc_col].mean()
    best_clf = agg.idxmax()
    best_acc = agg.max()
    kappa_col = "kappa_mean" if "kappa_mean" in df.columns else None
    best_kap = sub[sub["classifier"] == best_clf]["kappa_mean"].mean() if kappa_col else float("nan")
    return best_clf, best_acc, best_kap


def _flag(val: float, ref: float, threshold: float = 0.05) -> str:
    """Flag unexpected results (deviation > threshold from a reference)."""
    if abs(val - ref) > threshold:
        return " ⚠" if val < ref else " ★"
    return ""


# ─── Section builders ─────────────────────────────────────────────────────────

def section_exp2(df: pd.DataFrame) -> str:
    lines = ["## Exp 2 — Downstream Classification\n"]
    lines.append(f"{'Dataset':<22} {'Best BPE':>22} {'Best Baseline':>22} {'Gap':>8} {'Chance':>8}")
    lines.append("─" * 88)

    rows_for_table = []
    for ds in DATASETS:
        best_bpe    = _best(df, ds, BPE_CLASSIFIERS)
        best_base   = _best(df, ds, BASELINE_CLASSIFIERS)
        chance      = CHANCE[ds]

        bpe_str  = f"{best_bpe[0]} {_pct(best_bpe[1])} κ={best_bpe[2]:.2f}"   if best_bpe  else "—"
        base_str = f"{best_base[0]} {_pct(best_base[1])} κ={best_base[2]:.2f}" if best_base else "—"
        gap_str  = _pp(best_bpe[1] - best_base[1]) if (best_bpe and best_base) else "—"

        lines.append(f"{DS_LABEL[ds]:<22} {bpe_str:>32} {base_str:>32} {gap_str:>8} {_pct(chance):>8}")

        if best_bpe and best_base:
            rows_for_table.append({
                "dataset": ds, "paradigm": PARADIGM[ds],
                "best_bpe_clf": best_bpe[0], "best_bpe_acc": best_bpe[1],
                "best_bpe_kap": best_bpe[2],
                "best_base_clf": best_base[0], "best_base_acc": best_base[1],
                "best_base_kap": best_base[2],
                "gap_pp": 100 * (best_bpe[1] - best_base[1]),
                "chance": chance,
            })
    lines.append("")

    # Flags
    flags = []
    for r in rows_for_table:
        if r["best_bpe_acc"] > r["best_base_acc"]:
            flags.append(f"  ★  {DS_LABEL[r['dataset']]}: BPE ({_pct(r['best_bpe_acc'])}) BEATS baseline ({_pct(r['best_base_acc'])}) by {_pp(r['gap_pp']/100)}")
        elif r["best_bpe_acc"] < CHANCE[r["dataset"]] + 0.02:
            flags.append(f"  ⚠  {DS_LABEL[r['dataset']]}: Best BPE ({_pct(r['best_bpe_acc'])}) near chance ({_pct(CHANCE[r['dataset']])})")
    if flags:
        lines.append("**Flags:**")
        lines.extend(flags)
        lines.append("")

    # Per-classifier breakdown for each dataset
    lines.append("### Per-Classifier Breakdown\n")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].groupby("classifier")["accuracy_mean"].mean().sort_values(ascending=False)
        if sub.empty:
            continue
        lines.append(f"**{DS_LABEL[ds]}** (chance={_pct(CHANCE[ds])}):")
        for clf, acc in sub.items():
            marker = " ←BPE" if clf in BPE_CLASSIFIERS else ""
            lines.append(f"  {clf:<30s}  {_pct(acc)}{marker}")
        lines.append("")

    return "\n".join(lines)


def section_exp5(df: pd.DataFrame) -> str:
    lines = ["## Exp 5 — Scaling Laws (V* per Dataset)\n"]
    lines.append(f"{'Dataset':<22} {'V*':>8} {'Acc@V*':>10} {'Trend':>20}")
    lines.append("─" * 65)

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].sort_values("vocab_size")
        if sub.empty:
            lines.append(f"{DS_LABEL[ds]:<22}  {'(no data)':>8}")
            continue
        best_row = sub.loc[sub["accuracy_mean"].idxmax()]
        v_star = int(best_row["vocab_size"])
        acc_star = best_row["accuracy_mean"]

        # Trend: monotone increasing / peak-then-decay / flat
        accs = sub["accuracy_mean"].values
        vs   = sub["vocab_size"].values
        if len(accs) < 3:
            trend = "—"
        else:
            peak_idx = np.argmax(accs)
            if peak_idx == 0:
                trend = "monotone ↓"
            elif peak_idx == len(accs) - 1:
                trend = "monotone ↑"
            else:
                trend = f"peak@V={vs[peak_idx]:,}"

        lines.append(f"{DS_LABEL[ds]:<22} {v_star:>8,} {_pct(acc_star):>10} {trend:>20}")

    lines.append("")
    lines.append("*Note: V* measured with LogReg. RF-optimal V is lower (see A9, methodology_fixes.md).*")
    lines.append("")
    return "\n".join(lines)


def section_exp6(df: pd.DataFrame) -> str:
    lines = ["## Exp 6 — Alternative Tokenisation\n"]
    done_ds = df["dataset"].unique().tolist()
    if not done_ds:
        lines.append("*(no data yet)*\n")
        return "\n".join(lines)

    approach_col = "approach"
    acc_col = "accuracy_mean"

    lines.append(f"Datasets with results: {', '.join(done_ds)}\n")
    for ds in done_ds:
        sub = df[df["dataset"] == ds].groupby(approach_col)[acc_col].mean().sort_values(ascending=False)
        lines.append(f"**{DS_LABEL.get(ds, ds)}** (chance={_pct(CHANCE.get(ds, 0))}):")
        for approach, acc in sub.items():
            marker = " ← best" if acc == sub.max() else ""
            lines.append(f"  {approach:<30s}  {_pct(acc)}{marker}")
        lines.append("")

    return "\n".join(lines)


def section_ablations(dfs: dict[str, pd.DataFrame | None]) -> str:
    lines = ["## Ablations\n"]

    abl_path = LOGS / "ablation_results.csv"
    if not abl_path.exists():
        lines.append("*(ablation_results.csv not found — ablations may not have run yet)*\n")
        # try per-ablation files
        for name in ["ablation_A1_results.csv", "ablation_A2_results.csv",
                     "ablation_A8_results.csv", "ablation_A9_results.csv"]:
            p = LOGS / name
            if p.exists():
                lines.append(f"  Found: {name} ({sum(1 for _ in open(p))-1} rows)")
        lines.append("")
        return "\n".join(lines)

    df = pd.read_csv(abl_path)
    if "ablation" in df.columns:
        for abl in sorted(df["ablation"].unique()):
            sub = df[df["ablation"] == abl]
            lines.append(f"**{abl}:**  {len(sub)} rows, datasets: {', '.join(sub['dataset'].unique())}")
    lines.append("")
    return "\n".join(lines)


def section_summary(df2: pd.DataFrame | None) -> str:
    lines = ["## Key Findings Summary\n"]

    if df2 is None:
        lines.append("*(Exp2 data required)*\n")
        return "\n".join(lines)

    # Paradigm classification
    bpe_wins, bpe_fails = [], []
    for ds in DATASETS:
        best_bpe  = _best(df2, ds, BPE_CLASSIFIERS)
        best_base = _best(df2, ds, BASELINE_CLASSIFIERS)
        if best_bpe is None or best_base is None:
            continue
        if best_bpe[1] >= best_base[1]:
            bpe_wins.append(ds)
        else:
            bpe_fails.append(ds)

    lines.append("### BPE Works vs. Fails\n")
    if bpe_wins:
        lines.append("**BPE ≥ best baseline (amplitude-coded):**")
        for ds in bpe_wins:
            b = _best(df2, ds, BPE_CLASSIFIERS)
            lines.append(f"  ✓ {DS_LABEL[ds]}: BPE {_pct(b[1])}")
    if bpe_fails:
        lines.append("\n**BPE < best baseline (frequency-coded or other):**")
        for ds in bpe_fails:
            b  = _best(df2, ds, BPE_CLASSIFIERS)
            bl = _best(df2, ds, BASELINE_CLASSIFIERS)
            lines.append(f"  ✗ {DS_LABEL[ds]}: BPE {_pct(b[1])} vs {bl[0]} {_pct(bl[1])} ({_pp(b[1]-bl[1])})")

    lines.append("")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="EEG-BPE key findings extractor")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Write markdown to file (default: stdout)")
    parser.add_argument("--exp", nargs="*", default=["2", "5", "6", "ablations"],
                        help="Which experiments to include")
    args = parser.parse_args()

    out_lines = [
        f"# EEG-BPE Key Findings",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"(Read-only snapshot — partial results shown as-is)\n",
        "---\n",
    ]

    df2   = _load("exp2_downstream_results.csv")
    df5   = _load("exp5_scaling_results.csv")
    df6   = _load("exp6_alt_tokenization_results.csv")

    if "2" in args.exp and df2 is not None:
        out_lines.append(section_exp2(df2))

    if "5" in args.exp and df5 is not None:
        out_lines.append(section_exp5(df5))

    if "6" in args.exp and df6 is not None:
        out_lines.append(section_exp6(df6))

    if "ablations" in args.exp:
        out_lines.append(section_ablations({}))

    out_lines.append(section_summary(df2))

    result = "\n".join(out_lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Written to {args.output}")
    else:
        print(result)


if __name__ == "__main__":
    main()
