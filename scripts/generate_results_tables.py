"""
Auto-generate LaTeX result tables from experiment CSV logs.

Usage:
    python scripts/generate_results_tables.py
    python scripts/generate_results_tables.py --exp 2 6 7 8 9 ablations
    python scripts/generate_results_tables.py --output writing/tables_auto.tex

Reads:
    results/logs/exp2_downstream_results.csv
    results/logs/exp6_alt_tokenization_results.csv
    results/logs/exp7_fourier_bpe_results.csv
    results/logs/exp8_csp_bpe_results.csv
    results/logs/exp9_spatial_bpe_results.csv
    results/logs/ablation_A*.csv  (or ablation_A8_results.csv etc.)

Outputs LaTeX tabular blocks + a summary markdown report.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "results" / "logs"
OUT_DEFAULT = ROOT / "results" / "auto_tables.tex"

DATASET_ORDER = [
    "bci_iv_2a", "physionet_mi", "sleep_edf",
    "mental_arithmetic", "epfl_p300", "ssvep_nakanishi",
]
DATASET_LABELS = {
    "bci_iv_2a":        "BCI-IV-2a",
    "physionet_mi":     "PhysioNet-MI",
    "sleep_edf":        "Sleep-EDF",
    "mental_arithmetic":"Mental Arith.",
    "epfl_p300":        "EPFL P300",
    "ssvep_nakanishi":  "SSVEP",
}
CHANCE = {
    "bci_iv_2a": 0.25,
    "physionet_mi": 0.25,
    "sleep_edf": 0.20,
    "mental_arithmetic": 0.50,
    "epfl_p300": 0.50,
    "ssvep_nakanishi": 1/12,
}

# ─── helpers ──────────────────────────────────────────────────────────────────

def pct(v, decimals=1):
    return f"{v*100:.{decimals}f}\\%"

def kap(v):
    if v is None or np.isnan(v):
        return "--"
    return f"{v:.3f}"

def color_cell(acc, chance, content):
    if acc >= chance + 0.05:
        return f"\\cellcolor{{good}}{content}"
    elif acc <= chance:
        return f"\\cellcolor{{bad}}{content}"
    return content

def bold_if_best(val, best_val, content, tol=0.002):
    if abs(val - best_val) <= tol:
        return f"\\textbf{{{content}}}"
    return content

def section(title):
    return f"\n% {'─'*70}\n% {title}\n% {'─'*70}\n"

# ─── Exp 2 ────────────────────────────────────────────────────────────────────

def table_exp2(df: pd.DataFrame) -> str:
    """Best method per dataset × classifier (mean over seeds)."""
    grp = df.groupby(["dataset", "classifier"]).agg(
        acc=("accuracy_mean", "mean"),
        kappa=("kappa_mean", "mean"),
        std=("accuracy_std", "mean"),
    ).reset_index()

    clfs = [
        ("BPE_Hist_LogReg",      "BPE", "Hist\\_LogReg"),
        ("BPE_Hist_RF",          "BPE", "Hist\\_RF"),
        ("BPE_Windowed_LogReg",  "BPE", "Windowed\\_LogReg"),
        ("BPE_WindowedSeq_CNN",  "BPE", "WindSeqCNN"),
        ("BPE_Bigram_LogReg",    "BPE", "Bigram\\_LogReg"),
        ("BPE_Seq_CNN",          "BPE", "Seq\\_CNN"),
        ("BPE_Seq_Transformer",  "BPE", "Seq\\_Transformer"),
        ("BPE_Seq_CW_Transformer","BPE","CW\\_Transformer"),
        ("CSP_LDA",              "Base","CSP+LDA"),
        ("EEGNet",               "Base","EEGNet"),
        ("Patching_LogReg",      "Base","Patching\\_LogReg"),
        ("VQ_LogReg",            "Base","VQ\\_LogReg"),
        ("Chronos_Binning",      "Base","Chronos\\_Binning"),
        ("PSD_LogReg",           "Base","PSD\\_LogReg"),
        ("SSVEP_FFT_LogReg",     "Base","SSVEP\\_FFT\\_LogReg"),
    ]

    datasets_a = ["bci_iv_2a", "physionet_mi", "sleep_edf"]
    datasets_b = ["mental_arithmetic", "epfl_p300", "ssvep_nakanishi"]

    def get(ds, clf):
        r = grp[(grp.dataset == ds) & (grp.classifier == clf)]
        if r.empty:
            return None, None
        return float(r.iloc[0]["acc"]), float(r.iloc[0]["kappa"])

    def make_subtable(datasets):
        col_header = " & ".join(
            f"\\multicolumn{{2}}{{c}}{{{DATASET_LABELS[d]}}}" for d in datasets
        )
        cmidrule = "".join(
            f"\\cmidrule(lr){{{3+2*i}-{4+2*i}}}"
            for i in range(len(datasets))
        )
        lines = [
            "\\begin{tabular}{@{}ll" + "rr" * len(datasets) + "@{}}",
            "  \\toprule",
            f"  & & {col_header} \\\\",
            f"  {cmidrule}",
            "  Type & Classifier" + " & Acc & $\\kappa$" * len(datasets) + " \\\\",
            "  \\midrule",
        ]

        # best per dataset
        best = {}
        for ds in datasets:
            vals = [get(ds, c)[0] for c, _, _ in clfs]
            vals = [v for v in vals if v is not None]
            best[ds] = max(vals) if vals else 0

        prev_type = None
        for clf_id, clf_type, clf_label in clfs:
            row = "  "
            if clf_type != prev_type:
                if prev_type is not None:
                    lines.append("  \\midrule")
                row += f"\\multirow{{8}}{{*}}{{{clf_type}}} " if clf_type == "BPE" else f"\\multirow{{7}}{{*}}{{{clf_type}}} "
                prev_type = clf_type
            else:
                row += "  "

            row += f"& {clf_label} "
            for ds in datasets:
                acc, kap_val = get(ds, clf_id)
                if acc is None:
                    row += "& --- & --- "
                else:
                    chance = CHANCE[ds]
                    acc_str = pct(acc)
                    acc_str = color_cell(acc, chance, acc_str)
                    acc_str = bold_if_best(acc, best[ds], acc_str)
                    row += f"& {acc_str} & {kap(kap_val)} "
            lines.append(row + "\\\\")

        lines += [
            "  \\midrule",
            "  \\multicolumn{2}{c}{\\textit{Chance}}" +
            "".join(f" & \\multicolumn{{2}}{{c}}{{{pct(CHANCE[d])}}}" for d in datasets) +
            " \\\\",
            "  \\bottomrule",
            "\\end{tabular}",
        ]
        return "\n".join(lines)

    out = section("Exp 2: Main Downstream Classification Results (full run, V=4096, 5 seeds)")
    out += "\n% --- Table A: BCI-IV-2a, PhysioNet-MI, Sleep-EDF ---\n"
    out += make_subtable(datasets_a)
    out += "\n\n% --- Table B: Mental Arith, P300, SSVEP ---\n"
    out += make_subtable(datasets_b)
    return out


# ─── Exp 5 ────────────────────────────────────────────────────────────────────

def table_exp5(df: pd.DataFrame) -> str:
    out = section("Exp 5: Scaling Laws (BPE_Hist_LogReg accuracy vs V)")
    rows = []
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].sort_values("vocab_size")
        if sub.empty:
            continue
        best_row = sub.loc[sub["accuracy_mean"].idxmax()]
        curve = {int(r["vocab_size"]): f"{r['accuracy_mean']*100:.1f}\\%" for _, r in sub.iterrows()}
        rows.append({
            "Dataset": DATASET_LABELS.get(ds, ds),
            "V*": int(best_row["vocab_size"]),
            "Best Acc": pct(best_row["accuracy_mean"]),
            "Curve (512→65536)": " | ".join(f"{v}: {a}" for v, a in sorted(curve.items())),
        })

    out += "\n" + pd.DataFrame(rows).to_string(index=False)
    return out


# ─── Exp 6 ────────────────────────────────────────────────────────────────────

def table_exp6(df: pd.DataFrame) -> str:
    grp = df.groupby(["dataset", "approach", "approach_label"]).agg(
        acc=("accuracy_mean", "mean"),
        kappa=("kappa_mean", "mean"),
    ).reset_index().sort_values(["dataset", "acc"], ascending=[True, False])

    out = section("Exp 6: Alternative Tokenisation Approaches (mean over seeds)")
    lines = [
        "\\begin{tabular}{@{}llrrrr@{}}",
        "  \\toprule",
        "  Dataset & Approach & Acc & $\\kappa$ & Comp. & Token\\_ms \\\\",
        "  \\midrule",
    ]
    for ds in DATASET_ORDER:
        sub = grp[grp["dataset"] == ds]
        if sub.empty:
            continue
        best_acc = sub["acc"].max()
        lines.append(f"  \\multirow{{{len(sub)}}}{{*}}{{{DATASET_LABELS.get(ds,ds)}}}")
        for i, (_, row) in enumerate(sub.iterrows()):
            label = row["approach_label"].replace("_", "\\_").replace("&", "\\&")
            chance = CHANCE.get(ds, 0.25)
            acc_str = pct(row["acc"])
            acc_str = color_cell(row["acc"], chance, acc_str)
            acc_str = bold_if_best(row["acc"], best_acc, acc_str)
            prefix = "  " if i > 0 else ""
            lines.append(f"  {prefix}& {label} & {acc_str} & {kap(row['kappa'])} \\\\")
        lines.append("  \\midrule")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out += "\n" + "\n".join(lines)
    return out


# ─── Exp 7 ────────────────────────────────────────────────────────────────────

def table_exp7(df: pd.DataFrame) -> str:
    grp = df.groupby(["dataset", "win_ms"]).agg(
        acc=("accuracy_mean", "mean"),
        kappa=("kappa_mean", "mean"),
    ).reset_index()

    out = section("Exp 7: Fourier-guided BPE (mean over seeds)")
    lines = [
        "\\begin{tabular}{@{}lrrr@{}}",
        "  \\toprule",
        "  Dataset & Win=250ms & Win=500ms & Better \\\\",
        "  \\midrule",
    ]
    for ds in DATASET_ORDER:
        sub = grp[grp["dataset"] == ds]
        a250 = sub[sub["win_ms"] == 250]["acc"].mean() if 250 in sub["win_ms"].values else None
        a500 = sub[sub["win_ms"] == 500]["acc"].mean() if 500 in sub["win_ms"].values else None
        chance = CHANCE.get(ds, 0.25)
        s250 = pct(a250) if a250 is not None else "---"
        s500 = pct(a500) if a500 is not None else "---"
        if a250 is not None:
            s250 = color_cell(a250, chance, s250)
        if a500 is not None:
            s500 = color_cell(a500, chance, s500)
        better = "250ms" if (a250 or 0) > (a500 or 0) else "500ms"
        lines.append(f"  {DATASET_LABELS.get(ds,ds)} & {s250} & {s500} & {better} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out += "\n" + "\n".join(lines)
    return out


# ─── Exp 8 ────────────────────────────────────────────────────────────────────

def table_exp8(df: pd.DataFrame) -> str:
    grp = df.groupby(["dataset", "classifier"]).agg(
        acc=("accuracy_mean", "mean"),
        kappa=("kappa_mean", "mean"),
    ).reset_index().sort_values(["dataset", "acc"], ascending=[True, False])

    out = section("Exp 8: CSP + BPE on Motor Imagery (mean over seeds)")
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "  \\toprule",
        "  Dataset / Classifier & Acc & $\\kappa$ \\\\",
        "  \\midrule",
    ]
    for ds in ["bci_iv_2a", "physionet_mi"]:
        sub = grp[grp["dataset"] == ds]
        if sub.empty:
            continue
        lines.append(f"  \\multicolumn{{3}}{{l}}{{\\textit{{{DATASET_LABELS.get(ds,ds)}}}}}\\ \\\\")
        for _, row in sub.iterrows():
            chance = CHANCE.get(ds, 0.25)
            acc_str = color_cell(row["acc"], chance, pct(row["acc"]))
            lines.append(f"  \\quad {row['classifier']} & {acc_str} & {kap(row['kappa'])} \\\\")
        lines.append("  \\midrule")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out += "\n" + "\n".join(lines)
    return out


# ─── Exp 9 ────────────────────────────────────────────────────────────────────

def table_exp9(df: pd.DataFrame) -> str:
    grp = df.groupby(["dataset", "classifier"]).agg(
        acc=("accuracy_mean", "mean"),
        kappa=("kappa_mean", "mean"),
    ).reset_index().sort_values(["dataset", "acc"], ascending=[True, False])

    out = section("Exp 9: Spatial BPE (mean over seeds)")
    lines = [
        "\\begin{tabular}{@{}lrr@{}}",
        "  \\toprule",
        "  Dataset / Classifier & Acc & $\\kappa$ \\\\",
        "  \\midrule",
    ]
    for ds in DATASET_ORDER:
        sub = grp[grp["dataset"] == ds]
        if sub.empty:
            continue
        lines.append(f"  \\multicolumn{{3}}{{l}}{{\\textit{{{DATASET_LABELS.get(ds,ds)}}}}}\\ \\\\")
        for _, row in sub.iterrows():
            chance = CHANCE.get(ds, 0.25)
            acc_str = color_cell(row["acc"], chance, pct(row["acc"]))
            lines.append(f"  \\quad {row['classifier']} & {acc_str} & {kap(row['kappa'])} \\\\")
        lines.append("  \\midrule")
    lines += ["  \\bottomrule", "\\end{tabular}"]
    out += "\n" + "\n".join(lines)
    return out


# ─── Ablations summary ────────────────────────────────────────────────────────

def table_ablations(logs_dir: Path) -> str:
    out = section("Ablation Studies Summary")

    ablation_files = {
        "A1": "ablation_A1_quantization_results.csv",
        "A2": "ablation_A2_bins_results.csv",
        "A3": "ablation_A3_cross_condition_results.csv",
        "A4": "ablation_A4_spatial_results.csv",
        "A5": "ablation_A5_channel_importance_results.csv",
        "A7": "ablation_A7_bpe_vs_nobpe_results.csv",
        "A8": "ablation_A8_results.csv",
        "A9": "ablation_A9_grid_results.csv",
    }

    for abl, fname in ablation_files.items():
        fp = logs_dir / fname
        if not fp.exists():
            out += f"\n% {abl}: {fname} not found\n"
            continue
        try:
            df = pd.read_csv(fp, on_bad_lines="skip", engine="python")
            out += f"\n% {abl} ({len(df)} rows, columns: {list(df.columns)[:6]})\n"

            # Best result per dataset
            if "accuracy_mean" in df.columns and "dataset" in df.columns:
                grp = df.groupby("dataset")["accuracy_mean"].max().reset_index()
                for _, row in grp.iterrows():
                    out += f"%   {row['dataset']}: best acc = {row['accuracy_mean']*100:.1f}%\n"
        except Exception as e:
            out += f"\n% {abl}: error reading {fname}: {e}\n"

    return out


# ─── Markdown summary ─────────────────────────────────────────────────────────

def markdown_summary(logs_dir: Path) -> str:
    lines = ["# Full-Run Results Summary", ""]

    # Exp 2
    fp = logs_dir / "exp2_downstream_results.csv"
    if fp.exists():
        df = pd.read_csv(fp, on_bad_lines="skip", engine="python")
        grp = df.groupby(["dataset", "classifier"])["accuracy_mean"].mean().reset_index()
        lines.append("## Exp 2: Best per dataset (full run, V=4096)")
        for ds in DATASET_ORDER:
            sub = grp[grp["dataset"] == ds].sort_values("accuracy_mean", ascending=False)
            if sub.empty:
                continue
            top3 = sub.head(3)
            lines.append(f"### {DATASET_LABELS.get(ds, ds)} (chance {CHANCE.get(ds,0.25)*100:.0f}%)")
            for _, r in top3.iterrows():
                marker = " ← **BPE**" if r["classifier"].startswith("BPE") else ""
                lines.append(f"- {r['classifier']}: {r['accuracy_mean']*100:.1f}%{marker}")
        lines.append("")

    # Exp 5
    fp = logs_dir / "exp5_scaling_results.csv"
    if fp.exists():
        df = pd.read_csv(fp, on_bad_lines="skip", engine="python")
        lines.append("## Exp 5: Optimal V* per dataset")
        for ds in DATASET_ORDER:
            sub = df[df["dataset"] == ds].sort_values("accuracy_mean", ascending=False)
            if sub.empty:
                continue
            best = sub.iloc[0]
            lines.append(f"- {DATASET_LABELS.get(ds,ds)}: V*={int(best.vocab_size)}, acc={best.accuracy_mean*100:.1f}%")
        lines.append("")

    # Other exps
    for exp, fname, label in [
        ("6", "exp6_alt_tokenization_results.csv", "Exp 6: Alt Tokenisation"),
        ("7", "exp7_fourier_bpe_results.csv",       "Exp 7: Fourier-BPE"),
        ("8", "exp8_csp_bpe_results.csv",            "Exp 8: CSP+BPE"),
        ("9", "exp9_spatial_bpe_results.csv",        "Exp 9: Spatial BPE"),
    ]:
        fp = logs_dir / fname
        if not fp.exists():
            lines.append(f"## {label}: *pending*")
            lines.append("")
            continue
        df = pd.read_csv(fp, on_bad_lines="skip", engine="python")
        lines.append(f"## {label} ({len(df)} rows)")
        if "accuracy_mean" in df.columns and "dataset" in df.columns:
            key_col = "classifier" if "classifier" in df.columns else "approach"
            if key_col in df.columns:
                grp = df.groupby(["dataset", key_col])["accuracy_mean"].mean().reset_index()
                for ds in DATASET_ORDER:
                    sub = grp[grp["dataset"] == ds].sort_values("accuracy_mean", ascending=False)
                    if sub.empty:
                        continue
                    best = sub.iloc[0]
                    lines.append(
                        f"- {DATASET_LABELS.get(ds,ds)}: best {best[key_col]} "
                        f"= {best['accuracy_mean']*100:.1f}%"
                    )
        lines.append("")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables from experiment CSVs")
    parser.add_argument("--exp", nargs="+", default=["2", "5", "6", "7", "8", "9", "ablations"],
                        help="Which experiments to include")
    parser.add_argument("--output", default=str(OUT_DEFAULT),
                        help="Output .tex file path")
    parser.add_argument("--markdown", action="store_true",
                        help="Also write markdown summary")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = [
        "% Auto-generated by scripts/generate_results_tables.py",
        "% Run: python scripts/generate_results_tables.py",
        f"% Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    def try_load(fname):
        fp = LOGS / fname
        if not fp.exists():
            print(f"  [skip] {fname} not found", file=sys.stderr)
            return None
        df = pd.read_csv(fp, on_bad_lines="skip", engine="python")
        print(f"  [load] {fname}: {len(df)} rows", file=sys.stderr)
        return df

    if "2" in args.exp:
        df = try_load("exp2_downstream_results.csv")
        if df is not None:
            sections.append(table_exp2(df))

    if "5" in args.exp:
        df = try_load("exp5_scaling_results.csv")
        if df is not None:
            sections.append(table_exp5(df))

    if "6" in args.exp:
        df = try_load("exp6_alt_tokenization_results.csv")
        if df is not None:
            sections.append(table_exp6(df))

    if "7" in args.exp:
        df = try_load("exp7_fourier_bpe_results.csv")
        if df is not None:
            sections.append(table_exp7(df))

    if "8" in args.exp:
        df = try_load("exp8_csp_bpe_results.csv")
        if df is not None:
            sections.append(table_exp8(df))

    if "9" in args.exp:
        df = try_load("exp9_spatial_bpe_results.csv")
        if df is not None:
            sections.append(table_exp9(df))

    if "ablations" in args.exp:
        sections.append(table_ablations(LOGS))

    tex_out = "\n\n".join(sections)
    out_path.write_text(tex_out, encoding="utf-8")
    print(f"LaTeX tables written to: {out_path}", file=sys.stderr)

    if args.markdown:
        md_path = out_path.with_suffix(".md")
        md_path.write_text(markdown_summary(LOGS), encoding="utf-8")
        print(f"Markdown summary written to: {md_path}", file=sys.stderr)

    # Always print markdown to stdout
    print(markdown_summary(LOGS))


if __name__ == "__main__":
    main()
