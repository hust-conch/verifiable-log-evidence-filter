from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from scipy.stats import wilcoxon


METHOD_LABELS = {
    "rank_v0_anchor_first": "Anchor-first",
    "rank_v0_original": "Original RANK",
    "severity_only": "Severity-only",
    "keyword_only": "Keyword-only",
    "random_same_count": "Random-count",
    "random_same_span": "Random-span",
    "template_rarity": "Template-rarity",
    "tfidf_log_odds": "TF-IDF/log-odds",
    "single_line": "Single-line",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate paper tables from HDFS experiment outputs.")
    parser.add_argument("--runs-root", default="runs", help="Root directory containing timestamped run folders.")
    parser.add_argument("--hdfs-main-dir", default=None, help="Directory from run_hdfs_main.py.")
    parser.add_argument("--ablation-dir", default=None, help="Directory from run_ablation.py.")
    parser.add_argument("--verifier-dir", default=None, help="Directory from run_verifier_robustness.py.")
    parser.add_argument("--output", default="paper_tables", help="Output directory for regenerated tables.")
    return parser


def _latest(root: Path, prefix: str) -> Path:
    candidates = sorted([p for p in root.glob(f"{prefix}*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directory matching {prefix}* under {root}")
    return candidates[-1]


def _fmt_mean_std(mean: float, std: Optional[float], digits: int = 4) -> str:
    if std is None or pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def _write(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.md").write_text(df.to_markdown(index=False), encoding="utf-8")


def make_table_iii(summary: pd.DataFrame) -> pd.DataFrame:
    order = [
        "rank_v0_anchor_first",
        "template_rarity",
        "rank_v0_original",
        "severity_only",
        "keyword_only",
        "random_same_count",
        "random_same_span",
        "tfidf_log_odds",
    ]
    sub = summary[summary["budget_ratio"].round(2) == 0.30].copy()
    rows = []
    for method in order:
        row = sub[sub["method"] == method]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "Method": METHOD_LABELS.get(method, method),
                "Selected-only F1": f"{r['full_on_topk_macro_f1_mean']:.3f}",
                "Evidence-removed F1": f"{r['full_on_removed_topk_macro_f1_mean']:.3f}",
                "Necessity Drop": f"{r['necessity_f1_drop_mean']:.3f}",
                "Gap vs Count": f"{r['necessity_gap_vs_random_count_mean']:.3f}",
                "Gap vs Span": f"{r['necessity_gap_vs_random_span_mean']:.3f}",
                "Noise-like Ratio": f"{r['noise_like_ratio_mean']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def make_table_v(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for budget in sorted(summary["budget_ratio"].unique()):
        def pick(method: str):
            row = summary[(summary["budget_ratio"] == budget) & (summary["method"] == method)]
            return row.iloc[0] if not row.empty else None

        anchor = pick("rank_v0_anchor_first")
        template = pick("template_rarity")
        original = pick("rank_v0_original")
        severity = pick("severity_only")
        if anchor is None:
            continue
        rows.append(
            {
                "Budget": f"{budget:.2f}",
                "Anchor Drop": _fmt_mean_std(anchor["necessity_f1_drop_mean"], anchor["necessity_f1_drop_std"]),
                "Template Drop": _fmt_mean_std(template["necessity_f1_drop_mean"], template["necessity_f1_drop_std"]) if template is not None else "",
                "Original Drop": _fmt_mean_std(original["necessity_f1_drop_mean"], original["necessity_f1_drop_std"]) if original is not None else "",
                "Severity Drop": _fmt_mean_std(severity["necessity_f1_drop_mean"], severity["necessity_f1_drop_std"]) if severity is not None else "",
                "Gap Count": f"{anchor['necessity_gap_vs_random_count_mean']:.4f}",
                "Gap Span": f"{anchor['necessity_gap_vs_random_span_mean']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def make_table_vii(raw: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("random_same_count", "vs. Random-count"),
        ("random_same_span", "vs. Random-span"),
        ("severity_only", "vs. Severity-only"),
        ("rank_v0_original", "vs. Original RANK"),
        ("template_rarity", "vs. Template-rarity"),
        ("tfidf_log_odds", "vs. TF-IDF/log-odds"),
    ]
    anchor = raw[raw["method"] == "rank_v0_anchor_first"][
        ["seed", "budget_ratio", "necessity_f1_drop"]
    ].rename(columns={"necessity_f1_drop": "anchor_drop"})
    rows = []
    for method, label in comparisons:
        other = raw[raw["method"] == method][["seed", "budget_ratio", "necessity_f1_drop"]].rename(
            columns={"necessity_f1_drop": "other_drop"}
        )
        paired = anchor.merge(other, on=["seed", "budget_ratio"], how="inner")
        diff = paired["anchor_drop"] - paired["other_drop"]
        wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        losses = int((diff < 0).sum())
        p_value = 1.0
        if len(diff) > 0 and (diff != 0).any():
            alternative = "greater"
            try:
                p_value = float(wilcoxon(diff, alternative=alternative, zero_method="wilcox").pvalue)
            except ValueError:
                p_value = 1.0
        rows.append(
            {
                "Comparison": label,
                "Win/Tie/Loss": f"{wins}/{ties}/{losses}",
                "Mean Diff.": f"{float(diff.mean()):.4f}",
                "Wilcoxon p": f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.2f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    hdfs_main_dir = Path(args.hdfs_main_dir) if args.hdfs_main_dir else _latest(runs_root, "phase4_hdfs_main_")
    ablation_dir = Path(args.ablation_dir) if args.ablation_dir else _latest(runs_root, "hdfs_ablation_")
    verifier_dir = Path(args.verifier_dir) if args.verifier_dir else _latest(runs_root, "hdfs_verifier_robustness_")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_summary = pd.read_csv(hdfs_main_dir / "03_summary_mean_std_by_method_budget.csv")
    main_raw = pd.read_csv(hdfs_main_dir / "02_raw_records.csv")
    table_vi = pd.read_csv(ablation_dir / "03_ablation_table.csv")
    table_viii = pd.read_csv(verifier_dir / "03_verifier_robustness_table.csv")

    _write(make_table_iii(main_summary), out_dir, "table_iii_hdfs_main_budget0p30")
    _write(make_table_v(main_summary), out_dir, "table_v_hdfs_multi_budget")
    _write(table_vi, out_dir, "table_vi_ablation")
    _write(make_table_vii(main_raw), out_dir, "table_vii_paired_comparison")
    _write(table_viii, out_dir, "table_viii_verifier_robustness")

    manifest = pd.DataFrame(
        [
            {"Table": "III", "Source": str(hdfs_main_dir / "03_summary_mean_std_by_method_budget.csv")},
            {"Table": "V", "Source": str(hdfs_main_dir / "03_summary_mean_std_by_method_budget.csv")},
            {"Table": "VI", "Source": str(ablation_dir / "03_ablation_table.csv")},
            {"Table": "VII", "Source": str(hdfs_main_dir / "02_raw_records.csv")},
            {"Table": "VIII", "Source": str(verifier_dir / "03_verifier_robustness_table.csv")},
        ]
    )
    _write(manifest, out_dir, "manifest")
    print(f"Tables written to {out_dir}")


if __name__ == "__main__":
    main()
