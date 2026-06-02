from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from classifier import _eval_with_model, _fit_model, split_samples
from preprocess import add_normalized_views
from hdfs_loader import _load_hdfs_samples
from run_hdfs_main import (
    ABNORMAL_RE,
    ENTITY_RE,
    NOISE_RE,
    _add_gaps,
    _budget,
    _build_texts,
    _failure_score,
    _fit_log_odds,
    _keyword_scores,
    _log_odds_scores,
    _mean_ratio,
    _prediction_change,
    _sample_random_count,
    _sample_random_span,
    _select_anchor_first_indices,
    _select_rank_package_indices,
    _select_top_indices,
    _severity_scores,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HDFS budget=0.30 anchor-first ablation.")
    p.add_argument("--hdfs-root", default="data/HDFS")
    p.add_argument("--output", default="runs")
    p.add_argument("--writing-assets", default="paper_tables")
    p.add_argument("--seeds", default="13,21,42,87,123")
    p.add_argument("--budget-ratio", type=float, default=0.30)
    p.add_argument("--balanced-sample-per-class", type=int, default=20000)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--max-budget-lines", type=int, default=512)
    return p


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _make_out_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"hdfs_ablation_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _score_variant(lines: Sequence[str], templates: Sequence[str], variant: str, context_window: int) -> List[float]:
    tpl_counts = Counter(str(t) for t in templates)
    kw = _keyword_scores(lines)
    sev = _severity_scores(lines)
    scores: List[float] = []

    for i, line in enumerate(lines):
        lo = max(0, i - context_window)
        hi = min(len(lines), i + context_window + 1)
        context_signal = sum(1 for v in kw[lo:hi] if v > 0) * 0.5
        context_support = min(context_signal, 0.5)
        entities = set()
        for j in range(lo, hi):
            entities.update(m.group(0).lower() for m in ENTITY_RE.finditer(str(lines[j])))
        entity_score = min(3.0, 0.5 * len(entities))

        raw_burst = math.log1p(tpl_counts[str(templates[i])])
        if variant == "without_burst_cap":
            burst = raw_burst
        else:
            burst = min(raw_burst, 0.5) if (ABNORMAL_RE.search(str(line)) or sev[i] > 0 or entity_score > 0) else 0.0

        severity_coef = 0.4 if variant == "without_severity_boost" else 2.0
        failure_coef = 0.0 if variant == "without_failure_boost" else 1.5
        noise_coef = 0.0 if variant == "without_noise_penalty" else 1.5
        noise_penalty = 1.0 if NOISE_RE.search(str(line)) else 0.0

        score = (
            1.0 * kw[i]
            + severity_coef * sev[i]
            + failure_coef * _failure_score(str(line))
            + 0.7 * entity_score
            + 0.2 * burst
            + 0.2 * context_support
            - noise_coef * noise_penalty
        )
        scores.append(float(score))
    return scores


def _select_variant(
    method: str,
    lines: Sequence[str],
    templates: Sequence[str],
    k: int,
    log_odds: Dict[str, float],
) -> List[int]:
    if method == "random_same_count" or method == "random_same_span":
        raise ValueError("random handled outside")
    if method == "tfidf_log_odds":
        return _select_top_indices(_log_odds_scores(lines, log_odds), k)

    context_window = 1
    variant = method
    if method == "anchor_first_full":
        variant = "anchor_first_full"
    elif method == "without_anchor_first_allocation":
        variant = "anchor_first_full"
    elif method == "context_window_0":
        variant = "anchor_first_full"
        context_window = 0
    elif method == "context_window_2":
        variant = "anchor_first_full"
        context_window = 2

    scores = _score_variant(lines, templates, variant, context_window)
    if method == "without_anchor_first_allocation":
        return _select_rank_package_indices(lines, scores, k, radius=context_window)
    return _select_anchor_first_indices(lines, scores, k, radius=context_window)


def _selected_line_stats(df: pd.DataFrame, selected: List[List[int]]) -> Dict[str, float]:
    signal_flags: List[bool] = []
    noise_flags: List[bool] = []
    for (_, row), indices in zip(df.iterrows(), selected):
        lines = list(row["normalized_lines"])
        kw = _keyword_scores(lines)
        sev = _severity_scores(lines)
        for idx in indices:
            if idx < 0 or idx >= len(lines):
                continue
            line = str(lines[idx])
            signal_flags.append(bool(kw[idx] > 0 or sev[idx] > 0))
            noise_flags.append(bool(NOISE_RE.search(line)))
    denom = max(1, len(signal_flags))
    return {
        "signal_line_ratio": float(sum(signal_flags) / denom),
        "noise_like_ratio": float(sum(noise_flags) / denom),
    }


def _stable_method_offset(method: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(method)) % 10000


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    order = [
        "anchor_first_full",
        "without_anchor_first_allocation",
        "without_noise_penalty",
        "without_burst_cap",
        "without_severity_boost",
        "without_failure_boost",
        "context_window_0",
        "context_window_2",
        "tfidf_log_odds",
    ]
    labels = {
        "anchor_first_full": "Anchor-first full",
        "without_anchor_first_allocation": "w/o anchor-first allocation",
        "without_noise_penalty": "w/o noise penalty",
        "without_burst_cap": "w/o burst cap",
        "without_severity_boost": "w/o severity boost",
        "without_failure_boost": "w/o failure cue boost",
        "context_window_0": "context window = 0",
        "context_window_2": "context window = 2",
        "tfidf_log_odds": "TF-IDF/log-odds",
    }
    out = summary[summary["method"].isin(order)].copy()
    out["order"] = out["method"].map({m: i for i, m in enumerate(order)})
    out = out.sort_values("order")
    rows = []
    for _, r in out.iterrows():
        rows.append(
            {
                "Variant": labels.get(r["method"], r["method"]),
                "Necessity Drop": f"{r['necessity_f1_drop_mean']:.4f} +/- {r['necessity_f1_drop_std']:.4f}",
                "Gap vs Random Count": f"{r['necessity_gap_vs_random_count_mean']:.4f}",
                "Noise-like Ratio": f"{r['noise_like_ratio_mean']:.4f}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    out_dir = _make_out_dir(Path(args.output))
    writing_assets = Path(args.writing_assets)
    writing_assets.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(args.seeds)

    samples = _load_hdfs_samples(Path(args.hdfs_root), per_class=args.balanced_sample_per_class, sample_seed=args.sample_seed)
    samples = add_normalized_views(samples)
    methods = [
        "anchor_first_full",
        "without_anchor_first_allocation",
        "without_noise_penalty",
        "without_burst_cap",
        "without_severity_boost",
        "without_failure_boost",
        "context_window_0",
        "context_window_2",
        "tfidf_log_odds",
        "random_same_count",
        "random_same_span",
    ]

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        split = split_samples(samples, test_size=args.test_size, seed=seed)
        y_train = split.train_df["fault_type"].astype(str).tolist()
        y_test = split.test_df["fault_type"].astype(str).tolist()
        vec, clf, fit_meta = _fit_model(split.train_df["normalized_sequence"].astype(str).tolist(), y_train, seed=seed)
        full_eval = _eval_with_model(vec, clf, split.test_df["normalized_sequence"].astype(str).tolist(), y_test)
        log_odds = _fit_log_odds(split.train_df)

        for method in methods:
            selected: List[List[int]] = []
            counts: List[int] = []
            totals: List[int] = []
            for pos, (_, row) in enumerate(split.test_df.iterrows()):
                lines = list(row["normalized_lines"])
                templates = list(row["template_lines"])
                k = _budget(len(lines), args.budget_ratio, args.max_budget_lines)
                if method == "random_same_count":
                    import random

                    rng = random.Random(seed + int(args.budget_ratio * 10000) + _stable_method_offset(method) + pos)
                    idx = _sample_random_count(len(lines), k, rng)
                elif method == "random_same_span":
                    import random

                    rng = random.Random(seed + int(args.budget_ratio * 10000) + _stable_method_offset(method) + pos)
                    idx = _sample_random_span(len(lines), k, rng)
                else:
                    idx = _select_variant(method, lines, templates, k, log_odds)
                selected.append(idx)
                counts.append(len(idx))
                totals.append(len(lines))
            top_texts, removed_texts, _, _ = _build_texts(split.test_df, selected)
            top_eval = _eval_with_model(vec, clf, top_texts, y_test)
            rem_eval = _eval_with_model(vec, clf, removed_texts, y_test)
            stats = _selected_line_stats(split.test_df, selected)
            rows.append(
                {
                    "dataset": "hdfs_full_preprocessed",
                    "seed": seed,
                    "budget_ratio": args.budget_ratio,
                    "method": method,
                    "full_on_full_macro_f1": float(full_eval["macro_f1"]),
                    "full_on_topk_macro_f1": float(top_eval["macro_f1"]),
                    "full_on_removed_topk_macro_f1": float(rem_eval["macro_f1"]),
                    "necessity_f1_drop": float(full_eval["macro_f1"] - rem_eval["macro_f1"]),
                    "prediction_change_rate_removed": _prediction_change(full_eval["pred"], rem_eval["pred"]),
                    "actual_budget_lines_mean": float(sum(counts) / len(counts)) if counts else 0.0,
                    "actual_budget_ratio_mean": _mean_ratio(counts, totals),
                    "signal_line_ratio": stats["signal_line_ratio"],
                    "noise_like_ratio": stats["noise_like_ratio"],
                    "num_test_samples": int(len(split.test_df)),
                    "classifier_selected_C": fit_meta.get("selected_C"),
                }
            )

    raw = _add_gaps(pd.DataFrame(rows))
    raw.to_csv(out_dir / "01_raw_records.csv", index=False)
    summary = (
        raw.groupby(["method", "budget_ratio"])
        .agg(
            necessity_f1_drop_mean=("necessity_f1_drop", "mean"),
            necessity_f1_drop_std=("necessity_f1_drop", "std"),
            necessity_gap_vs_random_count_mean=("necessity_gap_vs_random_count", "mean"),
            necessity_gap_vs_random_span_mean=("necessity_gap_vs_random_span", "mean"),
            prediction_change_rate_removed_mean=("prediction_change_rate_removed", "mean"),
            noise_like_ratio_mean=("noise_like_ratio", "mean"),
            signal_line_ratio_mean=("signal_line_ratio", "mean"),
            actual_budget_lines_mean=("actual_budget_lines_mean", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "02_summary_budget0p30.csv", index=False)
    paper = _paper_table(summary)
    paper.to_csv(out_dir / "03_ablation_table.csv", index=False)
    (out_dir / "03_ablation_table.md").write_text(paper.to_markdown(index=False), encoding="utf-8")
    paper.to_csv(writing_assets / "table7_hdfs_ablation.csv", index=False)
    (writing_assets / "table7_hdfs_ablation.md").write_text(paper.to_markdown(index=False), encoding="utf-8")

    config = {
        "dataset": "HDFS full preprocessed block/session level",
        "role": "budget=0.30 design ablation",
        "budget_ratio": args.budget_ratio,
        "seeds": seeds,
        "balanced_sample_per_class": args.balanced_sample_per_class,
        "main_method": "anchor_first_full",
        "methods": methods,
        "scope_guard": {
            "single_budget_only": True,
            "does_not_change_frozen_main_method": True,
            "uses_frozen_verifier_protocol": True,
        },
        "outputs": {
            "raw_records": str(out_dir / "01_raw_records.csv"),
            "summary": str(out_dir / "02_summary_budget0p30.csv"),
            "table": str(out_dir / "03_ablation_table.csv"),
        },
    }
    (out_dir / "00_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    notes = [
        "# HDFS Ablation Notes",
        "",
        "Role: design ablation at budget=0.30. This does not change the frozen main method.",
        "",
        paper.to_markdown(index=False),
    ]
    (out_dir / "04_notes.md").write_text("\n".join(notes), encoding="utf-8")

    print("=" * 60)
    print("HDFS Ablation Completed")
    print("=" * 60)
    print(f"Out dir: {out_dir}")
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
