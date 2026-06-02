from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from classifier import _eval_with_model, _fit_model, split_samples
from preprocess import add_normalized_views
from hdfs_loader import _load_hdfs_samples


KEYWORDS = {
    "exception": 3.0,
    "failed": 3.0,
    "failure": 3.0,
    "error": 2.0,
    "timeout": 2.5,
    "refused": 2.5,
    "unreachable": 3.0,
    "fserror": 4.0,
    "could not": 2.5,
}
SEVERITY = {"fatal": 6.0, "critical": 6.0, "severe": 5.0, "error": 4.0, "warn": 2.0, "warning": 2.0}
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.$-]*")
ENTITY_RE = re.compile(r"\b(?:blk_-?\d+|E\d+|DataNode|FSNamesystem|NameSystem|DFSClient)\b", re.I)
NOISE_RE = re.compile(r"receiving block|verification succeeded|packetresponder|blockmap updated|served block", re.I)
ABNORMAL_RE = re.compile(r"exception|failed|failure|error|timeout|refused|unreachable|fserror|could not", re.I)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 4 HDFS evidence validity experiment.")
    p.add_argument("--hdfs-root", default="data/HDFS")
    p.add_argument("--output", default="runs")
    p.add_argument("--mode", choices=["smoke", "main"], default="smoke")
    p.add_argument("--seeds", default="13")
    p.add_argument("--budget-ratios", default="0.30")
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--balanced-sample-per-class", type=int, default=3000)
    p.add_argument("--smoke-sample-per-class", type=int, default=300)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--context-window", type=int, default=1)
    p.add_argument("--max-budget-lines", type=int, default=512)
    return p


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def _make_out_dir(base: Path, mode: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"phase4_hdfs_{mode}_{ts}"
    (out / "01_splits").mkdir(parents=True, exist_ok=False)
    (out / "plots").mkdir(parents=True, exist_ok=False)
    (out / "case_candidates").mkdir(parents=True, exist_ok=False)
    return out


def _budget(n_lines: int, ratio: float, max_budget_lines: int) -> int:
    if n_lines <= 0:
        return 0
    k = max(1, int(math.ceil(n_lines * float(ratio))))
    if max_budget_lines > 0:
        k = min(k, int(max_budget_lines))
    return min(n_lines, k)


def _tokens(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(str(text))]


def _fit_log_odds(train_df: pd.DataFrame) -> Dict[str, float]:
    pos: Counter[str] = Counter()
    neg: Counter[str] = Counter()
    vocab = set()
    for _, row in train_df.iterrows():
        is_pos = str(row["fault_type"]) != "normal"
        for line in row["normalized_lines"]:
            toks = set(_tokens(line))
            vocab.update(toks)
            if is_pos:
                pos.update(toks)
            else:
                neg.update(toks)
    v = max(1, len(vocab))
    pos_total = sum(pos.values())
    neg_total = sum(neg.values())
    weights: Dict[str, float] = {}
    for tok in vocab:
        pos_rate = (pos[tok] + 1.0) / (pos_total + v)
        neg_rate = (neg[tok] + 1.0) / (neg_total + v)
        weights[tok] = math.log(pos_rate / neg_rate)
    return weights


def _fit_template_rarity(train_df: pd.DataFrame) -> Tuple[Dict[str, int], int, int]:
    """Fit label-free template frequencies on the training split only."""
    counts: Counter[str] = Counter()
    total = 0
    for _, row in train_df.iterrows():
        for tpl in row["template_lines"]:
            counts[str(tpl)] += 1
            total += 1
    return dict(counts), total, max(1, len(counts))


def _keyword_scores(lines: Sequence[str]) -> List[float]:
    out = []
    for line in lines:
        s = str(line).lower()
        out.append(sum(w for kw, w in KEYWORDS.items() if kw in s))
    return out


def _severity_scores(lines: Sequence[str]) -> List[float]:
    out = []
    for line in lines:
        s = str(line).lower()
        out.append(max([w for level, w in SEVERITY.items() if re.search(rf"\b{level}\b", s)] or [0.0]))
    return out


def _log_odds_scores(lines: Sequence[str], weights: Dict[str, float]) -> List[float]:
    scores = []
    for line in lines:
        vals = sorted((weights.get(tok, 0.0) for tok in set(_tokens(line))), reverse=True)
        scores.append(float(sum(v for v in vals[:8] if v > 0)))
    return scores


def _template_rarity_scores(
    templates: Sequence[str],
    template_counts: Dict[str, int],
    total_templates: int,
    vocab_size: int,
) -> List[float]:
    denom = max(1, int(total_templates) + int(vocab_size))
    return [float(-math.log((template_counts.get(str(tpl), 0) + 1.0) / denom)) for tpl in templates]


def _rank_v0_scores(lines: Sequence[str], templates: Sequence[str], context_window: int) -> List[float]:
    tpl_counts = Counter(str(t) for t in templates)
    kw = _keyword_scores(lines)
    sev = _severity_scores(lines)
    scores: List[float] = []
    for i, line in enumerate(lines):
        lo = max(0, i - context_window)
        hi = min(len(lines), i + context_window + 1)
        context_kw = sum(1 for v in kw[lo:hi] if v > 0) * 0.5
        entities = set()
        for j in range(lo, hi):
            entities.update(m.group(0).lower() for m in ENTITY_RE.finditer(str(lines[j])))
        entity_score = min(3.0, 0.5 * len(entities))
        burst = math.log1p(tpl_counts[str(templates[i])])
        scores.append(float(kw[i] + 0.4 * sev[i] + burst + entity_score + context_kw))
    return scores


def _failure_score(line: str) -> float:
    low = str(line).lower()
    score = 0.0
    if "fail" in low:
        score += 1.0
    if "exception" in low:
        score += 0.8
    if "io" in low or "could not" in low:
        score += 0.8
    if "refused" in low or "unreachable" in low:
        score += 0.6
    return score


def _rank_v0_anchor_scores(lines: Sequence[str], templates: Sequence[str], context_window: int) -> List[float]:
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
        anchor_has_abnormal = bool(ABNORMAL_RE.search(str(line)))
        raw_burst = math.log1p(tpl_counts[str(templates[i])])
        template_burst = min(raw_burst, 0.5) if (anchor_has_abnormal or sev[i] > 0 or entity_score > 0) else 0.0
        noise_penalty = 1.0 if NOISE_RE.search(str(line)) else 0.0
        scores.append(
            float(
                1.0 * kw[i]
                + 2.0 * sev[i]
                + 1.5 * _failure_score(str(line))
                + 0.7 * entity_score
                + 0.2 * template_burst
                + 0.2 * context_support
                - 1.5 * noise_penalty
            )
        )
    return scores


def _select_top_indices(scores: Sequence[float], k: int) -> List[int]:
    if k <= 0:
        return []
    ranked = sorted(range(len(scores)), key=lambda i: (float(scores[i]), -i), reverse=True)
    selected = [i for i in ranked if float(scores[i]) > 0][:k]
    if len(selected) < k:
        seen = set(selected)
        for i in ranked:
            if i in seen:
                continue
            selected.append(i)
            seen.add(i)
            if len(selected) >= k:
                break
    return sorted(selected)


def _select_rank_package_indices(lines: Sequence[str], scores: Sequence[float], k: int, radius: int) -> List[int]:
    ranked = sorted(range(len(scores)), key=lambda i: (float(scores[i]), -i), reverse=True)
    selected = set()
    for anchor in ranked:
        if float(scores[anchor]) <= 0 and selected:
            continue
        lo = max(0, anchor - radius)
        hi = min(len(lines), anchor + radius + 1)
        for idx in range(lo, hi):
            if len(selected) >= k:
                break
            selected.add(idx)
        if len(selected) >= k:
            break
    if not selected:
        return _select_top_indices(scores, k)
    return sorted(selected)


def _select_anchor_first_indices(lines: Sequence[str], scores: Sequence[float], k: int, radius: int) -> List[int]:
    anchors = _select_top_indices(scores, k)
    selected = list(anchors)
    seen = set(selected)
    for anchor in anchors:
        if len(selected) >= k:
            break
        for idx in range(max(0, anchor - radius), min(len(lines), anchor + radius + 1)):
            if idx in seen:
                continue
            selected.append(idx)
            seen.add(idx)
            if len(selected) >= k:
                break
    return sorted(selected[:k])


def _sample_random_count(n_lines: int, count: int, rng: random.Random) -> List[int]:
    if n_lines <= 0 or count <= 0:
        return []
    return sorted(rng.sample(range(n_lines), min(n_lines, count)))


def _sample_random_span(n_lines: int, count: int, rng: random.Random) -> List[int]:
    if n_lines <= 0 or count <= 0:
        return []
    count = min(n_lines, count)
    start = 0 if count >= n_lines else rng.randint(0, n_lines - count)
    return list(range(start, start + count))


def _build_texts(df: pd.DataFrame, selected: List[List[int]]) -> Tuple[List[str], List[str], List[int], List[int]]:
    top_texts: List[str] = []
    removed_texts: List[str] = []
    counts: List[int] = []
    n_lines_list: List[int] = []
    for (_, row), indices in zip(df.iterrows(), selected):
        lines = list(row["normalized_lines"])
        keep = set(indices)
        top = [line for i, line in enumerate(lines) if i in keep]
        rem = [line for i, line in enumerate(lines) if i not in keep]
        top_texts.append("\n".join(top))
        removed_texts.append("\n".join(rem))
        counts.append(len(keep))
        n_lines_list.append(len(lines))
    return top_texts, removed_texts, counts, n_lines_list


def _select_for_method(
    df: pd.DataFrame,
    method: str,
    ratio: float,
    seed: int,
    max_budget_lines: int,
    context_window: int,
    log_odds_weights: Dict[str, float],
    template_rarity_model: Tuple[Dict[str, int], int, int],
) -> List[List[int]]:
    rng = random.Random(seed + int(ratio * 10000) + _stable_method_offset(method))
    selected: List[List[int]] = []
    for _, row in df.iterrows():
        lines = list(row["normalized_lines"])
        templates = list(row["template_lines"])
        k = _budget(len(lines), ratio, max_budget_lines)
        if method in {"rank_v0_package", "rank_v0_original"}:
            scores = _rank_v0_scores(lines, templates, context_window=context_window)
            idx = _select_rank_package_indices(lines, scores, k, radius=context_window)
        elif method == "rank_v0_anchor_first":
            scores = _rank_v0_anchor_scores(lines, templates, context_window=context_window)
            idx = _select_anchor_first_indices(lines, scores, k, radius=context_window)
        elif method == "single_line":
            scores = _keyword_scores(lines)
            idx = _select_top_indices(scores, k)
        elif method == "keyword_only":
            idx = _select_top_indices(_keyword_scores(lines), k)
        elif method == "severity_only":
            idx = _select_top_indices(_severity_scores(lines), k)
        elif method == "tfidf_log_odds":
            idx = _select_top_indices(_log_odds_scores(lines, log_odds_weights), k)
        elif method == "template_rarity":
            tpl_counts, tpl_total, tpl_vocab = template_rarity_model
            idx = _select_top_indices(_template_rarity_scores(templates, tpl_counts, tpl_total, tpl_vocab), k)
        elif method == "random_same_count":
            idx = _sample_random_count(len(lines), k, rng)
        elif method == "random_same_span":
            idx = _sample_random_span(len(lines), k, rng)
        else:
            raise ValueError(f"Unknown method: {method}")
        selected.append(idx)
    return selected


def _selected_line_stats(df: pd.DataFrame, selected: List[List[int]]) -> Dict[str, float]:
    signal_flags: List[bool] = []
    noise_flags: List[bool] = []
    entity_flags: List[bool] = []
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
            entity_flags.append(bool(ENTITY_RE.search(line)))
    denom = max(1, len(signal_flags))
    return {
        "signal_line_ratio": float(sum(signal_flags) / denom),
        "noise_like_ratio": float(sum(noise_flags) / denom),
        "entity_line_ratio": float(sum(entity_flags) / denom),
    }


def _stable_method_offset(method: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(method)) % 10000


def _prediction_change(a: Sequence[str], b: Sequence[str]) -> float:
    n = max(1, len(a))
    return sum(x != y for x, y in zip(a, b)) / n


def _mean_ratio(counts: Sequence[int], totals: Sequence[int]) -> float:
    vals = [(c / n) for c, n in zip(counts, totals) if n > 0]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["necessity_gap_vs_random_count"] = 0.0
    out["necessity_gap_vs_random_span"] = 0.0
    key_cols = ["seed", "budget_ratio"]
    rc = out[out["method"] == "random_same_count"][key_cols + ["necessity_f1_drop"]].rename(
        columns={"necessity_f1_drop": "rc_drop"}
    )
    rs = out[out["method"] == "random_same_span"][key_cols + ["necessity_f1_drop"]].rename(
        columns={"necessity_f1_drop": "rs_drop"}
    )
    out = out.merge(rc, on=key_cols, how="left").merge(rs, on=key_cols, how="left")
    out["necessity_gap_vs_random_count"] = out["necessity_f1_drop"] - out["rc_drop"].fillna(0.0)
    out["necessity_gap_vs_random_span"] = out["necessity_f1_drop"] - out["rs_drop"].fillna(0.0)
    return out.drop(columns=["rc_drop", "rs_drop"])


def _plot(summary: pd.DataFrame, value: str, out: Path, ylabel: str) -> None:
    plt.figure(figsize=(8.8, 4.8))
    for method in sorted(summary["method"].unique()):
        sub = summary[summary["method"] == method].sort_values("budget_ratio")
        col = f"{value}_mean"
        if col not in sub.columns or sub[col].isna().all():
            continue
        plt.plot(sub["budget_ratio"], sub[col], marker="o", label=method)
    plt.xlabel("Budget Ratio")
    plt.ylabel(ylabel)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def main() -> None:
    args = build_parser().parse_args()
    seeds = _parse_ints(args.seeds)
    budgets = _parse_floats(args.budget_ratios)
    sample_per_class = args.smoke_sample_per_class if args.mode == "smoke" else args.balanced_sample_per_class
    out_dir = _make_out_dir(Path(args.output), args.mode)

    samples = _load_hdfs_samples(Path(args.hdfs_root), per_class=sample_per_class, sample_seed=args.sample_seed)
    samples = add_normalized_views(samples)
    methods = [
        "rank_v0_anchor_first",
        "rank_v0_original",
        "random_same_count",
        "random_same_span",
        "keyword_only",
        "severity_only",
        "template_rarity",
        "tfidf_log_odds",
        "single_line",
    ]
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        split = split_samples(samples, test_size=args.test_size, seed=seed)
        split_ids = {
            "seed": seed,
            "unit": "block_id",
            "split_mode": split.split_meta,
            "train_sample_ids": split.train_df["sample_id"].astype(str).tolist(),
            "test_sample_ids": split.test_df["sample_id"].astype(str).tolist(),
        }
        (out_dir / "01_splits" / f"seed_{seed}_split.json").write_text(
            json.dumps(split_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        y_train = split.train_df["fault_type"].astype(str).tolist()
        y_test = split.test_df["fault_type"].astype(str).tolist()
        vec_full, clf_full, fit_meta = _fit_model(split.train_df["normalized_sequence"].astype(str).tolist(), y_train, seed=seed)
        full_eval = _eval_with_model(vec_full, clf_full, split.test_df["normalized_sequence"].astype(str).tolist(), y_test)
        log_odds = _fit_log_odds(split.train_df)
        template_rarity = _fit_template_rarity(split.train_df)

        for ratio in budgets:
            for method in methods:
                selected = _select_for_method(
                    split.test_df,
                    method=method,
                    ratio=ratio,
                    seed=seed,
                    max_budget_lines=args.max_budget_lines,
                    context_window=args.context_window,
                    log_odds_weights=log_odds,
                    template_rarity_model=template_rarity,
                )
                top_texts, removed_texts, counts, totals = _build_texts(split.test_df, selected)
                line_stats = _selected_line_stats(split.test_df, selected)
                top_eval = _eval_with_model(vec_full, clf_full, top_texts, y_test)
                removed_eval = _eval_with_model(vec_full, clf_full, removed_texts, y_test)
                rows.append(
                    {
                        "dataset": "hdfs_full_preprocessed",
                        "seed": seed,
                        "budget_ratio": ratio,
                        "method": method,
                        "full_on_full_macro_f1": float(full_eval["macro_f1"]),
                        "full_on_topk_macro_f1": float(top_eval["macro_f1"]),
                        "full_on_removed_topk_macro_f1": float(removed_eval["macro_f1"]),
                        "necessity_f1_drop": float(full_eval["macro_f1"] - removed_eval["macro_f1"]),
                        "sufficiency_f1_drop": float(full_eval["macro_f1"] - top_eval["macro_f1"]),
                        "prediction_change_rate_topk": _prediction_change(full_eval["pred"], top_eval["pred"]),
                        "prediction_change_rate_removed": _prediction_change(full_eval["pred"], removed_eval["pred"]),
                        "actual_budget_lines_mean": float(sum(counts) / len(counts)) if counts else 0.0,
                        "actual_budget_ratio_mean": _mean_ratio(counts, totals),
                        "signal_line_ratio": line_stats["signal_line_ratio"],
                        "noise_like_ratio": line_stats["noise_like_ratio"],
                        "entity_line_ratio": line_stats["entity_line_ratio"],
                        "num_test_samples": int(len(split.test_df)),
                        "num_train_samples": int(len(split.train_df)),
                        "classifier_selected_C": fit_meta.get("selected_C"),
                        "classifier_selected_class_weight": fit_meta.get("selected_class_weight"),
                    }
                )

    raw = _add_gaps(pd.DataFrame(rows))
    raw_file = out_dir / "02_raw_records.csv"
    raw.to_csv(raw_file, index=False)

    summary = (
        raw.groupby(["dataset", "method", "budget_ratio"])
        .agg(
            full_on_full_macro_f1_mean=("full_on_full_macro_f1", "mean"),
            full_on_full_macro_f1_std=("full_on_full_macro_f1", "std"),
            full_on_topk_macro_f1_mean=("full_on_topk_macro_f1", "mean"),
            full_on_topk_macro_f1_std=("full_on_topk_macro_f1", "std"),
            full_on_removed_topk_macro_f1_mean=("full_on_removed_topk_macro_f1", "mean"),
            full_on_removed_topk_macro_f1_std=("full_on_removed_topk_macro_f1", "std"),
            necessity_f1_drop_mean=("necessity_f1_drop", "mean"),
            necessity_f1_drop_std=("necessity_f1_drop", "std"),
            necessity_gap_vs_random_count_mean=("necessity_gap_vs_random_count", "mean"),
            necessity_gap_vs_random_span_mean=("necessity_gap_vs_random_span", "mean"),
            prediction_change_rate_topk_mean=("prediction_change_rate_topk", "mean"),
            prediction_change_rate_removed_mean=("prediction_change_rate_removed", "mean"),
            actual_budget_lines_mean=("actual_budget_lines_mean", "mean"),
            actual_budget_ratio_mean=("actual_budget_ratio_mean", "mean"),
            signal_line_ratio_mean=("signal_line_ratio", "mean"),
            noise_like_ratio_mean=("noise_like_ratio", "mean"),
            entity_line_ratio_mean=("entity_line_ratio", "mean"),
            num_test_samples=("num_test_samples", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "03_summary_mean_std_by_method_budget.csv", index=False)
    summary[summary["budget_ratio"] == 0.30].to_csv(out_dir / "04_main_table_budget0p30.csv", index=False)
    summary[
        [
            "dataset",
            "method",
            "budget_ratio",
            "necessity_f1_drop_mean",
            "necessity_gap_vs_random_count_mean",
            "necessity_gap_vs_random_span_mean",
        ]
    ].to_csv(out_dir / "05_necessity_gap_by_budget.csv", index=False)
    summary[
        ["dataset", "method", "budget_ratio", "prediction_change_rate_topk_mean", "prediction_change_rate_removed_mean"]
    ].to_csv(out_dir / "06_prediction_change_by_budget.csv", index=False)

    _plot(summary, "full_on_topk_macro_f1", out_dir / "plots" / "curve_sufficiency_by_budget.png", "Top-k Macro-F1")
    _plot(summary, "necessity_f1_drop", out_dir / "plots" / "curve_necessity_drop_by_budget.png", "Necessity F1 Drop")
    _plot(
        summary,
        "necessity_gap_vs_random_count",
        out_dir / "plots" / "curve_necessity_gap_by_budget.png",
        "Necessity Gap vs Random Count",
    )
    _plot(summary, "prediction_change_rate_removed", out_dir / "plots" / "curve_prediction_change.png", "Prediction Change")

    rank_cases = raw[raw["method"] == "rank_v0_anchor_first"].sort_values("necessity_gap_vs_random_count", ascending=False)
    rank_cases.head(50).to_csv(out_dir / "case_candidates" / "top_positive_gap_cases.csv", index=False)
    rank_cases.tail(50).to_csv(out_dir / "case_candidates" / "failed_cases.csv", index=False)

    config = {
        "mode": args.mode,
        "dataset": "HDFS full preprocessed block/session level",
        "hdfs_root": str(args.hdfs_root),
        "unit": "block_id",
        "split": "stratified_by_label_if_possible",
        "seeds": seeds,
        "budget_ratios": budgets,
        "methods": methods,
        "main_method": "rank_v0_anchor_first",
        "rank_ablation_method": "rank_v0_original",
        "sample_per_class": sample_per_class,
        "test_size": args.test_size,
        "context_window": args.context_window,
        "max_budget_lines": args.max_budget_lines,
        "protocol_guards": {
            "no_hdfs_5k_for_main": True,
            "split_unit": "block_id",
            "tfidf_log_odds_fit_on_train_only": True,
            "same_line_budget_for_all_methods": True,
            "frozen_full_model_verifier": True,
        },
        "n_samples": int(len(samples)),
        "label_counts": samples["fault_type"].value_counts().to_dict(),
        "outputs": {
            "raw_records": str(raw_file),
            "summary": str(out_dir / "03_summary_mean_std_by_method_budget.csv"),
        },
    }
    (out_dir / "00_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "RESULT_INDEX.md").write_text(
        "\n".join(
            [
                "# Phase 4 HDFS Result Index",
                "",
                "- `00_config.json`: run configuration and protocol guards",
                "- `01_splits/`: block/session-level split ids",
                "- `02_raw_records.csv`: seed x budget x method records",
                "- `03_summary_mean_std_by_method_budget.csv`: aggregate summary",
                "- `04_main_table_budget0p30.csv`: budget=0.30 table",
                "- `05_necessity_gap_by_budget.csv`: necessity gap table",
                "- `06_prediction_change_by_budget.csv`: prediction change table",
                "- Main method: `rank_v0_anchor_first`; ablation: `rank_v0_original`",
                "- `plots/`: main curves",
                "- `case_candidates/`: best/worst rank_v0_anchor_first cases",
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("Phase 4 HDFS Completed")
    print("=" * 60)
    print(f"Out dir: {out_dir}")
    print(f"Records: {len(raw)}")
    print(summary.to_string(index=False, max_colwidth=80))


if __name__ == "__main__":
    main()
