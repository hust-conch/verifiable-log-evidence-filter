from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.svm import LinearSVC

from classifier import split_samples
from preprocess import add_normalized_views
from hdfs_loader import _load_hdfs_samples
from run_hdfs_main import (
    _budget,
    _build_texts,
    _fit_log_odds,
    _fit_template_rarity,
    _prediction_change,
    _sample_random_count,
    _sample_random_span,
    _select_for_method,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HDFS verifier robustness at budget=0.30.")
    p.add_argument("--hdfs-root", default="data/HDFS")
    p.add_argument("--output", default="runs")
    p.add_argument("--writing-assets", default="paper_tables")
    p.add_argument("--seeds", default="13,21,42,87,123")
    p.add_argument("--budget-ratio", type=float, default=0.30)
    p.add_argument("--balanced-sample-per-class", type=int, default=20000)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.3)
    p.add_argument("--context-window", type=int, default=1)
    p.add_argument("--max-budget-lines", type=int, default=512)
    p.add_argument("--rf-n-estimators", type=int, default=80)
    p.add_argument("--rf-max-depth", type=int, default=32)
    return p


def _parse_ints(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _make_out_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"hdfs_verifier_robustness_{ts}"
    out.mkdir(parents=True, exist_ok=False)
    return out


def _stable_method_offset(method: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(method)) % 10000


def _fit_verifier(name: str, x_train, y_train: Sequence[str], seed: int, args: argparse.Namespace):
    if name == "logistic_regression":
        clf = LogisticRegression(max_iter=2000, C=0.25, class_weight="balanced", solver="lbfgs")
    elif name == "linear_svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=seed, max_iter=5000)
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown verifier: {name}")
    clf.fit(x_train, list(y_train))
    return clf


def _eval(clf, vec: TfidfVectorizer, texts: Sequence[str], y_true: Sequence[str]) -> Dict[str, object]:
    x = vec.transform(texts)
    pred = clf.predict(x)
    p_macro, r_macro, _, _ = precision_recall_fscore_support(y_true, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "pred": list(pred),
    }


def _select_random(method: str, n_lines: int, k: int, seed: int, pos: int, budget_ratio: float) -> List[int]:
    rng = random.Random(seed + int(budget_ratio * 10000) + _stable_method_offset(method) + pos)
    if method == "random_same_count":
        return _sample_random_count(n_lines, k, rng)
    if method == "random_same_span":
        return _sample_random_span(n_lines, k, rng)
    raise ValueError(method)


def _selected_for_method(
    split_df: pd.DataFrame,
    method: str,
    seed: int,
    log_odds: Dict[str, float],
    template_rarity_model,
    args,
) -> List[List[int]]:
    if method in {"random_same_count", "random_same_span"}:
        selected: List[List[int]] = []
        for pos, (_, row) in enumerate(split_df.iterrows()):
            lines = list(row["normalized_lines"])
            k = _budget(len(lines), args.budget_ratio, args.max_budget_lines)
            selected.append(_select_random(method, len(lines), k, seed, pos, args.budget_ratio))
        return selected
    return _select_for_method(
        split_df,
        method,
        args.budget_ratio,
        seed,
        args.max_budget_lines,
        args.context_window,
        log_odds,
        template_rarity_model,
    )


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    anchor = summary[summary["method"] == "rank_v0_anchor_first"].copy()
    sev = summary[summary["method"] == "severity_only"][["verifier", "necessity_f1_drop_mean"]].rename(
        columns={"necessity_f1_drop_mean": "severity_drop"}
    )
    out = anchor.merge(sev, on="verifier", how="left")
    labels = {
        "logistic_regression": "Logistic Regression",
        "linear_svm": "Linear SVM",
        "random_forest": "Random Forest",
    }
    rows = []
    for _, r in out.iterrows():
        beats = "Yes" if float(r["necessity_f1_drop_mean"]) >= float(r["severity_drop"]) - 1e-12 else "No"
        rows.append(
            {
                "Verifier": labels.get(str(r["verifier"]), str(r["verifier"])),
                "Anchor-first Drop": f"{r['necessity_f1_drop_mean']:.4f} +/- {r['necessity_f1_drop_std']:.4f}",
                "Gap vs Count": f"{r['necessity_gap_vs_random_count_mean']:.4f}",
                "Gap vs Span": f"{r['necessity_gap_vs_random_span_mean']:.4f}",
                "Beats Severity": beats,
            }
        )
    return pd.DataFrame(rows)


def _add_verifier_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    key_cols = ["seed", "budget_ratio", "verifier"]
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


def main() -> None:
    args = build_parser().parse_args()
    out_dir = _make_out_dir(Path(args.output))
    writing_assets = Path(args.writing_assets)
    writing_assets.mkdir(parents=True, exist_ok=True)

    seeds = _parse_ints(args.seeds)
    methods = ["rank_v0_anchor_first", "rank_v0_original", "severity_only", "random_same_count", "random_same_span"]
    verifiers = ["logistic_regression", "linear_svm", "random_forest"]

    samples = _load_hdfs_samples(Path(args.hdfs_root), per_class=args.balanced_sample_per_class, sample_seed=args.sample_seed)
    samples = add_normalized_views(samples)

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        split = split_samples(samples, test_size=args.test_size, seed=seed)
        train_texts = split.train_df["normalized_sequence"].astype(str).tolist()
        test_texts = split.test_df["normalized_sequence"].astype(str).tolist()
        y_train = split.train_df["fault_type"].astype(str).tolist()
        y_test = split.test_df["fault_type"].astype(str).tolist()
        log_odds = _fit_log_odds(split.train_df)
        template_rarity = _fit_template_rarity(split.train_df)

        vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=1)
        x_train = vec.fit_transform(train_texts)
        selected_by_method: Dict[str, List[List[int]]] = {
            method: _selected_for_method(split.test_df, method, seed, log_odds, template_rarity, args) for method in methods
        }
        texts_by_method: Dict[str, Tuple[List[str], List[str], List[int], List[int]]] = {
            method: _build_texts(split.test_df, selected) for method, selected in selected_by_method.items()
        }

        for verifier in verifiers:
            clf = _fit_verifier(verifier, x_train, y_train, seed, args)
            full_eval = _eval(clf, vec, test_texts, y_test)
            for method in methods:
                top_texts, removed_texts, counts, totals = texts_by_method[method]
                top_eval = _eval(clf, vec, top_texts, y_test)
                rem_eval = _eval(clf, vec, removed_texts, y_test)
                rows.append(
                    {
                        "dataset": "hdfs_full_preprocessed",
                        "seed": seed,
                        "budget_ratio": args.budget_ratio,
                        "verifier": verifier,
                        "method": method,
                        "full_on_full_macro_f1": float(full_eval["macro_f1"]),
                        "full_on_topk_macro_f1": float(top_eval["macro_f1"]),
                        "full_on_removed_topk_macro_f1": float(rem_eval["macro_f1"]),
                        "necessity_f1_drop": float(full_eval["macro_f1"] - rem_eval["macro_f1"]),
                        "prediction_change_rate_removed": _prediction_change(full_eval["pred"], rem_eval["pred"]),
                        "actual_budget_lines_mean": float(sum(counts) / len(counts)) if counts else 0.0,
                        "actual_budget_ratio_mean": float(sum(c / n for c, n in zip(counts, totals) if n > 0) / len(totals)),
                        "num_test_samples": int(len(split.test_df)),
                    }
                )

    raw = _add_verifier_gaps(pd.DataFrame(rows))
    raw.to_csv(out_dir / "01_raw_records.csv", index=False)
    summary = (
        raw.groupby(["verifier", "method", "budget_ratio"])
        .agg(
            necessity_f1_drop_mean=("necessity_f1_drop", "mean"),
            necessity_f1_drop_std=("necessity_f1_drop", "std"),
            necessity_gap_vs_random_count_mean=("necessity_gap_vs_random_count", "mean"),
            necessity_gap_vs_random_span_mean=("necessity_gap_vs_random_span", "mean"),
            prediction_change_rate_removed_mean=("prediction_change_rate_removed", "mean"),
            full_on_full_macro_f1_mean=("full_on_full_macro_f1", "mean"),
            num_test_samples=("num_test_samples", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "02_summary_budget0p30.csv", index=False)
    paper = _paper_table(summary)
    paper.to_csv(out_dir / "03_verifier_robustness_table.csv", index=False)
    (out_dir / "03_verifier_robustness_table.md").write_text(paper.to_markdown(index=False), encoding="utf-8")
    paper.to_csv(writing_assets / "table11_verifier_robustness.csv", index=False)
    (writing_assets / "table11_verifier_robustness.md").write_text(paper.to_markdown(index=False), encoding="utf-8")

    config = {
        "dataset": "HDFS full preprocessed block/session level",
        "role": "verifier robustness, budget=0.30",
        "budget_ratio": args.budget_ratio,
        "seeds": seeds,
        "methods": methods,
        "verifiers": verifiers,
        "scope_guard": {
            "does_not_change_frozen_main_method": True,
            "single_budget_robustness_only": True,
            "tfidf_vectorizer_shared_within_verifier": True,
        },
        "outputs": {
            "raw_records": str(out_dir / "01_raw_records.csv"),
            "summary": str(out_dir / "02_summary_budget0p30.csv"),
            "paper_table": str(out_dir / "03_verifier_robustness_table.csv"),
        },
    }
    (out_dir / "00_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "04_notes.md").write_text(
        "\n".join(
            [
                "# HDFS Verifier Robustness Notes",
                "",
                "Role: robustness check for whether necessity gaps depend on one frozen Logistic Regression verifier.",
                "",
                paper.to_markdown(index=False),
            ]
        ),
        encoding="utf-8",
    )

    print("=" * 60)
    print("HDFS Verifier Robustness Completed")
    print("=" * 60)
    print(f"Out dir: {out_dir}")
    print(paper.to_string(index=False))


if __name__ == "__main__":
    main()
