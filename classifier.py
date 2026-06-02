from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


@dataclass
class SplitData:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    split_meta: Dict[str, object]


def split_samples(samples: pd.DataFrame, test_size: float = 0.3, seed: int = 42) -> SplitData:
    """Split samples into train/test partitions and record split metadata."""
    counts = samples["fault_type"].astype(str).value_counts()
    n_classes = int(counts.shape[0])
    min_count = int(counts.min()) if len(counts) else 0
    n_total = int(len(samples))

    # Stratified split may fail when classes are very small or test split is too tiny.
    can_stratify = min_count >= 2 and n_total >= n_classes * 2
    min_test_ratio = (n_classes / n_total) if n_total > 0 else test_size
    effective_test_size = max(float(test_size), float(min_test_ratio))
    effective_test_size = min(effective_test_size, 0.5)

    split_mode = "stratified"
    try:
        train_df, test_df = train_test_split(
            samples,
            test_size=effective_test_size,
            random_state=seed,
            stratify=samples["fault_type"] if can_stratify else None,
        )
        if not can_stratify:
            split_mode = "random_fallback_small_class"
    except ValueError:
        train_df, test_df = train_test_split(
            samples,
            test_size=min(0.3, max(0.2, test_size)),
            random_state=seed,
            stratify=None,
        )
        split_mode = "random_fallback_exception"

    split_meta = {
        "split_mode": split_mode,
        "requested_test_size": float(test_size),
        "effective_test_size": float(effective_test_size),
        "n_total": n_total,
        "n_classes": n_classes,
        "class_counts": counts.to_dict(),
    }
    return SplitData(
        train_df=train_df.reset_index(drop=True),
        test_df=test_df.reset_index(drop=True),
        split_meta=split_meta,
    )


def _train_and_eval(train_text, train_y, test_text, test_y) -> Dict[str, float]:
    vec, clf, _ = _fit_model(train_text, train_y)
    x_test = vec.transform(test_text)
    pred = clf.predict(x_test)

    acc = accuracy_score(test_y, pred)
    macro_f1 = f1_score(test_y, pred, average="macro")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(test_y, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(macro_f1),
    }

def _fit_model(train_text, train_y, seed: int = 42) -> Tuple[TfidfVectorizer, LogisticRegression, Dict[str, object]]:
    """Fit the frozen TF-IDF + Logistic Regression verifier on training text only."""
    vec = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=1,
    )
    x_train = vec.fit_transform(train_text)

    y = np.array(list(train_y))
    unique, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    can_validate = len(y) >= 16 and min_count >= 2 and len(unique) >= 2

    candidates: List[Dict[str, object]] = [
        {"C": 0.25, "class_weight": "balanced"},
        {"C": 0.5, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 2.0, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": None},
    ]

    best_cfg = {"C": 1.0, "class_weight": "balanced"}
    best_score = -1.0

    if can_validate:
        x_tr, x_val, y_tr, y_val = train_test_split(
            x_train,
            y,
            test_size=0.25,
            random_state=seed,
            stratify=y,
        )
        for cfg in candidates:
            clf = LogisticRegression(
                max_iter=2000,
                C=float(cfg["C"]),
                class_weight=cfg["class_weight"],
                solver="lbfgs",
            )
            clf.fit(x_tr, y_tr)
            pred_val = clf.predict(x_val)
            score = f1_score(y_val, pred_val, average="macro")
            if score > best_score:
                best_score = float(score)
                best_cfg = cfg
    else:
        best_cfg = {"C": 1.0, "class_weight": "balanced"}
        best_score = float("nan")

    clf = LogisticRegression(
        max_iter=2000,
        C=float(best_cfg["C"]),
        class_weight=best_cfg["class_weight"],
        solver="lbfgs",
    )
    clf.fit(x_train, y)
    fit_meta = {
        "selected_C": float(best_cfg["C"]),
        "selected_class_weight": str(best_cfg["class_weight"]),
        "val_macro_f1": None if np.isnan(best_score) else float(best_score),
    }
    return vec, clf, fit_meta

def _eval_with_model(vec, clf, test_text, test_y) -> Dict[str, float]:
    """Evaluate a fixed verifier on one input view."""
    x_test = vec.transform(test_text)
    pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None
    acc = accuracy_score(test_y, pred)
    macro_f1 = f1_score(test_y, pred, average="macro")
    p_macro, r_macro, _, _ = precision_recall_fscore_support(test_y, pred, average="macro", zero_division=0)
    out = {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(macro_f1),
        "pred": list(pred),
    }
    if proba is not None:
        out["confidence"] = list(np.max(proba, axis=1))
    else:
        out["confidence"] = []
    return out

def run_classification_experiments(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Legacy helper retained for compatibility with earlier experiments."""
    y_train = train_df["fault_type"].astype(str).tolist()
    y_test = test_df["fault_type"].astype(str).tolist()

    full_train_text = train_df["normalized_sequence"].astype(str).tolist()
    full_test_text = test_df["normalized_sequence"].astype(str).tolist()
    topk_train_text = train_df["topk_sequence"].astype(str).tolist()
    topk_test_text = test_df["topk_sequence"].astype(str).tolist()
    removed_train_text = train_df["removed_topk_sequence"].astype(str).tolist()
    removed_test_text = test_df["removed_topk_sequence"].astype(str).tolist()
    removed_rand_count_train_text = (
        train_df["removed_random_count_sequence"].astype(str).tolist()
        if "removed_random_count_sequence" in train_df.columns
        else []
    )
    removed_rand_count_test_text = (
        test_df["removed_random_count_sequence"].astype(str).tolist()
        if "removed_random_count_sequence" in test_df.columns
        else []
    )
    removed_rand_span_train_text = (
        train_df["removed_random_span_sequence"].astype(str).tolist()
        if "removed_random_span_sequence" in train_df.columns
        else []
    )
    removed_rand_span_test_text = (
        test_df["removed_random_span_sequence"].astype(str).tolist()
        if "removed_random_span_sequence" in test_df.columns
        else []
    )

    # Version A: dedicated full-sequence classifier
    full_res = _train_and_eval(full_train_text, y_train, full_test_text, y_test)

    # Version B: dedicated top-k classifier
    topk_res = _train_and_eval(
        topk_train_text,
        y_train,
        topk_test_text,
        y_test,
    )

    removed_res = _train_and_eval(
        removed_train_text,
        y_train,
        removed_test_text,
        y_test,
    )

    # Strict evidence validation:
    # Keep full model fixed, evaluate on top-k / removed-top-k inputs.
    vec_full, clf_full, fit_meta_full = _fit_model(full_train_text, y_train)
    full_model_on_full_raw = _eval_with_model(vec_full, clf_full, full_test_text, y_test)
    full_model_on_topk_raw = _eval_with_model(vec_full, clf_full, topk_test_text, y_test)
    full_model_on_removed_raw = _eval_with_model(vec_full, clf_full, removed_test_text, y_test)
    full_model_on_removed_rand_count_raw = (
        _eval_with_model(vec_full, clf_full, removed_rand_count_test_text, y_test)
        if removed_rand_count_test_text
        else None
    )
    full_model_on_removed_rand_span_raw = (
        _eval_with_model(vec_full, clf_full, removed_rand_span_test_text, y_test)
        if removed_rand_span_test_text
        else None
    )

    suff_drop = full_model_on_full_raw["macro_f1"] - full_model_on_topk_raw["macro_f1"]
    nec_drop = full_model_on_full_raw["macro_f1"] - full_model_on_removed_raw["macro_f1"]

    pred_full = full_model_on_full_raw.get("pred", [])
    pred_topk = full_model_on_topk_raw.get("pred", [])
    pred_removed = full_model_on_removed_raw.get("pred", [])
    pred_removed_rand_count = full_model_on_removed_rand_count_raw.get("pred", []) if full_model_on_removed_rand_count_raw else []
    pred_removed_rand_span = full_model_on_removed_rand_span_raw.get("pred", []) if full_model_on_removed_rand_span_raw else []
    n = max(1, len(pred_full))
    pred_change_topk = sum(a != b for a, b in zip(pred_full, pred_topk)) / n if pred_topk else 0.0
    pred_change_removed = sum(a != b for a, b in zip(pred_full, pred_removed)) / n if pred_removed else 0.0
    pred_change_removed_rand_count = (
        sum(a != b for a, b in zip(pred_full, pred_removed_rand_count)) / n if pred_removed_rand_count else 0.0
    )
    pred_change_removed_rand_span = (
        sum(a != b for a, b in zip(pred_full, pred_removed_rand_span)) / n if pred_removed_rand_span else 0.0
    )

    conf_full = full_model_on_full_raw.get("confidence", [])
    conf_topk = full_model_on_topk_raw.get("confidence", [])
    conf_removed = full_model_on_removed_raw.get("confidence", [])
    conf_removed_rand_count = full_model_on_removed_rand_count_raw.get("confidence", []) if full_model_on_removed_rand_count_raw else []
    conf_removed_rand_span = full_model_on_removed_rand_span_raw.get("confidence", []) if full_model_on_removed_rand_span_raw else []
    mean_conf_full = float(np.mean(conf_full)) if conf_full else 0.0
    mean_conf_topk = float(np.mean(conf_topk)) if conf_topk else 0.0
    mean_conf_removed = float(np.mean(conf_removed)) if conf_removed else 0.0
    mean_conf_removed_rand_count = float(np.mean(conf_removed_rand_count)) if conf_removed_rand_count else 0.0
    mean_conf_removed_rand_span = float(np.mean(conf_removed_rand_span)) if conf_removed_rand_span else 0.0

    nec_drop_rand_count = (
        float(full_model_on_full_raw["macro_f1"] - full_model_on_removed_rand_count_raw["macro_f1"])
        if full_model_on_removed_rand_count_raw
        else 0.0
    )
    nec_drop_rand_span = (
        float(full_model_on_full_raw["macro_f1"] - full_model_on_removed_rand_span_raw["macro_f1"])
        if full_model_on_removed_rand_span_raw
        else 0.0
    )

    def _strip_aux(d: Dict) -> Dict[str, float]:
        keep = {}
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            keep[k] = float(d[k])
        return keep

    return {
        "full_sequence": full_res,
        "topk_only": topk_res,
        "removed_topk": removed_res,
        "full_model_on_full": _strip_aux(full_model_on_full_raw),
        "full_model_on_topk": _strip_aux(full_model_on_topk_raw),
        "full_model_on_removed_topk": _strip_aux(full_model_on_removed_raw),
        "full_model_on_removed_random_count": _strip_aux(full_model_on_removed_rand_count_raw)
        if full_model_on_removed_rand_count_raw
        else {},
        "full_model_on_removed_random_span": _strip_aux(full_model_on_removed_rand_span_raw)
        if full_model_on_removed_rand_span_raw
        else {},
        "sufficiency_f1_drop": float(suff_drop),
        "necessity_f1_drop": float(nec_drop),
        "necessity_f1_drop_random_count": float(nec_drop_rand_count),
        "necessity_f1_drop_random_span": float(nec_drop_rand_span),
        "prediction_change_rate_topk": float(pred_change_topk),
        "prediction_change_rate_removed_topk": float(pred_change_removed),
        "prediction_change_rate_removed_random_count": float(pred_change_removed_rand_count),
        "prediction_change_rate_removed_random_span": float(pred_change_removed_rand_span),
        "confidence_mean_full": mean_conf_full,
        "confidence_mean_topk": mean_conf_topk,
        "confidence_mean_removed_topk": mean_conf_removed,
        "confidence_mean_removed_random_count": mean_conf_removed_rand_count,
        "confidence_mean_removed_random_span": mean_conf_removed_rand_span,
        "confidence_drop_topk": float(mean_conf_full - mean_conf_topk),
        "confidence_drop_removed_topk": float(mean_conf_full - mean_conf_removed),
        "confidence_drop_removed_random_count": float(mean_conf_full - mean_conf_removed_rand_count),
        "confidence_drop_removed_random_span": float(mean_conf_full - mean_conf_removed_rand_span),
        "classifier_meta_full": fit_meta_full,
    }
