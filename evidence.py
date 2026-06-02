from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import pandas as pd


KEYWORD_WEIGHTS = {
    "fatal": 4.0,
    "error": 2.0,
    "warn": 1.0,
    "warning": 1.0,
    "failed": 3.0,
    "failure": 3.0,
    "exception": 3.0,
    "timeout": 2.5,
    "refused": 2.5,
    "no space": 4.5,
    "disk full": 5.0,
    "network": 2.0,
    "unreachable": 3.0,
    "down": 2.0,
    # Type-discriminative strong patterns
    "no route to host": 6.0,
    "deadnodes": 4.5,
    "bad connect ack": 5.0,
    "failed on local exception": 4.5,
    "fserror": 4.5,
    "spill failed": 4.0,
    "there is not enough space on the disk": 7.0,
    "forcibly closed by the remote host": 3.5,
}

BENIGN_PATTERNS = {
    "metrics system shutdown complete": 2.0,
    "down to the last merge-pass": 1.5,
    "opening proxy": 1.0,
    "resolved": 1.0,
    "maxtaskfailurespernode": 1.5,
}


def _keyword_score(line: str) -> float:
    s = line.lower()
    score = 0.0
    for kw, w in KEYWORD_WEIGHTS.items():
        if kw in s:
            score += w
    return score


def _benign_penalty(line: str) -> float:
    s = line.lower()
    penalty = 0.0
    for p, w in BENIGN_PATTERNS.items():
        if p in s:
            penalty += w
    return penalty


def build_template_prior(train_df: pd.DataFrame) -> Dict[str, float]:
    """
    Estimate anomaly prior per template from training samples.
    prior = log((abnormal_count+1)/(normal_count+1))
    """
    counts: Dict[str, List[int]] = {}
    for _, row in train_df.iterrows():
        is_abn = 1 if int(row["label"]) == 1 else 0
        seen = set()
        for tpl in row["template_lines"]:
            t = str(tpl)
            if t in seen:
                continue
            seen.add(t)
            if t not in counts:
                counts[t] = [0, 0]  # normal, abnormal
            counts[t][1 if is_abn else 0] += 1

    prior: Dict[str, float] = {}
    for t, (n_norm, n_abn) in counts.items():
        prior[t] = math.log((n_abn + 1.0) / (n_norm + 1.0))
    return prior


def score_lines(
    normalized_lines: List[str],
    template_lines: List[str],
    template_prior: Dict[str, float],
) -> List[float]:
    n = len(normalized_lines)
    if n == 0:
        return []

    scores: List[float] = []
    for i, (line, tpl) in enumerate(zip(normalized_lines, template_lines)):
        kw = _keyword_score(line)
        benign = _benign_penalty(line)
        prior = template_prior.get(str(tpl), 0.0) * 1.5
        # Later lines often contain direct failure outcome for jobs.
        tail = (i + 1) / n
        tail_bonus = 1.5 * tail
        scores.append(kw + prior + tail_bonus - benign)
    return scores


def select_topk_indices(
    scores: List[float],
    k: int,
    template_lines: List[str] | None = None,
    max_per_template: int = 3,
) -> List[int]:
    if not scores:
        return []
    k = max(1, min(k, len(scores)))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if not template_lines:
        top = ranked[:k]
        return sorted(top)

    top: List[int] = []
    tpl_count: Dict[str, int] = {}
    for i in ranked:
        tpl = str(template_lines[i])
        c = tpl_count.get(tpl, 0)
        if c >= max_per_template:
            continue
        top.append(i)
        tpl_count[tpl] = c + 1
        if len(top) >= k:
            break
    if len(top) < k:
        seen = set(top)
        for i in ranked:
            if i not in seen:
                top.append(i)
            if len(top) >= k:
                break
    top_sorted = sorted(top)
    return top_sorted


def _resolve_budget_k(
    n_lines: int,
    top_k: int | None = None,
    budget_ratio: float | None = None,
    max_budget_lines: int | None = None,
) -> int:
    if n_lines <= 0:
        return 0
    if budget_ratio is not None:
        ratio = max(0.0, min(1.0, float(budget_ratio)))
        k = max(1, min(n_lines, int(math.ceil(n_lines * ratio))))
        if max_budget_lines is not None and max_budget_lines > 0:
            k = min(k, int(max_budget_lines))
        return k
    if top_k is not None:
        k = max(1, min(n_lines, int(top_k)))
        if max_budget_lines is not None and max_budget_lines > 0:
            k = min(k, int(max_budget_lines))
        return k
    return max(1, min(n_lines, 20))


def _expand_with_radius(indices: List[int], n_lines: int, radius: int) -> List[int]:
    if radius <= 0:
        return sorted(set(indices))
    out = set()
    for i in indices:
        lo = max(0, i - radius)
        hi = min(n_lines - 1, i + radius)
        for j in range(lo, hi + 1):
            out.add(j)
    return sorted(out)


def _to_segments(indices: List[int]) -> List[Tuple[int, int]]:
    if not indices:
        return []
    idx = sorted(set(indices))
    segs: List[Tuple[int, int]] = []
    s = idx[0]
    p = idx[0]
    for i in idx[1:]:
        if i == p + 1:
            p = i
            continue
        segs.append((s, p))
        s = i
        p = i
    segs.append((s, p))
    return segs


def _sample_random_count_indices(n_lines: int, count: int, rng: random.Random) -> List[int]:
    if n_lines <= 0 or count <= 0:
        return []
    c = min(n_lines, count)
    return sorted(rng.sample(range(n_lines), c))


def _sample_random_span_indices(n_lines: int, source_indices: List[int], rng: random.Random) -> List[int]:
    if n_lines <= 0 or not source_indices:
        return []
    segs = _to_segments(source_indices)
    lengths = [b - a + 1 for a, b in segs]
    out = set()
    max_trials = 100
    for ln in lengths:
        if ln <= 0:
            continue
        placed = False
        for _ in range(max_trials):
            if ln >= n_lines:
                s = 0
            else:
                s = rng.randint(0, n_lines - ln)
            cand = set(range(s, s + ln))
            if out.isdisjoint(cand):
                out |= cand
                placed = True
                break
        if not placed:
            remain = [i for i in range(n_lines) if i not in out]
            take = min(len(remain), ln)
            out |= set(remain[:take])
    return sorted(out)


def attach_topk_evidence(
    df: pd.DataFrame,
    template_prior: Dict[str, float],
    top_k: int | None = None,
    budget_ratio: float | None = None,
    max_budget_lines: int | None = None,
    package_radius: int = 0,
    removal_radius: int = 0,
    random_seed: int = 42,
) -> pd.DataFrame:
    out = df.copy()

    topk_idx_col: List[List[int]] = []
    topk_package_idx_col: List[List[int]] = []
    removed_topk_idx_col: List[List[int]] = []
    removed_rand_count_idx_col: List[List[int]] = []
    removed_rand_span_idx_col: List[List[int]] = []
    topk_norm_col: List[str] = []
    removed_norm_col: List[str] = []
    removed_rand_count_col: List[str] = []
    removed_rand_span_col: List[str] = []
    topk_count_col: List[int] = []
    removed_count_col: List[int] = []
    evidence_rows: List[Tuple] = []
    rng = random.Random(random_seed)

    for _, row in out.iterrows():
        norm_lines = list(row["normalized_lines"])
        tpl_lines = list(row["template_lines"])
        scores = score_lines(norm_lines, tpl_lines, template_prior)
        k = _resolve_budget_k(
            len(norm_lines),
            top_k=top_k,
            budget_ratio=budget_ratio,
            max_budget_lines=max_budget_lines,
        )
        idx = select_topk_indices(scores, k, template_lines=tpl_lines)
        pkg_idx = _expand_with_radius(idx, len(norm_lines), package_radius)
        removal_idx = _expand_with_radius(pkg_idx, len(norm_lines), removal_radius)
        removal_set = set(removal_idx)

        top_lines = [norm_lines[i] for i in pkg_idx]
        rem_lines = [norm_lines[i] for i in range(len(norm_lines)) if i not in removal_set]

        rand_count_idx = _sample_random_count_indices(len(norm_lines), len(removal_set), rng=rng)
        rand_span_idx = _sample_random_span_indices(len(norm_lines), sorted(removal_set), rng=rng)
        rand_count_set = set(rand_count_idx)
        rand_span_set = set(rand_span_idx)
        rem_rand_count_lines = [norm_lines[i] for i in range(len(norm_lines)) if i not in rand_count_set]
        rem_rand_span_lines = [norm_lines[i] for i in range(len(norm_lines)) if i not in rand_span_set]

        topk_idx_col.append(idx)
        topk_package_idx_col.append(pkg_idx)
        removed_topk_idx_col.append(removal_idx)
        removed_rand_count_idx_col.append(rand_count_idx)
        removed_rand_span_idx_col.append(rand_span_idx)
        topk_norm_col.append("\n".join(top_lines))
        removed_norm_col.append("\n".join(rem_lines))
        removed_rand_count_col.append("\n".join(rem_rand_count_lines))
        removed_rand_span_col.append("\n".join(rem_rand_span_lines))
        topk_count_col.append(int(len(pkg_idx)))
        removed_count_col.append(int(len(removal_set)))

        for rank, i in enumerate(sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:k], start=1):
            evidence_rows.append(
                (
                    row["sample_id"],
                    row["fault_type"],
                    rank,
                    i,
                    float(scores[i]),
                    norm_lines[i],
                )
            )

    out["topk_indices"] = topk_idx_col
    out["topk_package_indices"] = topk_package_idx_col
    out["removed_topk_indices"] = removed_topk_idx_col
    out["removed_random_count_indices"] = removed_rand_count_idx_col
    out["removed_random_span_indices"] = removed_rand_span_idx_col
    out["topk_sequence"] = topk_norm_col
    out["removed_topk_sequence"] = removed_norm_col
    out["removed_random_count_sequence"] = removed_rand_count_col
    out["removed_random_span_sequence"] = removed_rand_span_col
    out["topk_count"] = topk_count_col
    out["removed_count"] = removed_count_col

    evidence_df = pd.DataFrame(
        evidence_rows,
        columns=["sample_id", "fault_type", "rank", "line_index", "score", "normalized_line"],
    )
    return out, evidence_df
