from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load LogHub/LogPai HDFS preprocessed block-level data.")
    parser.add_argument("--hdfs-root", default="data/HDFS", help="Path containing preprocessed/Event_traces.csv.")
    parser.add_argument("--per-class", type=int, default=20000)
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser


def _parse_event_ids(features: str) -> List[str]:
    text = str(features).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [x.strip() for x in text.split(",") if x.strip()]


def _load_hdfs_samples(hdfs_root: Path, per_class: int, sample_seed: int) -> pd.DataFrame:
    """Load HDFS block/session samples from the public LogHub/LogPai preprocessed files.

    Expected files:
      - <hdfs_root>/preprocessed/Event_traces.csv
      - <hdfs_root>/preprocessed/HDFS.log_templates.csv

    The split unit is BlockId. The returned rows are sorted by sample_id so that
    downstream stratified splits are reproducible for fixed seeds.
    """

    preprocessed = hdfs_root / "preprocessed"
    traces = pd.read_csv(preprocessed / "Event_traces.csv", usecols=["BlockId", "Label", "Features"])
    templates = pd.read_csv(preprocessed / "HDFS.log_templates.csv")
    template_map: Dict[str, str] = {
        str(row["EventId"]): str(row["EventTemplate"]) for _, row in templates.iterrows()
    }

    traces["fault_type"] = traces["Label"].map(
        lambda x: "normal" if str(x).lower() in {"success", "normal"} else "hdfs_fail"
    )
    traces["label"] = traces["fault_type"].map(lambda x: 0 if x == "normal" else 1)
    traces["label_str"] = traces["label"].map(lambda y: "normal" if int(y) == 0 else "abnormal")

    grouped = []
    for _, group in traces.groupby("fault_type", sort=True):
        grouped.append(group.sample(n=min(len(group), per_class), random_state=sample_seed))
    sampled = pd.concat(grouped, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=sample_seed).reset_index(drop=True)

    rows = []
    for _, row in sampled.iterrows():
        event_ids = _parse_event_ids(row["Features"])
        lines = [template_map.get(event_id, event_id) for event_id in event_ids]
        if not lines:
            continue
        block_id = str(row["BlockId"])
        rows.append(
            {
                "sample_id": block_id,
                "entity_id": block_id,
                "label": int(row["label"]),
                "label_str": str(row["label_str"]),
                "fault_type": str(row["fault_type"]),
                "workload": "hdfs",
                "num_logs": int(len(lines)),
                "log_sequence": "\n".join(lines),
            }
        )
    return pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)


def main() -> None:
    args = build_parser().parse_args()
    samples = _load_hdfs_samples(Path(args.hdfs_root), per_class=args.per_class, sample_seed=args.sample_seed)
    print(samples[["sample_id", "fault_type", "num_logs"]].head().to_string(index=False))
    print(f"loaded_samples={len(samples)}")


if __name__ == "__main__":
    main()
