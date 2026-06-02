from __future__ import annotations

import re
from typing import List

import pandas as pd


class LogNormalizer:
    IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
    HEX = re.compile(r"\b(?:0x[0-9a-fA-F]+|[0-9a-fA-F]{8,})\b")
    DATETIME = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)?\b")
    TIME = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
    PATH = re.compile(r"(?:/[A-Za-z0-9._-]+){2,}")
    NUM = re.compile(r"\b\d+\b")
    SPACES = re.compile(r"\s+")

    def normalize_line(self, line: str) -> str:
        s = str(line).strip()
        s = self.DATETIME.sub("<DATETIME>", s)
        s = self.TIME.sub("<TIME>", s)
        s = self.UUID.sub("<UUID>", s)
        s = self.IPV4.sub("<IP>", s)
        s = self.PATH.sub("<PATH>", s)
        s = self.HEX.sub("<HEX>", s)
        s = self.NUM.sub("<NUM>", s)
        s = self.SPACES.sub(" ", s).strip()
        return s

    def template_line(self, normalized_line: str) -> str:
        # Minimal template for MVP
        s = str(normalized_line)
        s = re.sub(r"<NUM>", "<*>", s)
        s = re.sub(r"<HEX>", "<*>", s)
        s = re.sub(r"<IP>", "<*>", s)
        s = re.sub(r"<UUID>", "<*>", s)
        return s


def add_normalized_views(samples: pd.DataFrame) -> pd.DataFrame:
    normalizer = LogNormalizer()
    out = samples.copy()

    raw_seq_list: List[List[str]] = []
    norm_seq_list: List[List[str]] = []
    tpl_seq_list: List[List[str]] = []
    norm_text_list: List[str] = []

    for _, row in out.iterrows():
        lines = str(row["log_sequence"]).splitlines()
        raw_lines = [x for x in lines if x.strip()]
        norm_lines = [normalizer.normalize_line(x) for x in raw_lines]
        tpl_lines = [normalizer.template_line(x) for x in norm_lines]
        raw_seq_list.append(raw_lines)
        norm_seq_list.append(norm_lines)
        tpl_seq_list.append(tpl_lines)
        norm_text_list.append("\n".join(norm_lines))

    out["raw_lines"] = raw_seq_list
    out["normalized_lines"] = norm_seq_list
    out["template_lines"] = tpl_seq_list
    out["normalized_sequence"] = norm_text_list
    return out

