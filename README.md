# EvidenceLog Reproduction Guide

This reproduction artifact accompanies the paper
**A Verifiable Pre-Diagnostic Evidence Filtering Framework for System Log Anomaly Analysis**.

It is written as a reviewer reproduction guide. It is not a general-purpose
project README.

## 1. What This Repository Reproduces

This repository reproduces the HDFS core tables reported in the paper:

- Table III: HDFS main evidence-validity result at budget ratio 0.30.
- Table V: HDFS multi-budget robustness over five seeds and five budgets.
- Table VI: anchor-first design ablation and context-policy sensitivity.
- Table VII: paired comparison over 5 seeds x 5 budgets.
- Table VIII: robustness under multiple frozen verifiers.

The artifact focuses on the pre-diagnostic evidence filter. It does not run an
LLM diagnosis system. The optional Qwen/Ollama pilot in the paper is
hardware-dependent and is not required for reproducing Tables III, V, VI, VII,
or VIII.

The intended claim is limited: anchor-first is a low-noise, auditable,
cold-start evidence filter with positive necessity gaps against random removals
and simple severity rules under the fixed HDFS protocol. The paper does not
claim that anchor-first is the strongest label-free salience method or that
selected evidence alone is sufficient for final diagnosis.

## 2. Environment Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required packages are listed in `requirements.txt`:

```text
pandas
numpy
scikit-learn
scipy
matplotlib
tabulate
```

The scripts are CPU-only. Runtime numbers may vary by machine, CPU, BLAS
backend, disk speed, and Python package versions.

## 3. Data Preparation from LogHub/LogPai

This artifact does not redistribute LogHub/LogPai data. Reviewers should
download the public HDFS preprocessed data from the LogHub/LogPai collection.

Expected layout:

```text
data/HDFS/
  preprocessed/
    Event_traces.csv
    HDFS.log_templates.csv
```

You may also keep the dataset elsewhere and pass:

```bash
--hdfs-root /path/to/HDFS
```

The code expects the two files above:

- `Event_traces.csv`: block/session-level event traces and labels.
- `HDFS.log_templates.csv`: event ID to template mapping.

Do not upload full HDFS, BGL, Thunderbird, Spirit, or other LogHub raw datasets
into a review repository unless the dataset license and venue policy
explicitly permit redistribution. The safer reproduction path is to provide
download instructions and require reviewers to place the files in `data/HDFS/`.

LogHub/LogPai citation note: the LogHub repository and Zenodo record ask users
to refer to the LogHub repository URL and cite the LogHub paper where
applicable. In a paper, cite:

```text
J. Zhu, S. He, P. He, J. Liu, and M. R. Lyu,
"Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics,"
Proc. IEEE ISSRE, 2023, pp. 355-366.
```

Repository URL:

```text
https://github.com/logpai/loghub
```

## 4. Quick Sanity Run

Use this smoke run to verify that the environment and HDFS path are correct.
This is not a paper result.

```bash
python run_hdfs_main.py \
  --hdfs-root data/HDFS \
  --output runs \
  --mode smoke \
  --seeds 13 \
  --budget-ratios 0.30 \
  --smoke-sample-per-class 300 \
  --sample-seed 42 \
  --test-size 0.30 \
  --context-window 1
```

Expected smoke output:

```text
runs/phase4_hdfs_smoke_<timestamp>/
  00_config.json
  01_splits/seed_13_split.json
  02_raw_records.csv
  03_summary_mean_std_by_method_budget.csv
```

If this run completes, the main reproduction commands below should be usable.

## 5. Full HDFS Reproduction

The fixed paper protocol is:

```text
Dataset: HDFS full preprocessed block/session data
Split unit: BlockId / block-level session
Seeds: 13, 21, 42, 87, 123
Test size: 0.30
Budget ratios for main run: 0.05, 0.10, 0.20, 0.30, 0.40
Budget ratio for ablation: 0.30
Budget ratio for verifier robustness: 0.30
Balanced sample cap per class: 20000
Sample seed: 42
Context window: 1 for the frozen main method
Main method: rank_v0_anchor_first
Main verifier: frozen TF-IDF + Logistic Regression
```

### 5.1 Main HDFS run: Tables III, V, and VII inputs

```bash
python run_hdfs_main.py \
  --hdfs-root data/HDFS \
  --output runs \
  --mode main \
  --seeds 13,21,42,87,123 \
  --budget-ratios 0.05,0.10,0.20,0.30,0.40 \
  --balanced-sample-per-class 20000 \
  --sample-seed 42 \
  --test-size 0.30 \
  --context-window 1
```

Key outputs:

```text
runs/phase4_hdfs_main_<timestamp>/
  00_config.json
  01_splits/seed_13_split.json
  01_splits/seed_21_split.json
  01_splits/seed_42_split.json
  01_splits/seed_87_split.json
  01_splits/seed_123_split.json
  02_raw_records.csv
  03_summary_mean_std_by_method_budget.csv
  04_main_table_budget0p30.csv
```

The split JSON files contain the exact train/test block IDs for each seed.

### 5.2 Ablation run: Table VI

```bash
python run_ablation.py \
  --hdfs-root data/HDFS \
  --output runs \
  --writing-assets paper_tables \
  --seeds 13,21,42,87,123 \
  --budget-ratio 0.30 \
  --balanced-sample-per-class 20000 \
  --sample-seed 42 \
  --test-size 0.30
```

Key output:

```text
runs/hdfs_ablation_<timestamp>/03_ablation_table.csv
```

The context-window rows in this table are sensitivity checks, not revisions to
the frozen main method.

### 5.3 Verifier robustness run: Table VIII

```bash
python run_verifier_robustness.py \
  --hdfs-root data/HDFS \
  --output runs \
  --writing-assets paper_tables \
  --seeds 13,21,42,87,123 \
  --budget-ratio 0.30 \
  --balanced-sample-per-class 20000 \
  --sample-seed 42 \
  --test-size 0.30
```

Key output:

```text
runs/hdfs_verifier_robustness_<timestamp>/03_verifier_robustness_table.csv
```

## 6. How to Regenerate Paper Tables

After running the three commands above, regenerate the paper tables with:

```bash
python make_tables.py --runs-root runs --output paper_tables
```

Or specify exact run directories:

```bash
python make_tables.py \
  --hdfs-main-dir runs/phase4_hdfs_main_<timestamp> \
  --ablation-dir runs/hdfs_ablation_<timestamp> \
  --verifier-dir runs/hdfs_verifier_robustness_<timestamp> \
  --output paper_tables
```

Generated files:

```text
paper_tables/table_iii_hdfs_main_budget0p30.csv
paper_tables/table_iii_hdfs_main_budget0p30.md
paper_tables/table_v_hdfs_multi_budget.csv
paper_tables/table_v_hdfs_multi_budget.md
paper_tables/table_vi_ablation.csv
paper_tables/table_vi_ablation.md
paper_tables/table_vii_paired_comparison.csv
paper_tables/table_vii_paired_comparison.md
paper_tables/table_viii_verifier_robustness.csv
paper_tables/table_viii_verifier_robustness.md
paper_tables/manifest.csv
```

## 7. Expected Runtime

Approximate cost ordering:

```text
Quick sanity run < ablation run < verifier robustness run < full HDFS main run
```

The full HDFS main run is the most expensive step because it evaluates:

```text
5 seeds x 5 budget ratios x multiple evidence methods
```

The verifier robustness run trains Logistic Regression, Linear SVM, and Random
Forest verifiers at budget 0.30.

Runtime and throughput numbers reported in the paper are machine-dependent.
They should be interpreted as implementation-cost evidence for the authors'
environment, not as hardware-independent constants.

The optional Qwen/Ollama evidence-grounding pilot is not required by this
artifact and is hardware-dependent. Its latency depends on local model serving,
model version, CPU/GPU availability, and Ollama configuration.

## 8. Known Limitations

- The artifact does not redistribute LogHub/LogPai datasets.
- The artifact reproduces HDFS core validity tables, not every supplementary
  experiment in the paper.
- The Qwen/Ollama pilot is optional and not included in the minimal reviewer
  reproduction path.
- Runtime numbers may vary by machine.
- Randomness is controlled through the fixed seeds listed above, but small
  numerical differences can still arise from library or platform versions.
- Template-rarity is a strong label-free corpus-level baseline on HDFS; the
  paper positions anchor-first as lower-noise and more auditable, not as the
  strongest label-free salience scorer.

## Leakage Guard Summary

The artifact enforces the following boundaries:

1. `anchor-first`
   - Label-free.
   - Uses rule cues, normalized text, templates, entity cues, and local context.
   - Does not access train labels, test labels, verifier predictions, or
     verifier coefficients.

2. `template-rarity`
   - Label-free corpus-level baseline.
   - Template frequency is fit only on the training split.
   - Test samples are scored using the train-fitted frequency table.

3. `TF-IDF/log-odds`
   - Supervised salience baseline.
   - Token salience is fit only on training split labels.
   - Test labels are never used for fitting salience.

4. Frozen verifier
   - TF-IDF vectorizer and classifier are fit on full training sequences.
   - The same frozen verifier evaluates full, selected-only, evidence-removed,
     and random-removed test sequences.
   - Evidence selectors do not read verifier coefficients.

## Repository Contents

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- classifier.py
|-- preprocess.py
|-- hdfs_loader.py
|-- evidence.py
|-- run_hdfs_main.py
|-- run_ablation.py
|-- run_verifier_robustness.py
`-- make_tables.py
```
