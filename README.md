# EvidenceLog Reproduction Guide

This reproduction artifact accompanies the paper
**A Verifiable Pre-Diagnostic Evidence Filtering Framework for System Log Anomaly Analysis**.

It is written as a reviewer reproduction guide. It is not a general-purpose
project README.

## 1. What This Repository Reproduces

This repository reproduces the HDFS core results reported in the paper:
- HDFS main evidence-validity result at budget ratio 0.30.
- HDFS multi-budget robustness over five seeds and five budgets.
- Anchor-first design ablation and context-policy sensitivity.
- Paired comparison over 5 seeds × 5 budgets.
- Robustness under multiple frozen verifiers.

The artifact focuses on the pre-diagnostic evidence filter. It does not run an
LLM diagnosis system. The optional Qwen/Ollama pilot in the paper is
hardware-dependent and is not required for reproducing Tables III, V, VI, VII,
or VIII.

The intended claim is limited: anchor-first is a low-noise, auditable,
cold-start evidence filter with positive necessity gaps against random removals
and simple severity rules under the fixed HDFS protocol. The paper does not
claim that anchor-first is the strongest label-free salience method or that
selected evidence alone is sufficient for final diagnosis.

No pre-trained model is required for the core artifact. The optional
Qwen/Ollama evidence-grounding pilot reported in the paper is an external,
hardware-dependent downstream feasibility check and is not part of the core
HDFS reproduction path.

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

## 7. Expected Results

The commands above should regenerate the core HDFS tables under `paper_tables/`.
Small numerical differences can arise from package versions or BLAS backends,
but the main trends should match the submission.

Key expected values for Table III at budget ratio 0.30:

```text
Method             Selected-only F1  Evidence-removed F1  Necessity Drop  Gap vs Count  Gap vs Span  Noise-like Ratio
Anchor-first       0.941             0.820                0.178           0.113         0.086        0.007
Template-rarity    0.481             0.664                0.334           0.269         0.242        0.137
Original RANK      0.560             0.856                0.142           0.078         0.050        0.589
Severity-only      0.436             0.861                0.137           0.072         0.045        0.694
Random-count       0.758             0.933                0.065           0.000        -0.028        0.491
Random-span        0.717             0.905                0.092           0.028         0.000        0.513
TF-IDF/log-odds    0.314             0.663                0.335           0.270         0.243        0.529
```

Key expected values for Table V:

```text
Budget  Anchor Drop mean +/- std  Gap vs Count  Gap vs Span
0.05    0.1288 +/- 0.0029         0.1173        0.1156
0.10    0.1338 +/- 0.0028         0.1113        0.1098
0.20    0.1718 +/- 0.0033         0.1280        0.1314
0.30    0.1779 +/- 0.0026         0.1133        0.0856
0.40    0.2340 +/- 0.0036         0.1403        0.0170
```

Key expected ablation values for Table VI at budget ratio 0.30:

```text
Variant                         Necessity Drop mean +/- std  Gap vs Count  Noise-like Ratio
Anchor-first full               0.1779 +/- 0.0026            0.1131        0.0070
w/o anchor-first allocation     0.1473 +/- 0.0029            0.0825        0.5324
w/o noise penalty               0.1350 +/- 0.0024            0.0702        0.5585
w/o burst cap                   0.1440 +/- 0.0023            0.0792        0.0032
w/o severity boost              0.1770 +/- 0.0025            0.1122        0.0070
w/o failure cue boost           0.1780 +/- 0.0026            0.1132        0.0027
```

Table VII should report 25/0/0 wins against both random baselines and original
RANK, and 22/3/0 against severity-only. Table VIII should report positive
anchor-first gaps under Logistic Regression, Linear SVM, and Random Forest.

## 8. Complexity

For a log sample with `n` lines, `m` candidate anchors, evidence budget `k`,
and context window `w`, anchor-first first scores anchors in `O(m)`, sorts them
in `O(m log m)`, and then expands context around selected anchors in `O(k w)`.
Since `m <= n` and `w` is fixed to 1 in the main experiments, the per-sample
time complexity is `O(m log m)` and the memory complexity is `O(n)` for
normalized lines and lightweight metadata. Across a dataset with total `N` log
lines, normalization and rule scoring are linear in `N`, while sorting is
performed independently within samples.

The frozen TF-IDF verifier additionally stores a vocabulary-limited sparse
representation with at most 30,000 features in the provided scripts.

## 9. Expected Runtime

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

## 10. Computing Infrastructure

The HDFS runtime numbers reported in the paper were measured on:

```text
CPU: Intel Core i9-10920X @ 3.50GHz, 12 cores / 24 threads
RAM: 62 GiB
OS: Linux 5.15.0-139-generic x86_64
Python: 3.12.7
GPU for core HDFS evidence generation: none; scripts are CPU-only
```

Runtime in the paper excludes data loading unless otherwise specified. Runtime
numbers may vary with CPU, RAM, disk speed, BLAS backend, and Python package
versions.

## 11. Optional LLM Grounding Pilot

The Qwen/Ollama grounding pilot is optional and is not required to reproduce the
main HDFS evidence-validity results. The main claims of the paper are based on
the frozen verifier deletion protocol, not on the LLM pilot.

For users who want to reproduce a similar optional pilot, use Qwen2.5-7B-Instruct
served by Ollama, e.g. the Ollama model tag `qwen2.5:7b`, with deterministic
decoding (`temperature = 0`). Because local model builds, serving backends, and
CPU/GPU configurations vary, small differences in optional pilot outputs and
latency are expected. The minimal reviewer artifact does not require this pilot
and does not use it to regenerate Tables III, V, VI, VII, or VIII.

## 12. Known Limitations

- The artifact does not redistribute LogHub/LogPai datasets.
- The artifact reproduces HDFS core validity tables, not every supplementary
  experiment in the paper.
- The Qwen/Ollama pilot is optional and not included in the minimal reviewer
  reproduction path.
- Runtime numbers may vary by machine and software stack.
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
