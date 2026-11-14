# I-NAS Project: Architecture & Reproduction Guide

## 1. Project Overview

This repository implements **I-NAS (Improved Novel Algorithmic Similarity)**, an enhanced user-based collaborative filtering similarity measure that extends the NAS method of Birim & Tümtürk (2022). The system benchmarks I-NAS against six classical similarity measures (COS, PCC, CPC, PIP, NHSM, Baseline NAS) on MovieLens 100K and FilmTrust datasets.

For the full mathematical derivation and research methodology, see [`explanation.md`](explanation.md).

---

## 2. Repository Structure

```
RDSProject/
├── explanation.md                       # Research paper: derivations, results, analysis
├── ai_agent_implementation_plan.md      # This file: architecture and reproduction guide
├── 10.21541-apjess.1060744-2203668.pdf  # Original NAS paper (Birim & Tümtürk, 2022)
├── dashboard.html                       # Interactive results dashboard
├── requirements.txt                     # Python dependencies
│
├── run_experiments.py                   # CLI benchmark runner (all 7 methods, 70/30 split)
├── run_grand_synthesis_benchmark.py     # 5-fold CV runner, ablation study, hypothesis testing
│
├── data/
│   ├── ml-100k/                         # MovieLens 100K (u.data + 5 official CV folds)
│   └── filmtrust/                       # FilmTrust (ratings.txt + trust.txt)
│
├── src/
│   ├── config.py                        # Dataset configs, K values, metric lists, split ratios
│   ├── data_loader.py                   # load_dataset(), split_train_test()
│   ├── metrics.py                       # MAE, RMSE, Precision@K, Recall@K, F1@K, MAP, MRR, FCP
│   ├── recommender.py                   # Top-K neighbour selection and weighted-average prediction
│   ├── evaluation_v3.py                 # 5-fold CV engine, paired t-test, merge eligibility check
│   ├── visualize.py                     # Figure generation (benchmark + I-NAS vs NAS comparison)
│   └── similarities/
│       ├── nas.py                       # Baseline NAS (Birim & Tümtürk, 2022)
│       ├── nas_v3.py                    # I-NAS implementation (Families A–F, NASv3Config)
│       ├── traditional.py               # COS, PCC, CPC
│       └── heuristic.py                 # PIP, NHSM
│
├── tests/
│   ├── conftest.py                      # Pytest fixtures and synthetic data generators
│   ├── test_table1_sample.py            # Verifies NAS against Table 3 of the original paper
│   ├── test_pipeline.py                 # End-to-end pipeline integration tests
│   ├── test_nas_v3.py                   # I-NAS unit tests: bounds, symmetry, zero-leakage
│   ├── test_metrics.py                  # Metric computation verification
│   ├── test_nas_original_regression.py  # Baseline NAS regression guards
│   └── test_recommender.py             # Recommender module tests
│
├── results/
│   ├── benchmark_results.csv            # Tabular results for all 7 methods × 6 K values × 2 datasets
│   ├── grand_synthesis_metrics.csv      # I-NAS vs Baseline NAS detailed CV metrics
│   ├── grand_synthesis_results.json     # Full structured CV payload with p-values
│   ├── baseline_nas_cv.json             # Official baseline 5-fold CV reference numbers
│   └── ablation_study.csv              # Step 0–5 sequential ablation data
│
└── figures/
    ├── Figure_1a_MAE_ml100k.png         # Paper figure reproductions (7 methods)
    ├── Figure_1b_MAE_filmtrust.png
    ├── Figure_2a_RMSE_ml100k.png
    ├── Figure_2b_RMSE_filmtrust.png
    ├── Figure_3a_Recall_ml100k.png
    ├── Figure_3b_Recall_filmtrust.png
    ├── Figure_4a_Precision_ml100k.png
    ├── Figure_4b_Precision_filmtrust.png
    ├── benchmark_comparison_master.png   # 4×2 master grid of all metrics
    ├── inas_vs_baseline_nas_ml100k.png   # Dedicated: I-NAS vs NAS on ML-100K
    ├── inas_vs_baseline_nas_filmtrust.png
    ├── inas_vs_baseline_error_reduction.png  # MAE/RMSE bar chart with p-values
    └── inas_ablation_progression.png     # Step-by-step waterfall chart
```

---

## 3. Quick Start

### 3.1 Prerequisites

- Python 3.10+
- System packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `seaborn`, `pytest`
- Optional (for hyperparameter search): `optuna` (installed in `.packages/`)

### 3.2 Run Tests

```bash
PYTHONPATH=.packages:. pytest
# Expected: 27 tests passed
```

### 3.3 Reproduce Benchmark (70/30 Split)

```bash
python3 run_experiments.py
# Runs all 7 methods on ML-100K and FilmTrust
# Generates figures/ and results/benchmark_results.csv
```

### 3.4 Reproduce 5-Fold CV and Ablation

```bash
python3 run_grand_synthesis_benchmark.py
# Runs ablation study + full CV + statistical tests
# Generates results/grand_synthesis_results.json
```

---

## 4. Key Module Reference

### `src/similarities/nas_v3.py`

The core I-NAS implementation. Key classes and functions:

- **`NASv3Config`** — Dataclass holding all 25+ hyperparameters organised by family (A–F).
- **`compute_nas_v3_similarity(R, r_med, cfg, trust_matrix)`** — Computes the full N×N similarity matrix. Vectorised with NumPy; the PS kernel iterates over users but vectorises item-level operations.
- **`predict_ratings_v3(sim_matrix, R_train, R_test_train, R_test_eval, K, cfg)`** — Predicts holdout ratings using Family E's case-amplified baseline predictor.
- **`compute_global_baseline(R_train, reg_u, reg_i)`** — Computes the $\mu + b_u + b_i$ baseline model with L2 regularisation.
- **`compute_katz_trust(trust_matrix, beta)`** — Computes the 2-hop Katz trust similarity for Family F.

### `src/evaluation_v3.py`

- **`run_5fold_cv_ml100k(cfg, K_neighbors, k_values)`** — Runs official 5-fold CV on ML-100K.
- **`run_5fold_cv_filmtrust(cfg, K_neighbors, k_values)`** — Runs 5-fold CV on FilmTrust with trust network.
- **`compute_statistical_acceptance(baseline_data, candidate_data)`** — Paired t-test, F1@10 regression check.

### `src/visualize.py`

- **`generate_all_paper_figures(df_results)`** — Produces Figures 1a–4b + master comparison.
- **`generate_inas_vs_baseline_figures()`** — Produces the 4 dedicated I-NAS vs Baseline NAS figures.

---

## 5. The I-NAS Grand Synthesis Configuration

The production-optimal hyperparameter set (selected via 600 Optuna trials across 6 families):

```python
NASv3Config(
    alpha_ps=1.25,      beta_svs=0.50,       gamma_ovl=0.35,       # Family A
    blend_mode="multiplicative",
    svs_mode="dual",    eta_disp=0.5,        eta_mean=0.2,          # Family B
    overlap_type="shrinkage", shrinkage_tau=12.0,                    # Family C
    use_surprisal=True, var_weight_kappa=0.0, rho_raw_vs_dev=0.70,  # Family D
    dist_p=1.0,         ps_zeta=1.5,         cross_median_penalty_xi=0.5,
    predictor_mode="case_amplification",                             # Family E
    case_amp_q=1.0,     epsilon_reg=0.001,
    baseline_reg_u=15.0, baseline_reg_i=25.0, blend_baseline_weight=0.35,
    use_trust=True,     beta_katz=0.1,       gamma_cold=0.5,        # Family F
    trust_fusion_mode="linear_blend"
)
```

---

## 6. Results Summary

| Dataset | Metric | Baseline NAS | I-NAS | Δ | p-value |
|:---|:---|:---:|:---:|:---:|:---:|
| ML-100K | MAE | 0.7658 | **0.7387** | −3.55% | 4.69×10⁻⁵ |
| ML-100K | RMSE | 0.9801 | **0.9431** | −3.77% | — |
| FilmTrust | MAE | 0.7178 | **0.6567** | −8.51% | 1.57×10⁻⁴ |
| FilmTrust | RMSE | 0.9920 | **0.8803** | −11.26% | — |

All improvements are statistically significant ($p < 0.05$) with no F1@10 regression.
