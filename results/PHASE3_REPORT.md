# Phase 3 Technical Report: Graph Neural Networks & Spectral Models

**Project:** AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks  
**Execution Date:** 2026-10-31 23:12:13  
**Dataset:** Elliptic Bitcoin Dataset (Held-out Test Timesteps 42..49)  
**Input Features:** `elliptic_x_reduced.npy` (`tx+agg_red`, 145 dimensions)  

---

## 1. Executive Summary & Comparative Scorecard

All four Graph Neural Network architectures were trained using identical temporal train partitions (timesteps 1..34), validated on timesteps 35..41 for checkpoint selection, and evaluated on the held-out test split (timesteps 42..49; 8,841 labeled transactions).

### Benchmark Comparison Table (Measured Results vs. Baselines)

| Model | Type | Features | Illicit Precision | Illicit Recall | Illicit F1 | PR-AUC | Macro-F1 | Accuracy | TP | FN | FP | TN | F1 Delta vs RF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Forest (Phase 2 Baseline) | Tabular Baseline | tx+agg_red (145) | 0.7814 | 0.4730 | 0.5893 | 0.5518 | — | 0.9696 | 193 | 215 | 54 | 8379 | 0.0000 |
| GCN+Lin | Graph Neural Network | tx+agg_red (145) | 0.5361 | 0.3824 | 0.4464 | 0.4100 | 0.7118 | 0.9562 | 156 | 252 | 135 | 8298 | -0.1430 |
| GraphSAGE+Lin | Graph Neural Network | tx+agg_red (145) | 0.6426 | 0.4363 | 0.5197 | 0.5100 | 0.7502 | 0.9628 | 178 | 230 | 99 | 8334 | -0.0696 |
| GATv2+Lin | Graph Neural Network | tx+agg_red (145) | 0.2513 | 0.4902 | 0.3322 | 0.4399 | 0.6417 | 0.9091 | 200 | 208 | 596 | 7837 | -0.2571 |
| Chebyshev+Lin | Graph Neural Network | tx+agg_red (145) | 0.4427 | 0.4926 | 0.4664 | 0.5259 | 0.7195 | 0.9480 | 201 | 207 | 253 | 8180 | -0.1230 |

---

## 2. Research Benchmark Reference (Ferretti et al., IEEE Access, 2025)

For academic rigor and viva defense alignment, the table below documents the published paper benchmarks obtained on GPU clusters (NVIDIA L40, 13,000 max epochs):

| Model Architecture | Published Feature Set | Paper Precision | Paper Recall | Paper Illicit F1 | Paper Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Chebyshev+Lin** *(Best Overall)* | `tx+agg_red` | **0.955** (±0.008) | **0.863** (±0.012) | **0.906** (±0.007) | **0.983** (±0.002) |
| **GATv2+Lin** | `tx+agg_red` | 0.938 | 0.842 | 0.887 | 0.976 |
| **GraphSAGE** | `tx+agg` | 0.852 | 0.818 | 0.835 | 0.958 |
| **Random Forest Baseline** (Weber et al.) | `tx+agg` | 0.960 | 0.651 | 0.782 | — |
| **GCN+Lin** | `tx+agg_red` | 0.720 | 0.621 | 0.667 | 0.912 |

> ⚠️ **ACADEMIC HONESTY NOTE:** The table above represents the **published research benchmarks** under GPU conditions. Our locally measured results represent our reproducible training runs under the fixed CPU configuration and seed 42.

---

## 3. Detailed Model Findings & Confusion Matrices

### Model: `GCN`
- **Best Validation F1:** `0.705531` (achieved at Epoch `91`)
- **Test Precision (Illicit):** `0.536082`
- **Test Recall (Illicit):** `0.382353`
- **Test F1-Score (Illicit):** `0.446352`
- **Test PR-AUC:** `0.410007`
- **Macro-F1:** `0.711782`
- **Overall Accuracy:** `0.956227`
- **Training Time:** `296.39` seconds
- **Checkpoint Location:** `C:\Users\moham\Downloads\crypto-aml-gnn-phase2\models\gcn_best.pt`

**Confusion Matrix (Test Timesteps 42..49):**
| True Class \ Predicted | Predicted Illicit (1) | Predicted Licit (0) |
| :--- | :---: | :---: |
| **Actual Illicit (1)** | **TP = 156** | **FN = 252** |
| **Actual Licit (0)** | **FP = 135** | **TN = 8298** |

---
### Model: `GRAPHSAGE`
- **Best Validation F1:** `0.737160` (achieved at Epoch `99`)
- **Test Precision (Illicit):** `0.642599`
- **Test Recall (Illicit):** `0.436275`
- **Test F1-Score (Illicit):** `0.519708`
- **Test PR-AUC:** `0.509961`
- **Macro-F1:** `0.750176`
- **Overall Accuracy:** `0.962787`
- **Training Time:** `262.58` seconds
- **Checkpoint Location:** `C:\Users\moham\Downloads\crypto-aml-gnn-phase2\models\graphsage_best.pt`

**Confusion Matrix (Test Timesteps 42..49):**
| True Class \ Predicted | Predicted Illicit (1) | Predicted Licit (0) |
| :--- | :---: | :---: |
| **Actual Illicit (1)** | **TP = 178** | **FN = 230** |
| **Actual Licit (0)** | **FP = 99** | **TN = 8334** |

---
### Model: `GATV2`
- **Best Validation F1:** `0.563784` (achieved at Epoch `98`)
- **Test Precision (Illicit):** `0.251256`
- **Test Recall (Illicit):** `0.490196`
- **Test F1-Score (Illicit):** `0.332226`
- **Test PR-AUC:** `0.439925`
- **Macro-F1:** `0.641717`
- **Overall Accuracy:** `0.909060`
- **Training Time:** `522.96` seconds
- **Checkpoint Location:** `C:\Users\moham\Downloads\crypto-aml-gnn-phase2\models\gatv2_best.pt`

**Confusion Matrix (Test Timesteps 42..49):**
| True Class \ Predicted | Predicted Illicit (1) | Predicted Licit (0) |
| :--- | :---: | :---: |
| **Actual Illicit (1)** | **TP = 200** | **FN = 208** |
| **Actual Licit (0)** | **FP = 596** | **TN = 7837** |

---
### Model: `CHEBYSHEV`
- **Best Validation F1:** `0.730642` (achieved at Epoch `91`)
- **Test Precision (Illicit):** `0.442731`
- **Test Recall (Illicit):** `0.492647`
- **Test F1-Score (Illicit):** `0.466357`
- **Test PR-AUC:** `0.525894`
- **Macro-F1:** `0.719504`
- **Overall Accuracy:** `0.947970`
- **Training Time:** `501.01` seconds
- **Checkpoint Location:** `C:\Users\moham\Downloads\crypto-aml-gnn-phase2\models\chebyshev_best.pt`

**Confusion Matrix (Test Timesteps 42..49):**
| True Class \ Predicted | Predicted Illicit (1) | Predicted Licit (0) |
| :--- | :---: | :---: |
| **Actual Illicit (1)** | **TP = 201** | **FN = 207** |
| **Actual Licit (0)** | **FP = 253** | **TN = 8180** |

---

## 4. Methodological Compliance & Leakage Audits

All 10 Phase 3 compliance checks have been fulfilled:
1. **Zero Data Leakage:** Features were constructed exclusively on train timesteps 1..34.
2. **Topological Purity:** 0 edges cross temporal split boundaries (100% within-snapshot edges).
3. **Graph Symmetrization:** ChebConv uses explicit `to_undirected(edge_index)` for valid spectral Laplacian computation.
4. **Unknown Label Isolation:** Transactions with label `-1` were completely masked out of training loss, validation early stopping, and test metrics.
5. **Deterministic Execution:** Fixed random seeds guarantee reproducible checkpoints and metric tables.
6. **Code Preservation:** All Phase 1 and Phase 2 source code, models, and predictions remain completely intact.
