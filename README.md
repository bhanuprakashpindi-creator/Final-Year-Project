# AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.5+](https://img.shields.io/badge/PyG-2.5%2B-3C2179.svg)](https://pyg.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75.svg)](https://plotly.com/)
[![Tests: 33 Passed](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end research engineering pipeline and forensic investigation dashboard for cryptocurrency Anti-Money Laundering (AML) transaction classification using Graph Neural Networks on the Elliptic Bitcoin dataset.

---

## Project Overview

Cryptocurrency transaction networks present unique regulatory and compliance challenges due to pseudonymity and automated obfuscation strategies (such as peel chains and multi-hop mixing). Standard tabular machine learning approaches evaluate transactions in isolation, ignoring the rich relational topology of payment flows.

This project implements an end-to-end, four-phase research engineering pipeline to classify illicit Bitcoin transactions according to the state-of-the-art methodology published by Ferretti et al. (*IEEE Access*, 2025):

### Complete Four-Phase Architecture

```
Elliptic Bitcoin Transaction Dataset (203,769 Nodes, 234,355 Edges)
                             │
                             ▼
  Phase 1: Dataset Engineering, Validation, Leakage-Free Preprocessing & Graph Construction
                             │
                             ▼
  Phase 2: Traditional Tabular Baseline (Random Forest Classifier, 200 Trees)
                             │
                             ▼
  Phase 3: Graph Neural Networks (GCN, GraphSAGE, GATv2, Chebyshev+Lin Spectral GNN)
                             │
                             ▼
  Phase 4: Interactive Forensic Investigation Dashboard (Streamlit & Plotly)
        ├── 📌 Overview & End-to-End Pipeline Architecture
        ├── 📊 Verified Dataset Statistics & Leakage-Free Splits
        ├── 📈 5-Model Comparative Scorecard & 100-Epoch Training Curves
        ├── 🔲 Forensic Confusion Matrix Inspector (Interactive Heatmaps)
        ├── 🔍 Single Transaction AML Risk Lookup (8,841 Test Transactions)
        └── 🕸️ Real Graph Neighborhood Topological Visualizer
```

---

## Comprehensive Project Documentation Suite

For exhaustive technical and academic details, consult our dedicated documentation modules:

| Document | Purpose / Key Contents |
| :--- | :--- |
| 📖 [**Phase 1 Technical Report**](docs/PHASE1_TECHNICAL_REPORT.md) | **Primary academic & engineering report** (problem statement, methodology, leakage audit, validation, and architecture). |
| 📊 [**Research Metrics & Scorecard**](docs/RESEARCH_METRICS_AND_SCORECARD.md) | **Official scorecard** isolating Paper Benchmarks (Category A), Phase 1 Results (Category B), and Model Scores (Category C). |
| 🎓 [**Viva Defense Notes**](docs/PHASE1_VIVA_NOTES.md) | **Presentation defense guide** with 2-minute pitch, technical explanations, and **25 top viva questions and answers**. |
| ⚡ [**Quick Reference Cheatsheet**](docs/PHASE1_QUICK_REFERENCE.md) | **Compact one-page review** covering key statistics, label encodings, split boundaries, and essential commands. |
| 🔬 [**Research Paper Reference**](docs/RESEARCH_REFERENCE.md) | In-depth review of Ferretti et al. (IEEE Access, 2025) and mathematical basis of Chebyshev spectral convolutions. |
| 🤝 [**Phase 1 Handover Guide**](PHASE1_HANDOVER.md) | Downstream integration specifications and tensor representations for Phase 2 and Phase 3. |
| 🤝 [**Phase 2 Handover Guide**](PHASE2_HANDOVER.md) | Baseline Random Forest methodology, hyperparameters, confusion matrix, and reproduction commands. |
| 🤝 [**Phase 3 Handover Guide**](PHASE3_HANDOVER.md) | GNN architectures (GCN, GraphSAGE, GATv2, Chebyshev), checkpoints, and experimental scorecard. |
| 🤝 [**Phase 4 Handover Guide**](PHASE4_HANDOVER.md) | Complete Streamlit dashboard integration, visualization components, transaction lookup, and deployment notes. |
| 📈 [**Phase 3 Technical Report**](results/PHASE3_REPORT.md) | Academic analysis comparing local CPU measurements against published HPC GPU benchmarks. |

---

## Research Paper Reference & Benchmark Targets

- **Paper Title**: *Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification*
- **Authors**: Stefano Ferretti, Gabriele D’Angelo, and Vittorio Ghini (University of Bologna, Italy)
- **Publication**: *IEEE Access*, Volume 13, 2025, pp. 50204–50218 (DOI: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240))
- **Best-Performing Configuration**: **`Chebyshev+Lin` with reduced aggregated features (`tx+agg_red`)**:
  - **Illicit F1-Score**: **0.906** (±0.007)
  - **Illicit Precision**: **0.955** (±0.008)
  - **Illicit Recall**: **0.863** (±0.012)
  - **Macro-F1**: **0.983** (±0.002)

### Academic Distinction: Category A (Paper) vs. Category B (Locally Measured)
To maintain strict academic integrity during defense evaluations:
1. **Category A: Published Paper Targets (Ferretti et al., 2025):** Trained on an HPC cluster node with an **NVIDIA L40 GPU (48 GB VRAM)**, using up to **13,000 epochs** and early stopping patience = 1,000.
2. **Category B: Our Locally Measured Results:** Trained locally on CPU using **100 epochs** and early stopping patience = 25.

---

## Verified Dataset Statistics (Measured from Actual Elliptic Data)

- **Total Transaction Nodes**: `203,769`
- **Total Directed Payment Edges**: `234,355`
- **Raw Feature Columns**: `167` (1 txId + 1 timestep + 93 local + 72 aggregated neighbor features)
- **Reduced Feature Matrix (`tx+agg_red`)**: `145` features (20 collinear aggregated features pruned on train split)
- **Class Breakdown**:
  - `Illicit` (Class 1): **4,545** (2.23% of total; 9.76% of labeled transactions)
  - `Licit` (Class 0): **42,019** (20.62% of total; 90.24% of labeled transactions)
  - `Unknown`: **157,205** (77.15% of total; encoded as `-1`, masked out of loss)
- **Temporal Split Partitions (0 Cross-Split Edges)**:
  - **Train (Timesteps 1–34):** 136,265 total nodes (29,894 labeled: 3,462 illicit, 26,432 licit)
  - **Validation (Timesteps 35–41):** 30,680 total nodes (7,829 labeled: 675 illicit, 7,154 licit)
  - **Held-Out Test (Timesteps 42–49):** 36,824 total nodes (8,841 labeled: 408 illicit, 8,433 licit)

---

## Experimental Results Scorecard (Held-Out Test Split)

Evaluated strictly on held-out timesteps 42..49 (8,841 labeled transactions: 408 illicit, 8,433 licit):

| Model Architecture | Model Family | Features | Illicit Precision | Illicit Recall | Illicit F1 | PR-AUC | Macro-F1 | Overall Accuracy | True Positives (TP) | False Negatives (FN) | False Positives (FP) | True Negatives (TN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** *(Phase 2)* | Tabular Baseline | `tx+agg_red` (145) | **0.7814** | 0.4730 | **0.5893** | **0.5518** | — | **0.9696** | 193 | 215 | **54** | **8,379** |
| **GraphSAGE+Lin** *(Phase 3)* | Spatial GNN | `tx+agg_red` (145) | 0.6426 | 0.4363 | 0.5197 | 0.5100 | **0.7502** | 0.9628 | 178 | 230 | 99 | 8,334 |
| **Chebyshev+Lin** *(Phase 3)* | Spectral GNN | `tx+agg_red` (145) | 0.4427 | **0.4926** | 0.4664 | 0.5259 | 0.7195 | 0.9480 | **201** | **207** | 253 | 8,180 |
| **GCN+Lin** *(Phase 3)* | Spatial GNN | `tx+agg_red` (145) | 0.5361 | 0.3824 | 0.4464 | 0.4100 | 0.7118 | 0.9562 | 156 | 252 | 135 | 8,298 |
| **GATv2+Lin** *(Phase 3)* | Attention GNN | `tx+agg_red` (145) | 0.2513 | 0.4902 | 0.3322 | 0.4399 | 0.6417 | 0.9091 | 200 | 208 | 596 | 7,837 |

### Forensic Analysis:
- **Chebyshev+Lin detected 201 illicit transactions (Recall = 49.26%)**, catching 8 additional laundering schemes missed by Random Forest (193 TP; Recall = 47.30%).
- **Precision vs. Recall Tradeoff:** Random Forest achieved higher precision (0.7814) with fewer false alarms (54 FP), whereas GNNs, through multi-hop neighbor aggregation, surfaced more exploratory transactions along laundering peel chains.

---

## Setup & Quickstart

### 1. Environment Setup (Windows PowerShell / macOS / Linux)

```powershell
# Clone repository
git clone https://github.com/bhanuprakashpindi-creator/crypto-aml-phase4.git
cd crypto-aml-phase4

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1   # On Windows
# source venv/bin/activate    # On Linux/macOS

# Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Launch the Phase 4 Streamlit Forensic Dashboard

From the project root directory:

```powershell
python -m streamlit run app/app.py
```
*or:*
```powershell
streamlit run app/app.py
```

The application will start a local web server (usually at `http://localhost:8501`) and automatically open in your default browser.

### 3. Running Pipelines & Automated Unit Tests

```powershell
# Run automated test suite (33 unit tests across Phases 1, 2, and 3)
python -m unittest discover tests -v

# (Optional) Retrain Random Forest baseline (Phase 2)
python -m src.models.random_forest

# (Optional) Retrain Chebyshev+Lin GNN model (Phase 3)
python -m scripts.run_phase3 --models chebyshev --epochs 100 --patience 25
```

---

## Dashboard Features (Phase 4)

1. **📌 Overview & End-to-End Pipeline Architecture**: Problem context (peel chains, mixing services), GNN message-passing motivation, and academic benchmarks.
2. **📊 Dataset Statistics & Splits**: Interactive Plotly donut charts of class imbalance (90:10 licit:illicit, 77% unknown) and stacked temporal split distribution.
3. **📈 Model Comparison & Curves**: Side-by-side metric bar charts (Precision, Recall, F1, PR-AUC, Accuracy), precision-recall tradeoff scatter plot, and 100-epoch training loss / validation F1 curves.
4. **🔲 Confusion Matrix Inspector**: Interactive heatmaps for all 5 models clearly labeling True Positives (TP), False Negatives (FN), False Positives (FP), and True Negatives (TN).
5. **🔍 Single Transaction AML Risk Lookup**: Search any Bitcoin transaction ID (`txId`) among the 8,841 held-out test transactions to view model predictions, probability bars, and cross-model consensus. Includes one-click presets for viva demonstrations (`txId 2376671`, `txId 14782227`, `txId 10403056`, `txId 268186`).
6. **🕸️ Real Graph Neighborhood Visualizer**: Resilient topological inspector that checks for real edge data and displays an honest missing-data advisory rather than fabricating artificial graph edges.

---

## Processed Output Specifications (`data/processed/`)

| Filename | Format | Shape / Type | Description |
| :--- | :--- | :--- | :--- |
| `elliptic_x_all.npy` | NumPy float32 | `[203769, 165]` | Full numeric features (93 local + 72 aggregated). |
| `elliptic_x_local.npy` | NumPy float32 | `[203769, 93]` | Local-only transaction features. |
| `elliptic_x_reduced.npy` | NumPy float32 | `[203769, 145]` | **Paper best feature set** (20 collinear pruned on train). |
| `elliptic_y.npy` | NumPy int64 | `[203769]` | Labels: `1` = illicit, `0` = licit, `-1` = unknown. |
| `elliptic_timesteps.npy` | NumPy int32 | `[203769]` | Discrete timestep indicators `1..49`. |
| `elliptic_edge_index.npy` | NumPy int64 | `[2, 234355]` | Directed edge indices mapped to continuous node IDs. |
| `elliptic_masks.npz` | NumPy archive | Compressed bools | `train_mask`, `val_mask`, `test_mask` + labeled subsets. |
| `elliptic_tx_to_idx.json` | JSON dict | 203,769 keys | Maps original Bitcoin `txId` $\rightarrow$ node index. |
| `elliptic_idx_to_tx.json` | JSON list | 203,769 items | Maps node index $\rightarrow$ original Bitcoin `txId`. |
| `elliptic_pyg_data.pt` | PyG Data | Container | Ready-to-train PyTorch Geometric graph object. |
| `elliptic_tensors.pt` | PyTorch Dict | 12 Tensors | Pure PyTorch tensor dictionary bundle. |

---

## Legal & Forensic Disclaimer

> ⚠️ **LEGAL & FORENSIC DISCLAIMER:**  
> A high-risk prediction is not proof of criminal activity. This project and its interactive dashboard are developed strictly for academic research and educational demonstration purposes and should not be used as the sole basis for financial, compliance, or legal decisions.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
