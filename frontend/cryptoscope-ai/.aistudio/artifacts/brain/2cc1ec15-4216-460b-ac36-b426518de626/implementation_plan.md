# CryptoScope AI — Anti-Money Laundering Forensic Intelligence Platform
*(Revised Technical Architecture & Implementation Blueprint)*

CryptoScope AI is an academic and operational anti-money laundering (AML) forensic platform built on the **Elliptic Bitcoin Transaction Dataset** and Graph Neural Networks. This blueprint incorporates all ten technical corrections regarding dataset authenticity, model checkpoint integrity, graph extraction mechanics, explicit telemetry status, and research metric fidelity.

---

## User Review & Critical Technical Corrections

> [!IMPORTANT]
> The following non-negotiable technical requirements are enforced across the entire architecture:

1. **Strict Data Authenticity**: No synthetic Bitcoin transaction hashes, wallet identities, fiat amounts, real-world timestamps, or fictitious KYC/sanctions tags will be introduced. The Elliptic dataset operates strictly on dataset transaction IDs (`txId`), timestep indices ($t \in [1, 49]$), 145 anonymized local/aggregated graph features, and class labels (`licit`, `illicit`, `unknown`).
2. **Dataset Transaction Identifiers vs. On-Chain Hashes**: The UI explicitly labels identifiers as *Elliptic Dataset Transaction IDs* (numerical identifiers such as `230425980`, `5530414`, `232022460`) and clearly distinguishes them from 64-character raw Bitcoin transaction hashes.
3. **Graph Architecture**: Cytoscape.js renders the directed transaction topology in the frontend. Directed edges strictly reflect verified connections from `elliptic_edge_index.npy`. The graph starts at 1-hop direct counterparties (inflows and outflows) with on-demand 2-hop expansion.
4. **FastAPI Backend Integration Contract**:
   - `GET /api/health`: Health status, active device, and loaded model checkpoints (Random Forest, GCN, GraphSAGE, GATv2, Chebyshev).
   - `GET /api/transaction/{tx_id}`: Dataset metadata, timestep ($1 \le t \le 49$), dataset split, and ground-truth label (if labeled).
   - `POST /api/predict`: Real-time inference using the 5 trained model checkpoints; returns illicit and licit probabilities, consensus, and risk classification.
   - `GET /api/graph/{tx_id}?hops=1`: Verified 1-hop or 2-hop directed neighborhood extracted via NetworkX from `elliptic_edge_index.npy`.
   - `GET /api/metrics`: Verified evaluation metrics (Precision, Recall, Illicit F1, PR-AUC, Macro-F1, Accuracy) and confusion matrices.
   - `GET /api/dataset/stats`: Real Elliptic dataset statistics and temporal distribution.
5. **Transparent 3-State Telemetry**: The application explicitly and unmistakably distinguishes:
   - `🟢 LIVE BACKEND`: Active connection to the running FastAPI server; real-time inference executed by PyTorch Geometric / Scikit-Learn checkpoints.
   - `🔵 VERIFIED BENCHMARK`: Read-only examination of authentic test-split transactions with recorded ground-truth and frozen model evaluation outputs. Clearly badged as historical experiment data; never misrepresented as live inference.
   - `🔴 OFFLINE`: Backend unreachable; clear diagnostic instructions provided without mock data fallback.
6. **Unmodified Empirical Results**: Baseline Random Forest and GNN (GCN+Lin, GraphSAGE+Lin, GATv2+Lin, Chebyshev+Lin) metrics remain 100% faithful to actual experimental logs. We do not alter, smooth, or inflate GNN scores over Random Forest.

---

## 1. Overview & Core Concept

- **What It Does**: Provides a rigorous forensic interface for analyzing Bitcoin transactions from the Elliptic dataset (203,769 transactions, 234,355 directed edges, 49 timesteps). Enables multi-model classification, topological network inspection, experimental validation, and academic literature analysis.
- **Target Audience**: Blockchain forensic investigators, compliance officers, and academic researchers evaluating GNN performance on dynamic transaction graphs.
- **Key Value**: Exposes structural UTXO flow topology (peeling chains, mixer patterns, dispersion) while preserving absolute empirical truth and model auditability.

---

## 2. User Experience & Visual Design

Following the Google Stitch specification (`/public/assets/aistudio/Image_13_de29f876.markdown`) and SaaS dashboard guidelines:

### Visual Language & Design Tokens
- **Canvas Base**: `#080C14` / `#0F131C` (obsidian canvas).
- **Surface Elevation 1**: `#111827` (command bar, primary container shells).
- **Surface Elevation 2**: `#1E293B` (interactive cards, input fields, table header rows).
- **Surface Elevation 3**: `#334155` (active tab segments, hover highlights, flyout sheets).
- **Forensic Semantics**:
  - Primary Cyber Cyan: `#00F2FE` (primary actions, focus rings, trace anchors).
  - Electric Blue: `#3B82F6` (verified topological links).
  - Licit Label: `#10B981` (verified licit class in dataset).
  - Elevated Warning: `#F59E0B` (medium illicit probability).
  - Critical Illicit Label: `#EF4444` (verified illicit class in dataset).
  - Unknown Label: `#64748B` (unlabeled in dataset).
- **Typography**: `Geist` / `Plus Jakarta Sans` for interface copy, paired with `JetBrains Mono` and `tabular-nums` for all transaction IDs, timestamps, and probabilities.

### Page Specifications (All 6 Pages)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TOP COMMAND BAR (56px)                          │
│ [CryptoScope AI]  ──  Home · Investigate · Network · Intelligence ...  │
│                       [Telemetry: LIVE / BENCHMARK / OFFLINE]          │
└──────┬──────────────────────────────────────────────────────────┬──────┘
       │                                                          │
       ▼                                                          ▼
┌───────────────┐                                          ┌───────────────┐
│   HOME VIEW   │ ── [Submit TxID / Select Preset] ──────► │  INVESTIGATE  │
└───────────────┘                                          └───────┬───────┘
                                                                  │
                                      [View in Graph Explorer]    │
                                                                  ▼
┌───────────────┐                                          ┌───────────────┐
│   DATASET &   │ ◄─────────────────────────────────────── │    NETWORK    │
│   RESEARCH    │                                          │   EXPLORER    │
└───────────────┘                                          └───────────────┘
```

1. **Home Page**:
   - Institutional banner: *"Intelligence Beyond Transactions."*
   - Subtitle: *"Investigate cryptocurrency transaction networks using advanced machine learning and graph intelligence."*
   - Search input for Elliptic Transaction ID with validation and quick-pick presets from the test split.
   - Interactive animated topological network preview illustrating UTXO flow.
   - Overview of the 5-model evaluation suite.
2. **Investigate Page**:
   - Investigation HUD: Dataset Transaction ID, Timestep ($t \in [1, 49]$), Split (Train / Val / Test), Ground Truth (if labeled).
   - Mode Indicator: Explicitly displays `LIVE INFERENCE` (with latency and model device) or `VERIFIED BENCHMARK RECORD`.
   - Five Model Cards: Random Forest, GCN+Lin, GraphSAGE+Lin, GATv2+Lin, Chebyshev+Lin with animated illicit/licit probability bars.
   - Model Consensus & Risk Assessment: Average illicit probability, vote agreement ($x/5$), and risk bracket.
   - Quick navigation into the Network Explorer for the analyzed transaction.
3. **Network Explorer Page**:
   - Full Cytoscape.js interactive graph workspace with pan, zoom, fit, and node physics.
   - Verified 1-hop neighborhood rendered by default (incoming and outgoing transaction edges with arrows).
   - On-demand 2-hop expansion button.
   - Node Inspector Drawer: Selected node details, in-degree, out-degree, timestep, dataset classification, and local feature summary.
4. **Intelligence Page**:
   - Exact experimental metric comparison table: Precision, Recall, Illicit F1, PR-AUC, Macro-F1, Accuracy for all 5 models.
   - Full $2 \times 2$ Confusion Matrices for all five architectures.
   - Training loss curves and validation F1 trajectories.
   - Metric comparison toggle (Radar view / Multi-bar view).
5. **Dataset Page**:
   - Elliptic dataset macro statistics (203,769 transactions, 234,355 directed edges, 145 features, 49 timesteps).
   - Class breakdown: 4,545 Illicit (2.2%), 42,019 Licit (20.6%), 157,205 Unknown (77.1%).
   - Temporal split architecture (Train: $t \le 34$, Val: $35 \le t \le 41$, Test: $42 \le t \le 49$).
   - Critical data leakage audit detailing why random k-fold splits cause severe temporal leakage and how time-step splitting ensures realistic evaluation.
6. **Research Page**:
   - Theoretical problem statement: Graph topology vs. flat tabular features in Bitcoin AML.
   - IEEE Access research citation (Ferretti et al., 2025, DOI: 10.1109/ACCESS.2025.3552240).
   - Mathematical formulations of GCN, GraphSAGE, GATv2, and Chebyshev spectral filters.
   - Published IEEE Access results vs. measured local experimental results table with transparent discussion of differences.

---

## 3. Technical Architecture & Integration Strategy

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND (React 19)                           │
│                                                                          │
│  - Navigation with Live Prober (polls /api/health every 15s)             │
│  - Views: Home, Investigate, NetworkExplorer, Intelligence,              │
│           Dataset, Research                                              │
│  - Graph Canvas: Cytoscape.js directed dagre/cose layout                │
│  - Telemetry Banner: LIVE / VERIFIED BENCHMARK / OFFLINE                 │
│  - Elliptic Benchmark Ledger (offline-safe verified test transactions)   │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │ HTTP (Configurable: /api or :8000)
                                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         FASTAPI BACKEND (Python)                         │
│                                                                          │
│  - main.py: FastAPI app with CORS middleware                             │
│  - models/: Checkpoint loader for RF (.joblib/.pkl) & PyTorch GNNs (.pt) │
│  - graph/: NetworkX subgraph extractor from elliptic_edge_index.npy     │
│  - data/: Transaction index mappings (tx_to_idx.json, idx_to_tx.json)   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Contracts & API Specification
- All endpoints return structured JSON with explicit source metadata:
  ```json
  {
    "status": "success",
    "source": "live_model_inference", // or "verified_benchmark_record"
    "transaction_id": "230425980",
    "timestep": 43,
    "split": "test",
    "ground_truth": "illicit",
    "predictions": {
      "random_forest": { "illicit_prob": 0.88, "licit_prob": 0.12, "prediction": "illicit" },
      "gcn": { "illicit_prob": 0.84, "licit_prob": 0.16, "prediction": "illicit" },
      "graphsage": { "illicit_prob": 0.89, "licit_prob": 0.11, "prediction": "illicit" },
      "gatv2": { "illicit_prob": 0.86, "licit_prob": 0.14, "prediction": "illicit" },
      "chebyshev": { "illicit_prob": 0.82, "licit_prob": 0.18, "prediction": "illicit" }
    },
    "consensus": {
      "average_illicit_prob": 0.858,
      "consensus_class": "illicit",
      "agreement": "5/5",
      "risk_level": "critical"
    }
  }
  ```

---

## 4. Verification & Testing Protocol

Before marking any stage complete:
- **Build & Lint Verification**: Zero TypeScript errors, zero missing imports, zero lint warnings (`compile_applet` & `lint_applet`).
- **Interactive Testing**: Verify search validation, preset selection, probability bar animations, Cytoscape graph rendering (zoom, pan, node click), 2-hop expansion, confusion matrix modal/toggles, and responsive layout across mobile and desktop.
- **Telemetry Verification**: Verify that toggling backend connection settings accurately shifts status between LIVE, BENCHMARK, and OFFLINE without data corruption or false claims.
