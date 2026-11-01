
# PHASE 4 HANDOVER
## AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks

**Project Phase:** Phase 4 – Integration, Dashboard Development and Visualization  
**Technology:** Python, Streamlit, Plotly, Pandas, PyTorch, Scikit-learn  
**Dataset:** Elliptic Bitcoin Transaction Dataset

---

# 1. Phase 4 Objective

The main objective of Phase 4 is to integrate the outputs of Phase 1, Phase 2 and Phase 3 into a single interactive web-based dashboard.

The dashboard provides a user-friendly interface to explore cryptocurrency transaction data, compare machine learning and graph neural network models, visualize model performance, and inspect individual transaction predictions.

Phase 4 focuses on presenting the research results in an understandable and interactive format without retraining or changing the models developed in the previous phases.

### Main Objectives

- Integrate the outputs generated in Phases 1, 2 and 3.
- Develop an interactive Streamlit dashboard.
- Display dataset statistics and transaction information.
- Compare Random Forest and Graph Neural Network models.
- Visualize confusion matrices and training curves.
- Provide transaction-level prediction and risk analysis.
- Visualize transaction graph neighborhoods when graph data is available.
- Provide clear setup instructions and project documentation.

# 2. Integration of Previous Phases

Phase 4 uses the outputs from the previous three phases.

### Phase 1 – Data Preprocessing and Graph Construction

Phase 1 is responsible for preparing the Elliptic Bitcoin transaction dataset.

Main activities:
- Dataset loading and preprocessing.
- Handling transaction features and labels.
- Feature reduction.
- Graph construction.
- Creating transaction-to-index mappings.
- Preparing graph data for Graph Neural Networks.

Important expected files:

```text
data/processed/
    elliptic_x_all.npy
    elliptic_x_local.npy
    elliptic_x_reduced.npy
    elliptic_y.npy
    elliptic_masks.npz
    elliptic_edge_index.npy
    elliptic_pyg_data.pt
    elliptic_tensors.pt
    elliptic_tx_to_idx.json
    elliptic_idx_to_tx.json
```

Labels:
- 1 – Illicit transaction
- 0 – Licit transaction
- -1 – Unknown transaction

Unknown transactions must not be treated as licit transactions during supervised evaluation.

### Phase 2 – Random Forest Baseline

Phase 2 developed a Random Forest classification model using the processed transaction features.

Model configuration:
- Algorithm: Random Forest Classifier
- Number of estimators: 200
- Class weight: Balanced
- Random state: 42

Test performance:

| Metric | Result |
|---|---:|
| Precision | 0.781377 |
| Recall | 0.473039 |
| F1-Score | 0.589313 |
| PR-AUC | 0.551840 |

Main output files:

```text
models/random_forest.joblib
results/tables/random_forest_metrics.csv
results/random_forest_predictions.csv
results/random_forest_confusion_matrix.csv
```

### Phase 3 – Graph Neural Network Models

Phase 3 developed and evaluated four Graph Neural Network architectures.

Models:
1. GCN + Linear
2. GraphSAGE + Linear
3. GATv2 + Linear
4. Chebyshev + Linear

Test performance:

| Model | Precision | Recall | F1-Score | PR-AUC |
|---|---:|---:|---:|---:|
| GCN + Linear | 0.5361 | 0.3824 | 0.4464 | 0.4100 |
| GraphSAGE + Linear | 0.6426 | 0.4363 | 0.5197 | 0.5100 |
| GATv2 + Linear | 0.2513 | 0.4902 | 0.3322 | 0.4399 |
| Chebyshev + Linear | 0.4427 | 0.4926 | 0.4664 | 0.5259 |
| Random Forest | 0.7814 | 0.4730 | 0.5893 | 0.5518 |

These are the locally measured results from the project experiments, not the published research paper's results.

Main model files:

```text
models/
    gcn_best.pt
    graphsage_best.pt
    gatv2_best.pt
    chebyshev_best.pt
    random_forest.joblib
```

Prediction and evaluation files are stored under:

```text
results/
    predictions/
    tables/
    confusion_matrices/
    training_curves/
    data_summary/
```

# 3. Phase 4 System Architecture

The Phase 4 dashboard follows this general workflow:

```text
Elliptic Bitcoin Transaction Dataset
                 |
                 v
       Phase 1: Data Processing
                 |
                 v
        Phase 2: Random Forest
                 |
                 v
       Phase 3: GNN Model Results
                 |
                 v
       Phase 4: Data Integration
                 |
                 v
       Streamlit Dashboard
                 |
       +---------+----------+
       |         |          |
       v         v          v
   Dataset   Model       Transaction
 Statistics Comparison    Lookup
       |         |          |
       v         v          v
   Charts    Metrics    Risk Analysis
                 |
                 v
          Graph Visualization
```

# 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data loading and processing |
| NumPy | Numerical operations |
| Plotly | Interactive charts and visualizations |
| Matplotlib | Data visualization support |
| Scikit-learn | Random Forest and evaluation |
| PyTorch | Deep learning framework |
| PyTorch Geometric | Graph Neural Network support |
| NetworkX | Graph analysis and visualization support |

# 5. Dashboard Modules

The dashboard is implemented using Streamlit and reusable Python components.

### 5.1 Overview

The Overview section provides a general introduction to the project.

It displays:
- Project title and objective.
- Dataset overview.
- Model information.
- Research context.
- Overall project pipeline.

### 5.2 Dataset Statistics

This section presents information about the Elliptic dataset.

Dataset summary:

| Description | Count |
|---|---:|
| Total transactions | 203,769 |
| Illicit transactions | 4,545 |
| Licit transactions | 42,019 |
| Unknown transactions | 157,205 |
| Total edges | 234,355 |
| Timesteps | 49 |

The dashboard uses the available dataset summary information and visualizes the transaction distribution.

### 5.3 Model Comparison

This module compares the performance of the five models.

Models displayed:
- Random Forest
- GCN + Linear
- GraphSAGE + Linear
- GATv2 + Linear
- Chebyshev + Linear

Evaluation metrics:
- Precision
- Recall
- F1-Score
- PR-AUC
- Accuracy, where available

The dashboard uses the saved experimental metric files instead of generating artificial values.

### 5.4 Confusion Matrix Inspector

This module displays the confusion matrix for each model.

The four confusion matrix components are:

- True Positive (TP)
- True Negative (TN)
- False Positive (FP)
- False Negative (FN)

The user can select different models to inspect their classification performance.

### 5.5 Training Curves

The Training Curves section visualizes the training progress of the GNN models.

It uses the saved training history files.

Available information includes:
- Training loss
- Training F1-score
- Validation loss
- Validation precision
- Validation recall
- Validation F1-score
- Validation PR-AUC

The training curves help understand how the models performed during training and validation.

### 5.6 Transaction AML Lookup

This module allows users to search for a transaction using its transaction ID (`txId`).

It displays:
- Transaction ID
- Model prediction
- Illicit probability
- Risk classification
- Model-by-model comparison
- Average predicted illicit probability
- Risk score distribution

The dashboard uses saved prediction CSV files from the previous phases.

It does not claim to generate new predictions for transaction IDs that are not available in the saved prediction files.

**Example transaction:** `10403056`

The available saved predictions show that all five models classify this transaction as licit.

| Model | Prediction | Illicit Probability |
|---|---|---:|
| Random Forest | Licit | 0.00% |
| GCN + Linear | Licit | 0.46% |
| GraphSAGE + Linear | Licit | 0.00% |
| GATv2 + Linear | Licit | 0.13% |
| Chebyshev + Linear | Licit | 0.00% |

Average illicit probability: approximately 0.12%.

### 5.7 Graph Neighborhood Visualization

This module is designed to visualize the local graph neighborhood of a selected transaction.

It uses:
- Transaction-to-index mappings.
- Graph edge information.
- Transaction node information.
- Local graph connections.

The graph visualization requires the actual Phase 1 processed graph files.

If those files are unavailable, the dashboard should display a clear missing-data message rather than generate fake graph connections.

# 6. Project Directory Structure

The combined project follows this structure:

```text
crypto-aml-phase4/
│
├── app/
│   ├── app.py
│   ├── data_loader.py
│   │
│   └── components/
│       ├── overview.py
│       ├── dataset_stats.py
│       ├── model_comparison.py
│       ├── confusion_matrix.py
│       ├── transaction_lookup.py
│       └── graph_visualization.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── random_forest.joblib
│   ├── gcn_best.pt
│   ├── graphsage_best.pt
│   ├── gatv2_best.pt
│   └── chebyshev_best.pt
│
├── results/
│   ├── predictions/
│   ├── tables/
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── data_summary/
│
├── src/
├── tests/
│
├── requirements.txt
├── README.md
├── PHASE1_HANDOVER.md
├── PHASE2_HANDOVER.md
├── PHASE3_HANDOVER.md
└── PHASE4_HANDOVER.md
```

# 7. Installation and Execution

### Prerequisites

- Python installed.
- Git installed.
- Project repository cloned locally.

### Step 1: Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Open the project folder

```bash
cd crypto-aml-phase4
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run the dashboard

```bash
python -m streamlit run app/app.py
```

### Step 5: Open the dashboard

Open the local URL displayed in the terminal.

Usually:

```text
http://localhost:8501
```

# 8. Testing and Validation

The following dashboard checks have been performed or are included in the Phase 4 testing plan.

| Test | Purpose |
|---|---|
| Dashboard startup | Verify Streamlit application launches |
| Overview | Verify project information is displayed |
| Dataset Statistics | Verify dataset summary is loaded |
| Model Comparison | Verify saved model metrics are displayed |
| Confusion Matrix | Verify confusion matrix visualization |
| Training Curves | Verify saved training history is displayed |
| Transaction Lookup | Verify saved transaction predictions |
| Invalid Transaction ID | Verify friendly error handling |
| Graph Neighborhood | Verify graph visualization or missing-data handling |
| Disclaimer | Verify risk disclaimer is displayed |

The dashboard was launched locally and the transaction lookup was checked using transaction ID `10403056`.

Further end-to-end testing should be completed for all dashboard modules before final submission.

# 9. Limitations

1. Graph Neighborhood visualization depends on the availability of the processed Phase 1 graph files.
2. Transaction lookup is limited to transaction IDs present in the saved prediction files.
3. Model predictions depend on the locally saved Phase 2 and Phase 3 experimental results.
4. The locally measured GNN results may differ from the results reported in the research paper because of differences in experimental configurations and training conditions.
5. A model's high-risk prediction is not proof of criminal activity.

# 10. Disclaimer

This project is developed for academic research and educational purposes.

A high-risk prediction does not prove that a transaction is involved in money laundering or any other criminal activity.

The dashboard should not be used as the sole basis for financial, legal, or law-enforcement decisions.

# 11. Phase 4 Deliverables

The expected Phase 4 deliverables are:

- Integrated Streamlit dashboard.
- Reusable visualization components.
- Dataset statistics page.
- Model comparison interface.
- Confusion matrix visualization.
- GNN training curve visualization.
- Transaction-level AML lookup.
- Graph neighborhood visualization when source graph data is available.
- Updated requirements file.
- Project README.
- Phase 4 handover documentation.

# 12. Conclusion

Phase 4 brings together the data preprocessing, machine learning and graph neural network outputs developed in the earlier phases.

The Streamlit dashboard provides a centralized interface for exploring the Elliptic Bitcoin transaction dataset, comparing model performance, inspecting transaction predictions and visualizing graph information when the required data is available.

This integration makes the research project easier to understand, demonstrate and evaluate during the final-year project review.

**Phase 4 Contribution:** Dashboard development, integration, visualization, transaction lookup, model comparison and project documentation.
## 13. Automated Test Results

The project automated tests were executed using the following command:

```bash
python -m pytest tests/ -v
```

**Test execution summary:**

| Item | Result |
|---|---:|
| Total tests collected | 33 |
| Tests passed | 33 |
| Tests failed | 0 |
| Warnings | 262 |
| Execution time | 33.12 seconds |

**Result:** All 33 automated tests passed successfully.

The warnings were related to PyTorch and PyTorch Geometric deprecation notices. They did not cause any test failures.

This confirms that the tested Phase 1, Phase 2, and Phase 3 components passed their existing automated test suite. The Streamlit dashboard should also be checked separately through manual testing.