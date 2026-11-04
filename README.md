# AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks

An end-to-end research engineering pipeline and forensic investigation dashboard for cryptocurrency Anti-Money Laundering (AML) transaction classification using Graph Neural Networks on the Elliptic Bitcoin dataset.

## Project Overview

This project implements a four-phase research engineering pipeline:

1. **Phase 1 – Dataset Engineering:** Dataset validation, preprocessing, leakage-free splitting, and graph construction.
2. **Phase 2 – Traditional Machine Learning:** Random Forest baseline classification.
3. **Phase 3 – Graph Neural Networks:** GCN, GraphSAGE, GATv2, and Chebyshev spectral GNN.
4. **Phase 4 – Interactive Dashboard:** Streamlit and Plotly-based forensic investigation dashboard.

## Technology Stack

- Python
- PyTorch
- PyTorch Geometric
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Plotly
- NetworkX

## Dataset

Elliptic Bitcoin Transaction Dataset:

- Transaction nodes: 203,769
- Directed edges: 234,355
- Features: 167
- Classes: Licit, Illicit, Unknown

## Model Architectures

- Random Forest
- Graph Convolutional Network (GCN)
- GraphSAGE
- Graph Attention Network v2 (GATv2)
- Chebyshev Spectral GNN

## Dashboard Features

- Dataset statistics and visualization.
- Model performance comparison.
- Training and validation curves.
- Interactive confusion matrices.
- Individual transaction AML risk lookup.
- Graph neighborhood visualization.

## Installation

```bash
git clone https://github.com/bhanuprakashpindi-creator/Final-Year-Project.git

cd Final-Year-Project

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

## Run Tests

```bash
python -m unittest discover tests -v
```

## Research Disclaimer

This project is developed strictly for academic research and educational purposes. A high-risk prediction does not establish criminal activity and should not be used as the sole basis for financial, compliance, or legal decisions.

## License

MIT License.
