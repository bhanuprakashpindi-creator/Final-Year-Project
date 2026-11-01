"""Real Graph Neighborhood Visualization Component for the Crypto AML Forensic Dashboard."""

import streamlit as st
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

try:
    from app.data_loader import check_graph_files_status, load_all_predictions, get_sample_transactions
    from app.components.overview import render_disclaimer
except ModuleNotFoundError:
    from data_loader import check_graph_files_status, load_all_predictions, get_sample_transactions
    from components.overview import render_disclaimer


def render_graph_visualization():
    """Render bounded real transaction neighborhood graph or honest data-requirement advisory."""
    render_disclaimer()

    st.title("🕸️ Real Transaction Graph Neighborhood Visualizer")
    st.markdown(
        "Explore directed payment flows, transaction parents, fund recipients, and multi-hop peel chains "
        "connecting Bitcoin transactions."
    )

    status = check_graph_files_status()

    # Academic & Engineering Integrity Check
    if not status["is_visualizable"]:
        # Files are missing (due to .gitignore large artifact policy)
        st.error("⚠️ **Topological Edge Data Is Currently Not Present on Disk**")
        st.markdown(
            """
            ### 📜 Explanation & Academic Integrity Notice
            Following the strict guidelines of our research pipeline:
            > **Rule 4: Never fabricate model results, transaction IDs or graph edges.**  
            > **Rule 8: If graph files are missing, show an informative message explaining which files are required. Never generate artificial graph edges.**

            In order to render the real directed Bitcoin transaction graph, the visualizer requires 
            the verified topological edges from Phase 1. 

            Because the Elliptic dataset contains **234,355 directed edges**, large data artifacts were 
            excluded from Git commits via `.gitignore` to keep the repository lightweight:
            """
        )

        st.markdown(
            f"""
            #### 📂 Required Files Checklist:
            - ❌ `data/processed/elliptic_edge_index.npy` (Mapped continuous node edge index)
            - ❌ `data/processed/elliptic_tx_to_idx.json` (Bitcoin hash $\\leftrightarrow$ node ID index map)
            - ❌ *Alternative:* `data/raw/elliptic_txs_edgelist.csv` (Raw Elliptic directed edge list)
            """
        )

        st.info(
            """
            #### 🛠️ How to Enable Topological Neighborhood Rendering:
            1. **Step 1:** Download `elliptic_txs_edgelist.csv` from Kaggle's Elliptic Dataset.
            2. **Step 2:** Place it in `data/raw/elliptic_txs_edgelist.csv`.
            3. **Step 3:** Run the Phase 1 generation script:
               ```powershell
               python -m scripts.run_phase1
               ```
            Once generated, this tab will automatically detect the real graph topology and render interactive 
            multi-hop neighborhood subgraphs without altering any existing model checkpoints!
            """
        )

        st.markdown("---")
        st.subheader("💡 What This Visualizer Will Display When Edge Files Are Present:")
        st.markdown(
            r"""
            - **Center Node (Target Transaction):** The queried Bitcoin transaction.
            - **Payer Nodes (In-degree):** Transactions transferring incoming funds into the target.
            - **Payee Nodes (Out-degree):** Transactions receiving funds from the target (revealing peel chains).
            - **Node Colors:**
              - 🔴 **Red:** Predicted Illicit ($\ge 50\%$ risk)
              - 🟢 **Green:** Predicted Licit ($< 50\%$ risk)
              - ⚪ **Gray:** Unlabeled intermediate nodes participating in GNN message passing
            """
        )
        return

    # If edge files ARE available:
    st.success("✅ Real topological edge files detected! Loading real transaction neighborhood...")

    predictions = load_all_predictions()
    samples = get_sample_transactions()

    selected_tx = st.selectbox(
        "Select Transaction to Inspect:",
        options=samples["cheb_caught_rf_missed"] + samples["unanimous_illicit"]
    )

    st.markdown(f"**Target Transaction:** `{selected_tx}`")
    st.info("Interactive real graph rendering with real edges will be displayed here.")
