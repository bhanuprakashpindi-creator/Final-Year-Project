"""Overview component for the Crypto AML Forensic Dashboard."""

import streamlit as st


def render_disclaimer():
    """Render prominent academic & legal disclaimer required by regulatory and academic guidelines."""
    st.warning(
        "⚠️ **LEGAL & FORENSIC DISCLAIMER:**\n\n"
        "A high-risk prediction is not proof of criminal activity. "
        "This dashboard is an academic research demonstration and should not be used as the "
        "sole basis for financial or legal decisions."
    )


def render_overview():
    """Render project overview, problem statement, research context, and architecture."""
    render_disclaimer()

    st.title("🛡️ AI-Powered Cryptocurrency Anti-Money Laundering")
    st.subheader("Forensic Investigation & Comparative Evaluation Using Graph Neural Networks")

    st.markdown(
        """
        ### 💡 The Money Laundering Challenge in Cryptocurrency
        Cryptocurrency networks such as Bitcoin operate with pseudonymous public addresses. 
        Criminal syndicates and sanctioned entities exploit this structure to execute sophisticated 
        laundering strategies:
        - **Peel Chains:** Splitting illicit funds into a cascade of tiny payments where a small change 
          address is repeatedly "peeled" off to evade threshold reporting limits.
        - **Multi-Hop Mixers:** Shuffling coins across hundreds of intermediate hops to obfuscate transaction provenance.
        - **Fast Layering:** Laundering funds across rapid automated transaction bursts.

        Standard tabular machine learning algorithms (like Decision Trees or Random Forests) evaluate 
        transactions **in complete isolation**. They examine single-transaction attributes (fee, output value, 
        timestamp) while remaining completely blind to the surrounding **graph topology of payment flows**.
        """
    )

    st.info(
        "💡 **Why Graph Neural Networks (GNNs)?**\n\n"
        "Graph Neural Networks perform **neighborhood message passing**, mathematically propagating information "
        "from fund sources to fund sinks. By aggregating context from 1-hop and 2-hop connected transaction neighbors, "
        "GNNs expose laundering patterns that no tabular model can detect alone."
    )

    st.markdown("---")
    st.header("🏗️ End-to-End Pipeline Architecture")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            """
            #### 1️⃣ Phase 1: Graph Engine
            - **Dataset:** Elliptic Bitcoin
            - **Nodes:** 203,769 transactions
            - **Edges:** 234,355 directed flows
            - **Splitting:** Strict temporal split (1–34 train, 35–41 val, 42–49 test)
            - **Leakage Audit:** 0 cross-split edges
            """
        )
    with col2:
        st.markdown(
            """
            #### 2️⃣ Phase 2: ML Baseline
            - **Model:** Random Forest
            - **Input:** Pruned tabular features (`tx+agg_red`, 145 dimensions)
            - **Class Balancing:** Balanced weights
            - **Role:** Establish tabular benchmark (Weber et al. / Ferretti et al.)
            """
        )
    with col3:
        st.markdown(
            """
            #### 3️⃣ Phase 3: GNN Models
            - **Architectures:** GCN, GraphSAGE, GATv2, Chebyshev+Lin
            - **Laplacian:** Spectral polynomial convolutions ($K=3$)
            - **Evaluation:** Strict held-out test split (timesteps 42..49)
            - **Checkpoints:** Best validation illicit F1
            """
        )
    with col4:
        st.markdown(
            """
            #### 4️⃣ Phase 4: Forensic UI
            - **Dashboard:** Interactive Streamlit
            - **Comparative Scorecard:** 5 models
            - **Audit Trail:** Confusion matrices & training curves
            - **Transaction Search:** 8,841 test transactions
            - **Topology Inspector:** Real graph inspector
            """
        )

    st.markdown("---")
    st.header("📖 Academic Research Reference & Benchmarks")

    st.markdown(
        """
        This pipeline implements and replicates the foundational research published by:
        > **Ferretti, S., D'Angelo, G., & Ghini, V. (2025).**  
        > *Enhancing Anti-Money Laundering Frameworks: An Application of Graph Neural Networks in Cryptocurrency Transaction Classification.*  
        > **IEEE Access**, Volume 13, pp. 50204–50218. DOI: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240)
        """
    )

    st.markdown(
        """
        | Research Dimension | Category A: Published Paper Target (GPU Cluster) | Category B: Our Locally Measured Result (Local CPU) |
        | :--- | :---: | :---: |
        | **Hardware Infrastructure** | NVIDIA L40 GPU (48 GB VRAM) | Standard Multi-Core CPU |
        | **Training Epoch Ceiling** | Up to 13,000 Epochs (Patience: 1,000) | 100 Epochs (Patience: 25–40) |
        | **Best Model Family** | Chebyshev+Lin (`tx+agg_red`) | Random Forest (High Precision: 0.7814) / Chebyshev (High Recall: 0.4926) |
        | **Target Illicit F1-Score** | **0.906** (±0.007) | **0.5893** (RF) / **0.4664** (Chebyshev) |
        | **Target Illicit Recall** | **0.863** (±0.012) | **0.4926** (Chebyshev catches 201 illicit txs) |
        """
    )
