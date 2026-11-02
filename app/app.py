"""AI-Powered Cryptocurrency Anti-Money Laundering Forensic Dashboard.

Phase 4 Interactive Dashboard integrating Phase 1, Phase 2, and Phase 3 artifacts.
Replication and forensic extension of Ferretti et al. (IEEE Access, 2025).
"""

import os
import sys
from pathlib import Path

# Fix Streamlit import paths and sys.modules conflict
ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# If Streamlit added APP_DIR to sys.path, remove it so 'app' doesn't collide with 'app.py'
while str(APP_DIR) in sys.path:
    sys.path.remove(str(APP_DIR))

# If 'app' is in sys.modules as a single file module (app.py) without __path__, remove it
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

# Ensure repository root is the first entry in sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
else:
    sys.path.remove(str(ROOT_DIR))
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

try:
    from app.components.overview import render_overview, render_disclaimer
    from app.components.dataset_stats import render_dataset_stats
    from app.components.model_comparison import render_model_comparison
    from app.components.confusion_matrix import render_confusion_matrix
    from app.components.transaction_lookup import render_transaction_lookup
    from app.components.graph_visualization import render_graph_visualization
    from app.data_loader import load_dataset_manifest
except ModuleNotFoundError:
    from components.overview import render_overview, render_disclaimer
    from components.dataset_stats import render_dataset_stats
    from components.model_comparison import render_model_comparison
    from components.confusion_matrix import render_confusion_matrix
    from components.transaction_lookup import render_transaction_lookup
    from components.graph_visualization import render_graph_visualization
    from data_loader import load_dataset_manifest


def setup_page():
    """Configure Streamlit layout, page titles, and styling."""
    st.set_page_config(
        page_title="Crypto AML GNN Forensic Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Modern Custom Styling
    st.markdown(
        """
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #f8fafc;
            border-radius: 8px;
            padding: 16px;
            border: 1px solid #e2e8f0;
        }
        .footer-text {
            font-size: 0.85rem;
            color: #6b7280;
            text-align: center;
            padding-top: 2rem;
            border-top: 1px solid #e5e7eb;
            margin-top: 3rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_sidebar() -> str:
    """Render sidebar navigation, quick stats snapshot, and research citation."""
    with st.sidebar:
        st.markdown("## 🛡️ Crypto AML GNN")
        st.markdown("**Forensic Intelligence System**")
        st.caption("Phase 4 Dashboard Integration")

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        section = st.radio(
            "Go to section:",
            options=[
                "🏛️ Overview & Architecture",
                "📊 Dataset Statistics & Splits",
                "📈 Model Comparison & Curves",
                "🎯 Confusion Matrix Inspector",
                "🔍 Transaction AML Risk Lookup",
                "🌐 Real Graph Neighborhood"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("### 📊 Quick Facts Snapshot")
        manifest = load_dataset_manifest()
        if manifest:
            st.markdown(
                f"""
                - **Transactions:** 203,769
                - **Graph Edges:** 234,355
                - **Timesteps:** 49
                - **Test Split (T42–49):** 8,841 txs
                - **Models Benchmarked:** 5
                - **SOTA Target:** Chebyshev+Lin
                """
            )

        st.markdown("---")
        st.markdown("### 📖 Research Reference")
        st.caption(
            "Ferretti et al., *IEEE Access*, 2025. "
            "DOI: [10.1109/ACCESS.2025.3552240](https://doi.org/10.1109/ACCESS.2025.3552240)"
        )

        st.markdown("---")
        st.caption("v1.0.0 (Phase 4 Final Integrated)")

    return section


def render_footer():
    """Render academic footer at the bottom of the page."""
    st.markdown(
        """
        <div class="footer-text">
            <b>AI-Powered Cryptocurrency Anti-Money Laundering Using Graph Neural Networks</b><br>
            Academic Research Pipeline & Forensic Investigation Dashboard &bull; Phase 4 Final Integrated Implementation<br>
            <i>Strict Leakage-Free Temporal Partitioning &bull; Elliptic Bitcoin Dataset &bull; MIT License</i>
        </div>
        """,
        unsafe_allow_html=True
    )


def main():
    setup_page()
    active_section = render_sidebar()

    # Route navigation
    if active_section == "🏛️ Overview & Architecture":
        render_overview()
    elif active_section == "📊 Dataset Statistics & Splits":
        render_dataset_stats()
    elif active_section == "📈 Model Comparison & Curves":
        render_model_comparison()
    elif active_section == "🎯 Confusion Matrix Inspector":
        render_confusion_matrix()
    elif active_section == "🔍 Transaction AML Risk Lookup":
        render_transaction_lookup()
    elif active_section == "🌐 Real Graph Neighborhood":
        render_graph_visualization()

    render_footer()


if __name__ == "__main__":
    main()
