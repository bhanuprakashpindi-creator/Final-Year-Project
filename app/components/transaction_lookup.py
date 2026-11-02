"""Single Transaction AML Lookup and Risk Scoring Component for the Crypto AML Forensic Dashboard."""

import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np

try:
    from app.data_loader import load_all_predictions, get_sample_transactions, get_project_root
    from app.components.overview import render_disclaimer
except ModuleNotFoundError:
    from data_loader import load_all_predictions, get_sample_transactions, get_project_root
    from components.overview import render_disclaimer


@st.cache_data(show_spinner=False)
def load_all_dataset_tx_indices():
    """Load global mapping of all 203,769 transactions, labels, and timesteps."""
    root = get_project_root()
    p_map = root / "data" / "processed" / "elliptic_tx_to_idx.json"
    p_y = root / "data" / "processed" / "elliptic_y.npy"
    p_t = root / "data" / "processed" / "elliptic_timesteps.npy"
    if p_map.exists() and p_y.exists() and p_t.exists():
        with open(p_map, "r", encoding="utf-8") as f:
            tx_to_idx = json.load(f)
        y = np.load(p_y)
        timesteps = np.load(p_t)
        return tx_to_idx, y, timesteps
    return {}, None, None


def render_transaction_lookup():
    """Render interactive single-transaction lookup, cross-model risk scoring, and consensus analysis."""
    render_disclaimer()

    st.title("🔍 Single Bitcoin Transaction AML Audit & Risk Lookup")
    st.markdown(
        "Enter any Bitcoin transaction identifier (`txId`) from the held-out test split (timesteps 42..49) "
        "to inspect individual model predictions, risk confidence scores, and cross-architecture consensus."
    )

    predictions = load_all_predictions()
    samples = get_sample_transactions()
    tx_to_idx, y_all, timesteps_all = load_all_dataset_tx_indices()

    if not predictions:
        st.error("Transaction prediction CSV files could not be loaded from `results/`.")
        return

    # Quick Sample Selector for Evaluators & Defense Viva
    st.markdown("#### 🎯 Quick Audit Presets (Representative Test Split Transactions)")
    c1, c2, c3, c4 = st.columns(4)

    preset_tx = None
    with c1:
        if st.button("🚨 Unanimous Illicit (5/5)", help="All 5 models agree this is illicit"):
            preset_tx = samples["unanimous_illicit"][0] if samples["unanimous_illicit"] else "72629711"
    with c2:
        if st.button("💡 Cheb Caught, RF Missed", help="Chebyshev GNN caught laundering where Random Forest predicted licit"):
            preset_tx = "106359119"
    with c3:
        if st.button("✅ Unanimous Licit (5/5)", help="All 5 models agree this is clean"):
            preset_tx = samples["unanimous_licit"][0] if samples["unanimous_licit"] else "218047773"
    with c4:
        if st.button("⚖️ High Model Disagreement", help="Split decisions across architectures"):
            preset_tx = samples["disagreed"][0] if samples["disagreed"] else "71450526"

    # Search Bar
    default_tx = preset_tx if preset_tx else "106359119"
    search_query = st.text_input(
        "Enter Bitcoin Transaction ID (txId):",
        value=default_tx,
        help="Search among the 8,841 held-out test transactions evaluated in Phase 2 and Phase 3."
    ).strip()

    if not search_query:
        st.info("Please enter a transaction ID or click one of the quick presets above.")
        return

    # Check if transaction exists in test set predictions
    if search_query not in predictions:
        # Check if transaction exists anywhere in the 203,769 node graph
        if tx_to_idx and search_query in tx_to_idx:
            idx = tx_to_idx[search_query]
            ts = int(timesteps_all[idx])
            lbl = int(y_all[idx])
            split_name = "Train Partition (Timesteps 1–34)" if ts <= 34 else ("Validation Partition (Timesteps 35–41)" if ts <= 41 else "Test Partition (Timesteps 42–49)")
            lbl_name = "Illicit (1)" if lbl == 1 else ("Licit (0)" if lbl == 0 else "Unknown / Unlabeled (-1)")
            
            st.info(
                f"ℹ️ **Transaction ID `{search_query}` Found in Global Dataset (Node #{idx:,})**\n\n"
                f"- **Timestep:** `T{ts}`\n"
                f"- **Split:** `{split_name}`\n"
                f"- **Ground-Truth Label:** `{lbl_name}`\n\n"
                f"**Methodological Note:** Under strict temporal partitioning, model performance scorecards and "
                f"pre-computed multi-model consensus predictions are compiled exclusively for the **held-out test split (Timesteps 42–49; 8,841 labeled transactions)**. "
                f"To inspect live multi-model predictions, select one of the test presets above or try: `106359119`, `72629711`, `218047773`, or `71450526`."
            )
            return
        else:
            st.error(
                f"⚠️ **Transaction ID '{search_query}' Not Found in the Elliptic Dataset!**\n\n"
                "Please verify the transaction ID. You can click any of the quick preset buttons above or try searching for: `106359119`, `72629711`, `218047773`, or `2376671`."
            )
            return

    record = predictions[search_query]

    # Transaction Forensic Card
    st.markdown("---")
    st.header(f"📁 Forensic Audit Dossier: Bitcoin `txId {search_query}`")

    # High-level Risk Summary
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Consensus Verdict", record["consensus"])
    with k2:
        st.metric("Illicit Vote Count", f"{record['illicit_votes']} / 5 Models")
    with k3:
        st.metric("Average Illicit Probability", f"{record['avg_prob']*100:.1f}%")
    with k4:
        st.metric("Risk Classification", record["risk_tier"])

    # Color badge for risk level
    if record["risk_tier"] == "Critical Risk":
        st.error(f"🚨 **CRITICAL RISK ALERT:** {record['illicit_votes']} of 5 models flag this transaction as illicit. Average confidence: {record['avg_prob']*100:.2f}%. Immediate compliance audit recommended.")
    elif record["risk_tier"] == "High Risk":
        st.warning(f"⚠️ **HIGH RISK:** Majority of models flag this transaction as suspicious ({record['illicit_votes']}/5). Average confidence: {record['avg_prob']*100:.2f}%.")
    elif record["risk_tier"] == "Elevated Suspicion":
        st.info(f"🔍 **ELEVATED SUSPICION:** At least one model detected laundering patterns ({record['illicit_votes']}/5).")
    else:
        st.success(f"✅ **CLEAN / LOW RISK:** Models agree this transaction exhibits legitimate characteristics (Average Illicit Risk: {record['avg_prob']*100:.2f}%).")

    # Side-by-side Table & Radar/Bar Chart
    col_table, col_chart = st.columns([1.2, 1])

    with col_table:
        st.subheader("Model-by-Model Predictions & Risk Scores")
        model_rows = [
            {
                "Model Architecture": "Random Forest (Phase 2 Baseline)",
                "Prediction": "🚨 Illicit (1)" if record["rf_pred"] == 1 else "✅ Licit (0)",
                "Illicit Probability": f"{record['rf_prob']*100:.2f}%",
                "Model Type": "Tabular Machine Learning"
            },
            {
                "Model Architecture": "GCN+Lin",
                "Prediction": "🚨 Illicit (1)" if record["gcn_pred"] == 1 else "✅ Licit (0)",
                "Illicit Probability": f"{record['gcn_prob']*100:.2f}%",
                "Model Type": "Spatial Graph Neural Network"
            },
            {
                "Model Architecture": "GraphSAGE+Lin",
                "Prediction": "🚨 Illicit (1)" if record["sage_pred"] == 1 else "✅ Licit (0)",
                "Illicit Probability": f"{record['sage_prob']*100:.2f}%",
                "Model Type": "Spatial Graph Neural Network"
            },
            {
                "Model Architecture": "GATv2+Lin",
                "Prediction": "🚨 Illicit (1)" if record["gat_pred"] == 1 else "✅ Licit (0)",
                "Illicit Probability": f"{record['gat_prob']*100:.2f}%",
                "Model Type": "Attention Graph Neural Network"
            },
            {
                "Model Architecture": "Chebyshev+Lin (Paper SOTA)",
                "Prediction": "🚨 Illicit (1)" if record["cheb_pred"] == 1 else "✅ Licit (0)",
                "Illicit Probability": f"{record['cheb_prob']*100:.2f}%",
                "Model Type": "Spectral Graph Neural Network"
            },
        ]
        st.table(model_rows)

    with col_chart:
        st.subheader("Risk Score Distribution (%)")
        chart_models = ["Random Forest", "GCN+Lin", "GraphSAGE", "GATv2+Lin", "Chebyshev"]
        chart_probs = [
            record["rf_prob"] * 100,
            record["gcn_prob"] * 100,
            record["sage_prob"] * 100,
            record["gat_prob"] * 100,
            record["cheb_prob"] * 100
        ]
        bar_colors = ["#EF4444" if p >= 50.0 else "#10B981" for p in chart_probs]

        fig_risk = go.Figure(data=[go.Bar(
            x=chart_models,
            y=chart_probs,
            marker_color=bar_colors,
            text=[f"{p:.1f}%" for p in chart_probs],
            textposition="auto"
        )])
        fig_risk.add_hline(y=50.0, line_dash="dot", line_color="red", annotation_text="Classification Threshold (50%)")
        fig_risk.update_layout(
            yaxis_title="Illicit Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=340,
            margin=dict(t=30, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_risk, width="stretch")

    # Disagreement Deep-Dive Callout
    if record["cheb_pred"] == 1 and record["rf_pred"] == 0:
        st.warning(
            "💡 **Topological Advantage Detected:**\n\n"
            f"For transaction `{search_query}`, **Chebyshev+Lin flagged Illicit ({record['cheb_prob']*100:.1f}%)**, "
            f"whereas **Random Forest predicted Licit ({record['rf_prob']*100:.1f}%)**.\n\n"
            "This illustrates the core motivation of Graph Neural Networks: by aggregating payment flows from 1-hop and 2-hop "
            "neighbors, the spectral GNN identified criminal layering that tabular models completely missed."
        )
