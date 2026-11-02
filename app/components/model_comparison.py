"""Model Comparison and Evaluation Component for the Crypto AML Forensic Dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from app.data_loader import load_model_comparison, load_training_curves
except ModuleNotFoundError:
    from data_loader import load_model_comparison, load_training_curves


def render_model_comparison():
    """Render 5-model comparative evaluation, interactive metrics charts, and training convergence curves."""
    records = load_model_comparison()
    curves = load_training_curves()

    st.title("📈 Comparative Model Evaluation & Convergence")
    st.markdown(
        "Compare traditional machine learning (**Random Forest baseline**) against Graph Neural Networks "
        "(**GCN+Lin, GraphSAGE+Lin, GATv2+Lin, Chebyshev+Lin**) on the held-out temporal test set "
        "(timesteps 42..49; 8,841 labeled transactions)."
    )

    if not records:
        st.error("Comparative metrics table (`gnn_model_comparison.csv`) could not be loaded.")
        return

    # Comparative Metrics Table
    st.header("📋 Unified Evaluation Scorecard")
    st.markdown(
        "*Evaluated on strictly held-out timesteps 42..49 (8,841 labeled transactions: 408 illicit, 8,433 licit)*"
    )

    table_data = []
    for r in records:
        table_data.append({
            "Model Architecture": r.get("Model", "Unknown"),
            "Model Family": r.get("Type", "Unknown"),
            "Illicit Precision": f"{r.get('Illicit Precision', 0):.4f}",
            "Illicit Recall": f"{r.get('Illicit Recall', 0):.4f}",
            "Illicit F1-Score": f"{r.get('Illicit F1', 0):.4f}",
            "PR-AUC": f"{r.get('PR-AUC', 0):.4f}",
            "Macro-F1": f"{r.get('Macro-F1', 0):.4f}" if r.get("Macro-F1") is not None else "N/A",
            "Accuracy": f"{r.get('Accuracy', 0):.4f}",
            "True Positives (TP)": f"{int(r.get('TP', 0))}",
            "False Negatives (FN)": f"{int(r.get('FN', 0))}",
            "False Positives (FP)": f"{int(r.get('FP', 0))}",
            "True Negatives (TN)": f"{int(r.get('TN', 0))}",
        })
    st.table(table_data)

    st.markdown("---")

    # Interactive Metric Comparison Charts
    st.header("📊 Interactive Metric Comparisons")

    metric_tab1, metric_tab2, metric_tab3 = st.tabs([
        "📊 F1, Precision & Recall",
        "🎯 PR-AUC & Accuracy",
        "⚖️ Precision vs. Recall Tradeoff"
    ])

    models = [r["Model"] for r in records]
    precisions = [r["Illicit Precision"] for r in records]
    recalls = [r["Illicit Recall"] for r in records]
    f1s = [r["Illicit F1"] for r in records]
    pr_aucs = [r["PR-AUC"] for r in records]
    accuracies = [r["Accuracy"] for r in records]
    tps = [int(r["TP"]) for r in records]

    colors = ["#2b5c8f", "#e67e22", "#27ae60", "#8e44ad", "#c0392b"]

    with metric_tab1:
        fig_f1 = go.Figure()
        fig_f1.add_trace(go.Bar(name="Illicit Precision", x=models, y=precisions, marker_color="#3498db"))
        fig_f1.add_trace(go.Bar(name="Illicit Recall", x=models, y=recalls, marker_color="#e67e22"))
        fig_f1.add_trace(go.Bar(name="Illicit F1-Score", x=models, y=f1s, marker_color="#2ecc71"))
        fig_f1.update_layout(
            barmode="group",
            title="Illicit Class Performance (Precision, Recall, F1)",
            yaxis_title="Score (0.0 to 1.0)",
            height=400,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_f1, width="stretch")

    with metric_tab2:
        fig_auc = go.Figure()
        fig_auc.add_trace(go.Bar(name="PR-AUC", x=models, y=pr_aucs, marker_color="#9b59b6"))
        fig_auc.add_trace(go.Bar(name="Overall Accuracy", x=models, y=accuracies, marker_color="#1abc9c"))
        fig_auc.update_layout(
            barmode="group",
            title="Area Under Precision-Recall Curve (PR-AUC) & Overall Accuracy",
            yaxis_title="Score (0.0 to 1.0)",
            height=400,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_auc, width="stretch")

    with metric_tab3:
        fig_scatter = go.Figure()
        for idx, (m, p, r, tp) in enumerate(zip(models, precisions, recalls, tps)):
            fig_scatter.add_trace(go.Scatter(
                x=[r],
                y=[p],
                mode="markers+text",
                name=m,
                text=[f"<b>{m}</b><br>TP: {tp}/408"],
                textposition="top right",
                marker=dict(size=14, color=colors[idx % len(colors)]),
                hoverinfo="text",
                hovertext=f"{m}<br>Precision: {p:.4f}<br>Recall: {r:.4f}<br>Caught: {tp}/408 illicit txs"
            ))
        fig_scatter.update_layout(
            title="Precision vs. Recall Tradeoff (Top Right is Ideal)",
            xaxis_title="Illicit Recall (Detection Sensitivity)",
            yaxis_title="Illicit Precision (Alert Accuracy)",
            xaxis=dict(range=[0.30, 0.55]),
            yaxis=dict(range=[0.20, 0.85]),
            height=430,
            margin=dict(t=40, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_scatter, width="stretch")

    # Key Academic Insight Box
    st.info(
        "💡 **Key Forensic Insight for Examiners:**\n\n"
        "- **Chebyshev+Lin caught the highest number of illicit transactions (201 True Positives; Recall = 49.26%)**, "
        "surpassing the Random Forest baseline (193 True Positives; Recall = 47.30%).\n"
        "- **Why Random Forest F1 is higher (0.5893 vs 0.4664):** Random Forest produces fewer false alarms (54 FP vs 253 FP), "
        "giving it higher precision (0.7814). GNNs, by aggregating neighborhood flows, flag more exploratory transactions "
        "associated with multi-hop laundering chains.\n"
        "- In high-stakes financial compliance, **catching additional true laundering networks is often prioritized** "
        "over minimizing investigative reviews."
    )

    st.markdown("---")

    # Training Curves Section
    st.header("🔄 GNN Training Dynamics & Convergence (100 Epochs)")
    st.markdown(
        "Track the training and validation progression recorded epoch-by-epoch during Phase 3. "
        "Checkpoints were saved on **best validation illicit F1** (monitored on timesteps 35..41)."
    )

    if not curves:
        st.warning("Training curves JSON files were not found.")
        return

    c_left, c_right = st.columns(2)

    with c_left:
        # Loss Curves (Train Loss vs Val Loss)
        fig_loss = go.Figure()
        model_display_names = {
            "chebyshev": "Chebyshev+Lin",
            "gatv2": "GATv2+Lin",
            "gcn": "GCN+Lin",
            "graphsage": "GraphSAGE+Lin"
        }
        for k, v in curves.items():
            fig_loss.add_trace(go.Scatter(
                x=v["epoch"],
                y=v["train_loss"],
                mode="lines",
                name=f"{model_display_names.get(k, k)} (Train)",
                line=dict(width=1.5)
            ))
            fig_loss.add_trace(go.Scatter(
                x=v["epoch"],
                y=v["val_loss"],
                mode="lines",
                name=f"{model_display_names.get(k, k)} (Val)",
                line=dict(dash="dot", width=1.5)
            ))
        fig_loss.update_layout(
            title="Class-Weighted NLL Loss across 100 Epochs",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=420,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4)
        )
        st.plotly_chart(fig_loss, width="stretch")

    with c_right:
        # Validation F1 Progression
        fig_val_f1 = go.Figure()
        for k, v in curves.items():
            fig_val_f1.add_trace(go.Scatter(
                x=v["epoch"],
                y=v["val_f1"],
                mode="lines",
                name=model_display_names.get(k, k),
                line=dict(width=2)
            ))
        fig_val_f1.update_layout(
            title="Validation Illicit F1-Score Progression (Early Stopping Criterion)",
            xaxis_title="Epoch",
            yaxis_title="Validation Illicit F1",
            height=420,
            margin=dict(t=40, b=20, l=20, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4)
        )
        st.plotly_chart(fig_val_f1, width="stretch")
