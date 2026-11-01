"""Confusion Matrix Inspection Component for the Crypto AML Forensic Dashboard."""

import streamlit as st
import plotly.graph_objects as go

try:
    from app.data_loader import load_confusion_matrices
except ModuleNotFoundError:
    from data_loader import load_confusion_matrices


def render_confusion_matrix():
    """Render interactive confusion matrix heatmaps and diagnostic classification metrics for all 5 models."""
    matrices = load_confusion_matrices()

    st.title("🔲 Forensic Confusion Matrix Inspector")
    st.markdown(
        "Inspect the classification outcomes across all five models on the held-out test split "
        "(timesteps 42..49; exactly **408 actual illicit** and **8,433 actual licit** transactions)."
    )

    if not matrices:
        st.error("Confusion matrix CSV files could not be loaded from `results/`.")
        return

    # Visual explanation of confusion matrix terms in simple English
    with st.expander("ℹ️ Understanding Confusion Matrix Terms (Simple English Guide)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                - **True Positive (TP) — Successful Detection:**  
                  The transaction was **truly illicit**, and the model **correctly flagged it**.
                - **False Negative (FN) — Missed Laundering:**  
                  The transaction was **illicit**, but the model mistakenly predicted it as **licit**.  
                  *(In AML compliance, this is the most dangerous error because criminal funds escape).*
                """
            )
        with c2:
            st.markdown(
                """
                - **False Positive (FP) — False Alarm:**  
                  The transaction was **legitimate**, but the model falsely flagged it as **illicit**.  
                  *(Requires human analyst review, increasing operational cost).*
                - **True Negative (TN) — Clean Pass:**  
                  The transaction was **legitimate**, and the model **correctly identified it as licit**.
                """
            )

    st.markdown("---")

    # Model selector
    model_names = list(matrices.keys())
    selected_model = st.selectbox(
        "Select Model to Inspect:",
        options=model_names,
        index=min(4, len(model_names) - 1) if model_names else 0  # Default to Chebyshev+Lin if present
    )

    if selected_model not in matrices:
        st.error(f"Confusion matrix for '{selected_model}' is not available.")
        return

    data = matrices[selected_model]
    tp = data.get("TP", 0)
    fn = data.get("FN", 0)
    fp = data.get("FP", 0)
    tn = data.get("TN", 0)
    actual_illicit = data.get("actual_illicit", tp + fn)
    actual_licit = data.get("actual_licit", fp + tn)
    total = data.get("total", actual_illicit + actual_licit)

    precision = data.get("precision", tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    recall = data.get("recall", tp / actual_illicit if actual_illicit > 0 else 0.0)
    f1 = data.get("f1", 0.0)
    accuracy = data.get("accuracy", (tp + tn) / total if total > 0 else 0.0)

    tp_pct = (tp / actual_illicit * 100) if actual_illicit > 0 else 0.0
    fn_pct = (fn / actual_illicit * 100) if actual_illicit > 0 else 0.0
    fp_pct = (fp / actual_licit * 100) if actual_licit > 0 else 0.0
    tn_pct = (tn / actual_licit * 100) if actual_licit > 0 else 0.0

    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        # Interactive Heatmap using plotly.graph_objects.Heatmap
        z_values = [[tp, fn], [fp, tn]]
        x_labels = ["Predicted Illicit (1)", "Predicted Licit (0)"]
        y_labels = ["Actual Illicit (1)", "Actual Licit (0)"]

        annotations_text = [
            [
                f"<b>True Positive (TP)</b><br>{tp:,}<br>({tp_pct:.1f}% of illicit)",
                f"<b>False Negative (FN)</b><br>{fn:,}<br>({fn_pct:.1f}% missed)"
            ],
            [
                f"<b>False Positive (FP)</b><br>{fp:,}<br>({fp_pct:.2f}% false alarm)",
                f"<b>True Negative (TN)</b><br>{tn:,}<br>({tn_pct:.1f}% clean)"
            ]
        ]

        hover_text = [
            [
                f"Actual: Illicit | Predicted: Illicit<br>Count: {tp:,} ({tp_pct:.2f}% of actual illicit)",
                f"Actual: Illicit | Predicted: Licit<br>Count: {fn:,} ({fn_pct:.2f}% of actual illicit)"
            ],
            [
                f"Actual: Licit | Predicted: Illicit<br>Count: {fp:,} ({fp_pct:.2f}% of actual licit)",
                f"Actual: Licit | Predicted: Licit<br>Count: {tn:,} ({tn_pct:.2f}% of actual licit)"
            ]
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            text=annotations_text,
            texttemplate="%{text}",
            textfont=dict(size=13),
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
            hoverinfo="text",
            hovertext=hover_text
        ))

        fig.update_layout(
            title=f"Confusion Matrix: {selected_model}",
            yaxis=dict(autorange="reversed"),
            xaxis=dict(side="bottom"),
            height=380,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.markdown(f"#### Diagnostic Scorecard: {selected_model}")
        st.markdown(
            f"""
            - **Illicit Precision:** `{precision:.4f}` ({tp:,} / {(tp + fp):,} alerts correct)
            - **Illicit Recall (Sensitivity):** `{recall:.4f}` ({tp:,} / {actual_illicit:,} illicit caught)
            - **Illicit F1-Score:** `{f1:.4f}`
            - **Overall Accuracy:** `{accuracy:.4f}` ({(tp + tn):,} / {total:,} correct)
            - **Total Test Labeled Transactions:** `{total:,}`
            """
        )

        # Detection Rate Progress Bar
        st.markdown(f"**Illicit Catch Rate (Recall):** `{recall * 100:.2f}%`")
        st.progress(float(min(1.0, max(0.0, recall))))

        # False Alarm Rate
        fpr = fp / actual_licit if actual_licit > 0 else 0.0
        st.markdown(f"**False Alarm Rate (Over Licit):** `{fpr * 100:.2f}%`")
        st.progress(float(min(1.0, max(0.0, fpr * 5))))  # Visual scaling for minority rate

    st.markdown("---")

    # Side-by-Side 5-Model Overview Table
    st.header("⚖️ 5-Model Confusion Matrix Comparison Grid")

    grid_cols = st.columns(len(matrices))
    rf_tp = matrices.get("Random Forest", {}).get("TP", 193)
    for idx, (m_name, m_data) in enumerate(matrices.items()):
        with grid_cols[idx]:
            st.markdown(f"**{m_name}**")
            delta_val = f"{m_data['TP'] - rf_tp:+d} vs RF" if m_name != "Random Forest" else None
            st.metric("TP (Caught)", f"{m_data['TP']:,}", delta=delta_val)
            st.metric("FN (Missed)", f"{m_data['FN']:,}")
            st.metric("FP (False Alarms)", f"{m_data['FP']:,}")
            st.metric("TN (Cleared)", f"{m_data['TN']:,}")
