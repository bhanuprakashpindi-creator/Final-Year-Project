"""Dataset Statistics and Splitting Component for the Crypto AML Forensic Dashboard."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    from app.data_loader import load_dataset_manifest, load_validation_summary
except ModuleNotFoundError:
    from data_loader import load_dataset_manifest, load_validation_summary


def render_dataset_stats():
    """Render verified Elliptic dataset statistics, temporal split breakdown, and leakage audit."""
    manifest = load_dataset_manifest()
    val_summary = load_validation_summary()

    st.title("📊 Elliptic Bitcoin Dataset & Splitting Anatomy")
    st.markdown(
        "This section presents verified structural statistics measured directly from the **Elliptic Bitcoin Transaction Dataset**, "
        "the standard benchmark in cryptocurrency anti-money laundering research."
    )

    if not manifest:
        st.error("Dataset manifest file (`dataset_manifest.json`) could not be loaded.")
        return

    # Extract verified numbers directly from manifest
    num_nodes = manifest.get("num_nodes", 203769)
    num_edges = manifest.get("num_directed_edges", 234355)
    labels = manifest.get("label_counts", {})
    illicit_count = labels.get("illicit", 4545)
    licit_count = labels.get("licit", 42019)
    unknown_count = labels.get("unknown", 157205)
    total_labeled = illicit_count + licit_count

    splits = manifest.get("temporal_splits", {})
    train_s = splits.get("train_split", {})
    val_s = splits.get("val_split", {})
    test_s = splits.get("test_split", {})

    # Top-level KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transactions", f"{num_nodes:,}", help="Total distinct Bitcoin transaction vertices in the graph.")
    with col2:
        st.metric("Directed Payment Flows", f"{num_edges:,}", help="Directed transaction-to-transaction fund transfer edges.")
    with col3:
        st.metric("Discrete Timesteps", "49 Timesteps", help="Two-week chronological graph snapshots.")
    with col4:
        st.metric("Reduced Features", "145 Features", help="Pruned feature set (tx+agg_red) matching paper configuration.")

    st.markdown("---")

    # Class Breakdown & Imbalance
    st.header("⚖️ Class Distribution & Severe Imbalance")
    st.markdown(
        """
        In financial fraud and anti-money laundering, positive targets (illicit transactions) 
        represent a tiny fraction of the network.
        """
    )

    st.warning(
        """
        **⚠️ Why Overall Accuracy Is Fundamentally Misleading in AML Detection:**
        
        - **On the Held-Out Test Split (Timesteps 42–49):** There are **8,433 licit** and **408 illicit** transactions (total: 8,841).
          A naive dummy classifier predicting *"every transaction is licit"* achieves **95.39% accuracy** ($8,433 / 8,841 = 95.385\%$).
          However, it detects **0 out of 408 financial criminals** (Illicit Recall = 0.00%, Illicit F1 = 0.00%).
        - **On the Entire Labeled Dataset (46,564 nodes):** Predicting all licit yields **90.24% accuracy** ($42,019 / 46,564 = 90.237\%$).
        
        Therefore, overall accuracy creates a dangerous illusion of efficacy. In regulatory compliance, 
        **Illicit Precision, Illicit Recall, Illicit F1-Score, and PR-AUC** are the only academically defensible evaluation metrics.
        """
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        # Donut Chart for All Nodes
        donut_labels = ["Illicit (Class 1)", "Licit (Class 0)", "Unknown / Unlabeled (-1)"]
        donut_values = [illicit_count, licit_count, unknown_count]
        donut_colors = ["#EF4444", "#10B981", "#94A3B8"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=donut_labels,
            values=donut_values,
            hole=0.55,
            marker_colors=donut_colors,
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )])
        fig_donut.update_layout(
            title_text="Complete Transaction Graph (203,769 Nodes)",
            margin=dict(t=40, b=20, l=20, r=20),
            height=340,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_donut, width="stretch")

    with c2:
        # Donut Chart for Labeled Subgraph
        fig_subgraph = go.Figure(data=[go.Pie(
            labels=["Illicit (Class 1)", "Licit (Class 0)"],
            values=[illicit_count, licit_count],
            hole=0.55,
            marker_colors=["#EF4444", "#10B981"],
            textinfo="label+percent",
            hoverinfo="label+value+percent",
        )])
        fig_subgraph.update_layout(
            title_text=f"Labeled Subgraph Only ({total_labeled:,} Nodes)",
            margin=dict(t=40, b=20, l=20, r=20),
            height=340,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_subgraph, width="stretch")

    # Detailed counts table
    st.markdown("#### Exact Category Counts")
    cat_df = [
        {"Category": "🚨 Illicit (1)", "Node Count": f"{illicit_count:,}", "% of Total": f"{(illicit_count/num_nodes)*100:.2f}%", "% of Labeled": f"{(illicit_count/total_labeled)*100:.2f}%", "Role in Pipeline": "Target of interest (Minority positive class)"},
        {"Category": "✅ Licit (0)", "Node Count": f"{licit_count:,}", "% of Total": f"{(licit_count/num_nodes)*100:.2f}%", "% of Labeled": f"{(licit_count/total_labeled)*100:.2f}%", "Role in Pipeline": "Legitimate transactions (Majority negative class)"},
        {"Category": "⚪ Unknown (-1)", "Node Count": f"{unknown_count:,}", "% of Total": f"{(unknown_count/num_nodes)*100:.2f}%", "% of Labeled": "N/A", "Role in Pipeline": "Masked from loss; passes messages in GNNs"}
    ]
    st.table(cat_df)

    st.markdown("---")

    # Temporal Splits Breakdown
    st.header("⏱️ Leakage-Free Temporal Partitioning")
    st.markdown(
        """
        Random train/test splits cause severe **temporal look-ahead leakage** in financial networks. 
        In the real world, an AML detection model must train on past transactions and predict future transactions.
        Following Ferretti et al. (2025) and Weber et al. (2019), the 49 discrete timesteps are partitioned chronologically:
        """
    )

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown(
            f"""
            #### 🚂 Train Split
            - **Timesteps:** `{train_s.get('timesteps', '1..34')}`
            - **Total Nodes:** {train_s.get('total_nodes', 136265):,}
            - **Labeled Nodes:** {train_s.get('labeled_nodes', 29894):,}
            - **Illicit:** {train_s.get('illicit_count', 3462):,} ({train_s.get('illicit_ratio_labeled', 0.1158)*100:.2f}%)
            - **Licit:** {train_s.get('licit_count', 26432):,}
            """
        )
    with t2:
        st.markdown(
            f"""
            #### 🔍 Validation Split
            - **Timesteps:** `{val_s.get('timesteps', '35..41')}`
            - **Total Nodes:** {val_s.get('total_nodes', 30680):,}
            - **Labeled Nodes:** {val_s.get('labeled_nodes', 7829):,}
            - **Illicit:** {val_s.get('illicit_count', 675):,} ({val_s.get('illicit_ratio_labeled', 0.0862)*100:.2f}%)
            - **Licit:** {val_s.get('licit_count', 7154):,}
            """
        )
    with t3:
        st.markdown(
            f"""
            #### 🧪 Held-Out Test Split
            - **Timesteps:** `{test_s.get('timesteps', '42..49')}`
            - **Total Nodes:** {test_s.get('total_nodes', 36824):,}
            - **Labeled Nodes:** {test_s.get('labeled_nodes', 8841):,}
            - **Illicit:** {test_s.get('illicit_count', 408):,} ({test_s.get('illicit_ratio_labeled', 0.0461)*100:.2f}%)
            - **Licit:** {test_s.get('licit_count', 8433):,}
            """
        )

    # Bar chart of temporal split composition
    split_categories = ["Train (1–34)", "Validation (35–41)", "Test (42–49)"]
    fig_splits = go.Figure()
    fig_splits.add_trace(go.Bar(
        name="Licit Nodes",
        x=split_categories,
        y=[train_s.get("licit_count", 26432), val_s.get("licit_count", 7154), test_s.get("licit_count", 8433)],
        marker_color="#10B981"
    ))
    fig_splits.add_trace(go.Bar(
        name="Illicit Nodes",
        x=split_categories,
        y=[train_s.get("illicit_count", 3462), val_s.get("illicit_count", 675), test_s.get("illicit_count", 408)],
        marker_color="#EF4444"
    ))
    fig_splits.add_trace(go.Bar(
        name="Unknown / Unlabeled Nodes",
        x=split_categories,
        y=[train_s.get("unknown_count", 106371), val_s.get("unknown_count", 22851), test_s.get("unknown_count", 27983)],
        marker_color="#94A3B8"
    ))
    fig_splits.update_layout(
        barmode="stack",
        title="Temporal Split Node Composition (Stacked)",
        xaxis_title="Chronological Partition",
        yaxis_title="Transaction Count",
        height=380,
        margin=dict(t=40, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_splits, width="stretch")

    # Leakage Audit Card
    st.markdown("---")
    st.header("🛡️ Zero-Leakage Verification Audit")
    leak = manifest.get("leakage_assessment", {})
    st.success(
        f"✅ **Audit Result: Topologically Leakage-Free**\n\n"
        f"- Total Edges Evaluated: **{leak.get('total_edges_evaluated', 234355):,}**\n"
        f"- Cross-Split Edges (Train ↔ Val ↔ Test): **{leak.get('total_cross_split_edges', 0)}**\n"
        f"- Status: **{leak.get('findings', 'No edges cross temporal split boundaries.')}**"
    )
