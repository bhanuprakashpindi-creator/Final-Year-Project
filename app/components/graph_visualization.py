"""Real Graph Neighborhood Visualization Component for the Crypto AML Forensic Dashboard.

Renders real directed Bitcoin transaction topologies from the Elliptic dataset using NetworkX and Plotly.
Supports central transaction targeting, in-degree/out-degree inspection, 1-hop and 2-hop neighborhoods,
directed payment flow arrows, model prediction overlays, and graph topological statistics.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st

try:
    from app.data_loader import (
        check_graph_files_status,
        load_all_predictions,
        get_sample_transactions,
        get_project_root
    )
    from app.components.overview import render_disclaimer
except ModuleNotFoundError:
    from data_loader import (
        check_graph_files_status,
        load_all_predictions,
        get_sample_transactions,
        get_project_root
    )
    from components.overview import render_disclaimer


@st.cache_data(show_spinner=False)
def load_graph_topology() -> Tuple[np.ndarray, Dict[str, int], Dict[int, str], np.ndarray, np.ndarray]:
    """Load verified graph topology arrays and node mappings from disk."""
    root = get_project_root()
    edge_index = np.load(root / "data" / "processed" / "elliptic_edge_index.npy")
    
    with open(root / "data" / "processed" / "elliptic_tx_to_idx.json", "r", encoding="utf-8") as f:
        tx_to_idx = json.load(f)
        
    with open(root / "data" / "processed" / "elliptic_idx_to_tx.json", "r", encoding="utf-8") as f:
        idx_to_tx_raw = json.load(f)
        if isinstance(idx_to_tx_raw, list):
            idx_to_tx = {i: str(tx) for i, tx in enumerate(idx_to_tx_raw)}
        else:
            idx_to_tx = {int(k): str(v) for k, v in idx_to_tx_raw.items()}
        
    y = np.load(root / "data" / "processed" / "elliptic_y.npy")
    timesteps = np.load(root / "data" / "processed" / "elliptic_timesteps.npy")
    
    return edge_index, tx_to_idx, idx_to_tx, y, timesteps


def extract_ego_subgraph(
    target_idx: int,
    edge_index: np.ndarray,
    hops: int = 1,
    direction: str = "Both",
    max_nodes: int = 40
) -> Tuple[Set[int], np.ndarray, np.ndarray]:
    """Extract directed multi-hop ego neighborhood around target_idx."""
    src, dst = edge_index[0], edge_index[1]
    
    current_layer = {target_idx}
    visited = {target_idx}
    
    for _ in range(hops):
        next_layer = set()
        for node in current_layer:
            if direction in ["Both", "In (Sources / Payers)"]:
                in_neighbors = src[dst == node].tolist()
                for n in in_neighbors:
                    if n not in visited and len(visited) + len(next_layer) < max_nodes:
                        next_layer.add(n)
            if direction in ["Both", "Out (Destinations / Payees)"]:
                out_neighbors = dst[src == node].tolist()
                for n in out_neighbors:
                    if n not in visited and len(visited) + len(next_layer) < max_nodes:
                        next_layer.add(n)
        visited.update(next_layer)
        current_layer = next_layer
        if not current_layer or len(visited) >= max_nodes:
            break
            
    node_list = list(visited)
    mask = np.isin(src, node_list) & np.isin(dst, node_list)
    sub_src, sub_dst = src[mask], dst[mask]
    
    return visited, sub_src, sub_dst


def build_hierarchical_positions(
    target_idx: int,
    nodes: Set[int],
    sub_src: np.ndarray,
    sub_dst: np.ndarray
) -> Dict[int, Tuple[float, float]]:
    """Create left-to-right flow layout: incoming on left, target at center, outgoing on right."""
    pos = {target_idx: (0.0, 0.0)}
    
    in_neighbors = set(sub_src[sub_dst == target_idx])
    out_neighbors = set(sub_dst[sub_src == target_idx])
    
    # Layer 1 Left: direct payers
    in_list = [n for n in in_neighbors if n != target_idx]
    if in_list:
        ys = np.linspace(-1.0, 1.0, len(in_list))
        for idx, y in zip(in_list, ys):
            pos[idx] = (-1.5, float(y))
            
    # Layer 1 Right: direct payees
    out_list = [n for n in out_neighbors if n != target_idx]
    if out_list:
        ys = np.linspace(-1.0, 1.0, len(out_list))
        for idx, y in zip(out_list, ys):
            pos[idx] = (1.5, float(y))
            
    # Remaining nodes (multi-hop / peer flow)
    remaining = [n for n in nodes if n not in pos]
    if remaining:
        rem_in = [n for n in remaining if n in sub_src and np.isin(sub_dst[sub_src == n], in_list).any()]
        rem_out = [n for n in remaining if n in sub_dst and np.isin(sub_src[sub_dst == n], out_list).any()]
        other = [n for n in remaining if n not in rem_in and n not in rem_out]
        
        if rem_in:
            ys = np.linspace(-1.2, 1.2, len(rem_in))
            for idx, y in zip(rem_in, ys):
                pos[idx] = (-3.0, float(y))
        if rem_out:
            ys = np.linspace(-1.2, 1.2, len(rem_out))
            for idx, y in zip(rem_out, ys):
                pos[idx] = (3.0, float(y))
        if other:
            ys = np.linspace(-1.5, 1.5, len(other))
            for idx, y in zip(other, ys):
                pos[idx] = (0.0, float(y if abs(y) > 0.3 else 1.2))
                
    return pos


def render_graph_visualization():
    """Render bounded real transaction neighborhood graph or honest data-requirement advisory."""
    render_disclaimer()

    st.title("🌐 Real Transaction Graph Neighborhood Visualizer")
    st.markdown(
        "Explore verified directed payment flows, transaction parents, fund recipients, and multi-hop peel chains "
        "connecting Bitcoin transactions from the Elliptic dataset."
    )

    status = check_graph_files_status()

    # Academic & Engineering Integrity Check
    if not status["is_visualizable"]:
        st.error("⚠️ **Topological Edge Data Is Currently Not Present on Disk**")
        st.markdown(
            """
            ### ℹ️ Academic Integrity Notice
            Following the strict guidelines of our research pipeline:
            > **Rule 4: Never fabricate model results, transaction IDs or graph edges.**  
            > **Rule 8: If graph files are missing, show an informative message explaining which files are required. Never generate artificial graph edges.**

            Because the Elliptic dataset contains **234,355 directed edges**, large data artifacts were 
            excluded from Git commits via `.gitignore` to keep the repository lightweight.
            """
        )
        st.markdown(
            """
            #### Required Files Checklist:
            - `data/processed/elliptic_edge_index.npy` (Mapped continuous node edge index)
            - `data/processed/elliptic_tx_to_idx.json` (Bitcoin hash ↔ node ID index map)
            - *Alternative:* `data/raw/elliptic_txs_edgelist.csv` (Raw Elliptic directed edge list)
            """
        )
        st.info(
            """
            #### How to Enable Topological Neighborhood Rendering:
            Run the Phase 1 generation script from the project root:
            ```powershell
            python -m scripts.run_phase1
            ```
            Once generated, this tab will automatically detect the real graph topology and render interactive 
            multi-hop neighborhood subgraphs without altering any existing model checkpoints!
            """
        )
        return

    # Edge files ARE available
    edge_index, tx_to_idx, idx_to_tx, y_labels, timesteps = load_graph_topology()
    predictions = load_all_predictions()
    samples = get_sample_transactions()

    st.markdown("#### 🔍 Transaction Selection & Audit Presets")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    preset_tx = None
    with p_col1:
        if st.button("🚨 Unanimous Illicit", help="Transaction flagged illicit by all models"):
            preset_tx = samples["unanimous_illicit"][0] if samples["unanimous_illicit"] else "72629711"
    with p_col2:
        if st.button("💡 Cheb Caught, RF Missed", help="Chebyshev GNN detected laundering where Random Forest missed"):
            preset_tx = "106359119"
    with p_col3:
        if st.button("✅ Unanimous Licit", help="Transaction classified licit by all models"):
            preset_tx = "218047773"
    with p_col4:
        if st.button("⚖️ High Disagreement", help="Mixed predictions across architectures"):
            preset_tx = "71450526"

    # Search bar & Preset
    default_tx = preset_tx if preset_tx else "106359119"
    selected_tx = st.text_input(
        "Enter Bitcoin Transaction ID (txId):",
        value=default_tx,
        help="Enter any valid transaction ID from the Elliptic dataset (203,769 total nodes)."
    ).strip()

    if selected_tx not in tx_to_idx:
        st.error(
            f"Transaction ID `{selected_tx}` not found in the Elliptic dataset index! "
            f"Please check the ID or try one of the presets above (e.g. `106359119`, `72629711`, `218047773`, `230425980`)."
        )
        return

    target_idx = tx_to_idx[selected_tx]

    # Subgraph Configuration Controls
    st.markdown("---")
    st.subheader("⚙️ Graph Subgraph Controls")
    c_ctrl1, c_ctrl2, c_ctrl3, c_ctrl4 = st.columns(4)
    
    with c_ctrl1:
        direction_mode = st.selectbox(
            "Flow Direction:",
            options=["Both", "In (Sources / Payers)", "Out (Destinations / Payees)"],
            index=0,
            help="Choose whether to explore payer inputs, payee outputs, or full bidirectional flow."
        )
    with c_ctrl2:
        hop_radius = st.slider(
            "Neighborhood Hops:",
            min_value=1,
            max_value=2,
            value=2,
            help="1-hop shows direct transaction neighbors; 2-hop reveals multi-hop laundering cascades."
        )
    with c_ctrl3:
        max_nodes = st.slider(
            "Max Rendered Nodes:",
            min_value=10,
            max_value=60,
            value=35,
            step=5,
            help="Bounding the subgraph ensures instant, responsive rendering in the browser."
        )
    with c_ctrl4:
        color_mode = st.selectbox(
            "Color Scheme:",
            options=["Model Consensus Risk", "Ground Truth Label"],
            index=0,
            help="Color nodes by live model predictions or verified ground-truth labels."
        )

    # Extract Subgraph
    nodes, sub_src, sub_dst = extract_ego_subgraph(
        target_idx=target_idx,
        edge_index=edge_index,
        hops=hop_radius,
        direction=direction_mode,
        max_nodes=max_nodes
    )

    # Compute Layout Positions
    layout_mode = "Hierarchical Flow (Left-to-Right)"
    if layout_mode == "Hierarchical Flow (Left-to-Right)" and len(nodes) > 1:
        pos = build_hierarchical_positions(target_idx, nodes, sub_src, sub_dst)
    else:
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for s, d in zip(sub_src, sub_dst):
            G.add_edge(s, d)
        pos = nx.spring_layout(G, seed=42)

    # Compute Node Level Statistics
    src_all, dst_all = edge_index[0], edge_index[1]
    direct_in_deg = int(np.sum(dst_all == target_idx))
    direct_out_deg = int(np.sum(src_all == target_idx))
    target_timestep = int(timesteps[target_idx])
    target_label = int(y_labels[target_idx])

    # Subgraph Stats KPIs
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Nodes in View", f"{len(nodes):,}")
    with m2:
        st.metric("Directed Edges", f"{len(sub_src):,}")
    with m3:
        st.metric("Target In-Degree", f"{direct_in_deg} Payers")
    with m4:
        st.metric("Target Out-Degree", f"{direct_out_deg} Payees")
    with m5:
        split_name = "Train (1-34)" if target_timestep <= 34 else ("Val (35-41)" if target_timestep <= 41 else "Test (42-49)")
        st.metric("Timestep / Split", f"T{target_timestep} ({split_name})")

    # Construct Plotly Figure
    fig = go.Figure()

    # Draw Directed Edges & Arrows
    annotations = []
    for s, d in zip(sub_src, sub_dst):
        if s in pos and d in pos:
            x0, y0 = pos[s]
            x1, y1 = pos[d]
            
            # Line trace
            fig.add_trace(go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#475569", width=1.5),
                hoverinfo="none",
                showlegend=False
            ))
            
            # Midpoint directional arrow
            annotations.append(dict(
                ax=x0,
                ay=y0,
                x=(x0 + 2 * x1) / 3,
                y=(y0 + 2 * y1) / 3,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.2,
                arrowwidth=1.5,
                arrowcolor="#94A3B8"
            ))

    # Node Scatter Traces Grouped by Category for Interactive Legend
    category_nodes = {
        "Target Transaction": [],
        "Predicted / Actual Illicit": [],
        "Predicted / Actual Licit": [],
        "Unknown / Intermediate Node": []
    }

    colors = {
        "Target Transaction": "#F59E0B",         # Amber/Gold
        "Predicted / Actual Illicit": "#EF4444", # Crimson Red
        "Predicted / Actual Licit": "#10B981",   # Emerald Green
        "Unknown / Intermediate Node": "#6B7280" # Slate Gray
    }

    node_data_map = {}
    for n in nodes:
        tx_str = idx_to_tx.get(n, str(n))
        n_label = int(y_labels[n])
        n_ts = int(timesteps[n])
        pred_item = predictions.get(tx_str, None)

        # Categorize
        if n == target_idx:
            cat = "Target Transaction"
        elif color_mode == "Model Consensus Risk":
            if pred_item is not None:
                cat = "Predicted / Actual Illicit" if pred_item["avg_prob"] >= 0.50 else "Predicted / Actual Licit"
            else:
                if n_label == 1:
                    cat = "Predicted / Actual Illicit"
                elif n_label == 0:
                    cat = "Predicted / Actual Licit"
                else:
                    cat = "Unknown / Intermediate Node"
        else: # Ground Truth
            if n_label == 1:
                cat = "Predicted / Actual Illicit"
            elif n_label == 0:
                cat = "Predicted / Actual Licit"
            else:
                cat = "Unknown / Intermediate Node"

        category_nodes[cat].append(n)
        
        # Tooltip content
        gt_str = "Illicit (1)" if n_label == 1 else ("Licit (0)" if n_label == 0 else "Unknown (-1)")
        if pred_item:
            pred_str = f"{pred_item['consensus']} (Risk: {pred_item['avg_prob']*100:.1f}%)"
        else:
            pred_str = "Train / Val Split (Supervised Training Node)" if n_ts <= 41 else "Unlabeled Test Node"
            
        role = "Target Transaction" if n == target_idx else ("Payer (In)" if n in sub_src[sub_dst == target_idx] else "Payee (Out)")
        tooltip = (
            f"<b>txId:</b> {tx_str}<br>"
            f"<b>Role:</b> {role}<br>"
            f"<b>Timestep:</b> {n_ts}<br>"
            f"<b>Ground Truth:</b> {gt_str}<br>"
            f"<b>Model Verdict:</b> {pred_str}<br>"
            f"<b>In-Degree:</b> {int(np.sum(dst_all == n))}, <b>Out-Degree:</b> {int(np.sum(src_all == n))}"
        )
        node_data_map[n] = tooltip

    for cat_name, n_list in category_nodes.items():
        if not n_list:
            continue
        xs = [pos[n][0] for n in n_list]
        ys = [pos[n][1] for n in n_list]
        texts = [node_data_map[n] for n in n_list]
        
        is_target = (cat_name == "Target Transaction")
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text" if is_target else "markers",
            marker=dict(
                size=24 if is_target else 16,
                color=colors[cat_name],
                symbol="diamond" if is_target else "circle",
                line=dict(color="#FFFFFF", width=2 if is_target else 1)
            ),
            text=["🎯 Target" if is_target else "" for _ in xs],
            textposition="top center",
            hovertext=texts,
            hoverinfo="text",
            name=cat_name
        ))

    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=f"Real Directed Transaction Subgraph &bull; Bitcoin txId: {selected_tx}",
            font=dict(size=16, color="#1E293B")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=40, b=50, l=20, r=20),
        height=540,
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#FFFFFF"
    )

    st.plotly_chart(fig, width="stretch")

    # Forensic Interpretation Guide
    st.markdown("---")
    st.subheader("🧠 Forensic Graph Intelligence & Topology Insights")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(
            """
            #### 🔄 Understanding Directed Flow in Anti-Money Laundering
            - **Incoming Arrows (Left → Center):** Represent predecessor transactions transferring funds **into** the target transaction (funding inputs).
            - **Outgoing Arrows (Center → Right):** Represent successor transactions receiving funds **from** the target transaction (peel chains, change outputs, or layering sinks).
            - **Peel Chain Detection:** A high out-degree coupled with small consecutive outputs indicates rapid automated fund dissipation to avoid AML reporting thresholds.
            """
        )
    with col_info2:
        st.markdown(
            """
            #### 🛡️ The GNN Neighborhood Advantage
            - **Tabular Models (RF):** Evaluate each node in isolation based solely on its local features (fee, size, count), ignoring who sent the funds or where they traveled.
            - **Graph Neural Networks (Chebyshev/GCN):** Perform spectral polynomial aggregation across the edges rendered above, gathering multi-hop context from tainted neighbors even if the target transaction crafted clean local features.
            """
        )

    st.info(
        "📌 **Academic Note:** The Elliptic dataset models **transaction-to-transaction** payment flows "
        "(UTXO inputs and outputs), not complete wallet-address clusters. Unknown nodes (-1) are masked "
        "from supervised loss calculation but actively participate in message passing to propagate topological context."
    )
