"""Dedicated unit and integration tests for real graph neighborhood visualization."""

import unittest
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import networkx as nx

from app.data_loader import get_project_root, load_all_predictions
from app.components.graph_visualization import (
    load_graph_topology,
    extract_incoming_neighbors,
    extract_outgoing_neighbors,
    extract_ego_subgraph,
    build_hierarchical_positions,
    build_neighborhood_figure
)


class TestGraphVisualization(unittest.TestCase):
    """Test suite covering transaction lookup, neighborhood extraction, and Plotly figure generation."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()
        cls.edge_index, cls.tx_to_idx, cls.idx_to_tx, cls.y_labels, cls.timesteps = load_graph_topology()
        cls.predictions = load_all_predictions()

    def test_transaction_id_lookup_valid(self):
        """Verify transaction ID 94992016 maps to a valid integer node index."""
        tx_id = "94992016"
        self.assertIn(tx_id, self.tx_to_idx)
        idx = self.tx_to_idx[tx_id]
        self.assertIsInstance(idx, int)
        self.assertTrue(0 <= idx < len(self.tx_to_idx))
        self.assertEqual(self.idx_to_tx[idx], tx_id)
        # Check timestep
        ts = self.timesteps[idx]
        self.assertEqual(ts, 43)

    def test_missing_transaction_id_handling(self):
        """Verify that looking up a non-existent transaction ID returns None."""
        invalid_tx = "non_existent_tx_9999999999"
        self.assertNotIn(invalid_tx, self.tx_to_idx)
        self.assertIsNone(self.tx_to_idx.get(invalid_tx))

    def test_incoming_neighbor_extraction(self):
        """Verify extracting incoming neighbors (payers/sources) for transaction 94992016."""
        idx = self.tx_to_idx["94992016"]
        in_neighbors = extract_incoming_neighbors(idx, self.edge_index)
        self.assertIsInstance(in_neighbors, np.ndarray)
        self.assertEqual(len(in_neighbors), 2)
        # Nodes 178980 and 178981 are known direct payers for 94992016
        in_tx_ids = [self.idx_to_tx[n] for n in in_neighbors]
        self.assertIn("99262906", in_tx_ids)
        self.assertIn("99262907", in_tx_tx := in_tx_ids)

    def test_outgoing_neighbor_extraction(self):
        """Verify extracting outgoing neighbors for transaction 94992016."""
        idx = self.tx_to_idx["94992016"]
        out_neighbors = extract_outgoing_neighbors(idx, self.edge_index)
        self.assertIsInstance(out_neighbors, np.ndarray)
        self.assertEqual(len(out_neighbors), 0)  # 94992016 is a terminal recipient

    def test_one_hop_extraction(self):
        """Verify 1-hop ego-network extraction for transaction 94992016."""
        idx = self.tx_to_idx["94992016"]
        nodes, sub_src, sub_dst = extract_ego_subgraph(
            target_idx=idx,
            edge_index=self.edge_index,
            hops=1,
            direction="Both",
            max_nodes=40
        )
        self.assertIn(idx, nodes)
        self.assertEqual(len(nodes), 3)  # Target + 2 incoming payers
        self.assertEqual(len(sub_src), 2)  # 2 incoming directed edges
        for s, d in zip(sub_src, sub_dst):
            self.assertEqual(d, idx)

    def test_two_hop_extraction(self):
        """Verify 2-hop ego-network extraction for transaction 94992016."""
        idx = self.tx_to_idx["94992016"]
        nodes, sub_src, sub_dst = extract_ego_subgraph(
            target_idx=idx,
            edge_index=self.edge_index,
            hops=2,
            direction="Both",
            max_nodes=40
        )
        self.assertIn(idx, nodes)
        self.assertEqual(len(nodes), 5)  # 5 nodes in 2-hop tree
        self.assertEqual(len(sub_src), 4)  # 4 directed edges
        # Verify real NetworkX DiGraph creation
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        for s, d in zip(sub_src, sub_dst):
            G.add_edge(s, d)
        self.assertEqual(len(G.nodes), 5)
        self.assertEqual(len(G.edges), 4)

    def test_empty_neighborhood_isolated_node(self):
        """Verify handling of an isolated node (0 incoming, 0 outgoing edges)."""
        # Node with in_deg=0 and out_deg=0
        src_all, dst_all = self.edge_index[0], self.edge_index[1]
        all_connected = np.unique(np.concatenate([src_all, dst_all]))
        all_indices = set(range(len(self.tx_to_idx)))
        isolated_indices = all_indices - set(all_connected)
        if isolated_indices:
            iso_idx = next(iter(isolated_indices))
            nodes, sub_src, sub_dst = extract_ego_subgraph(
                target_idx=iso_idx,
                edge_index=self.edge_index,
                hops=2,
                direction="Both",
                max_nodes=40
            )
            self.assertEqual(nodes, {iso_idx})
            self.assertEqual(len(sub_src), 0)
            self.assertEqual(len(sub_dst), 0)

    def test_plotly_figure_generation_94992016(self):
        """Verify complete Plotly figure generation for transaction 94992016."""
        tx_id = "94992016"
        target_idx = self.tx_to_idx[tx_id]
        nodes, sub_src, sub_dst = extract_ego_subgraph(
            target_idx=target_idx,
            edge_index=self.edge_index,
            hops=2,
            direction="Both",
            max_nodes=40
        )
        pos = build_hierarchical_positions(target_idx, nodes, sub_src, sub_dst)
        self.assertEqual(len(pos), len(nodes))

        fig = build_neighborhood_figure(
            target_idx=target_idx,
            selected_tx=tx_id,
            nodes=nodes,
            sub_src=sub_src,
            sub_dst=sub_dst,
            pos=pos,
            idx_to_tx=self.idx_to_tx,
            y_labels=self.y_labels,
            timesteps=self.timesteps,
            predictions=self.predictions,
            color_mode="Model Consensus Risk",
            edge_index=self.edge_index
        )
        self.assertIsInstance(fig, go.Figure)
        # Verify figure contains node and edge traces
        self.assertGreater(len(fig.data), 0)
        # Verify directed arrow annotations are present
        self.assertEqual(len(fig.layout.annotations), len(sub_src))
        self.assertEqual(len(fig.layout.annotations), 4)


if __name__ == "__main__":
    unittest.main()
