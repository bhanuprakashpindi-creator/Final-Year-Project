"""Integration tests for Phase 4 dashboard components, model checkpoints, and graph visualizer."""

import unittest
from pathlib import Path
import json
import joblib
import numpy as np
import torch

from app.data_loader import (
    get_project_root,
    check_graph_files_status,
    load_all_predictions,
    load_confusion_matrices,
    load_model_comparison,
    get_sample_transactions
)
from app.components.graph_visualization import (
    load_graph_topology,
    extract_ego_subgraph,
    build_hierarchical_positions
)
from src.models.gnn_architectures import build_model


class TestPhase4Integration(unittest.TestCase):
    """End-to-end integration tests verifying artifacts, models, and graph visualizer."""

    @classmethod
    def setUpClass(cls):
        cls.root = get_project_root()

    def test_check_graph_files_status(self):
        """Verify that processed topological graph files are present and detected."""
        status = check_graph_files_status()
        self.assertTrue(status["is_visualizable"])
        self.assertTrue(status["processed_edge_exists"])
        self.assertTrue(status["processed_mapping_exists"])

    def test_load_all_predictions(self):
        """Verify that all 8,841 held-out test predictions are loaded and valid."""
        predictions = load_all_predictions()
        self.assertEqual(len(predictions), 8841)

        sample_tx = next(iter(predictions.keys()))
        rec = predictions[sample_tx]
        self.assertIn("rf_pred", rec)
        self.assertIn("cheb_pred", rec)
        self.assertIn("avg_prob", rec)
        self.assertIn("illicit_votes", rec)
        self.assertTrue(0 <= rec["illicit_votes"] <= 5)
        self.assertTrue(0.0 <= rec["avg_prob"] <= 1.0)

    def test_load_confusion_matrices_sums(self):
        """Verify that confusion matrices for all 5 models sum exactly to 8,841."""
        cm_dict = load_confusion_matrices()
        expected_models = ["Random Forest", "GCN+Lin", "GraphSAGE+Lin", "GATv2+Lin", "Chebyshev+Lin"]
        for name in expected_models:
            self.assertIn(name, cm_dict, f"Missing confusion matrix for {name}")
            data = cm_dict[name]
            total = data["TP"] + data["FN"] + data["FP"] + data["TN"]
            self.assertEqual(total, 8841, f"{name} confusion matrix total {total} != 8841")
            self.assertEqual(data["actual_illicit"], 408)
            self.assertEqual(data["actual_licit"], 8433)

    def test_sample_transactions_availability(self):
        """Verify curated sample transactions are available."""
        samples = get_sample_transactions()
        self.assertIn("unanimous_illicit", samples)
        self.assertIn("cheb_caught_rf_missed", samples)
        self.assertIn("unanimous_licit", samples)
        self.assertGreater(len(samples["unanimous_illicit"]), 0)
        self.assertGreater(len(samples["cheb_caught_rf_missed"]), 0)

    def test_graph_visualization_subgraph_extraction(self):
        """Verify graph topology loading and directed ego-subgraph extraction."""
        edge_index, tx_to_idx, idx_to_tx, y, timesteps = load_graph_topology()
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(edge_index.shape[1], 234355)
        self.assertEqual(len(tx_to_idx), 203769)

        # Test node 182163 (tx 106359119)
        nodes, sub_src, sub_dst = extract_ego_subgraph(182163, edge_index, hops=2, direction="Both", max_nodes=35)
        self.assertIn(182163, nodes)
        self.assertGreaterEqual(len(nodes), 1)
        self.assertGreaterEqual(len(sub_src), 0)

        # Verify hierarchical layout calculation
        pos = build_hierarchical_positions(182163, nodes, sub_src, sub_dst)
        self.assertEqual(len(pos), len(nodes))
        self.assertEqual(pos[182163], (0.0, 0.0))

    def test_random_forest_checkpoint_loading(self):
        """Verify Random Forest model loads from joblib and evaluates dummy input."""
        rf_path = self.root / "models" / "random_forest.joblib"
        self.assertTrue(rf_path.exists())
        model = joblib.load(rf_path)
        dummy_x = np.zeros((2, 145), dtype=np.float32)
        preds = model.predict(dummy_x)
        probs = model.predict_proba(dummy_x)
        self.assertEqual(len(preds), 2)
        self.assertEqual(probs.shape, (2, 2))

    def test_gnn_checkpoints_loading_and_inference(self):
        """Verify all 4 GNN checkpoints load and perform forward inference."""
        models = {
            "gcn": self.root / "models" / "gcn_best.pt",
            "graphsage": self.root / "models" / "graphsage_best.pt",
            "gatv2": self.root / "models" / "gatv2_best.pt",
            "chebyshev": self.root / "models" / "chebyshev_best.pt",
        }
        dummy_x = torch.zeros((5, 145), dtype=torch.float32)
        dummy_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)

        for name, path in models.items():
            self.assertTrue(path.exists(), f"Missing checkpoint: {path}")
            model = build_model(name, in_channels=145, hidden_channels=110, out_channels=2)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            with torch.no_grad():
                out = model(dummy_x, dummy_edges)
            self.assertEqual(out.shape, (5, 2), f"{name} output shape incorrect: {out.shape}")


if __name__ == "__main__":
    unittest.main()
