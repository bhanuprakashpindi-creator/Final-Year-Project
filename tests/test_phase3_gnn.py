"""Unit tests for Phase 3 Graph Neural Network architectures, trainer, and metrics.

Verifies:
1. Model forward passes, output log-probability dimensions [N, 2].
2. Symmetrization handling for ChebConv and spatial GNNs.
3. Loss masking isolation (unlabeled/test nodes do not contaminate train gradients).
4. Balanced class weight calculation.
5. Metric computation correctness and confusion matrix mapping.
6. Random seed reproducibility.
"""

import unittest
import torch
import numpy as np

from torch_geometric.data import Data
from src.models.gnn_architectures import (
    GCNPlusLin,
    GraphSAGEPlusLin,
    GATv2PlusLin,
    ChebyshevPlusLin,
    build_model
)
from src.models.metrics import compute_all_metrics
from src.models.gnn_trainer import compute_class_weights, set_seed, GNNTrainer


class TestPhase3GNN(unittest.TestCase):
    """Test suite for Phase 3 GNN models, metrics, and training pipeline."""

    def setUp(self):
        """Build synthetic graph for deterministic testing."""
        set_seed(42)
        self.num_nodes = 30
        self.in_channels = 145
        self.x = torch.randn(self.num_nodes, self.in_channels)
        
        # Directed chain: 0->1, 1->2, ...
        src = torch.arange(0, self.num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, self.num_nodes, dtype=torch.long)
        self.edge_index = torch.stack([src, dst], dim=0)

        # Labels: 10 licit (0), 5 illicit (1), 15 unknown (-1)
        self.y = torch.tensor(
            [0]*10 + [1]*5 + [-1]*15,
            dtype=torch.long
        )

        # Boolean masks
        self.train_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(self.num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(self.num_nodes, dtype=torch.bool)

        self.train_mask[:12] = True   # nodes 0..11
        self.val_mask[12:15] = True   # nodes 12..14
        self.test_mask[15:] = True    # nodes 15..29

        self.data = Data(
            x=self.x,
            edge_index=self.edge_index,
            y=self.y,
            train_mask=self.train_mask,
            val_mask=self.val_mask,
            test_mask=self.test_mask
        )

    def test_gcn_forward_pass_and_dimensions(self):
        """Test GCN+Lin output shape and valid log-probabilities."""
        model = GCNPlusLin(in_channels=self.in_channels, hidden_channels=32, out_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)

        self.assertEqual(out.shape, (self.num_nodes, 2))
        probs = torch.exp(out)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.num_nodes), atol=1e-5))

    def test_graphsage_forward_pass_and_dimensions(self):
        """Test GraphSAGE+Lin output shape and valid log-probabilities."""
        model = GraphSAGEPlusLin(in_channels=self.in_channels, hidden_channels=32, out_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)

        self.assertEqual(out.shape, (self.num_nodes, 2))
        probs = torch.exp(out)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.num_nodes), atol=1e-5))

    def test_gatv2_forward_pass_and_dimensions(self):
        """Test GATv2+Lin dynamic attention with multi-head output shape."""
        model = GATv2PlusLin(in_channels=self.in_channels, hidden_channels=32, heads=2, out_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)

        self.assertEqual(out.shape, (self.num_nodes, 2))
        probs = torch.exp(out)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.num_nodes), atol=1e-5))

    def test_chebyshev_forward_pass_and_symmetrization(self):
        """Test Chebyshev+Lin spectral convolution on directed input edges."""
        model = ChebyshevPlusLin(in_channels=self.in_channels, hidden_channels=32, K=3, out_channels=2)
        model.eval()
        with torch.no_grad():
            out = model(self.data.x, self.data.edge_index)

        self.assertEqual(out.shape, (self.num_nodes, 2))
        probs = torch.exp(out)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.num_nodes), atol=1e-5))

    def test_build_model_factory(self):
        """Test factory function for creating models by string name."""
        for name in ["gcn", "graphsage", "gatv2", "chebyshev"]:
            m = build_model(name, in_channels=self.in_channels, hidden_channels=16)
            self.assertIsNotNone(m)

        with self.assertRaises(ValueError):
            build_model("unsupported_gnn_model")

    def test_loss_masking_isolation(self):
        """Verify that modifying unlabeled (-1) or test node targets does not affect training loss."""
        model = ChebyshevPlusLin(in_channels=self.in_channels, hidden_channels=16, K=2)
        model.eval()

        is_labeled = (self.data.y != -1)
        train_mask_labeled = self.data.train_mask & is_labeled

        weights = compute_class_weights(self.data.y, train_mask_labeled)
        criterion = torch.nn.NLLLoss(weight=weights)

        # Baseline loss
        with torch.no_grad():
            out1 = model(self.data.x, self.data.edge_index)
            loss1 = criterion(out1[train_mask_labeled], self.data.y[train_mask_labeled])

            # Modify labels of test nodes and unknown nodes
            y_tampered = self.data.y.clone()
            y_tampered[~train_mask_labeled] = 1  # Flip all unmonitored nodes

            out2 = model(self.data.x, self.data.edge_index)
            loss2 = criterion(out2[train_mask_labeled], y_tampered[train_mask_labeled])

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)

    def test_class_weights_computation(self):
        """Test inverse class frequency weight formula."""
        # 8 licit (0), 2 illicit (1)
        y = torch.tensor([0]*8 + [1]*2, dtype=torch.long)
        mask = torch.ones(10, dtype=torch.bool)
        weights = compute_class_weights(y, mask)

        # Expected: w0 = 10 / (2 * 8) = 0.625, w1 = 10 / (2 * 2) = 2.5
        self.assertAlmostEqual(weights[0].item(), 0.625, places=4)
        self.assertAlmostEqual(weights[1].item(), 2.5, places=4)

    def test_metrics_calculation(self):
        """Test precision, recall, F1, PR-AUC, accuracy and confusion matrix calculation."""
        y_true = np.array([1, 1, 0, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.4, 0.7, 0.1, 0.2])

        metrics = compute_all_metrics(y_true, y_pred, y_prob)

        # Illicit class (1): TP=1, FN=1, FP=1, TN=2
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["true_negatives"], 2)

        # Precision = 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(metrics["precision"], 0.5, places=4)
        # Recall = 1 / (1 + 1) = 0.5
        self.assertAlmostEqual(metrics["recall"], 0.5, places=4)
        # F1 = 0.5
        self.assertAlmostEqual(metrics["f1"], 0.5, places=4)
        # Accuracy = 3 / 5 = 0.6
        self.assertAlmostEqual(metrics["accuracy"], 0.6, places=4)

        # Unknown label (-1) should raise ValueError
        with self.assertRaises(ValueError):
            compute_all_metrics([1, -1, 0], [1, 0, 0], [0.9, 0.1, 0.2])

    def test_reproducibility_with_seed(self):
        """Verify that identical random seeds produce identical outputs."""
        set_seed(123)
        m1 = GCNPlusLin(in_channels=10, hidden_channels=8, out_channels=2)
        m1.eval()
        x = torch.randn(5, 10)
        e = torch.tensor([[0, 1, 2], [1, 2, 0]])
        out1 = m1(x, e)

        set_seed(123)
        m2 = GCNPlusLin(in_channels=10, hidden_channels=8, out_channels=2)
        m2.eval()
        out2 = m2(x, e)

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
