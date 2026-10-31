"""Training and evaluation engine for Graph Neural Networks.

Handles:
- Seed reproducibility (Python, NumPy, PyTorch)
- Class-weighted NLL loss calculation exclusively on labeled training nodes
- Strict isolation of validation and test masks
- Validation-driven early stopping and checkpointing (monitors Illicit F1)
- Comprehensive metric logging, loss curve recording, and artifact persistence
"""

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.metrics import (
    compute_all_metrics,
    save_model_metrics,
    save_model_confusion_matrix,
    save_model_predictions
)


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(y: torch.Tensor, train_mask_labeled: torch.Tensor) -> torch.Tensor:
    """
    Compute balanced inverse class frequency weights exclusively on labeled training nodes.
    
    Formula: weight[c] = total_labeled_train / (num_classes * count[c])
    """
    y_train = y[train_mask_labeled]
    n_total = float(len(y_train))
    n_licit = float((y_train == 0).sum())
    n_illicit = float((y_train == 1).sum())

    if n_licit == 0 or n_illicit == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    w0 = n_total / (2.0 * n_licit)
    w1 = n_total / (2.0 * n_illicit)
    weights = torch.tensor([w0, w1], dtype=torch.float32)
    return weights


class GNNTrainer:
    """Modular trainer for Graph Neural Network classification on Elliptic dataset."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        data: Any,
        idx_to_tx: Optional[List[str]] = None,
        lr: float = 0.009,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        patience: int = 40,
        seed: int = 42,
        device: Optional[torch.device] = None,
        output_dirs: Optional[Dict[str, Path]] = None
    ):
        self.seed = seed
        set_seed(self.seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model_name = model_name
        self.data = data.to(self.device)
        self.idx_to_tx = idx_to_tx

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience

        # Compute masks
        # Labeled masks: mask & (y != -1)
        is_labeled = (self.data.y != -1)
        self.train_mask = getattr(self.data, "train_mask_labeled", None)
        if self.train_mask is None:
            self.train_mask = (self.data.train_mask & is_labeled)

        self.val_mask = getattr(self.data, "val_mask_labeled", None)
        if self.val_mask is None:
            self.val_mask = (self.data.val_mask & is_labeled)

        self.test_mask = getattr(self.data, "test_mask_labeled", None)
        if self.test_mask is None:
            self.test_mask = (self.data.test_mask & is_labeled)

        # Compute class weights on labeled train nodes only
        self.class_weights = compute_class_weights(self.data.y, self.train_mask).to(self.device)
        self.criterion = nn.NLLLoss(weight=self.class_weights)

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=0.9,
            patience=max(5, patience // 3),
            min_lr=1e-5
        )

        # Output paths
        self.output_dirs = output_dirs or {}
        self.model_dir = self.output_dirs.get("models", Path("models"))
        self.results_dir = self.output_dirs.get("results", Path("results"))
        self.tables_dir = self.results_dir / "tables"
        self.predictions_dir = self.results_dir / "predictions"
        self.cm_dir = self.results_dir / "confusion_matrices"
        self.curves_dir = self.results_dir / "training_curves"

        for d in [self.model_dir, self.tables_dir, self.predictions_dir, self.cm_dir, self.curves_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_f1": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_pr_auc": [],
            "lr": []
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Perform one training epoch over the full graph with masked loss."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass on entire graph
        log_probs = self.model(self.data.x, self.data.edge_index)

        # Supervised loss computed ONLY on labeled training nodes
        train_logits = log_probs[self.train_mask]
        train_targets = self.data.y[self.train_mask]
        loss = self.criterion(train_logits, train_targets)

        loss.backward()
        self.optimizer.step()

        # Compute train illicit F1
        with torch.no_grad():
            preds = train_logits.argmax(dim=-1).cpu().numpy()
            targets = train_targets.cpu().numpy()
            train_metrics = compute_all_metrics(
                targets,
                preds,
                torch.exp(train_logits)[:, 1].cpu().numpy()
            )
            train_f1 = train_metrics["f1"]

        return float(loss.item()), float(train_f1)

    @torch.no_grad()
    def evaluate_mask(self, mask: torch.Tensor) -> Tuple[float, Dict[str, Any]]:
        """Evaluate model on a specified boolean mask."""
        self.model.eval()
        log_probs = self.model(self.data.x, self.data.edge_index)

        masked_logits = log_probs[mask]
        masked_targets = self.data.y[mask]
        loss = float(self.criterion(masked_logits, masked_targets).item())

        probs = torch.exp(masked_logits).cpu().numpy()
        preds = masked_logits.argmax(dim=-1).cpu().numpy()
        targets = masked_targets.cpu().numpy()

        metrics = compute_all_metrics(
            y_true=targets,
            y_pred=preds,
            y_prob_illicit=probs[:, 1]
        )
        return loss, metrics

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """Run complete training cycle with validation early stopping."""
        best_val_f1 = -1.0
        best_epoch = 0
        patience_counter = 0
        best_model_weights = None
        start_time = time.time()

        if verbose:
            print(f"\n--- Training {self.model_name} on {self.device} ---")
            print(f"Hyperparameters: LR={self.lr}, Max Epochs={self.max_epochs}, Patience={self.patience}")
            print(f"Class Weights (Licit=0, Illicit=1): {self.class_weights.tolist()}")
            print(f"Train nodes: {int(self.train_mask.sum()):,}, Val nodes: {int(self.val_mask.sum()):,}, Test nodes: {int(self.test_mask.sum()):,}")

        for epoch in range(1, self.max_epochs + 1):
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_metrics = self.evaluate_mask(self.val_mask)
            val_f1 = val_metrics["f1"]

            current_lr = float(self.optimizer.param_groups[0]["lr"])
            self.scheduler.step(val_f1)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["train_f1"].append(train_f1)
            self.history["val_loss"].append(val_loss)
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_f1)
            self.history["val_pr_auc"].append(val_metrics["pr_auc"])
            self.history["lr"].append(current_lr)

            # Checkpoint on best validation illicit F1
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                patience_counter = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1 or patience_counter == 0):
                print(
                    f"Epoch {epoch:03d}/{self.max_epochs:03d} | "
                    f"Train Loss: {train_loss:.4f} (F1: {train_f1:.4f}) | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Prec: {val_metrics['precision']:.4f} | "
                    f"Val Rec: {val_metrics['recall']:.4f} | "
                    f"Val F1: {val_f1:.4f} | "
                    f"Patience: {patience_counter}/{self.patience}"
                )

            if patience_counter >= self.patience:
                if verbose:
                    print(f"[*] Early stopping triggered at epoch {epoch}. Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}.")
                break

        total_time = time.time() - start_time

        # Restore best model weights
        if best_model_weights is not None:
            self.model.load_state_dict(best_model_weights)

        # Save model checkpoint
        checkpoint_path = self.model_dir / f"{self.model_name}_best.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "hyperparameters": {
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "seed": self.seed
            }
        }, checkpoint_path)

        # Final evaluation on held-out test split (timesteps 42..49)
        test_loss, test_metrics = self.evaluate_mask(self.test_mask)

        if verbose:
            print(f"\n[Test Evaluation for {self.model_name}] (Held-out Timesteps 42..49):")
            print(f"  Precision: {test_metrics['precision']:.6f}")
            print(f"  Recall:    {test_metrics['recall']:.6f}")
            print(f"  F1-Score:  {test_metrics['f1']:.6f}")
            print(f"  PR-AUC:    {test_metrics['pr_auc']:.6f}")
            print(f"  Macro-F1:  {test_metrics['macro_f1']:.6f}")
            print(f"  Accuracy:  {test_metrics['accuracy']:.6f}")
            cm = test_metrics["confusion_matrix"]
            print(f"  Confusion Matrix: TP={cm[0,0]}, FN={cm[0,1]}, FP={cm[1,0]}, TN={cm[1,1]}")

        # Save test artifacts
        metrics_csv_path = self.tables_dir / f"{self.model_name}_metrics.csv"
        save_model_metrics(test_metrics, metrics_csv_path)

        cm_csv_path = self.cm_dir / f"{self.model_name}_confusion_matrix.csv"
        save_model_confusion_matrix(test_metrics["confusion_matrix"], cm_csv_path)

        # Save predictions keyed by txId
        if self.idx_to_tx is not None:
            test_indices = self.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
            tx_ids = [self.idx_to_tx[i] for i in test_indices]
            
            with torch.no_grad():
                self.model.eval()
                all_log_probs = self.model(self.data.x, self.data.edge_index)
                test_probs = torch.exp(all_log_probs[self.test_mask])[:, 1].cpu().numpy()
                test_preds = all_log_probs[self.test_mask].argmax(dim=-1).cpu().numpy()

            pred_csv_path = self.predictions_dir / f"{self.model_name}_predictions.csv"
            save_model_predictions(tx_ids, test_preds, test_probs, pred_csv_path)

        # Save training curves
        curves_json_path = self.curves_dir / f"{self.model_name}_curves.json"
        with open(curves_json_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return {
            "model_name": self.model_name,
            "best_epoch": best_epoch,
            "best_val_f1": best_val_f1,
            "test_loss": test_loss,
            "test_metrics": test_metrics,
            "training_time": total_time,
            "checkpoint_path": str(checkpoint_path)
        }

