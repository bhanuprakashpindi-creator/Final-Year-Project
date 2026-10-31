"""Standardized evaluation metrics for Crypto AML models.

Computes classification metrics with specific focus on the minority illicit class (1),
macro-averaged metrics, accuracy, and confusion matrix.
"""

from typing import Dict, Any, Union, Optional
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)


def compute_all_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob_illicit: Union[np.ndarray, list]
) -> Dict[str, Any]:
    """
    Compute comprehensive AML evaluation metrics on labeled transactions.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 = licit, 1 = illicit). Unknowns (-1) must not be included.
    y_pred : np.ndarray
        Predicted binary labels (0 = licit, 1 = illicit).
    y_prob_illicit : np.ndarray
        Predicted probabilities for the illicit class (class 1), in range [0, 1].
        
    Returns
    -------
    dict containing:
        - precision: illicit class precision
        - recall: illicit class recall
        - f1: illicit class F1-score
        - pr_auc: Area Under Precision-Recall Curve (average precision) for illicit class
        - macro_f1: unweighted average of F1 for licit and illicit classes
        - accuracy: overall fraction of correctly classified transactions
        - confusion_matrix: 2x2 ndarray [[TP, FN], [FP, TN]] matching Phase 2 format
        - true_positives, false_positives, true_negatives, false_negatives
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob_illicit, dtype=float)

    # Validate inputs
    unique_labels = np.unique(y_true)
    if -1 in unique_labels:
        raise ValueError("Unknown label (-1) detected in evaluation input. Filter unknown labels first.")

    # Illicit-class (Class 1) metrics
    precision = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    recall = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    pr_auc = float(average_precision_score(y_true, y_prob))

    # Overall / macro metrics
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    # Confusion matrix in Phase 2 orientation: [[TP, FN], [FP, TN]]
    # labels=[1, 0] ensures row 0 = illicit, row 1 = licit; col 0 = pred illicit, col 1 = pred licit
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "macro_f1": macro_f1,
        "accuracy": acc,
        "confusion_matrix": cm,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "true_negatives": tn,
        "total_evaluated": int(len(y_true)),
        "illicit_count": int((y_true == 1).sum()),
        "licit_count": int((y_true == 0).sum())
    }


def save_model_metrics(
    metrics: Dict[str, Any],
    file_path: Union[str, Path]
) -> None:
    """Save metrics dictionary to a two-column CSV file matching Phase 2 format."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write("metric,value\n")
        f.write(f"precision,{metrics['precision']:.6f}\n")
        f.write(f"recall,{metrics['recall']:.6f}\n")
        f.write(f"f1,{metrics['f1']:.6f}\n")
        f.write(f"pr_auc,{metrics['pr_auc']:.6f}\n")
        f.write(f"macro_f1,{metrics['macro_f1']:.6f}\n")
        f.write(f"accuracy,{metrics['accuracy']:.6f}\n")


def save_model_confusion_matrix(
    cm: np.ndarray,
    file_path: Union[str, Path]
) -> None:
    """Save 2x2 confusion matrix to CSV matching Phase 2 structure."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write(",illicit,licit\n")
        f.write(f"illicit,{cm[0, 0]},{cm[0, 1]}\n")
        f.write(f"licit,{cm[1, 0]},{cm[1, 1]}\n")


def save_model_predictions(
    tx_ids: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    prob_illicit: Union[np.ndarray, list],
    file_path: Union[str, Path]
) -> None:
    """Save predictions keyed by transaction ID matching Phase 2 structure."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        f.write("txId,prediction,prob_illicit\n")
        for tx, pred, prob in zip(tx_ids, y_pred, prob_illicit):
            f.write(f"{tx},{int(pred)},{float(prob):.6f}\n")

