"""Data loading and caching utility module for the Crypto AML Forensic Dashboard.

Loads existing Phase 1, Phase 2, and Phase 3 artifacts from disk:
- Dataset statistics & validation manifests (JSON / Markdown)
- 5-Model comparative scorecard (CSV)
- Confusion matrices for all models (CSV)
- 100-epoch training curves for GNN architectures (JSON)
- Prediction records for 8,841 held-out test transactions (CSV)
- Topological graph data status checks
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    import streamlit as st
    cache_data = st.cache_data
except ImportError:
    # Fallback no-op decorator if executed outside Streamlit environment
    def cache_data(func):
        return func


def get_project_root() -> Path:
    """Return the absolute path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@cache_data
def load_dataset_manifest() -> Dict[str, Any]:
    """Load verified dataset summary manifest generated in Phase 1."""
    root = get_project_root()
    manifest_path = root / "results" / "data_summary" / "dataset_manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_dataset_statistics_md() -> str:
    """Load pre-formatted markdown summary of dataset statistics."""
    root = get_project_root()
    stats_path = root / "results" / "data_summary" / "dataset_statistics.md"
    if not stats_path.exists():
        return "Dataset statistics file not found."
    with open(stats_path, "r", encoding="utf-8") as f:
        return f.read()


@cache_data
def load_validation_summary() -> Dict[str, Any]:
    """Load Phase 1 dataset validation summary."""
    root = get_project_root()
    val_path = root / "results" / "data_summary" / "validation_summary.json"
    if not val_path.exists():
        return {}
    with open(val_path, "r", encoding="utf-8") as f:
        return json.load(f)


@cache_data
def load_model_comparison() -> List[Dict[str, Any]]:
    """Load unified 5-model comparative scorecard.
    
    Returns a list of dictionaries corresponding to:
    Random Forest, GCN+Lin, GraphSAGE+Lin, GATv2+Lin, Chebyshev+Lin.
    """
    root = get_project_root()
    csv_path = root / "results" / "tables" / "gnn_model_comparison.csv"
    if not csv_path.exists():
        return []

    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = None
                else:
                    try:
                        # Attempt int conversion then float conversion
                        if "." in v or "e" in v.lower():
                            parsed[k] = float(v)
                        else:
                            parsed[k] = int(v)
                    except ValueError:
                        parsed[k] = v
            records.append(parsed)
    return records


@cache_data
def load_training_curves() -> Dict[str, Dict[str, List[float]]]:
    """Load 100-epoch training curves for GNN models.
    
    Keys: 'chebyshev', 'gatv2', 'gcn', 'graphsage'
    Metrics per model: epoch, train_loss, train_f1, val_loss, val_precision, val_recall, val_f1, val_pr_auc, lr
    """
    root = get_project_root()
    curves_dir = root / "results" / "training_curves"
    curves = {}
    for model_key in ["chebyshev", "gatv2", "gcn", "graphsage"]:
        p = curves_dir / f"{model_key}_curves.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                curves[model_key] = json.load(f)
    return curves


@cache_data
def load_confusion_matrices() -> Dict[str, Dict[str, Any]]:
    """Load 2x2 confusion matrices for all five models.
    
    Structure per model:
    {
        'TP': int, # True Positives (Actual Illicit, Predicted Illicit)
        'FN': int, # False Negatives (Actual Illicit, Predicted Licit)
        'FP': int, # False Positives (Actual Licit, Predicted Illicit)
        'TN': int, # True Negatives (Actual Licit, Predicted Licit)
        'total': int,
        'actual_illicit': int,
        'actual_licit': int,
        'matrix': [[TP, FN], [FP, TN]]
    }
    """
    root = get_project_root()
    models_cm = {
        "Random Forest": root / "results" / "random_forest_confusion_matrix.csv",
        "GCN+Lin": root / "results" / "confusion_matrices" / "gcn_confusion_matrix.csv",
        "GraphSAGE+Lin": root / "results" / "confusion_matrices" / "graphsage_confusion_matrix.csv",
        "GATv2+Lin": root / "results" / "confusion_matrices" / "gatv2_confusion_matrix.csv",
        "Chebyshev+Lin": root / "results" / "confusion_matrices" / "chebyshev_confusion_matrix.csv",
    }

    result = {}
    for name, p in models_cm.items():
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            # Row 0: ,illicit,licit
            # Row 1: illicit, TP, FN
            # Row 2: licit, FP, TN
            tp = int(reader[1][1])
            fn = int(reader[1][2])
            fp = int(reader[2][1])
            tn = int(reader[2][2])
            actual_illicit = tp + fn
            actual_licit = fp + tn
            total = actual_illicit + actual_licit

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / total if total > 0 else 0.0

            result[name] = {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "total": total,
                "actual_illicit": actual_illicit,
                "actual_licit": actual_licit,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "accuracy": acc,
                "matrix": [[tp, fn], [fp, tn]]
            }
    return result


@cache_data
def load_all_predictions() -> Dict[str, Dict[str, Any]]:
    """Load test predictions across all 5 models for all 8,841 held-out test transactions.
    
    Returns a dict mapping:
    txId -> {
        'txId': str,
        'rf_pred': int, 'rf_prob': float,
        'gcn_pred': int, 'gcn_prob': float,
        'sage_pred': int, 'sage_prob': float,
        'gat_pred': int, 'gat_prob': float,
        'cheb_pred': int, 'cheb_prob': float,
        'illicit_votes': int, # Count of models predicting 1 (0 to 5)
        'avg_prob': float,    # Mean probability across all 5 models
        'max_prob': float,
        'risk_tier': str,     # 'Critical Risk', 'High Risk', 'Elevated Risk', 'Low Risk'
        'consensus': str
    }
    """
    root = get_project_root()
    files = {
        "rf": root / "results" / "random_forest_predictions.csv",
        "cheb": root / "results" / "predictions" / "chebyshev_predictions.csv",
        "gcn": root / "results" / "predictions" / "gcn_predictions.csv",
        "sage": root / "results" / "predictions" / "graphsage_predictions.csv",
        "gat": root / "results" / "predictions" / "gatv2_predictions.csv",
    }

    # Load each file into a temp map
    model_data = {}
    for key, p in files.items():
        m = {}
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) >= 3:
                        m[row[0].strip()] = (int(row[1]), float(row[2]))
        model_data[key] = m

    all_tx_ids = set()
    for m in model_data.values():
        all_tx_ids.update(m.keys())

    consolidated = {}
    for tx in all_tx_ids:
        rf_p, rf_prob = model_data["rf"].get(tx, (0, 0.0))
        gcn_p, gcn_prob = model_data["gcn"].get(tx, (0, 0.0))
        sage_p, sage_prob = model_data["sage"].get(tx, (0, 0.0))
        gat_p, gat_prob = model_data["gat"].get(tx, (0, 0.0))
        cheb_p, cheb_prob = model_data["cheb"].get(tx, (0, 0.0))

        preds = [rf_p, gcn_p, sage_p, gat_p, cheb_p]
        probs = [rf_prob, gcn_prob, sage_prob, gat_prob, cheb_prob]

        illicit_votes = sum(preds)
        avg_prob = sum(probs) / len(probs)
        max_prob = max(probs)

        if illicit_votes == 5:
            consensus = "Unanimous Illicit (5/5)"
        elif illicit_votes == 4:
            consensus = "Strong Illicit (4/5)"
        elif illicit_votes == 3:
            consensus = "Majority Illicit (3/5)"
        elif illicit_votes == 2:
            consensus = "Majority Licit (3/5 Licit)"
        elif illicit_votes == 1:
            consensus = "Strong Licit (4/5 Licit)"
        else:
            consensus = "Unanimous Licit (5/5 Licit)"

        if avg_prob >= 0.70 or illicit_votes >= 4:
            risk_tier = "Critical Risk"
        elif avg_prob >= 0.45 or illicit_votes >= 3:
            risk_tier = "High Risk"
        elif avg_prob >= 0.20 or illicit_votes >= 1:
            risk_tier = "Elevated Suspicion"
        else:
            risk_tier = "Low Risk / Clean"

        consolidated[tx] = {
            "txId": tx,
            "rf_pred": rf_p, "rf_prob": rf_prob,
            "gcn_pred": gcn_p, "gcn_prob": gcn_prob,
            "sage_pred": sage_p, "sage_prob": sage_prob,
            "gat_pred": gat_p, "gat_prob": gat_prob,
            "cheb_pred": cheb_p, "cheb_prob": cheb_prob,
            "illicit_votes": illicit_votes,
            "avg_prob": avg_prob,
            "max_prob": max_prob,
            "risk_tier": risk_tier,
            "consensus": consensus
        }

    return consolidated


@cache_data
def get_sample_transactions() -> Dict[str, List[str]]:
    """Return curated lists of representative test Bitcoin transactions for quick audit."""
    predictions = load_all_predictions()
    unanimous_illicit = []
    cheb_caught_rf_missed = []
    unanimous_licit = []
    disagreed = []

    for tx, item in predictions.items():
        if item["illicit_votes"] == 5:
            unanimous_illicit.append(tx)
        elif item["illicit_votes"] == 0:
            unanimous_licit.append(tx)

        if item["cheb_pred"] == 1 and item["rf_pred"] == 0:
            cheb_caught_rf_missed.append(tx)

        if item["illicit_votes"] in [2, 3]:
            disagreed.append(tx)

    return {
        "unanimous_illicit": unanimous_illicit[:10],
        "cheb_caught_rf_missed": cheb_caught_rf_missed[:10],
        "unanimous_licit": unanimous_licit[:10],
        "disagreed": disagreed[:10]
    }


def check_graph_files_status() -> Dict[str, Any]:
    """Check availability of topological graph artifacts required for neighborhood visualization."""
    root = get_project_root()
    processed_edge = root / "data" / "processed" / "elliptic_edge_index.npy"
    processed_mapping = root / "data" / "processed" / "elliptic_tx_to_idx.json"
    raw_edgelist = root / "data" / "raw" / "elliptic_txs_edgelist.csv"

    has_processed_edge = processed_edge.exists()
    has_processed_mapping = processed_mapping.exists()
    has_raw_edgelist = raw_edgelist.exists()

    is_visualizable = (has_processed_edge and has_processed_mapping) or has_raw_edgelist

    return {
        "is_visualizable": is_visualizable,
        "processed_edge_exists": has_processed_edge,
        "processed_mapping_exists": has_processed_mapping,
        "raw_edgelist_exists": has_raw_edgelist,
        "processed_edge_path": str(processed_edge),
        "processed_mapping_path": str(processed_mapping),
        "raw_edgelist_path": str(raw_edgelist),
    }
