"""Unit and integration tests for CryptoScope AI FastAPI Backend Service."""

import pytest
import sys
from urllib.parse import urlparse, parse_qs
from pathlib import Path

# Ensure root and backend are in path
ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "frontend" / "cryptoscope-ai" / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import main
from main import (
    app,
    STATE,
    init_engine,
    PredictPayload,
    health_check,
    get_transaction,
    predict_transaction,
    predict_random_forest,
    predict_gcn,
    predict_graphsage,
    predict_gatv2,
    predict_chebyshev,
    compare_models,
    get_risk_classification,
    get_graph_neighborhood,
    get_dataset_statistics,
    get_metrics,
    get_sample_transactions,
)


class MockResponse:
    def __init__(self, data, status_code: int = 200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


class DirectBackendClient:
    """
    Direct invocation client simulating ASGI HTTP client interface without 
    event-loop deadlocks on Windows/Python 3.13.
    """
    def __init__(self):
        init_engine()

    def get(self, url: str):
        parsed = urlparse(url)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            return MockResponse(health_check())
        elif path.startswith("/api/transaction/"):
            tx_id = path.split("/api/transaction/")[1]
            return MockResponse(get_transaction(tx_id))
        elif path.startswith("/api/predict/random-forest/"):
            tx_id = path.split("/api/predict/random-forest/")[1]
            return MockResponse(predict_random_forest(tx_id))
        elif path.startswith("/api/predict/gcn/"):
            tx_id = path.split("/api/predict/gcn/")[1]
            return MockResponse(predict_gcn(tx_id))
        elif path.startswith("/api/predict/graphsage/"):
            tx_id = path.split("/api/predict/graphsage/")[1]
            return MockResponse(predict_graphsage(tx_id))
        elif path.startswith("/api/predict/gatv2/"):
            tx_id = path.split("/api/predict/gatv2/")[1]
            return MockResponse(predict_gatv2(tx_id))
        elif path.startswith("/api/predict/chebyshev/"):
            tx_id = path.split("/api/predict/chebyshev/")[1]
            return MockResponse(predict_chebyshev(tx_id))
        elif path.startswith("/api/compare/"):
            tx_id = path.split("/api/compare/")[1]
            return MockResponse(compare_models(tx_id))
        elif path.startswith("/api/risk/"):
            tx_id = path.split("/api/risk/")[1]
            return MockResponse(get_risk_classification(tx_id))
        elif path.startswith("/api/graph/"):
            tx_id = path.split("/api/graph/")[1]
            hops = int(query.get("hops", [1])[0])
            max_nodes = int(query.get("max_nodes", [40])[0])
            return MockResponse(get_graph_neighborhood(tx_id, hops=hops, max_nodes=max_nodes))
        elif path == "/api/dataset/stats":
            return MockResponse(get_dataset_statistics())
        elif path == "/api/metrics":
            return MockResponse(get_metrics())
        elif path == "/api/transactions/sample":
            return MockResponse(get_sample_transactions())
        else:
            return MockResponse({"detail": "Not Found"}, status_code=404)

    def post(self, url: str, json: dict = None):
        parsed = urlparse(url)
        path = parsed.path
        if path == "/api/predict":
            payload = PredictPayload(**(json or {}))
            return MockResponse(predict_transaction(payload))
        return MockResponse({"detail": "Not Found"}, status_code=404)


@pytest.fixture(scope="module")
def client():
    """Create a test client with initialized backend state."""
    return DirectBackendClient()


def test_health_endpoint(client):
    """Test health telemetry probe."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["graph_loaded"] is True
    assert data["total_transactions"] == 203769
    assert data["total_edges"] == 234355
    assert data["models_loaded"]["random_forest"] is True
    assert data["models_loaded"]["gcn"] is True
    assert data["models_loaded"]["graphsage"] is True
    assert data["models_loaded"]["gatv2"] is True
    assert data["models_loaded"]["chebyshev"] is True


def test_transaction_lookup_real_dataset(client):
    """Test lookup of real Elliptic transaction (72629711 - Illicit)."""
    res = client.get("/api/transaction/72629711")
    assert res.status_code == 200
    data = res.json()
    assert data["txId"] == "72629711"
    assert data["timestep"] == 42
    assert data["split"] == "test"
    assert data["ground_truth"] == "illicit"
    assert "predictions" in data
    assert "random_forest" in data["predictions"]
    assert "gcn" in data["predictions"]
    assert "graphsage" in data["predictions"]
    assert "gatv2" in data["predictions"]
    assert "chebyshev" in data["predictions"]
    assert data["consensus"]["illicit_votes"] == 5
    assert data["consensus"]["risk_level"] == "critical"
    assert len(data["features_sample"]) >= 5


def test_transaction_lookup_preset(client):
    """Test lookup of verified benchmark preset (5530414 - Licit)."""
    res = client.get("/api/transaction/5530414")
    assert res.status_code == 200
    data = res.json()
    assert data["txId"] == "5530414"
    assert data["ground_truth"] == "licit"
    assert data["consensus"]["risk_level"] == "low"


def test_predict_post(client):
    """Test POST /api/predict with transaction ID."""
    res = client.post("/api/predict", json={"tx_id": "106359119"})
    assert res.status_code == 200
    data = res.json()
    assert data["txId"] == "106359119"
    assert "predictions" in data
    assert "consensus" in data


def test_individual_model_prediction_endpoints(client):
    """Test dedicated model prediction endpoints."""
    tx_id = "72629711"
    for model_name in ["random-forest", "gcn", "graphsage", "gatv2", "chebyshev"]:
        res = client.get(f"/api/predict/{model_name}/{tx_id}")
        assert res.status_code == 200, f"Failed for model {model_name}"
        data = res.json()
        assert "prediction" in data
        assert data["prediction"]["predicted_class"] in ["licit", "illicit"]


def test_model_comparison_and_risk(client):
    """Test compare and risk endpoints."""
    res_comp = client.get("/api/compare/72629711")
    assert res_comp.status_code == 200
    assert len(res_comp.json()["comparison"]) == 5

    res_risk = client.get("/api/risk/72629711")
    assert res_risk.status_code == 200
    assert res_risk.json()["risk_level"] == "critical"


def test_graph_neighborhood_extraction(client):
    """Test 1-hop and 2-hop graph extraction."""
    # 1-hop
    res1 = client.get("/api/graph/72629711?hops=1")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["central_node_id"] == "72629711"
    assert data1["hops"] == 1
    assert len(data1["nodes"]) >= 1
    assert data1["nodes"][0]["is_central"] is True

    # 2-hop
    res2 = client.get("/api/graph/72629711?hops=2")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["hops"] == 2
    assert len(data2["nodes"]) >= len(data1["nodes"])


def test_dataset_stats_and_metrics(client):
    """Test macro dataset statistics and benchmark scorecards."""
    res_stats = client.get("/api/dataset/stats")
    assert res_stats.status_code == 200
    assert res_stats.json()["total_transactions"] == 203769

    res_metrics = client.get("/api/metrics")
    assert res_metrics.status_code == 200
    assert len(res_metrics.json()) == 5


def test_sample_transactions(client):
    """Test sample transaction audit presets."""
    res = client.get("/api/transactions/sample")
    assert res.status_code == 200
    data = res.json()
    assert "unanimous_illicit" in data
    assert "unanimous_licit" in data
