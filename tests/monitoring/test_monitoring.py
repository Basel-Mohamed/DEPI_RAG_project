import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from main import app

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_grafana_service():
    with patch(
        "app.services.monitoring.grafana_service.GrafanaService.check_qdrant",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = {
            "reachable": True,
            "status_code": 200,
            "healthy": True,
        }
        yield mock


@pytest.fixture
def mock_grafana_service_down():
    with patch(
        "app.services.monitoring.grafana_service.GrafanaService.check_qdrant",
        new_callable=AsyncMock,
    ) as mock:
        mock.return_value = {
            "reachable": False,
            "status_code": None,
            "healthy": False,
        }
        yield mock


# ── /health ──────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status_ok(self):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_contains_environment(self):
        response = client.get("/health")
        data = response.json()
        assert "environment" in data

    def test_health_content_type_is_json(self):
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]


# ── /metrics ─────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_content_type_is_prometheus(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]

    def test_metrics_body_is_not_empty(self):
        response = client.get("/metrics")
        assert len(response.text) > 0

    def test_metrics_contains_request_counter(self):
        response = client.get("/metrics")
        assert "rag_requests_total" in response.text

    def test_metrics_contains_request_latency(self):
        response = client.get("/metrics")
        assert "rag_request_duration_seconds" in response.text

    def test_metrics_contains_pipeline_stage_metrics(self):
        response = client.get("/metrics")
        assert "embedding_duration_seconds" in response.text
        assert "qdrant_query_duration_seconds" in response.text
        assert "llm_response_duration_seconds" in response.text
        assert "reranking_duration_seconds" in response.text

    def test_metrics_contains_ingestion_counters(self):
        response = client.get("/metrics")
        assert "documents_ingested_total" in response.text
        assert "document_chunks_ingested_total" in response.text


# ── /monitoring/stats ────────────────────────────────────────────────────────

class TestMonitoringStatsEndpoint:

    def test_stats_returns_200(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        assert response.status_code == 200

    def test_stats_contains_qdrant_up_key(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert "qdrant_up" in data

    def test_stats_contains_metrics_key(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert "metrics" in data

    def test_stats_qdrant_up_is_boolean(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert isinstance(data["qdrant_up"], bool)

    def test_stats_metrics_is_dict(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert isinstance(data["metrics"], dict)

    def test_stats_qdrant_up_true_when_healthy(self, mock_grafana_service):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert data["qdrant_up"] is True

    def test_stats_qdrant_up_false_when_down(self, mock_grafana_service_down):
        response = client.get("/monitoring/stats")
        data = response.json()
        assert data["qdrant_up"] is False


# ── Middleware ────────────────────────────────────────────────────────────────

class TestMetricsMiddleware:

    def test_request_counter_increments_after_request(self):
        client.get("/health")
        metrics = client.get("/metrics").text
        assert "rag_requests_total" in metrics

    def test_request_counter_tracks_status_code(self):
        client.get("/health")
        metrics = client.get("/metrics").text
        assert 'status_code="200"' in metrics

    def test_request_counter_tracks_endpoint(self):
        client.get("/health")
        metrics = client.get("/metrics").text
        assert 'endpoint="/health"' in metrics

    def test_latency_histogram_recorded_after_request(self):
        client.get("/health")
        metrics = client.get("/metrics").text
        assert "rag_request_duration_seconds_bucket" in metrics

    def test_error_request_tracked_in_counter(self):
        client.get("/this-route-does-not-exist")
        metrics = client.get("/metrics").text
        assert "rag_requests_total" in metrics


# ── End to end ───────────────────────────────────────────────────────────────

class TestEndToEndMonitoring:

    def test_ingestion_increments_document_counter(self):
        client.post("/api/v1/documents/ingest", json={
            "title": "Test Doc",
            "content": "This is test content for monitoring.",
            "metadata": {}
        })
        metrics = client.get("/metrics").text
        assert "documents_ingested_total" in metrics

    def test_query_records_qdrant_latency(self):
        client.post("/api/v1/documents/ingest", json={
            "title": "Test Doc",
            "content": "This is test content for monitoring.",
            "metadata": {}
        })
        client.post("/api/v1/query", json={
            "question": "What is in the test doc?",
            "top_k": 1
        })
        metrics = client.get("/metrics").text
        assert "qdrant_query_duration_seconds_bucket" in metrics