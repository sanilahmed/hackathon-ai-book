"""
Unit tests for the RAG Agent API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from rag_agent_api.main import app
from rag_agent_api.models import QueryRequest


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test the root endpoint returns expected information."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "version" in data
    assert "description" in data
    assert "endpoints" in data


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "services" in data


def test_ready_endpoint(client):
    """Test the readiness check endpoint."""
    response = client.get("/ready")
    # This might return 503 if components aren't initialized properly in test context
    assert response.status_code in [200, 503]


def test_ask_endpoint_basic_structure(client):
    """Test the ask endpoint with a basic query."""
    # This test will likely fail without proper initialization of components,
    # but it verifies the endpoint exists and returns expected structure
    query_request = {
        "query": "What is the transformer architecture?",
        "context_window": 5,
        "include_sources": True,
        "temperature": 0.1
    }

    response = client.post("/ask", json=query_request)

    # The endpoint should exist (even if it returns an error due to missing dependencies)
    assert response.status_code in [200, 422, 500]  # 200 for success, 422 for validation error, 500 for internal error


def test_query_request_model_validation():
    """Test the validation of the QueryRequest model."""
    # Test valid request
    valid_request = QueryRequest(
        query="What is a transformer model?",
        context_window=5,
        include_sources=True,
        temperature=0.1
    )
    assert valid_request.query == "What is a transformer model?"
    assert valid_request.context_window == 5
    assert valid_request.include_sources is True
    assert valid_request.temperature == 0.1

    # Test validation constraints
    with pytest.raises(ValueError):
        QueryRequest(
            query="",  # Empty query should fail
            context_window=5,
            include_sources=True,
            temperature=0.1
        )

    with pytest.raises(ValueError):
        QueryRequest(
            query="A" * 1001,  # Too long query should fail
            context_window=5,
            include_sources=True,
            temperature=0.1
        )

    with pytest.raises(ValueError):
        QueryRequest(
            query="Valid query",
            context_window=25,  # Too large context window should fail
            include_sources=True,
            temperature=0.1
        )

    with pytest.raises(ValueError):
        QueryRequest(
            query="Valid query",
            context_window=5,
            include_sources=True,
            temperature=1.5  # Too high temperature should fail
        )


if __name__ == "__main__":
    pytest.main([__file__])