"""
Tests for REST API server.

Tests the FastAPI endpoints and ModelManager for serving Infinigram models.
"""

import pytest
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient

from infinigram.server.api import app, model_manager
from infinigram.server.models import ModelManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create test client and clean up models after each test."""
    # Clear any existing models
    for model_id in list(model_manager.models.keys()):
        model_manager.remove_model(model_id)

    with TestClient(app) as test_client:
        yield test_client

    # Cleanup
    for model_id in list(model_manager.models.keys()):
        model_manager.remove_model(model_id)


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4, 7]


@pytest.fixture
def client_with_model(client, sample_corpus):
    """Client with a pre-loaded model."""
    model_manager.add_model(
        model_id="test-model",
        corpus=sample_corpus,
        max_length=10,
        description="Test model"
    )
    return client


@pytest.fixture
def fresh_manager():
    """Fresh ModelManager instance for unit tests."""
    return ModelManager()


# ============================================================================
# ModelManager Unit Tests
# ============================================================================

class TestModelManager:
    """Tests for ModelManager class."""

    def test_add_model(self, fresh_manager, sample_corpus):
        """Test adding a model creates it with correct metadata."""
        fresh_manager.add_model(
            model_id="demo",
            corpus=sample_corpus,
            max_length=10,
            description="Demo model"
        )

        assert fresh_manager.has_model("demo")
        metadata = fresh_manager.metadata["demo"]
        assert metadata["id"] == "demo"
        assert metadata["corpus_size"] == len(sample_corpus)
        assert metadata["max_length"] == 10
        assert metadata["description"] == "Demo model"

    def test_add_duplicate_model_raises(self, fresh_manager, sample_corpus):
        """Test adding duplicate model raises ValueError."""
        fresh_manager.add_model("demo", sample_corpus)

        with pytest.raises(ValueError, match="already exists"):
            fresh_manager.add_model("demo", sample_corpus)

    def test_get_model(self, fresh_manager, sample_corpus):
        """Test retrieving a model by ID."""
        fresh_manager.add_model("demo", sample_corpus)

        model = fresh_manager.get_model("demo")
        assert model is not None
        assert len(model.corpus) == len(sample_corpus)

    def test_get_model_not_found_raises(self, fresh_manager):
        """Test getting nonexistent model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            fresh_manager.get_model("nonexistent")

    def test_remove_model(self, fresh_manager, sample_corpus):
        """Test removing a model."""
        fresh_manager.add_model("demo", sample_corpus)
        assert fresh_manager.has_model("demo")

        fresh_manager.remove_model("demo")
        assert not fresh_manager.has_model("demo")

    def test_remove_nonexistent_model_is_safe(self, fresh_manager):
        """Test removing nonexistent model doesn't raise."""
        # Should not raise
        fresh_manager.remove_model("nonexistent")

    def test_has_model(self, fresh_manager, sample_corpus):
        """Test has_model returns correct boolean."""
        assert not fresh_manager.has_model("demo")
        fresh_manager.add_model("demo", sample_corpus)
        assert fresh_manager.has_model("demo")

    def test_list_models(self, fresh_manager, sample_corpus):
        """Test listing all models returns metadata."""
        fresh_manager.add_model("model1", sample_corpus)
        fresh_manager.add_model("model2", [1, 2, 3])

        models = fresh_manager.list_models()
        assert len(models) == 2
        model_ids = {m["id"] for m in models}
        assert model_ids == {"model1", "model2"}

    def test_list_models_empty(self, fresh_manager):
        """Test list_models returns empty list when no models."""
        models = fresh_manager.list_models()
        assert models == []

    def test_save_and_load_model(self, fresh_manager, sample_corpus):
        """Test saving and loading a model from disk."""
        fresh_manager.add_model("demo", sample_corpus, max_length=5)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = Path(f.name)

        try:
            # Save
            fresh_manager.save_model("demo", model_path)
            assert model_path.exists()

            # Create new manager and load
            new_manager = ModelManager()
            new_manager.load_model("loaded", model_path, description="Loaded model")

            assert new_manager.has_model("loaded")
            loaded_model = new_manager.get_model("loaded")
            assert len(loaded_model.corpus) == len(sample_corpus)
        finally:
            model_path.unlink(missing_ok=True)

    def test_save_nonexistent_model_raises(self, fresh_manager):
        """Test saving nonexistent model raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            fresh_manager.save_model("nonexistent", Path("/tmp/test.pkl"))

    def test_load_duplicate_model_raises(self, fresh_manager, sample_corpus):
        """Test loading model with existing ID raises ValueError."""
        fresh_manager.add_model("demo", sample_corpus)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = Path(f.name)

        try:
            fresh_manager.save_model("demo", model_path)

            with pytest.raises(ValueError, match="already exists"):
                fresh_manager.load_model("demo", model_path)
        finally:
            model_path.unlink(missing_ok=True)


# ============================================================================
# API Root and Health Tests
# ============================================================================

class TestAPIRoot:
    """Tests for root and health endpoints."""

    def test_root_endpoint_returns_api_info(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Infinigram API"
        assert "version" in data
        assert "endpoints" in data
        assert "completions" in data["endpoints"]
        assert "models" in data["endpoints"]

    def test_health_endpoint_returns_healthy(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_health_endpoint_shows_model_count(self, client_with_model):
        """Test health endpoint shows correct model count."""
        response = client_with_model.get("/health")
        data = response.json()
        assert data["models_loaded"] == 1


# ============================================================================
# Models Endpoint Tests
# ============================================================================

class TestModelsEndpoints:
    """Tests for /v1/models endpoints."""

    def test_list_models_empty(self, client):
        """Test listing models when none are loaded."""
        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert data["data"] == []

    def test_list_models_with_loaded_models(self, client_with_model):
        """Test listing models returns loaded models."""
        response = client_with_model.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"

    def test_get_model_exists(self, client_with_model):
        """Test getting specific model info."""
        response = client_with_model.get("/v1/models/test-model")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "test-model"
        assert data["object"] == "model"
        assert "corpus_size" in data
        assert "vocab_size" in data

    def test_get_model_not_found_returns_404(self, client):
        """Test getting nonexistent model returns 404."""
        response = client.get("/v1/models/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_load_model_success(self, client, sample_corpus):
        """Test loading a new model via API."""
        response = client.post(
            "/v1/models/load",
            params={
                "model_id": "new-model",
                "description": "New test model"
            },
            json=sample_corpus
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "loaded"
        assert data["model_id"] == "new-model"

        # Verify it's in the list
        list_response = client.get("/v1/models")
        model_ids = [m["id"] for m in list_response.json()["data"]]
        assert "new-model" in model_ids

    def test_load_model_duplicate_returns_400(self, client_with_model, sample_corpus):
        """Test loading duplicate model returns 400."""
        response = client_with_model.post(
            "/v1/models/load",
            params={"model_id": "test-model"},
            json=sample_corpus
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_delete_model_success(self, client_with_model):
        """Test deleting a model."""
        response = client_with_model.delete("/v1/models/test-model")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "deleted"

        # Verify it's gone
        get_response = client_with_model.get("/v1/models/test-model")
        assert get_response.status_code == 404

    def test_delete_model_not_found_returns_404(self, client):
        """Test deleting nonexistent model returns 404."""
        response = client.delete("/v1/models/nonexistent")
        assert response.status_code == 404


# ============================================================================
# Completions Endpoint Tests
# ============================================================================

class TestCompletionsEndpoint:
    """Tests for /v1/completions endpoint."""

    def test_completion_requires_model(self, client, sample_corpus):
        """Test completion requires a valid model."""
        response = client.post(
            "/v1/completions",
            json={
                "model": "nonexistent",
                "prompt": [1, 2],
                "max_tokens": 5
            }
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_completion_with_valid_model(self, client_with_model):
        """Test completion with valid model returns predictions."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "finish_reason" in data["choices"][0]

    def test_completion_string_prompt_not_supported(self, client_with_model):
        """Test string prompts return error (not yet supported)."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello world",
                "max_tokens": 5
            }
        )
        assert response.status_code == 400
        assert "string prompts" in response.json()["detail"].lower()

    def test_completion_respects_max_tokens(self, client_with_model):
        """Test completion respects max_tokens parameter."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [1, 2],
                "max_tokens": 2
            }
        )
        assert response.status_code == 200

        data = response.json()
        tokens = data["choices"][0]["metadata"]["tokens"]
        assert len(tokens) <= 2

    def test_completion_with_weight_function(self, client_with_model):
        """Test completion with weight function parameter."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3,
                "weight_function": "quadratic"
            }
        )
        assert response.status_code == 200
        assert response.json()["choices"][0] is not None

    def test_completion_invalid_weight_function(self, client_with_model):
        """Test completion with invalid weight function returns 400."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3,
                "weight_function": "invalid_function"
            }
        )
        assert response.status_code == 400
        assert "unknown weight function" in response.json()["detail"].lower()

    def test_completion_returns_logprobs_when_requested(self, client_with_model):
        """Test completion returns logprobs when requested."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 2,
                "logprobs": 3
            }
        )
        assert response.status_code == 200

        data = response.json()
        logprobs = data["choices"][0]["logprobs"]
        assert logprobs is not None
        assert "content" in logprobs

    def test_completion_usage_stats_correct(self, client_with_model):
        """Test completion returns correct usage statistics."""
        prompt = [1, 2, 3]
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": prompt,
                "max_tokens": 2
            }
        )
        assert response.status_code == 200

        data = response.json()
        usage = data["usage"]
        assert usage["prompt_tokens"] == len(prompt)
        assert usage["completion_tokens"] >= 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completion_metadata_includes_confidence(self, client_with_model):
        """Test completion metadata includes confidence score."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 2
            }
        )
        assert response.status_code == 200

        data = response.json()
        metadata = data["choices"][0]["metadata"]
        assert "confidence" in metadata
        assert "match_length" in metadata
        assert 0.0 <= metadata["confidence"] <= 1.0

    def test_completion_finish_reason_length(self, client_with_model):
        """Test completion returns 'length' finish_reason when max_tokens reached."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 1
            }
        )
        assert response.status_code == 200

        data = response.json()
        # If we got exactly max_tokens, finish_reason should be "length"
        if len(data["choices"][0]["metadata"]["tokens"]) == 1:
            assert data["choices"][0]["finish_reason"] == "length"

    def test_completion_with_no_match_still_predicts(self, client):
        """Test completion still works when no exact match (uses smoothing)."""
        # Create model with very short corpus
        model_manager.add_model("tiny", [1, 2])

        try:
            response = client.post(
                "/v1/completions",
                json={
                    "model": "tiny",
                    "prompt": [99, 99, 99],  # No exact matches
                    "max_tokens": 2
                }
            )
            assert response.status_code == 200

            data = response.json()
            # Should still generate tokens (smoothing ensures all vocab has probability)
            assert len(data["choices"]) == 1
            # Finish reason should be "length" since we hit max_tokens
            # (smoothing ensures predictions are always available)
            assert data["choices"][0]["finish_reason"] in ["length", "stop"]
        finally:
            model_manager.remove_model("tiny")


# ============================================================================
# Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for the API."""

    def test_full_workflow(self, client, sample_corpus):
        """Test complete workflow: load, predict, delete."""
        # 1. List models (should be empty)
        response = client.get("/v1/models")
        assert len(response.json()["data"]) == 0

        # 2. Load a model
        response = client.post(
            "/v1/models/load",
            params={"model_id": "workflow-test"},
            json=sample_corpus
        )
        assert response.status_code == 200

        # 3. Get model info
        response = client.get("/v1/models/workflow-test")
        assert response.status_code == 200
        assert response.json()["corpus_size"] == len(sample_corpus)

        # 4. Make a completion
        response = client.post(
            "/v1/completions",
            json={
                "model": "workflow-test",
                "prompt": [2, 3],
                "max_tokens": 3
            }
        )
        assert response.status_code == 200
        assert len(response.json()["choices"]) == 1

        # 5. Delete the model
        response = client.delete("/v1/models/workflow-test")
        assert response.status_code == 200

        # 6. Verify it's gone
        response = client.get("/v1/models/workflow-test")
        assert response.status_code == 404

    def test_multiple_models(self, client):
        """Test handling multiple models simultaneously."""
        # Load two models
        client.post(
            "/v1/models/load",
            params={"model_id": "model-a"},
            json=[1, 2, 3, 4, 5]
        )
        client.post(
            "/v1/models/load",
            params={"model_id": "model-b"},
            json=[10, 20, 30, 40, 50]
        )

        # Both should be listed
        response = client.get("/v1/models")
        model_ids = {m["id"] for m in response.json()["data"]}
        assert model_ids == {"model-a", "model-b"}

        # Both should accept completions
        for model_id in ["model-a", "model-b"]:
            response = client.post(
                "/v1/completions",
                json={"model": model_id, "prompt": [1], "max_tokens": 1}
            )
            assert response.status_code == 200

        # Cleanup
        client.delete("/v1/models/model-a")
        client.delete("/v1/models/model-b")
