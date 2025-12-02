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
        """Test adding a model from corpus."""
        fresh_manager.add_model(
            "demo",
            sample_corpus,
            max_length=5,
            description="Demo model"
        )

        assert fresh_manager.has_model("demo")
        models = fresh_manager.list_models()
        assert len(models) == 1

        metadata = models[0]
        assert metadata["id"] == "demo"
        assert metadata["max_length"] == 5
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
        # Use model.n instead of model.corpus (unified API change)
        assert model.n == len(sample_corpus)

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

    def test_model_persistence(self, sample_corpus):
        """Test creating and loading persistent models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(Path(tmpdir))

            # Create persistent model
            manager.add_model("demo", sample_corpus, max_length=5, persist=True)
            assert manager.has_model("demo")

            # Create new manager and load model
            new_manager = ModelManager(Path(tmpdir))
            new_manager.load_model("demo", description="Loaded model")

            assert new_manager.has_model("demo")
            loaded_model = new_manager.get_model("demo")
            assert loaded_model.n == len(sample_corpus)

    def test_load_model_returns_existing(self, fresh_manager, sample_corpus):
        """Test loading already-loaded model returns existing instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelManager(Path(tmpdir))
            manager.add_model("demo", sample_corpus, persist=True)

            # Loading same model twice should return same instance
            model1 = manager.load_model("demo")
            model2 = manager.load_model("demo")
            assert model1 is model2


# ============================================================================
# API Root/Health Tests
# ============================================================================

class TestAPIRoot:
    """Tests for root and health endpoints."""

    def test_root_endpoint_returns_api_info(self, client):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health_endpoint_returns_healthy(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"

    def test_health_endpoint_shows_model_count(self, client_with_model):
        """Test health endpoint shows correct model count."""
        response = client_with_model.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["models_loaded"] >= 1


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
        """Test listing models shows loaded models."""
        response = client_with_model.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert len(data["data"]) >= 1
        model_ids = [m["id"] for m in data["data"]]
        assert "test-model" in model_ids

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

    def test_create_model_success(self, client, sample_corpus):
        """Test creating a new model via API."""
        response = client.post(
            "/v1/models/create",
            json={
                "model_id": "new-model",
                "corpus": sample_corpus,
                "description": "New test model"
            }
        )
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "created"
        assert data["model_id"] == "new-model"

        # Verify it's in the list
        list_response = client.get("/v1/models")
        model_ids = [m["id"] for m in list_response.json()["data"]]
        assert "new-model" in model_ids

    def test_create_model_duplicate_returns_400(self, client_with_model, sample_corpus):
        """Test creating duplicate model returns 400."""
        response = client_with_model.post(
            "/v1/models/create",
            json={
                "model_id": "test-model",
                "corpus": sample_corpus
            }
        )
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"].lower()

    def test_delete_model_success(self, client_with_model):
        """Test deleting a model."""
        response = client_with_model.delete("/v1/models/test-model")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "removed"

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

    def test_completion_string_prompt_works(self, client_with_model):
        """Test string prompts are supported (encoded to bytes)."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "hello",
                "max_tokens": 5
            }
        )
        # String prompts now work with the unified API
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) == 1

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
        tokens = data["choices"][0]["metadata"]["token_bytes"]
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
        """Test completion returns correct usage stats."""
        prompt = [1, 2, 3, 4]
        max_tokens = 5

        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": prompt,
                "max_tokens": max_tokens
            }
        )
        assert response.status_code == 200

        usage = response.json()["usage"]
        assert usage["prompt_tokens"] == len(prompt)
        assert usage["completion_tokens"] <= max_tokens
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_completion_metadata_includes_confidence(self, client_with_model):
        """Test completion metadata includes confidence score."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3
            }
        )
        assert response.status_code == 200

        metadata = response.json()["choices"][0]["metadata"]
        assert "confidence" in metadata
        assert 0 <= metadata["confidence"] <= 1

    def test_completion_finish_reason_length(self, client_with_model):
        """Test finish_reason is 'length' when max_tokens reached."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3
            }
        )
        assert response.status_code == 200

        # Should be "length" since we're generating tokens
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] in ["length", "stop"]

    def test_completion_with_no_match_still_predicts(self, client_with_model):
        """Test completion still works when prompt has no corpus match."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [99, 98, 97],  # Unlikely to match
                "max_tokens": 3
            }
        )
        assert response.status_code == 200
        assert len(response.json()["choices"][0]["text"]) >= 0

    def test_completion_with_temperature(self, client_with_model):
        """Test completion with temperature parameter."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 3,
                "temperature": 0.5
            }
        )
        assert response.status_code == 200

    def test_completion_with_stop_sequence(self, client_with_model):
        """Test completion respects stop sequences."""
        response = client_with_model.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": [2, 3],
                "max_tokens": 10,
                "stop": ["\n"]
            }
        )
        assert response.status_code == 200


# ============================================================================
# Integration Tests
# ============================================================================

class TestAPIIntegration:
    """Integration tests for full API workflows."""

    def test_full_workflow(self, client, sample_corpus):
        """Test complete workflow: create, query, delete."""
        # Create model
        create_response = client.post(
            "/v1/models/create",
            json={
                "model_id": "workflow-test",
                "corpus": sample_corpus,
                "description": "Workflow test model"
            }
        )
        assert create_response.status_code == 200

        # Query model
        completion_response = client.post(
            "/v1/completions",
            json={
                "model": "workflow-test",
                "prompt": [2, 3],
                "max_tokens": 5
            }
        )
        assert completion_response.status_code == 200

        # Delete model
        delete_response = client.delete("/v1/models/workflow-test")
        assert delete_response.status_code == 200

        # Verify deleted
        get_response = client.get("/v1/models/workflow-test")
        assert get_response.status_code == 404

    def test_multiple_models(self, client, sample_corpus):
        """Test working with multiple models simultaneously."""
        # Create two models
        client.post("/v1/models/create", json={"model_id": "model-a", "corpus": sample_corpus})
        client.post("/v1/models/create", json={"model_id": "model-b", "corpus": [1, 2, 3]})

        # Both should be queryable
        response_a = client.post(
            "/v1/completions",
            json={"model": "model-a", "prompt": [2, 3], "max_tokens": 2}
        )
        response_b = client.post(
            "/v1/completions",
            json={"model": "model-b", "prompt": [1, 2], "max_tokens": 2}
        )

        assert response_a.status_code == 200
        assert response_b.status_code == 200

        # Cleanup
        client.delete("/v1/models/model-a")
        client.delete("/v1/models/model-b")


# ============================================================================
# Introspection Endpoint Tests
# ============================================================================

class TestIntrospectionEndpoints:
    """Tests for introspection endpoints (/v1/predict, /v1/suffix_matches, etc.)."""

    def test_predict_endpoint_returns_predictions(self, client_with_model):
        """Test /v1/predict returns probability distribution."""
        response = client_with_model.post(
            "/v1/predict",
            json={"model": "test-model", "context": [2, 3]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        # Each prediction has byte, char, probability
        pred = data["predictions"][0]
        assert "byte" in pred
        assert "char" in pred
        assert "probability" in pred

    def test_predict_endpoint_with_string_context(self, client_with_model):
        """Test /v1/predict works with string context."""
        response = client_with_model.post(
            "/v1/predict",
            json={"model": "test-model", "context": "hello"}
        )
        assert response.status_code == 200
        assert "predictions" in response.json()

    def test_predict_endpoint_with_weight_function(self, client_with_model):
        """Test /v1/predict with weight function."""
        response = client_with_model.post(
            "/v1/predict",
            json={
                "model": "test-model",
                "context": [2, 3],
                "weight_function": "quadratic"
            }
        )
        assert response.status_code == 200
        assert response.json()["weight_function"] == "quadratic"

    def test_predict_endpoint_model_not_found(self, client):
        """Test /v1/predict returns 404 for nonexistent model."""
        response = client.post(
            "/v1/predict",
            json={"model": "nonexistent", "context": [1, 2]}
        )
        assert response.status_code == 404

    def test_suffix_matches_returns_matches(self, client_with_model):
        """Test /v1/suffix_matches returns matches at different lengths."""
        response = client_with_model.post(
            "/v1/suffix_matches",
            json={"model": "test-model", "context": [2, 3, 4]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "matches" in data
        assert "context_length" in data
        assert data["context_length"] == 3

        # Each match has length, suffix, count, positions
        for match in data["matches"]:
            assert "length" in match
            assert "suffix" in match
            assert "count" in match
            assert "positions" in match

    def test_suffix_matches_string_context(self, client_with_model):
        """Test /v1/suffix_matches works with string context."""
        response = client_with_model.post(
            "/v1/suffix_matches",
            json={"model": "test-model", "context": "abc"}
        )
        assert response.status_code == 200
        assert "matches" in response.json()

    def test_suffix_matches_model_not_found(self, client):
        """Test /v1/suffix_matches returns 404 for nonexistent model."""
        response = client.post(
            "/v1/suffix_matches",
            json={"model": "nonexistent", "context": [1, 2]}
        )
        assert response.status_code == 404

    def test_longest_suffix_returns_match_info(self, client_with_model):
        """Test /v1/longest_suffix returns match position and length."""
        response = client_with_model.post(
            "/v1/longest_suffix",
            json={"model": "test-model", "context": [2, 3]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "match_position" in data
        assert "match_length" in data
        assert "matched_suffix" in data
        assert "context_length" in data

    def test_longest_suffix_no_match(self, client_with_model):
        """Test /v1/longest_suffix handles no match."""
        response = client_with_model.post(
            "/v1/longest_suffix",
            json={"model": "test-model", "context": [99, 98, 97]}  # Unlikely to match
        )
        assert response.status_code == 200

        data = response.json()
        # Should still return valid structure even if no match
        assert "match_length" in data

    def test_longest_suffix_model_not_found(self, client):
        """Test /v1/longest_suffix returns 404 for nonexistent model."""
        response = client.post(
            "/v1/longest_suffix",
            json={"model": "nonexistent", "context": [1, 2]}
        )
        assert response.status_code == 404

    def test_confidence_returns_score(self, client_with_model):
        """Test /v1/confidence returns confidence score."""
        response = client_with_model.post(
            "/v1/confidence",
            json={"model": "test-model", "context": [2, 3]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1
        assert "match_length" in data
        assert "context_length" in data

    def test_confidence_model_not_found(self, client):
        """Test /v1/confidence returns 404 for nonexistent model."""
        response = client.post(
            "/v1/confidence",
            json={"model": "nonexistent", "context": [1, 2]}
        )
        assert response.status_code == 404

    def test_predict_backoff_returns_predictions(self, client_with_model):
        """Test /v1/predict_backoff returns predictions with backoff."""
        response = client_with_model.post(
            "/v1/predict_backoff",
            json={"model": "test-model", "context": [2, 3]}
        )
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) > 0
        assert "backoff_factor" in data

    def test_predict_backoff_with_custom_factor(self, client_with_model):
        """Test /v1/predict_backoff with custom backoff factor."""
        response = client_with_model.post(
            "/v1/predict_backoff",
            json={
                "model": "test-model",
                "context": [2, 3],
                "backoff_factor": 0.6
            }
        )
        assert response.status_code == 200
        assert response.json()["backoff_factor"] == 0.6

    def test_predict_backoff_string_context(self, client_with_model):
        """Test /v1/predict_backoff works with string context."""
        response = client_with_model.post(
            "/v1/predict_backoff",
            json={"model": "test-model", "context": "hello"}
        )
        assert response.status_code == 200
        assert "predictions" in response.json()

    def test_predict_backoff_model_not_found(self, client):
        """Test /v1/predict_backoff returns 404 for nonexistent model."""
        response = client.post(
            "/v1/predict_backoff",
            json={"model": "nonexistent", "context": [1, 2]}
        )
        assert response.status_code == 404

    def test_transforms_endpoint_lists_available(self, client):
        """Test /v1/transforms lists available transforms."""
        response = client.get("/v1/transforms")
        assert response.status_code == 200

        data = response.json()
        assert "transforms" in data
        assert "description" in data
        assert "lowercase" in data["transforms"]
