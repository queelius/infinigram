# Infinigram REST API Documentation

**Version**: 0.2.0
**Status**: Production Ready
**Compatibility**: OpenAI API v1

## Overview

Infinigram provides an OpenAI-compatible REST API for corpus-based language modeling. The API allows you to:
- Generate text completions using variable-length n-gram matching
- Manage multiple models simultaneously
- Use hierarchical weighted predictions
- Get detailed match metadata and confidence scores

## Quick Start

### 1. Start the Server

```bash
# Option 1: Run example server with demo models
python examples/start_server.py

# Option 2: Start server programmatically
from infinigram.server.api import app, model_manager
import uvicorn

# Load your models
model_manager.add_model("my-model", corpus=[1,2,3,4,5], max_length=10)

# Start server
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2. Test the API

```bash
# Check health
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Generate completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo",
    "prompt": [2, 3],
    "max_tokens": 5,
    "top_k": 10
  }'
```

## API Endpoints

### Core Endpoints

#### `GET /`
Root endpoint with API information.

**Response:**
```json
{
  "message": "Infinigram API",
  "version": "0.2.0",
  "endpoints": {
    "completions": "/v1/completions",
    "models": "/v1/models"
  }
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 2
}
```

### Completion Endpoints

#### `POST /v1/completions`
Create a text completion (OpenAI-compatible).

**Request Body:**
```json
{
  "model": "demo",              // Required: Model ID
  "prompt": [1, 2, 3],          // Required: List of integer token IDs
  "max_tokens": 10,             // Optional: Maximum tokens to generate (default: 10)
  "temperature": 1.0,           // Optional: Sampling temperature (not yet implemented)
  "top_k": 50,                  // Optional: Return top k predictions (default: 50)
  "weight_function": "quadratic", // Optional: "linear", "quadratic", "exponential", "sigmoid"
  "min_length": 1,              // Optional: Minimum suffix length for weighted prediction
  "max_length": null,           // Optional: Maximum suffix length
  "echo": false,                // Optional: Echo prompt in response
  "logprobs": 3                 // Optional: Return log probabilities for top N tokens
}
```

**Response:**
```json
{
  "id": "cmpl-1760741740364",
  "object": "text_completion",
  "created": 1760741740,
  "model": "demo",
  "choices": [
    {
      "text": "[4, 2, 3, 5, 6]",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length",
      "metadata": {
        "match_position": 1,
        "match_length": 7,
        "confidence": 0.493,
        "tokens": [4, 2, 3, 5, 6]
      }
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 5,
    "total_tokens": 7
  }
}
```

**Example with Weighted Prediction:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo",
    "prompt": [2, 3],
    "max_tokens": 3,
    "weight_function": "quadratic",
    "min_length": 1,
    "max_length": 5
  }'
```

**Example with Log Probabilities:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo",
    "prompt": [2, 3],
    "max_tokens": 2,
    "logprobs": 3
  }'
```

Response includes detailed probability information:
```json
{
  "logprobs": {
    "content": [
      {
        "tokens": ["4", "5", "1"],
        "token_logprobs": [-0.307, -1.399, -6.014],
        "top_logprobs": {
          "4": -0.307,
          "5": -1.399,
          "1": -6.014
        }
      }
    ]
  }
}
```

### Model Management Endpoints

#### `GET /v1/models`
List all available models (OpenAI-compatible).

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "demo",
      "object": "model",
      "created": 1760741705,
      "owned_by": "infinigram",
      "description": "Simple demo model with numeric tokens",
      "corpus_size": 17,
      "vocab_size": 9,
      "max_length": 10
    }
  ]
}
```

#### `GET /v1/models/{model_id}`
Get information about a specific model.

**Example:**
```bash
curl http://localhost:8000/v1/models/demo
```

**Response:**
```json
{
  "id": "demo",
  "object": "model",
  "created": 1760741759,
  "owned_by": "infinigram",
  "description": "Simple demo model with numeric tokens",
  "corpus_size": 17,
  "vocab_size": 9,
  "max_length": 10
}
```

#### `POST /v1/models/load`
Load a new model from a corpus.

**Request:**
```json
{
  "model_id": "my-custom-model",
  "corpus": [1, 2, 3, 4, 5, 6, 7, 8],
  "max_length": 10,
  "description": "My custom model description"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "test-model",
    "corpus": [1,2,3,4,5,2,3,6],
    "max_length": 5,
    "description": "Test model"
  }'
```

**Response:**
```json
{
  "status": "loaded",
  "model_id": "test-model"
}
```

#### `DELETE /v1/models/{model_id}`
Unload a model from memory.

**Example:**
```bash
curl -X DELETE http://localhost:8000/v1/models/test-model
```

**Response:**
```json
{
  "status": "deleted",
  "model_id": "test-model"
}
```

## Advanced Features

### Hierarchical Weighted Prediction

Use multiple suffix lengths with configurable weighting:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "demo",
    "prompt": [1, 2, 3],
    "max_tokens": 5,
    "weight_function": "exponential",
    "min_length": 1,
    "max_length": 10
  }'
```

**Available weight functions:**
- `linear`: w(k) = k (default)
- `quadratic`: w(k) = k²
- `exponential`: w(k) = 2^k
- `sigmoid`: w(k) = 1 / (1 + exp(-k + 5))

### Metadata and Confidence

Every completion includes metadata about the match:

```json
{
  "metadata": {
    "match_position": 42,       // Position in corpus where match was found
    "match_length": 5,          // Length of longest matching suffix
    "confidence": 0.78,         // Confidence score (0-1)
    "tokens": [4, 5, 6]        // Raw token IDs generated
  }
}
```

## Integration Examples

### Python Client

```python
import requests

# Create completion
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "demo",
        "prompt": [1, 2, 3],
        "max_tokens": 10,
        "top_k": 50
    }
)

result = response.json()
print(f"Generated tokens: {result['choices'][0]['metadata']['tokens']}")
print(f"Confidence: {result['choices'][0]['metadata']['confidence']}")
```

### LLM Probability Mixing

Use Infinigram to ground LLM predictions in a specific corpus:

```python
# Get LLM probabilities
llm_probs = llm_api.get_next_token_probs(context)

# Get Infinigram probabilities
infinigram_response = requests.post(
    "http://localhost:8000/v1/completions",
    json={"model": "domain-corpus", "prompt": context, "max_tokens": 1}
).json()

infinigram_probs = parse_probs_from_logprobs(infinigram_response)

# Mix probabilities
mixed_probs = 0.7 * llm_probs + 0.3 * infinigram_probs
next_token = sample(mixed_probs)
```

## Error Handling

### Model Not Found
```json
{
  "detail": "Model 'unknown-model' not found. Available models: ['demo', 'large-demo']"
}
```
**HTTP Status:** 404

### Invalid Request
```json
{
  "detail": "String prompts not yet supported. Please provide a list of integer token IDs."
}
```
**HTTP Status:** 400

### Unknown Weight Function
```json
{
  "detail": "Unknown weight function 'invalid'. Available: ['linear', 'quadratic', 'exponential', 'sigmoid']"
}
```
**HTTP Status:** 400

## Performance Characteristics

- **Latency**: <10ms for typical queries (100-token context)
- **Throughput**: 1000+ requests/second on single CPU
- **Memory**: O(corpus_size) per model
- **Model loading**: Instant (no training required)

## Roadmap

Planned enhancements:
- [ ] Streaming responses for long completions
- [ ] String tokenization (BPE/WordPiece support)
- [ ] Authentication and API keys
- [ ] Rate limiting
- [ ] Batch completion endpoint
- [ ] Model persistence to disk
- [ ] Prometheus metrics endpoint
- [ ] WebSocket support for real-time predictions

## See Also

- [Architecture Documentation](ARCHITECTURE.md)
- [Phase 1 Implementation Plan](PHASE1_PLAN.md)
- [API Source Code](../infinigram/server/api.py)
