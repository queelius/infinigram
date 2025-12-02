"""
FastAPI REST API for Infinigram.

Provides OpenAI-compatible endpoints for text completion and chat.
Supports both string prompts (UTF-8 encoded) and byte arrays.
"""

import math
import random
import time
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from infinigram.server.models import ModelManager
from infinigram.weighting import get_weight_function


# Global model manager
model_manager = ModelManager()


# Pydantic models for request/response validation
class CompletionRequest(BaseModel):
    """Request body for /v1/completions endpoint."""
    model: str
    prompt: Union[str, List[int]]
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: int = 50
    top_p: Optional[float] = None  # Nucleus sampling (not yet implemented)
    weight_function: Optional[str] = None
    min_length: int = 1
    max_length: Optional[int] = None
    echo: bool = False
    logprobs: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    smoothing: float = 0.0
    transforms: Optional[List[str]] = None  # Query transforms (e.g., ['lowercase'])
    search: Optional[List[str]] = None  # Beam search transforms (e.g., ['lowercase', 'strip'])
    search_max_depth: int = 2  # Max depth for beam search
    search_beam_width: int = 3  # Beam width for search


class CompletionChoice(BaseModel):
    """A single completion choice."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: str
    metadata: Optional[Dict[str, Any]] = None


class CompletionResponse(BaseModel):
    """Response body for /v1/completions endpoint."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    """Model metadata."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "infinigram"
    description: Optional[str] = None
    corpus_size: int
    vocab_size: int
    max_length: Optional[int] = None


class ModelList(BaseModel):
    """List of models."""
    object: str = "list"
    data: List[ModelInfo]


class LoadModelRequest(BaseModel):
    """Request to load a model."""
    model_id: str
    model_path: Optional[str] = None
    description: str = ""


class CreateModelRequest(BaseModel):
    """Request to create a model from corpus."""
    model_id: str
    corpus: Union[str, List[int]]
    max_length: Optional[int] = None
    description: str = ""
    persist: bool = False


class ContextRequest(BaseModel):
    """Request with context for introspection endpoints."""
    model: str
    context: Union[str, List[int]]
    transforms: Optional[List[str]] = None


class PredictRequest(ContextRequest):
    """Request for /v1/predict endpoint."""
    top_k: int = 50
    smoothing: float = 0.0
    weight_function: Optional[str] = None
    min_length: int = 1
    max_length: Optional[int] = None


class PredictBackoffRequest(ContextRequest):
    """Request for /v1/predict_backoff endpoint."""
    top_k: int = 50
    backoff_factor: float = 0.4
    min_count_threshold: int = 1
    smoothing: float = 0.0


# Initialize FastAPI app
app = FastAPI(
    title="Infinigram API",
    description="OpenAI-compatible REST API for Infinigram corpus-based language models",
    version="0.4.0"
)


def sample_from_distribution(probs: Dict[int, float], temperature: float = 1.0) -> int:
    """
    Sample a token from a probability distribution with temperature.

    Args:
        probs: Dict mapping token -> probability
        temperature: Temperature for sampling (0 = greedy, 1 = standard, >1 = more random)

    Returns:
        Sampled token
    """
    if not probs:
        return 0

    if temperature == 0:
        # Greedy decoding
        return max(probs.items(), key=lambda x: x[1])[0]

    if temperature != 1.0:
        # Apply temperature
        tokens = list(probs.keys())
        log_probs = [math.log(p + 1e-10) / temperature for p in probs.values()]
        max_log = max(log_probs)
        exp_probs = [math.exp(lp - max_log) for lp in log_probs]
        total = sum(exp_probs)
        probs = {t: p / total for t, p in zip(tokens, exp_probs)}

    # Sample from distribution
    r = random.random()
    cumulative = 0.0
    for token, prob in probs.items():
        cumulative += prob
        if r < cumulative:
            return token

    # Fallback
    return list(probs.keys())[-1]


def decode_bytes_to_text(byte_list: List[int]) -> str:
    """Decode a list of byte values to UTF-8 text, replacing invalid sequences."""
    return bytes(byte_list).decode('utf-8', errors='replace')


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Infinigram API",
        "version": "0.4.0",
        "description": "Corpus-based language model with OpenAI-compatible API",
        "endpoints": {
            "completions": "/v1/completions",
            "models": "/v1/models",
            "predict": "/v1/predict",
            "predict_backoff": "/v1/predict_backoff",
            "suffix_matches": "/v1/suffix_matches",
            "longest_suffix": "/v1/longest_suffix",
            "confidence": "/v1/confidence",
            "count": "/v1/count",
            "search": "/v1/search",
            "transforms": "/v1/transforms",
            "health": "/health"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.list_models()),
        "available_models": model_manager.list_available_models()
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Create a text completion (OpenAI-compatible endpoint).

    Supports both string prompts (UTF-8 encoded to bytes) and byte arrays.
    Returns decoded UTF-8 text in the response.

    Args:
        request: Completion request parameters

    Returns:
        CompletionResponse with generated text

    Example:
        ```bash
        curl -X POST http://localhost:8000/v1/completions \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "wikipedia",
            "prompt": "Albert Einstein was ",
            "max_tokens": 20,
            "temperature": 0.7
          }'
        ```
    """
    # Validate model exists
    if not model_manager.has_model(request.model):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available: {[m['id'] for m in model_manager.list_models()]}"
        )

    model = model_manager.get_model(request.model)

    # Convert prompt to bytes
    if isinstance(request.prompt, str):
        context = list(request.prompt.encode('utf-8'))
    else:
        context = list(request.prompt)

    # Get weight function if specified
    weight_fn = None
    if request.weight_function:
        try:
            weight_fn = get_weight_function(request.weight_function)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Parse stop sequences
    stop_bytes = []
    if request.stop:
        if isinstance(request.stop, str):
            stop_bytes = [list(request.stop.encode('utf-8'))]
        else:
            stop_bytes = [list(s.encode('utf-8')) for s in request.stop]

    # Generate completion tokens
    completion_tokens = []
    current_context = list(context)
    logprobs_data = [] if request.logprobs else None

    # Get transforms (use empty list if None to skip model defaults during generation)
    transforms = request.transforms if request.transforms is not None else []

    for i in range(request.max_tokens):
        # Get predictions - choose method based on parameters
        if request.search:
            # Beam search over transform space
            probs = model.predict_search(
                current_context,
                search=request.search,
                max_depth=request.search_max_depth,
                beam_width=request.search_beam_width,
                top_k=request.top_k,
                smoothing=request.smoothing
            )
        elif weight_fn:
            probs = model.predict_weighted(
                current_context,
                min_length=request.min_length,
                max_length=request.max_length,
                weight_fn=weight_fn,
                top_k=request.top_k,
                smoothing=request.smoothing,
                transforms=transforms
            )
        else:
            probs = model.predict(
                current_context,
                top_k=request.top_k,
                smoothing=request.smoothing,
                transforms=transforms
            )

        if not probs:
            break

        # Sample next token
        next_token = sample_from_distribution(probs, request.temperature)
        completion_tokens.append(next_token)

        # Store logprobs if requested
        if request.logprobs:
            top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:request.logprobs]
            logprobs_data.append({
                "tokens": [chr(t) if 32 <= t < 127 else f"<{t}>" for t, _ in top_probs],
                "token_logprobs": [math.log(p) if p > 0 else float('-inf') for _, p in top_probs],
                "top_logprobs": {
                    chr(t) if 32 <= t < 127 else f"<{t}>": math.log(p) if p > 0 else float('-inf')
                    for t, p in top_probs
                }
            })

        # Update context
        current_context.append(next_token)

        # Check for stop sequences
        if stop_bytes:
            for stop_seq in stop_bytes:
                if completion_tokens[-len(stop_seq):] == stop_seq:
                    # Remove stop sequence from output
                    completion_tokens = completion_tokens[:-len(stop_seq)]
                    break
            else:
                continue
            break

    # Decode completion to text
    completion_text = decode_bytes_to_text(completion_tokens)

    # Echo prompt if requested
    if request.echo:
        prompt_text = request.prompt if isinstance(request.prompt, str) else decode_bytes_to_text(request.prompt)
        completion_text = prompt_text + completion_text

    # Get match metadata
    pos, length = model.longest_suffix(current_context)
    confidence = model.confidence(current_context)

    # Handle chunked model position
    if isinstance(pos, tuple):
        pos_info = {"chunk": pos[0], "position": pos[1]}
    else:
        pos_info = pos

    finish_reason = "length" if len(completion_tokens) == request.max_tokens else "stop"

    choice = CompletionChoice(
        text=completion_text,
        index=0,
        logprobs={"content": logprobs_data} if logprobs_data else None,
        finish_reason=finish_reason,
        metadata={
            "match_position": pos_info if pos != -1 else None,
            "match_length": int(length),
            "confidence": float(confidence),
            "token_bytes": completion_tokens
        }
    )

    response = CompletionResponse(
        id=f"cmpl-{int(time.time() * 1000)}",
        created=int(time.time()),
        model=request.model,
        choices=[choice],
        usage={
            "prompt_tokens": len(context),
            "completion_tokens": len(completion_tokens),
            "total_tokens": len(context) + len(completion_tokens)
        }
    )

    return response


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List all loaded models."""
    models_data = []
    for meta in model_manager.list_models():
        models_data.append(ModelInfo(
            id=meta["id"],
            created=int(time.time()),
            description=meta.get("description", ""),
            corpus_size=meta["corpus_size"],
            vocab_size=meta["vocab_size"],
            max_length=meta.get("max_length")
        ))
    return ModelList(data=models_data)


@app.get("/v1/models/available")
async def list_available_models():
    """List models available in the default models directory."""
    return {
        "available": model_manager.list_available_models(),
        "loaded": [m["id"] for m in model_manager.list_models()]
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """Get information about a specific model."""
    if not model_manager.has_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    meta = next(m for m in model_manager.list_models() if m["id"] == model_id)

    return ModelInfo(
        id=meta["id"],
        created=int(time.time()),
        description=meta.get("description", ""),
        corpus_size=meta["corpus_size"],
        vocab_size=meta["vocab_size"],
        max_length=meta.get("max_length")
    )


@app.post("/v1/models/load")
async def load_model(request: LoadModelRequest):
    """
    Load a model from disk.

    Models are loaded from ~/.infinigram/models/<model_id>/ by default,
    or from a custom path if model_path is specified.
    """
    try:
        model_manager.load_model(
            request.model_id,
            model_path=request.model_path,
            description=request.description
        )
        return {"status": "loaded", "model_id": request.model_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/models/create")
async def create_model(request: CreateModelRequest):
    """
    Create a new model from corpus.

    If persist=True, the model is saved to ~/.infinigram/models/<model_id>/
    """
    try:
        model_manager.add_model(
            request.model_id,
            request.corpus,
            max_length=request.max_length,
            description=request.description,
            persist=request.persist
        )
        return {
            "status": "created",
            "model_id": request.model_id,
            "persisted": request.persist
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/v1/models/{model_id}")
async def delete_model(model_id: str):
    """Remove a model from memory (does not delete from disk)."""
    if not model_manager.has_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    model_manager.remove_model(model_id)
    return {"status": "removed", "model_id": model_id}


@app.get("/v1/transforms")
async def list_transforms():
    """List available query transforms."""
    from infinigram import Infinigram
    return {
        "transforms": Infinigram.list_transforms(),
        "description": {
            "lowercase": "Convert query to lowercase",
            "uppercase": "Convert query to uppercase",
            "casefold": "Unicode case folding (more aggressive than lowercase)",
            "strip": "Remove leading/trailing whitespace",
            "normalize_whitespace": "Collapse multiple spaces to single space",
        }
    }


@app.post("/v1/count")
async def count_pattern(
    model: str,
    pattern: str,
    transforms: Optional[List[str]] = Query(default=None)
):
    """Count occurrences of a pattern in the corpus."""
    if not model_manager.has_model(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    m = model_manager.get_model(model)
    # Use empty list if None to skip model defaults
    count = m.count(pattern, transforms=transforms if transforms else [])
    return {"model": model, "pattern": pattern, "count": count, "transforms": transforms}


@app.post("/v1/search")
async def search_pattern(
    model: str,
    pattern: str,
    limit: int = Query(default=10, le=1000),
    transforms: Optional[List[str]] = Query(default=None)
):
    """Search for pattern occurrences in the corpus."""
    if not model_manager.has_model(model):
        raise HTTPException(status_code=404, detail=f"Model '{model}' not found")

    m = model_manager.get_model(model)
    # Use empty list if None to skip model defaults
    results = m.search(pattern, transforms=transforms if transforms else [])[:limit]

    return {
        "model": model,
        "pattern": pattern,
        "count": len(results),
        "positions": results,
        "transforms": transforms
    }


# ============================================================================
# Introspection Endpoints
# ============================================================================

@app.post("/v1/suffix_matches")
async def get_suffix_matches(request: ContextRequest):
    """
    Find all matching suffixes at different lengths.

    For context "abc", searches for "abc", "bc", "c" and returns
    all matches with their corpus positions.

    Returns:
        Dict with suffix matches at each length, sorted by decreasing length.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/suffix_matches" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "wikipedia", "context": "the cat"}'
        ```
    """
    if not model_manager.has_model(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    m = model_manager.get_model(request.model)

    # Convert context to bytes
    if isinstance(request.context, str):
        context_bytes = list(request.context.encode('utf-8'))
    else:
        context_bytes = list(request.context)

    # Get suffix matches
    matches = m.find_all_suffix_matches(
        context_bytes,
        transforms=request.transforms if request.transforms else []
    )

    # Format response
    return {
        "model": request.model,
        "context": request.context,
        "context_length": len(context_bytes),
        "matches": [
            {
                "length": length,
                "suffix": bytes(context_bytes[-length:]).decode('utf-8', errors='replace') if length > 0 else "",
                "count": len(positions),
                "positions": positions[:100]  # Limit positions returned
            }
            for length, positions in matches
        ],
        "transforms": request.transforms
    }


@app.post("/v1/predict")
async def get_predictions(request: PredictRequest):
    """
    Get next-byte predictions for a context.

    Returns probability distribution over possible next bytes.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/predict" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "wikipedia", "context": "the cat"}'
        ```
    """
    if not model_manager.has_model(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    m = model_manager.get_model(request.model)

    # Convert context to bytes
    if isinstance(request.context, str):
        context_bytes = list(request.context.encode('utf-8'))
    else:
        context_bytes = list(request.context)

    # Get predictions
    if request.weight_function:
        try:
            weight_fn = get_weight_function(request.weight_function)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        probs = m.predict_weighted(
            context_bytes,
            min_length=request.min_length,
            max_length=request.max_length,
            weight_fn=weight_fn,
            top_k=request.top_k,
            smoothing=request.smoothing,
            transforms=request.transforms if request.transforms else []
        )
    else:
        probs = m.predict(
            context_bytes,
            top_k=request.top_k,
            smoothing=request.smoothing,
            transforms=request.transforms if request.transforms else []
        )

    # Format probabilities with readable byte representations
    predictions = []
    for byte_val, prob in probs.items():
        if 32 <= byte_val < 127:
            display = chr(byte_val)
        else:
            display = f"<0x{byte_val:02x}>"
        predictions.append({
            "byte": byte_val,
            "char": display,
            "probability": prob
        })

    return {
        "model": request.model,
        "context": request.context,
        "predictions": predictions,
        "transforms": request.transforms,
        "weight_function": request.weight_function
    }


@app.post("/v1/longest_suffix")
async def get_longest_suffix(request: ContextRequest):
    """
    Find the longest suffix of context that matches in the corpus.

    Returns the position and length of the longest match.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/longest_suffix" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "wikipedia", "context": "the cat sat on"}'
        ```
    """
    if not model_manager.has_model(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    m = model_manager.get_model(request.model)

    # Convert context to bytes
    if isinstance(request.context, str):
        context_bytes = list(request.context.encode('utf-8'))
    else:
        context_bytes = list(request.context)

    pos, length = m.longest_suffix(
        context_bytes,
        transforms=request.transforms if request.transforms else []
    )

    # Handle chunked model position
    if isinstance(pos, tuple):
        pos_info = {"chunk": pos[0], "position": pos[1]}
    else:
        pos_info = pos if pos != -1 else None

    return {
        "model": request.model,
        "context": request.context,
        "context_length": len(context_bytes),
        "match_position": pos_info,
        "match_length": int(length),
        "matched_suffix": bytes(context_bytes[-length:]).decode('utf-8', errors='replace') if length > 0 else "",
        "transforms": request.transforms
    }


@app.post("/v1/confidence")
async def get_confidence(request: ContextRequest):
    """
    Get confidence score for a context.

    Returns a 0-1 score based on match quality.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/confidence" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "wikipedia", "context": "the cat sat on"}'
        ```
    """
    if not model_manager.has_model(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    m = model_manager.get_model(request.model)

    # Convert context to bytes
    if isinstance(request.context, str):
        context_bytes = list(request.context.encode('utf-8'))
    else:
        context_bytes = list(request.context)

    confidence = m.confidence(
        context_bytes,
        transforms=request.transforms if request.transforms else []
    )

    pos, length = m.longest_suffix(
        context_bytes,
        transforms=request.transforms if request.transforms else []
    )

    return {
        "model": request.model,
        "context": request.context,
        "confidence": float(confidence),
        "match_length": int(length),
        "context_length": len(context_bytes),
        "transforms": request.transforms
    }


@app.post("/v1/predict_backoff")
async def get_predictions_backoff(request: PredictBackoffRequest):
    """
    Get next-byte predictions using Stupid Backoff smoothing.

    Uses longest matching suffix if it has enough counts, otherwise
    backs off to shorter suffixes with a penalty factor.

    Args:
        model: Model ID
        context: Context string or byte array
        top_k: Number of top predictions to return
        backoff_factor: Penalty factor for backing off (default 0.4)
        min_count_threshold: Minimum count before backing off
        smoothing: Laplace smoothing parameter
        transforms: Query transforms to apply

    Example:
        ```bash
        curl -X POST "http://localhost:8000/v1/predict_backoff" \\
          -H "Content-Type: application/json" \\
          -d '{"model": "wikipedia", "context": "the cat", "backoff_factor": 0.4}'
        ```
    """
    if not model_manager.has_model(request.model):
        raise HTTPException(status_code=404, detail=f"Model '{request.model}' not found")

    m = model_manager.get_model(request.model)

    # Convert context to bytes
    if isinstance(request.context, str):
        context_bytes = request.context.encode('utf-8')
    else:
        context_bytes = bytes(request.context)

    probs = m.predict_backoff(
        context_bytes,
        backoff_factor=request.backoff_factor,
        min_count_threshold=request.min_count_threshold,
        top_k=request.top_k,
        smoothing=request.smoothing,
        transforms=request.transforms if request.transforms else []
    )

    # Format probabilities with readable byte representations
    predictions = []
    for byte_val, prob in probs.items():
        if 32 <= byte_val < 127:
            display = chr(byte_val)
        else:
            display = f"<0x{byte_val:02x}>"
        predictions.append({
            "byte": byte_val,
            "char": display,
            "probability": prob
        })

    return {
        "model": request.model,
        "context": request.context if isinstance(request.context, str) else list(request.context),
        "predictions": predictions,
        "backoff_factor": request.backoff_factor,
        "min_count_threshold": request.min_count_threshold,
        "transforms": request.transforms
    }


def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    models_dir: Optional[str] = None,
    auto_load: bool = True
):
    """
    Start the Infinigram API server.

    Args:
        host: Host to bind to
        port: Port to bind to
        models_dir: Directory containing models (default: ~/.infinigram/models)
        auto_load: Whether to auto-load all models from models_dir
    """
    global model_manager

    if models_dir:
        model_manager = ModelManager(Path(models_dir))

    if auto_load:
        print(f"Auto-loading models from {model_manager.models_dir}...")
        model_manager.auto_load_all()
        print(f"Loaded {len(model_manager.list_models())} models")

    uvicorn.run(app, host=host, port=port)


def main():
    """Entry point for infinigram-serve command."""
    import argparse

    parser = argparse.ArgumentParser(description="Infinigram API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", help="Models directory")
    parser.add_argument("--no-auto-load", action="store_true", help="Don't auto-load models")

    args = parser.parse_args()

    start_server(
        host=args.host,
        port=args.port,
        models_dir=args.models_dir,
        auto_load=not args.no_auto_load
    )


if __name__ == "__main__":
    main()
