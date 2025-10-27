"""
FastAPI REST API for Infinigram.

Provides OpenAI-compatible endpoints for text completion and chat.
"""

from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
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
    weight_function: Optional[str] = None
    min_length: int = 1
    max_length: Optional[int] = None
    echo: bool = False
    logprobs: Optional[int] = None


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


# Initialize FastAPI app
app = FastAPI(
    title="Infinigram API",
    description="OpenAI-compatible REST API for Infinigram language models",
    version="0.2.0"
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Infinigram API",
        "version": "0.2.0",
        "endpoints": {
            "completions": "/v1/completions",
            "models": "/v1/models",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.list_models())
    }


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Create a text completion (OpenAI-compatible endpoint).

    This endpoint mimics OpenAI's /v1/completions API but uses Infinigram
    for predictions based on corpus patterns.

    Args:
        request: Completion request parameters

    Returns:
        CompletionResponse with predicted tokens

    Example curl:
        ```bash
        curl -X POST http://localhost:8000/v1/completions \\
          -H "Content-Type: application/json" \\
          -d '{
            "model": "demo",
            "prompt": [1, 2, 3],
            "max_tokens": 5,
            "top_k": 10
          }'
        ```
    """
    # Validate model exists
    if not model_manager.has_model(request.model):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model}' not found. Available models: {[m['id'] for m in model_manager.list_models()]}"
        )

    model = model_manager.get_model(request.model)

    # Convert prompt to token list if string
    if isinstance(request.prompt, str):
        raise HTTPException(
            status_code=400,
            detail="String prompts not yet supported. Please provide a list of integer token IDs."
        )

    context = request.prompt

    # Get weight function if specified
    weight_fn = None
    if request.weight_function:
        try:
            weight_fn = get_weight_function(request.weight_function)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Generate completion tokens
    completion_tokens = []
    current_context = list(context)
    logprobs_data = [] if request.logprobs else None

    for i in range(request.max_tokens):
        # Get predictions
        if weight_fn:
            probs = model.predict_weighted(
                current_context,
                min_length=request.min_length,
                max_length=request.max_length,
                weight_fn=weight_fn,
                top_k=request.top_k
            )
        else:
            probs = model.predict(current_context, top_k=request.top_k)

        if not probs:
            break  # No more predictions

        # Sample from distribution (temperature=1.0 for now, ignoring temperature parameter)
        # For deterministic behavior, just take the most likely token
        next_token = max(probs.items(), key=lambda x: x[1])[0]
        completion_tokens.append(next_token)

        # Store logprobs if requested
        if request.logprobs:
            import math
            top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:request.logprobs]
            logprobs_data.append({
                "tokens": [str(t) for t, _ in top_probs],
                "token_logprobs": [math.log(p) if p > 0 else float('-inf') for _, p in top_probs],
                "top_logprobs": {str(t): math.log(p) if p > 0 else float('-inf') for t, p in top_probs}
            })

        # Update context for next iteration
        current_context.append(next_token)

    # Build response
    completion_text = str(completion_tokens)  # For now, just stringify the tokens

    # Get match metadata
    pos, length = model.longest_suffix(current_context)
    confidence = model.confidence(current_context)

    choice = CompletionChoice(
        text=completion_text,
        index=0,
        logprobs={"content": logprobs_data} if logprobs_data else None,
        finish_reason="length" if len(completion_tokens) == request.max_tokens else "stop",
        metadata={
            "match_position": int(pos) if pos >= 0 else None,
            "match_length": int(length),
            "confidence": float(confidence),
            "tokens": completion_tokens
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
    """
    List available models (OpenAI-compatible endpoint).

    Returns:
        ModelList with metadata for all loaded models
    """
    models_data = []
    for meta in model_manager.list_models():
        model_info = ModelInfo(
            id=meta["id"],
            created=int(time.time()),
            description=meta.get("description", ""),
            corpus_size=meta["corpus_size"],
            vocab_size=meta["vocab_size"],
            max_length=meta["max_length"]
        )
        models_data.append(model_info)

    return ModelList(data=models_data)


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    """
    Get information about a specific model.

    Args:
        model_id: Model identifier

    Returns:
        Model metadata
    """
    if not model_manager.has_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    meta = next(m for m in model_manager.list_models() if m["id"] == model_id)

    return ModelInfo(
        id=meta["id"],
        created=int(time.time()),
        description=meta.get("description", ""),
        corpus_size=meta["corpus_size"],
        vocab_size=meta["vocab_size"],
        max_length=meta["max_length"]
    )


@app.post("/v1/models/load")
async def load_model(
    model_id: str,
    corpus: List[int],
    max_length: Optional[int] = None,
    description: str = ""
):
    """
    Load a new model from a corpus.

    Args:
        model_id: Unique identifier for the model
        corpus: List of integer tokens
        max_length: Maximum suffix length
        description: Human-readable description

    Returns:
        Model metadata
    """
    try:
        model_manager.add_model(
            model_id=model_id,
            corpus=corpus,
            max_length=max_length,
            description=description
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "loaded", "model_id": model_id}


@app.delete("/v1/models/{model_id}")
async def delete_model(model_id: str):
    """
    Unload a model from memory.

    Args:
        model_id: Model identifier

    Returns:
        Success message
    """
    if not model_manager.has_model(model_id):
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    model_manager.remove_model(model_id)
    return {"status": "deleted", "model_id": model_id}


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Start the Infinigram API server.

    Args:
        host: Host to bind to
        port: Port to listen on
        reload: Enable auto-reload for development
    """
    uvicorn.run(
        "infinigram.server.api:app",
        host=host,
        port=port,
        reload=reload
    )


if __name__ == "__main__":
    # For development
    start_server(reload=True)
