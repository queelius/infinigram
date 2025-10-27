#!/usr/bin/env python3
"""
Example: Start Infinigram REST API server with a demo model.

This script loads a simple demo model and starts the server on http://localhost:8000

You can then test with curl:
    curl -X POST http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model": "demo", "prompt": [2, 3], "max_tokens": 5}'
"""

from infinigram.server.api import app, model_manager
import uvicorn


def load_demo_models():
    """Load demonstration models into the server."""

    # Demo model 1: Simple numeric sequence
    print("Loading demo models...")

    demo_corpus = [1, 2, 3, 4, 2, 3, 5, 6, 2, 3, 4, 7, 8, 9, 2, 3, 4]
    model_manager.add_model(
        model_id="demo",
        corpus=demo_corpus,
        max_length=10,
        description="Simple demo model with numeric tokens"
    )
    print(f"  ✓ Loaded 'demo' model ({len(demo_corpus)} tokens)")

    # Demo model 2: Larger corpus with more patterns
    large_corpus = list(range(100)) * 5  # Repeated patterns
    model_manager.add_model(
        model_id="large-demo",
        corpus=large_corpus,
        max_length=20,
        description="Larger demo model with 500 tokens"
    )
    print(f"  ✓ Loaded 'large-demo' model ({len(large_corpus)} tokens)")

    print("\nModels loaded successfully!")
    print("\nAvailable endpoints:")
    print("  - GET  http://localhost:8000/")
    print("  - GET  http://localhost:8000/health")
    print("  - POST http://localhost:8000/v1/completions")
    print("  - GET  http://localhost:8000/v1/models")
    print("  - POST http://localhost:8000/v1/models/load")
    print("\nExample curl command:")
    print('  curl -X POST http://localhost:8000/v1/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"model": "demo", "prompt": [2, 3], "max_tokens": 5, "top_k": 10}\'')
    print()


if __name__ == "__main__":
    load_demo_models()

    print("Starting server on http://localhost:8000")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
