"""
Model management for Infinigram server.

Handles loading, caching, and serving multiple Infinigram models.
"""

from typing import Dict, Optional, List
from pathlib import Path
import pickle
import json
from infinigram import Infinigram


class ModelManager:
    """
    Manages multiple Infinigram models for the server.

    Supports:
    - Loading models from files or creating from corpora
    - Model caching
    - Model metadata
    """

    def __init__(self):
        """Initialize the model manager."""
        self.models: Dict[str, Infinigram] = {}
        self.metadata: Dict[str, dict] = {}

    def add_model(
        self,
        model_id: str,
        corpus: List[int],
        max_length: Optional[int] = None,
        description: str = "",
        **kwargs
    ) -> None:
        """
        Add a model from a corpus.

        Args:
            model_id: Unique identifier for the model
            corpus: Token sequence
            max_length: Maximum suffix length
            description: Human-readable description
            **kwargs: Additional Infinigram parameters
        """
        if model_id in self.models:
            raise ValueError(f"Model '{model_id}' already exists")

        model = Infinigram(corpus, max_length=max_length, **kwargs)
        self.models[model_id] = model
        self.metadata[model_id] = {
            "id": model_id,
            "description": description,
            "corpus_size": len(corpus),
            "vocab_size": model.vocab_size,
            "max_length": max_length,
        }

    def load_model(
        self,
        model_id: str,
        model_path: Path,
        description: str = ""
    ) -> None:
        """
        Load a saved model from disk.

        Args:
            model_id: Unique identifier
            model_path: Path to saved model
            description: Human-readable description
        """
        if model_id in self.models:
            raise ValueError(f"Model '{model_id}' already exists")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        self.models[model_id] = model
        self.metadata[model_id] = {
            "id": model_id,
            "description": description,
            "corpus_size": model.n,
            "vocab_size": model.vocab_size,
            "max_length": model.max_length,
        }

    def save_model(self, model_id: str, model_path: Path) -> None:
        """
        Save a model to disk.

        Args:
            model_id: Model identifier
            model_path: Path to save to
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")

        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_id], f)

    def get_model(self, model_id: str) -> Infinigram:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            Infinigram model

        Raises:
            ValueError: If model not found
        """
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found")
        return self.models[model_id]

    def remove_model(self, model_id: str) -> None:
        """
        Remove a model from memory.

        Args:
            model_id: Model identifier
        """
        if model_id in self.models:
            del self.models[model_id]
            del self.metadata[model_id]

    def list_models(self) -> List[dict]:
        """
        List all loaded models.

        Returns:
            List of model metadata dicts
        """
        return list(self.metadata.values())

    def has_model(self, model_id: str) -> bool:
        """Check if a model exists."""
        return model_id in self.models
