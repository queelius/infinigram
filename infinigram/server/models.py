"""
Model management for Infinigram server.

Handles loading, caching, and serving multiple Infinigram models.
Uses the unified mmap-backed Infinigram interface.
"""

from typing import Dict, Optional, List, Union
from pathlib import Path
import json
from infinigram import Infinigram


# Default models directory
DEFAULT_MODELS_DIR = Path.home() / ".infinigram" / "models"


class ModelManager:
    """
    Manages multiple Infinigram models for the server.

    Models can be:
    - Loaded from ~/.infinigram/models/<name>/ by name
    - Loaded from absolute paths
    - Built from corpora on-the-fly (creates temp models)
    """

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the model manager."""
        self.models: Dict[str, Infinigram] = {}
        self.metadata: Dict[str, dict] = {}
        self.models_dir = models_dir or DEFAULT_MODELS_DIR

    def load_model(
        self,
        model_id: str,
        model_path: Optional[Union[str, Path]] = None,
        description: str = ""
    ) -> Infinigram:
        """
        Load a model by name or path.

        Args:
            model_id: Unique identifier for the model
            model_path: Path to model directory (optional, will look in default dir if not provided)
            description: Human-readable description

        Returns:
            Loaded Infinigram model
        """
        if model_id in self.models:
            return self.models[model_id]

        # Determine path
        if model_path:
            path = Path(model_path)
        else:
            # Try default models directory
            path = self.models_dir / model_id

        if not path.exists():
            raise ValueError(f"Model not found at {path}")

        # Load using unified interface
        model = Infinigram.load(str(path))

        self.models[model_id] = model
        self.metadata[model_id] = {
            "id": model_id,
            "description": description,
            "corpus_size": model.n,
            "vocab_size": model.vocab_size,
            "max_length": model.max_length,
            "path": str(path),
            "is_chunked": model._is_chunked,
        }

        return model

    def add_model(
        self,
        model_id: str,
        corpus: Union[str, bytes, List[int]],
        max_length: Optional[int] = None,
        description: str = "",
        persist: bool = False,
        **kwargs
    ) -> Infinigram:
        """
        Add a model from a corpus.

        Args:
            model_id: Unique identifier for the model
            corpus: Text string, bytes, or list of byte values
            max_length: Maximum suffix length
            description: Human-readable description
            persist: If True, save to default models directory
            **kwargs: Additional Infinigram parameters

        Returns:
            Created Infinigram model
        """
        if model_id in self.models:
            raise ValueError(f"Model '{model_id}' already exists")

        if persist:
            # Build persistent model
            model_path = self.models_dir / model_id
            model = Infinigram.build(
                corpus, str(model_path),
                max_length=max_length,
                verbose=False,
                **kwargs
            )
        else:
            # Create temporary model (mmap-backed but temp dir)
            model = Infinigram(corpus, max_length=max_length, **kwargs)

        self.models[model_id] = model
        self.metadata[model_id] = {
            "id": model_id,
            "description": description,
            "corpus_size": model.n,
            "vocab_size": model.vocab_size,
            "max_length": max_length,
            "is_chunked": model._is_chunked,
        }

        return model

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
            self.models[model_id].close()
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
        """Check if a model is loaded."""
        return model_id in self.models

    def list_available_models(self) -> List[str]:
        """
        List models available in the default models directory.

        Returns:
            List of model names that can be loaded
        """
        return Infinigram.list_models(self.models_dir)

    def auto_load_all(self) -> None:
        """Load all models from the default models directory."""
        for name in self.list_available_models():
            if name not in self.models:
                try:
                    self.load_model(name)
                except Exception as e:
                    print(f"Failed to load model '{name}': {e}")
