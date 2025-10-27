"""
Infinigram REST API Server

OpenAI-compatible REST API for Infinigram language models.
"""

from infinigram.server.api import app, start_server
from infinigram.server.models import ModelManager

__all__ = ["app", "start_server", "ModelManager"]
