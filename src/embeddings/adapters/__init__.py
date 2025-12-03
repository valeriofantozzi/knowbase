"""
Embedding Model Adapters Package

This package provides adapters for different embedding models,
enabling consistent interfaces for model-specific behaviors.
"""

from .base_adapter import ModelAdapter
from .bge_adapter import BGEAdapter
from .embeddinggemma_adapter import EmbeddingGemmaAdapter

__all__ = [
    "ModelAdapter",
    "BGEAdapter",
    "EmbeddingGemmaAdapter"
]
