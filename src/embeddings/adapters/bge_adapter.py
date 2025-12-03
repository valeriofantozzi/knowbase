"""
BGE Model Adapter

Adapter implementation for BGE (BAAI General Embedding) models.
Handles BGE-specific prompt formatting and encoding behaviors.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer

from .base_adapter import ModelAdapter


class BGEAdapter(ModelAdapter):
    """
    Adapter for BGE (BAAI General Embedding) models.

    BGE models use simple instruction prefixes:
    - Query: "Represent this sentence for searching relevant passages: "
    - Document: Empty prefix (raw text)
    """

    # BGE-specific instruction prefixes
    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
    DOCUMENT_INSTRUCTION = ""

    def __init__(self, model: SentenceTransformer):
        """
        Initialize BGE adapter.

        Args:
            model: BGE SentenceTransformer model instance
        """
        super().__init__(model)

    def format_query_prompt(self, text: str) -> str:
        """
        Format text for BGE query embedding generation.

        Args:
            text: Raw query text

        Returns:
            Text with BGE query instruction prefix
        """
        return self.QUERY_INSTRUCTION + text

    def format_document_prompt(self, text: str, title: Optional[str] = None) -> str:
        """
        Format text for BGE document embedding generation.

        BGE models don't use document prefixes, so raw text is returned.

        Args:
            text: Raw document text
            title: Optional document title (ignored for BGE)

        Returns:
            Raw document text (no prefix)
        """
        return text

    def get_embedding_dimension(self) -> int:
        """
        Get BGE embedding dimension.

        Returns:
            1024 for standard BGE-large-en-v1.5
        """
        # BGE-large-en-v1.5 has 1024 dimensions
        # Could be made dynamic by checking model config
        return 1024

    def get_max_sequence_length(self) -> int:
        """
        Get BGE maximum sequence length.

        Returns:
            512 tokens for standard BGE models
        """
        return 512

    def get_precision_requirements(self) -> List[str]:
        """
        Get supported precision types for BGE models.

        Returns:
            List of supported precision strings
        """
        return ["float32", "float16"]

    def validate_precision(self, precision: str) -> bool:
        """
        Validate if precision is supported by BGE models.

        Args:
            precision: Precision string to validate

        Returns:
            True if precision is supported
        """
        return precision in self.get_precision_requirements()
