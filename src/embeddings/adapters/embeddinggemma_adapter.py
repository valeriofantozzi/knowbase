"""
EmbeddingGemma Model Adapter

Adapter implementation for Google EmbeddingGemma models.
Handles structured prompts, MRL (Matryoshka Representation Learning) dimensions,
and precision requirements specific to EmbeddingGemma models.
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from .base_adapter import ModelAdapter


class EmbeddingGemmaAdapter(ModelAdapter):
    """
    Adapter for Google EmbeddingGemma models.

    EmbeddingGemma uses structured prompts and supports MRL (Matryoshka
    Representation Learning) for different embedding dimensions.
    """

    # EmbeddingGemma structured prompts
    QUERY_PROMPT_TEMPLATE = "task: search result | query: {content}"
    DOCUMENT_PROMPT_TEMPLATE = "title: {title} | text: {content}"

    # MRL supported dimensions (sorted descending for truncation)
    MRL_DIMENSIONS = [768, 512, 256, 128]

    def __init__(
        self,
        model: SentenceTransformer,
        target_dimension: Optional[int] = None
    ):
        """
        Initialize EmbeddingGemma adapter.

        Args:
            model: EmbeddingGemma SentenceTransformer model instance
            target_dimension: Target embedding dimension (768, 512, 256, 128)
                            Defaults to 768 (full dimension)
        """
        super().__init__(model)

        if target_dimension is None:
            target_dimension = 768  # Default to full dimension

        if target_dimension not in self.MRL_DIMENSIONS:
            raise ValueError(
                f"Invalid target dimension {target_dimension}. "
                f"Supported dimensions: {self.MRL_DIMENSIONS}"
            )

        self.target_dimension = target_dimension

    def format_query_prompt(self, text: str) -> str:
        """
        Format text for EmbeddingGemma query embedding generation.

        Args:
            text: Raw query text

        Returns:
            Structured query prompt
        """
        return self.QUERY_PROMPT_TEMPLATE.format(content=text)

    def format_document_prompt(self, text: str, title: Optional[str] = None) -> str:
        """
        Format text for EmbeddingGemma document embedding generation.

        Args:
            text: Raw document text
            title: Optional document title (defaults to 'none')

        Returns:
            Structured document prompt
        """
        title = title or "none"
        return self.DOCUMENT_PROMPT_TEMPLATE.format(title=title, content=text)

    def get_embedding_dimension(self) -> int:
        """
        Get EmbeddingGemma embedding dimension.

        Returns:
            Configured target dimension (768, 512, 256, or 128)
        """
        return self.target_dimension

    def get_max_sequence_length(self) -> int:
        """
        Get EmbeddingGemma maximum sequence length.

        Returns:
            2048 tokens for EmbeddingGemma models
        """
        return 2048

    def get_precision_requirements(self) -> List[str]:
        """
        Get supported precision types for EmbeddingGemma models.

        EmbeddingGemma requires float32 or bfloat16, float16 is not supported.

        Returns:
            List of supported precision strings
        """
        return ["float32", "bfloat16"]

    def validate_precision(self, precision: str) -> bool:
        """
        Validate if precision is supported by EmbeddingGemma models.

        Args:
            precision: Precision string to validate

        Returns:
            True if precision is supported
        """
        return precision in self.get_precision_requirements()

    def encode_query(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode query texts with MRL dimension truncation.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings (truncated to target dimension)
        """
        embeddings = super().encode_query(
            texts=texts,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings
        )

        # Apply MRL dimension truncation if needed
        if self.target_dimension < 768:
            embeddings = embeddings[:, :self.target_dimension]

        return embeddings

    def encode_document(
        self,
        texts: Union[str, List[str]],
        titles: Optional[List[Optional[str]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode document texts with MRL dimension truncation.

        Args:
            texts: Single text or list of texts
            titles: Optional list of titles corresponding to texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings (truncated to target dimension)
        """
        embeddings = super().encode_document(
            texts=texts,
            titles=titles,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings
        )

        # Apply MRL dimension truncation if needed
        if self.target_dimension < 768:
            embeddings = embeddings[:, :self.target_dimension]

        return embeddings

    def set_target_dimension(self, dimension: int) -> None:
        """
        Change the target embedding dimension.

        Args:
            dimension: New target dimension (768, 512, 256, or 128)

        Raises:
            ValueError: If dimension is not supported
        """
        if dimension not in self.MRL_DIMENSIONS:
            raise ValueError(
                f"Invalid target dimension {dimension}. "
                f"Supported dimensions: {self.MRL_DIMENSIONS}"
            )
        self.target_dimension = dimension
