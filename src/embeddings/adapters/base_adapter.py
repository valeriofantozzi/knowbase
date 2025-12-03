"""
Base Adapter Module

Defines the abstract base class for all embedding model adapters.
Provides a common interface for different embedding models with their
specific behaviors (prompt formatting, precision requirements, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class ModelAdapter(ABC):
    """
    Abstract base class for embedding model adapters.

    This class defines the interface that all model adapters must implement.
    It provides methods for model-specific behaviors like prompt formatting,
    precision validation, and encoding strategies.
    """

    def __init__(self, model: SentenceTransformer):
        """
        Initialize adapter with model instance.

        Args:
            model: SentenceTransformer model instance
        """
        self.model = model

    @abstractmethod
    def format_query_prompt(self, text: str) -> str:
        """
        Format text for query embedding generation.

        Args:
            text: Raw query text

        Returns:
            Formatted text with appropriate instructions/prompts
        """
        pass

    @abstractmethod
    def format_document_prompt(self, text: str, title: Optional[str] = None) -> str:
        """
        Format text for document embedding generation.

        Args:
            text: Raw document text
            title: Optional document title (some models use title information)

        Returns:
            Formatted text with appropriate instructions/prompts
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for this model.

        Returns:
            Embedding dimension (e.g., 768 for EmbeddingGemma, 1024 for BGE)
        """
        pass

    @abstractmethod
    def get_max_sequence_length(self) -> int:
        """
        Get the maximum sequence length for this model.

        Returns:
            Maximum sequence length in tokens
        """
        pass

    @abstractmethod
    def get_precision_requirements(self) -> List[str]:
        """
        Get supported precision types for this model.

        Returns:
            List of supported precision strings (e.g., ["float32", "float16"])
        """
        pass

    @abstractmethod
    def validate_precision(self, precision: str) -> bool:
        """
        Validate if the given precision is supported by this model.

        Args:
            precision: Precision string to validate

        Returns:
            True if precision is supported, False otherwise
        """
        pass

    def encode_query(
        self,
        texts: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode query texts using model-specific formatting.

        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        formatted_texts = [self.format_query_prompt(text) for text in texts]

        try:
            return self.model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings
            )
        except Exception as e:
            raise self._handle_precision_error(e, "query encoding") from e

    def encode_document(
        self,
        texts: Union[str, List[str]],
        titles: Optional[List[Optional[str]]] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode document texts using model-specific formatting.

        Args:
            texts: Single text or list of texts
            titles: Optional list of titles corresponding to texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize_embeddings: Whether to normalize embeddings

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            titles = [titles] if titles is not None else None

        if titles is None:
            titles = [None] * len(texts)
        elif len(titles) != len(texts):
            raise ValueError("titles list must have same length as texts list")

        formatted_texts = [
            self.format_document_prompt(text, title)
            for text, title in zip(texts, titles)
        ]

        try:
            return self.model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings
            )
        except Exception as e:
            raise self._handle_precision_error(e, "document encoding") from e

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings to unit vectors.

        Args:
            embeddings: Embedding array

        Returns:
            Normalized embeddings
        """
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        return embeddings / norms

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Validate generated embeddings.

        Args:
            embeddings: Embedding array

        Raises:
            ValueError: If validation fails
        """
        if embeddings is None:
            raise ValueError("Embeddings are None")

        if len(embeddings) == 0:
            raise ValueError("Empty embeddings array")

        # Check for NaN or Inf values
        if not np.isfinite(embeddings).all():
            nan_count = np.isnan(embeddings).sum()
            inf_count = np.isinf(embeddings).sum()
            raise ValueError(
                f"Embeddings contain invalid values: {nan_count} NaN, {inf_count} Inf. "
                "This may indicate precision compatibility issues. "
                "Try using float32 precision or check device compatibility."
            )

        # Check embedding dimension
        expected_dim = self.get_embedding_dimension()
        if embeddings.shape[-1] != expected_dim:
            raise ValueError(
                f"Wrong embedding dimension: expected {expected_dim}, "
                f"got {embeddings.shape[-1]}. "
                "This suggests the model generated embeddings with unexpected dimensions."
            )

    def _handle_precision_error(self, error: Exception, operation: str) -> Exception:
        """
        Handle precision-related errors during encoding operations.

        Args:
            error: The original exception
            operation: Description of the operation being performed

        Returns:
            Enhanced exception with helpful error message
        """
        error_msg = str(error).lower()

        # Check for common precision-related error patterns
        if any(keyword in error_msg for keyword in ["bfloat16", "bf16", "precision", "dtype"]):
            enhanced_msg = (
                f"Precision error during {operation}: {str(error)}\n\n"
                "This model requires specific precision support:\n"
                f"- Required precisions: {self.get_precision_requirements()}\n"
                "- For EmbeddingGemma: requires bfloat16 support (not available on all GPUs)\n"
                "- For BGE: supports float32 and float16\n\n"
                "Solutions:\n"
                "1. Use a compatible device (GPU with bfloat16 support for EmbeddingGemma)\n"
                "2. Switch to a different model (BGE works on more devices)\n"
                "3. Use CPU mode (slower but more compatible)\n"
                "4. Check PyTorch/CUDA versions for precision support"
            )
            return RuntimeError(enhanced_msg)

        elif any(keyword in error_msg for keyword in ["cuda", "gpu", "device"]):
            enhanced_msg = (
                f"Device error during {operation}: {str(error)}\n\n"
                "Device compatibility issues detected:\n"
                "- Check that your GPU supports required precisions\n"
                "- Try using device='cpu' for CPU-only execution\n"
                "- Verify CUDA installation and compatibility"
            )
            return RuntimeError(enhanced_msg)

        elif any(keyword in error_msg for keyword in ["memory", "out of memory", "oom"]):
            enhanced_msg = (
                f"Memory error during {operation}: {str(error)}\n\n"
                "Memory issues detected:\n"
                "- Model may be too large for your device\n"
                "- Try using smaller batch sizes\n"
                "- Switch to CPU mode or a smaller model\n"
                "- Close other applications to free memory"
            )
            return RuntimeError(enhanced_msg)

        # Return original error if we can't enhance it
        return error

    def _get_model_info(self) -> Dict[str, Any]:
        """
        Get basic model information.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": getattr(self.model, 'model_name_or_path', 'unknown'),
            "embedding_dimension": self.get_embedding_dimension(),
            "max_sequence_length": self.get_max_sequence_length(),
            "precision_requirements": self.get_precision_requirements()
        }
