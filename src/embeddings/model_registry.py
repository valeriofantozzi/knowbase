"""
Model Registry Module

Centralized registry for embedding model metadata and adapter management.
Provides factory methods for creating model adapters and model detection logic.
"""

from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
import re
import logging
from sentence_transformers import SentenceTransformer

from .adapters.base_adapter import ModelAdapter
from .adapters.bge_adapter import BGEAdapter
from .adapters.embeddinggemma_adapter import EmbeddingGemmaAdapter


@dataclass
class ModelMetadata:
    """
    Metadata for an embedding model.
    """
    model_name: str
    embedding_dimension: int
    max_sequence_length: int
    precision_requirements: List[str]
    supports_query_method: bool = False
    supports_document_method: bool = False
    prompt_format: str = "generic"
    mrl_supported: bool = False
    adapter_class: Type[ModelAdapter] = ModelAdapter

    def validate(self) -> None:
        """
        Validate model metadata.

        Raises:
            ValueError: If metadata is invalid
        """
        if not self.model_name:
            raise ValueError("model_name cannot be empty")

        if self.embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive")

        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")

        if not self.precision_requirements:
            raise ValueError("precision_requirements cannot be empty")

        if not issubclass(self.adapter_class, ModelAdapter):
            raise ValueError("adapter_class must be a subclass of ModelAdapter")


class ModelRegistry:
    """
    Registry for managing embedding model metadata and adapters.

    Provides centralized access to model information and adapter instantiation.
    """

    # Model name validation patterns
    VALID_MODEL_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9._/-]{0,127}$')
    SUSPICIOUS_PATTERNS = [
        re.compile(r'\.\.'),  # Directory traversal
        re.compile(r'^/'),    # Absolute paths
        re.compile(r'\\'),    # Backslashes (Windows paths)
        re.compile(r'\$'),    # Shell variables
        re.compile(r'`'),     # Command substitution
        re.compile(r';'),     # Command chaining
        re.compile(r'\|'),    # Pipes
        re.compile(r'>'),     # Redirection
        re.compile(r'<'),     # Input redirection
    ]

    def __init__(self):
        """Initialize registry with empty model dictionary."""
        self._models: Dict[str, ModelMetadata] = {}
        self.logger = logging.getLogger(__name__)
        self._register_defaults()

    def register_model(self, metadata: ModelMetadata) -> None:
        """
        Register a model in the registry.

        Args:
            metadata: Model metadata to register

        Raises:
            ValueError: If metadata is invalid or model already exists
        """
        # Validate model name before other validations
        self.validate_model_name(metadata.model_name)

        metadata.validate()

        if metadata.model_name in self._models:
            raise ValueError(f"Model '{metadata.model_name}' is already registered")

        self._models[metadata.model_name] = metadata

    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a registered model.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetadata if found, None otherwise

        Raises:
            ValueError: If model name is invalid
        """
        self.validate_model_name(model_name)
        return self._models.get(model_name)

    def is_registered(self, model_name: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_name: Name of the model

        Returns:
            True if model is registered
        """
        return model_name in self._models

    def get_registered_models(self) -> List[str]:
        """
        Get list of all registered model names.

        Returns:
            List of registered model names
        """
        return list(self._models.keys())

    def get_adapter(
        self,
        model_name: str,
        model: SentenceTransformer,
        **adapter_kwargs
    ) -> ModelAdapter:
        """
        Create an adapter instance for the given model.

        Args:
            model_name: Name of the model
            model: SentenceTransformer model instance
            **adapter_kwargs: Additional arguments for adapter initialization

        Returns:
            ModelAdapter instance

        Raises:
            ValueError: If model is not registered or name is invalid
        """
        # Validate model name
        self.validate_model_name(model_name)

        metadata = self.get_model_metadata(model_name)
        if metadata is None:
            # Try to detect model type for unregistered models
            adapter_class = self._detect_adapter_class(model_name)
        else:
            adapter_class = metadata.adapter_class

        try:
            return adapter_class(model, **adapter_kwargs)
        except Exception as e:
            # If adapter creation fails and this is not the base adapter, try fallback
            if adapter_class != ModelAdapter:
                self.logger.warning(
                    f"Failed to create {adapter_class.__name__} for model '{model_name}': {e}. "
                    "Falling back to base ModelAdapter."
                )
                try:
                    return ModelAdapter(model, **adapter_kwargs)
                except Exception as fallback_e:
                    raise ValueError(
                        f"Failed to create adapter for model '{model_name}': {e}. "
                        f"Fallback to base adapter also failed: {fallback_e}"
                    ) from fallback_e
            else:
                raise ValueError(f"Failed to create adapter for model '{model_name}': {e}") from e

    def detect_model_type(self, model_name: str) -> str:
        """
        Detect the model type/family from model name.

        Args:
            model_name: HuggingFace model name or path

        Returns:
            Model type string (e.g., 'bge', 'embeddinggemma', 'generic')
        """
        model_name_lower = model_name.lower()

        # Check for BGE models
        if 'bge' in model_name_lower or 'baai' in model_name_lower:
            return 'bge'

        # Check for EmbeddingGemma models
        if 'embeddinggemma' in model_name_lower or 'gemma' in model_name_lower:
            return 'embeddinggemma'

        # Default to generic
        return 'generic'

    def _detect_adapter_class(self, model_name: str) -> Type[ModelAdapter]:
        """
        Detect appropriate adapter class for unregistered model.

        Args:
            model_name: Model name

        Returns:
            Adapter class to use
        """
        model_type = self.detect_model_type(model_name)

        if model_type == 'bge':
            return BGEAdapter
        elif model_type == 'embeddinggemma':
            return EmbeddingGemmaAdapter
        else:
            # Fallback to base adapter for unknown models
            return ModelAdapter

    def _register_defaults(self) -> None:
        """Register default model configurations."""
        # Register BGE-large-en-v1.5
        self.register_model(ModelMetadata(
            model_name="BAAI/bge-large-en-v1.5",
            embedding_dimension=1024,
            max_sequence_length=512,
            precision_requirements=["float32", "float16"],
            supports_query_method=False,
            supports_document_method=False,
            prompt_format="instruction_prefix",
            mrl_supported=False,
            adapter_class=BGEAdapter
        ))

        # Register EmbeddingGemma-300m
        self.register_model(ModelMetadata(
            model_name="google/embeddinggemma-300m",
            embedding_dimension=768,  # Default, but supports MRL
            max_sequence_length=2048,
            precision_requirements=["float32", "bfloat16"],
            supports_query_method=False,  # Uses structured prompts, not separate methods
            supports_document_method=False,
            prompt_format="structured",
            mrl_supported=True,
            adapter_class=EmbeddingGemmaAdapter
        ))

    def validate_model_name(self, model_name: str) -> None:
        """
        Validate a model name for security and correctness.

        Args:
            model_name: Model name to validate

        Raises:
            ValueError: If model name is invalid or potentially malicious
        """
        if not isinstance(model_name, str):
            raise ValueError(f"Model name must be a string, got {type(model_name)}")

        if not model_name.strip():
            raise ValueError("Model name cannot be empty or whitespace only")

        if len(model_name) > 128:
            raise ValueError(f"Model name too long: {len(model_name)} characters (max 128)")

        # Check basic pattern
        if not self.VALID_MODEL_NAME_PATTERN.match(model_name):
            raise ValueError(
                f"Invalid model name format: '{model_name}'. "
                "Must start with alphanumeric character and contain only: letters, numbers, dots, hyphens, underscores, slashes"
            )

        # Check for suspicious patterns
        for pattern in self.SUSPICIOUS_PATTERNS:
            if pattern.search(model_name):
                self.logger.warning(f"Suspicious pattern detected in model name: '{model_name}'")
                raise ValueError(
                    f"Potentially unsafe model name: '{model_name}'. "
                    "Model names should not contain path traversal, shell commands, or special characters"
                )

        # Check for common attack vectors
        if any(char in model_name for char in ['\n', '\r', '\t']):
            raise ValueError(f"Model name contains control characters: '{model_name}'")

        # Warn about unusual patterns but don't block
        if '..' in model_name:
            self.logger.warning(f"Model name contains '..': '{model_name}' - potential path traversal")
        elif model_name.startswith('./') or model_name.startswith('../'):
            self.logger.warning(f"Model name starts with relative path: '{model_name}'")

    def sanitize_model_name(self, model_name: str) -> str:
        """
        Sanitize a model name by removing potentially problematic characters.

        Args:
            model_name: Raw model name

        Returns:
            Sanitized model name
        """
        # Remove leading/trailing whitespace
        sanitized = model_name.strip()

        # Replace problematic characters with safe alternatives
        sanitized = re.sub(r'[^\w./-]', '_', sanitized)

        # Remove consecutive dots, slashes, etc.
        sanitized = re.sub(r'\.\.+', '.', sanitized)
        sanitized = re.sub(r'/+', '/', sanitized)

        return sanitized

    def get_models_by_type(self, model_type: str) -> List[str]:
        """
        Get all registered models of a specific type.

        Args:
            model_type: Model type to filter by

        Returns:
            List of model names matching the type
        """
        return [
            name for name, metadata in self._models.items()
            if self.detect_model_type(name) == model_type
        ]

    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare models and return comparison data.

        Args:
            model_names: List of model names to compare (all if None)

        Returns:
            Dictionary with comparison data
        """
        if model_names is None:
            model_names = self.get_registered_models()

        comparison = {}
        for name in model_names:
            metadata = self.get_model_metadata(name)
            if metadata:
                comparison[name] = {
                    "dimension": metadata.embedding_dimension,
                    "max_length": metadata.max_sequence_length,
                    "precision": metadata.precision_requirements,
                    "mrl": metadata.mrl_supported,
                    "type": self.detect_model_type(name)
                }

        return comparison


# Global registry instance
_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.

    Returns:
        ModelRegistry instance
    """
    return _registry
