"""
Unit tests for embeddings module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.embeddings.adapters.base_adapter import ModelAdapter
from src.embeddings.adapters.bge_adapter import BGEAdapter
from src.embeddings.adapters.embeddinggemma_adapter import EmbeddingGemmaAdapter
from src.embeddings.model_registry import ModelRegistry, get_model_registry
from src.embeddings.model_loader import ModelLoader
from src.embeddings.embedder import Embedder


class TestModelAdapter:
    """Test base ModelAdapter functionality."""

    def test_abstract_methods(self):
        """Test that ModelAdapter defines required abstract methods."""
        # This should raise TypeError since ModelAdapter is abstract
        with pytest.raises(TypeError):
            ModelAdapter(Mock())

    @patch('sentence_transformers.SentenceTransformer')
    def test_base_adapter_initialization(self, mock_model):
        """Test base adapter initialization."""
        # Create a concrete subclass for testing
        class ConcreteAdapter(ModelAdapter):
            def format_query_prompt(self, query: str) -> str:
                return f"Query: {query}"

            def format_document_prompt(self, document: str, title: str = None) -> str:
                return f"Document: {document}"

            def get_embedding_dimension(self) -> int:
                return 768

            def get_max_sequence_length(self) -> int:
                return 512

            def get_precision_requirements(self) -> list:
                return ["float32"]

            def validate_precision(self, embeddings: np.ndarray) -> None:
                pass

        adapter = ConcreteAdapter(mock_model)
        assert adapter.model == mock_model

    @patch('sentence_transformers.SentenceTransformer')
    def test_encode_query_default(self, mock_model):
        """Test default encode_query implementation."""
        class ConcreteAdapter(ModelAdapter):
            def format_query_prompt(self, query: str) -> str:
                return f"Query: {query}"

            def format_document_prompt(self, document: str, title: str = None) -> str:
                return f"Document: {document}"

            def get_embedding_dimension(self) -> int:
                return 768

            def get_max_sequence_length(self) -> int:
                return 512

            def get_precision_requirements(self) -> list:
                return ["float32"]

            def validate_precision(self, embeddings: np.ndarray) -> None:
                pass

        adapter = ConcreteAdapter(mock_model)
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])

        result = adapter.encode_query("test query")
        mock_model.encode.assert_called_once_with(["Query: test query"])
        assert isinstance(result, np.ndarray)


class TestBGEAdapter:
    """Test BGEAdapter functionality."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_bge_adapter_initialization(self, mock_model):
        """Test BGE adapter initialization."""
        mock_model.get_sentence_embedding_dimension.return_value = 1024
        mock_model.max_seq_length = 512

        adapter = BGEAdapter(mock_model)

        assert adapter.get_embedding_dimension() == 1024
        assert adapter.get_max_sequence_length() == 512
        assert adapter.get_precision_requirements() == ["float32"]

    @patch('sentence_transformers.SentenceTransformer')
    def test_bge_prompt_formatting(self, mock_model):
        """Test BGE prompt formatting."""
        adapter = BGEAdapter(mock_model)

        query_prompt = adapter.format_query_prompt("What is AI?")
        assert query_prompt == "Represent this sentence for searching relevant passages: What is AI?"

        doc_prompt = adapter.format_document_prompt("AI is artificial intelligence")
        assert doc_prompt == "AI is artificial intelligence"

        doc_prompt_with_title = adapter.format_document_prompt("AI is artificial intelligence", "About AI")
        assert doc_prompt_with_title == "AI is artificial intelligence"

    @patch('sentence_transformers.SentenceTransformer')
    def test_bge_precision_validation(self, mock_model):
        """Test BGE precision validation."""
        adapter = BGEAdapter(mock_model)

        # Valid float32 embeddings
        valid_embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        adapter.validate_precision(valid_embeddings)  # Should not raise

        # Invalid precision
        invalid_embeddings = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        with pytest.raises(ValueError, match="BGE embeddings must be float32"):
            adapter.validate_precision(invalid_embeddings)


class TestEmbeddingGemmaAdapter:
    """Test EmbeddingGemmaAdapter functionality."""

    @patch('sentence_transformers.SentenceTransformer')
    def test_gemma_adapter_initialization(self, mock_model):
        """Test EmbeddingGemma adapter initialization."""
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_model.max_seq_length = 512

        adapter = EmbeddingGemmaAdapter(mock_model)

        assert adapter.get_embedding_dimension() == 768
        assert adapter.get_max_sequence_length() == 512
        assert adapter.get_precision_requirements() == ["float32"]

    @patch('sentence_transformers.SentenceTransformer')
    def test_gemma_prompt_formatting(self, mock_model):
        """Test EmbeddingGemma prompt formatting."""
        adapter = EmbeddingGemmaAdapter(mock_model)

        query_prompt = adapter.format_query_prompt("What is AI?")
        assert query_prompt == "task: search result | query: What is AI?"

        doc_prompt = adapter.format_document_prompt("AI is artificial intelligence")
        assert doc_prompt == "title:  | text: AI is artificial intelligence"

        doc_prompt_with_title = adapter.format_document_prompt("AI is artificial intelligence", "About AI")
        assert doc_prompt_with_title == "title: About AI | text: AI is artificial intelligence"

    @patch('sentence_transformers.SentenceTransformer')
    def test_gemma_mrl_dimension(self, mock_model):
        """Test MRL dimension handling."""
        adapter = EmbeddingGemmaAdapter(mock_model)

        # Test valid MRL dimension
        adapter.set_mrl_dimension(256)
        assert adapter.get_embedding_dimension() == 256

        # Test invalid MRL dimension
        with pytest.raises(ValueError, match="Unsupported MRL dimension"):
            adapter.set_mrl_dimension(999)

    @patch('sentence_transformers.SentenceTransformer')
    def test_gemma_encode_with_mrl(self, mock_model):
        """Test encoding with MRL dimension."""
        mock_model.encode.return_value = np.random.rand(1, 768).astype(np.float32)
        adapter = EmbeddingGemmaAdapter(mock_model)

        # Test with MRL
        result = adapter.encode_query("test", mrl_dimension=256)
        assert result.shape[-1] == 256
        mock_model.encode.assert_called_once()


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    def test_registry_initialization(self):
        """Test registry initialization with default models."""
        registry = ModelRegistry()

        # Check that default models are registered
        bge_metadata = registry.get_model_metadata("BAAI/bge-large-en-v1.5")
        assert bge_metadata is not None
        assert bge_metadata.model_name == "BAAI/bge-large-en-v1.5"
        assert bge_metadata.embedding_dimension == 1024

        gemma_metadata = registry.get_model_metadata("google/embeddinggemma-300m")
        assert gemma_metadata is not None
        assert gemma_metadata.model_name == "google/embeddinggemma-300m"
        assert gemma_metadata.embedding_dimension == 768

    def test_model_registration(self):
        """Test registering custom models."""
        registry = ModelRegistry()

        # Register a custom model
        custom_metadata = Mock()
        custom_metadata.model_name = "custom/model-v1"
        custom_metadata.embedding_dimension = 512

        registry.register_model(custom_metadata)

        retrieved = registry.get_model_metadata("custom/model-v1")
        assert retrieved == custom_metadata

    def test_model_type_detection(self):
        """Test automatic model type detection."""
        registry = ModelRegistry()

        # Test BGE detection
        bge_adapter_class = registry.detect_model_type("BAAI/bge-large-en-v1.5")
        assert bge_adapter_class == BGEAdapter

        # Test Gemma detection
        gemma_adapter_class = registry.detect_model_type("google/embeddinggemma-300m")
        assert gemma_adapter_class == EmbeddingGemmaAdapter

        # Test unknown model (should fallback to base adapter)
        unknown_adapter_class = registry.detect_model_type("unknown/model")
        assert unknown_adapter_class == ModelAdapter

    def test_get_registered_models(self):
        """Test getting list of registered models."""
        registry = ModelRegistry()
        models = registry.get_registered_models()

        assert "BAAI/bge-large-en-v1.5" in models
        assert "google/embeddinggemma-300m" in models
        assert len(models) >= 2


class TestModelLoader:
    """Test ModelLoader functionality."""

    @patch('src.embeddings.model_registry.get_model_registry')
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loader_initialization(self, mock_model_class, mock_get_registry):
        """Test ModelLoader initialization."""
        # Mock registry
        mock_registry = Mock()
        mock_adapter = Mock()
        mock_registry.get_adapter.return_value = mock_adapter
        mock_get_registry.return_value = mock_registry

        loader = ModelLoader(model_name="test/model")

        assert loader.model_name == "test/model"
        mock_registry.get_adapter.assert_called_once_with("test/model")
        assert loader.adapter == mock_adapter

    @patch('src.embeddings.model_registry.get_model_registry')
    @patch('sentence_transformers.SentenceTransformer')
    def test_model_loading(self, mock_model_class, mock_get_registry):
        """Test model loading process."""
        # Mock registry and adapter
        mock_registry = Mock()
        mock_adapter = Mock()
        mock_adapter.validate_precision.return_value = None
        mock_registry.get_adapter.return_value = mock_adapter
        mock_get_registry.return_value = mock_registry

        # Mock SentenceTransformer
        mock_model_instance = Mock()
        mock_model_class.return_value = mock_model_instance

        loader = ModelLoader(model_name="test/model")
        model = loader.load_model()

        assert model == mock_model_instance
        mock_model_class.assert_called_once_with(
            "test/model",
            cache_folder=loader.cache_dir,
            device=loader.device
        )
        mock_model_instance.eval.assert_called_once()


class TestEmbedder:
    """Test Embedder functionality."""

    @patch('src.embeddings.model_loader.get_model_loader')
    def test_embedder_initialization(self, mock_get_loader):
        """Test Embedder initialization."""
        mock_loader = Mock()
        mock_adapter = Mock()
        mock_loader.adapter = mock_adapter
        mock_get_loader.return_value = mock_loader

        embedder = Embedder()

        assert embedder.model_loader == mock_loader
        assert embedder.adapter == mock_adapter

    @patch('src.embeddings.model_loader.get_model_loader')
    def test_embedder_model_property(self, mock_get_loader):
        """Test lazy loading of model."""
        mock_loader = Mock()
        mock_model = Mock()
        mock_loader.get_model.return_value = mock_model
        mock_get_loader.return_value = mock_loader

        embedder = Embedder()

        # First access should load model
        model1 = embedder.model
        assert model1 == mock_model
        mock_loader.get_model.assert_called_once()

        # Second access should return cached model
        model2 = embedder.model
        assert model2 == mock_model
        mock_loader.get_model.assert_called_once()  # Should not be called again

    @patch('src.embeddings.model_loader.get_model_loader')
    def test_embedder_set_model(self, mock_get_loader):
        """Test setting a different model."""
        # Initial loader
        mock_loader1 = Mock()
        mock_adapter1 = Mock()
        mock_loader1.adapter = mock_adapter1

        # New loader for set_model
        mock_loader2 = Mock()
        mock_adapter2 = Mock()
        mock_loader2.adapter = mock_adapter2
        mock_get_loader.return_value = mock_loader2

        embedder = Embedder(model_loader=mock_loader1)

        # Set new model
        embedder.set_model("new/model")

        # Should have new loader and adapter
        assert embedder.model_loader == mock_loader2
        assert embedder.adapter == mock_adapter2

        # Model cache should be cleared
        assert embedder._model is None

