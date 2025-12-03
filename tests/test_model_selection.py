"""
Unit tests for model selection and collection management.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.vector_store.chroma_manager import ChromaDBManager
from src.embeddings.model_registry import get_model_registry
from src.embeddings.model_manager import get_model_manager
from src.retrieval.query_engine import QueryEngine
from src.retrieval.similarity_search import SimilaritySearch


class TestChromaDBManagerModelSupport:
    """Test ChromaDBManager model-specific collection support."""

    @patch("chromadb.PersistentClient")
    def test_collection_name_generation(self, mock_client):
        """Test model-specific collection name generation."""
        manager = ChromaDBManager()

        # Test BGE model
        name = manager._generate_collection_name(
            "BAAI/bge-large-en-v1.5", "document_embeddings"
        )
        assert name == "document_embeddings_bge_large"

        # Test Gemma model
        name = manager._generate_collection_name(
            "google/embeddinggemma-300m", "document_embeddings"
        )
        assert name == "document_embeddings_gemma_300m"

        # Test unknown model
        name = manager._generate_collection_name("unknown/model", "document_embeddings")
        assert name == "document_embeddings_unknown"

    def test_model_slug_extraction(self):
        """Test model slug extraction from model names."""
        manager = ChromaDBManager()

        # Test various model names
        test_cases = [
            ("BAAI/bge-large-en-v1.5", "bge_large"),
            ("google/embeddinggemma-300m", "gemma_300m"),
            ("sentence-transformers/all-MiniLM-L6-v2", "all_minilm"),
            ("microsoft/DialoGPT-medium", "dialogpt_medium"),
            ("unknown-model-name", "unknown_model"),
        ]

        for model_name, expected_slug in test_cases:
            slug = manager._extract_model_slug(model_name)
            assert slug == expected_slug

    def test_collection_name_sanitization(self):
        """Test collection name sanitization."""
        manager = ChromaDBManager()

        test_cases = [
            ("valid_name", "valid_name"),
            ("name with spaces", "name_with_spaces"),
            ("name@with#special$chars", "name_with_special_chars"),
            ("_leading_underscore", "leading_underscore"),
            ("trailing_underscore_", "trailing_underscore"),
            ("multiple___underscores", "multiple_underscores"),
            ("", "default_collection"),
        ]

        for input_name, expected_output in test_cases:
            sanitized = manager._sanitize_collection_name(input_name)
            assert sanitized == expected_output
            # Ensure no invalid characters remain
            import re

            assert re.match(r"^[a-zA-Z0-9_-]+$", sanitized)

    @patch("chromadb.PersistentClient")
    def test_get_or_create_collection_with_model(self, mock_client):
        """Test creating collections with model metadata."""
        mock_collection = Mock()
        mock_client_instance = Mock()
        mock_client_instance.create_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        manager = ChromaDBManager()

        # Create collection with model
        collection = manager.get_or_create_collection(
            model_name="BAAI/bge-large-en-v1.5", embedding_dimension=1024
        )

        # Verify collection creation was called with metadata
        mock_client_instance.create_collection.assert_called_once()
        call_args = mock_client_instance.create_collection.call_args
        metadata = call_args[1]["metadata"]

        assert metadata["embedding_dimension"] == 1024
        assert metadata["model_name"] == "BAAI/bge-large-en-v1.5"
        assert "created_at" in metadata
        assert metadata["model_adapter"] == "BGEAdapter"

    @patch("chromadb.PersistentClient")
    def test_get_collection_for_model(self, mock_client):
        """Test getting collection for specific model."""
        manager = ChromaDBManager()

        collection = manager.get_collection_for_model("BAAI/bge-large-en-v1.5")

        # Should generate correct collection name and get/create it
        expected_name = "document_embeddings_bge_large"
        mock_client_instance = mock_client.return_value
        mock_client_instance.get_or_create_collection.assert_called_with(
            name=expected_name, model_name="BAAI/bge-large-en-v1.5"
        )

    @patch("chromadb.PersistentClient")
    def test_list_collections_by_model(self, mock_client):
        """Test listing collections grouped by model."""
        # Mock collections with different metadata
        mock_collections = [
            Mock(
                name="document_embeddings_bge_large",
                metadata={"model_name": "BAAI/bge-large-en-v1.5"},
            ),
            Mock(
                name="document_embeddings_gemma_300m",
                metadata={"model_name": "google/embeddinggemma-300m"},
            ),
            Mock(name="old_collection", metadata={}),  # No model metadata
        ]

        mock_client_instance = Mock()
        mock_client_instance.list_collections.return_value = mock_collections
        mock_client.return_value = mock_client_instance

        manager = ChromaDBManager()
        collections_by_model = manager.list_collections_by_model()

        assert "BAAI/bge-large-en-v1.5" in collections_by_model
        assert (
            collections_by_model["BAAI/bge-large-en-v1.5"]
            == "document_embeddings_bge_large"
        )
        assert "google/embeddinggemma-300m" in collections_by_model
        assert (
            collections_by_model["google/embeddinggemma-300m"]
            == "document_embeddings_gemma_300m"
        )
        assert "unknown" in collections_by_model

    @patch("chromadb.PersistentClient")
    def test_validate_collection_model(self, mock_client):
        """Test collection model validation."""
        # Mock collection with correct metadata
        mock_collection = Mock()
        mock_collection.metadata = {
            "model_name": "BAAI/bge-large-en-v1.5",
            "embedding_dimension": 1024,
        }
        mock_client_instance = Mock()
        mock_client_instance.get_collection.return_value = mock_collection
        mock_client.return_value = mock_client_instance

        manager = ChromaDBManager()

        # Valid validation
        result = manager.validate_collection_model(
            "test_collection", "BAAI/bge-large-en-v1.5"
        )
        assert result["valid"] is True
        assert result["model_valid"] is True

        # Invalid model name
        result = manager.validate_collection_model("test_collection", "different/model")
        assert result["valid"] is False
        assert result["model_valid"] is False
        assert "Model mismatch" in result["errors"][0]


class TestQueryEngineModelSupport:
    """Test QueryEngine model switching functionality."""

    @patch("src.retrieval.similarity_search.SimilaritySearch")
    def test_query_engine_model_initialization(self, mock_similarity_search):
        """Test QueryEngine initialization with model."""
        engine = QueryEngine(model_name="BAAI/bge-large-en-v1.5")

        mock_similarity_search.assert_called_once_with(
            model_name="BAAI/bge-large-en-v1.5"
        )
        assert engine.model_name == "BAAI/bge-large-en-v1.5"

    @patch("src.retrieval.similarity_search.SimilaritySearch")
    def test_query_engine_set_model(self, mock_similarity_search_class):
        """Test switching model in QueryEngine."""
        # Mock similarity search instance
        mock_similarity_search = Mock()
        mock_similarity_search_class.return_value = mock_similarity_search

        engine = QueryEngine()

        # Switch model
        engine.set_model("google/embeddinggemma-300m")

        # Should create new SimilaritySearch with new model
        assert mock_similarity_search_class.call_count == 2  # Initial + set_model
        last_call = mock_similarity_search_class.call_args_list[-1]
        assert last_call[1]["model_name"] == "google/embeddinggemma-300m"

    def test_query_cache_with_model(self):
        """Test that cache keys include model name."""
        from src.retrieval.query_engine import QueryEngine, SearchFilters

        engine = QueryEngine()
        engine.model_name = "test/model"

        # Generate cache key
        filters = SearchFilters()
        cache_key = engine._get_cache_key("test query", 10, None, filters, "test/model")

        # Cache key should be deterministic and include model
        assert isinstance(cache_key, str)
        assert len(cache_key) == 32  # MD5 hex length

        # Same inputs should produce same key
        cache_key2 = engine._get_cache_key(
            "test query", 10, None, filters, "test/model"
        )
        assert cache_key == cache_key2

        # Different model should produce different key
        cache_key3 = engine._get_cache_key(
            "test query", 10, None, filters, "different/model"
        )
        assert cache_key != cache_key3


class TestSimilaritySearchModelSupport:
    """Test SimilaritySearch model switching functionality."""

    @patch("src.vector_store.chroma_manager.ChromaDBManager")
    @patch("src.embeddings.embedder.Embedder")
    def test_similarity_search_model_initialization(
        self, mock_embedder, mock_chroma_manager
    ):
        """Test SimilaritySearch initialization with model."""
        search = SimilaritySearch(model_name="BAAI/bge-large-en-v1.5")

        assert search.model_name == "BAAI/bge-large-en-v1.5"

    @patch("src.vector_store.chroma_manager.ChromaDBManager")
    @patch("src.embeddings.embedder.Embedder")
    def test_similarity_search_set_model(
        self, mock_embedder_class, mock_chroma_manager_class
    ):
        """Test switching model in SimilaritySearch."""
        # Mock instances
        mock_embedder = Mock()
        mock_embedder_class.return_value = mock_embedder
        mock_chroma_manager = Mock()
        mock_chroma_manager_class.return_value = mock_chroma_manager

        search = SimilaritySearch()

        # Switch model
        search.set_model("google/embeddinggemma-300m")

        # Should call set_model on embedder
        mock_embedder.set_model.assert_called_once_with("google/embeddinggemma-300m")

        # Should reset collection cache
        assert search._collection is None


class TestModelManagerIntegration:
    """Test ModelManager integration with caching."""

    @patch("src.embeddings.model_manager.time.time")
    @patch("src.embeddings.model_loader.ModelLoader")
    def test_model_manager_caching(self, mock_loader_class, mock_time):
        """Test model caching in ModelManager."""
        mock_time.return_value = 1000.0

        # Mock loader
        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader

        from src.embeddings.model_manager import ModelManager

        manager = ModelManager(max_cache_size=2)

        # First access - should create new loader
        loader1 = manager.get_model_loader("BAAI/bge-large-en-v1.5")
        assert loader1 == mock_loader
        assert mock_loader_class.call_count == 1

        # Second access - should return cached loader
        loader2 = manager.get_model_loader("BAAI/bge-large-en-v1.5")
        assert loader2 == mock_loader
        assert mock_loader_class.call_count == 1  # Not called again

    @patch("src.embeddings.model_manager.time.time")
    @patch("src.embeddings.model_loader.ModelLoader")
    def test_model_manager_cache_eviction(self, mock_loader_class, mock_time):
        """Test LRU cache eviction in ModelManager."""
        mock_time.return_value = 1000.0

        from src.embeddings.model_manager import ModelManager

        manager = ModelManager(max_cache_size=2)

        # Load two models
        manager.get_model_loader("model1")
        manager.get_model_loader("model2")

        assert len(manager._model_cache) == 2

        # Load third model - should evict oldest
        manager.get_model_loader("model3")

        assert len(manager._model_cache) == 2
        assert "model1" not in manager._model_cache  # Should be evicted
        assert "model2" in manager._model_cache
        assert "model3" in manager._model_cache
