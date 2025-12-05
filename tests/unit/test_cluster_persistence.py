import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from src.clustering.cluster_manager import ClusterManager
from src.clustering.pipeline import ClusteringResult

class TestClusterPersistence:
    
    @pytest.fixture
    def chunk_ids(self):
        return ["id1", "id2", "id3", "id4"]
    
    @pytest.fixture
    def labels(self):
        return np.array([0, 1, 0, -1])
    
    @pytest.fixture
    def probabilities(self):
        return np.array([0.9, 0.8, 0.95, 0.0])
    
    @pytest.fixture
    def cluster_names(self):
        return {0: "Topic A", 1: "Topic B"}
    
    @pytest.fixture
    def topics(self):
        return {
            0: [("keyword1", 0.5), ("keyword2", 0.4)],
            1: [("keyword3", 0.6)]
        }

    def test_store_cluster_labels_with_topics(self, chunk_ids, labels, probabilities, cluster_names, topics):
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        
        # Mock existing data return
        mock_collection.get.return_value = {
            "ids": chunk_ids,
            "metadatas": [{"existing": "data"}] * len(chunk_ids)
        }
        
        manager = ClusterManager(chroma_manager=mock_chroma)
        
        manager.store_cluster_labels(
            chunk_ids=chunk_ids,
            labels=labels,
            probabilities=probabilities,
            cluster_names=cluster_names,
            topics=topics
        )
        
        # Verify update was called
        mock_collection.update.assert_called_once()
        call_args = mock_collection.update.call_args[1]
        updated_metadatas = call_args["metadatas"]
        
        assert len(updated_metadatas) == 4
        
        # Check first item (Cluster 0, Topic A)
        meta0 = updated_metadatas[0]
        assert meta0["cluster_id"] == 0
        assert meta0["cluster_topic"] == "Topic A"
        assert meta0["cluster_keywords"] == "keyword1, keyword2"
        
        # Check last item (Outlier)
        meta3 = updated_metadatas[3]
        assert meta3["cluster_id"] == -1
        assert "cluster_topic" not in meta3
        assert "cluster_keywords" not in meta3

    def test_load_clustering_results(self):
        mock_chroma = MagicMock()
        mock_collection = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        
        # Mock fetch data
        mock_collection.get.return_value = {
            "ids": ["id1", "id2", "id3"],
            "metadatas": [
                {"cluster_id": 0, "cluster_topic": "Topic A", "cluster_keywords": "k1, k2", "cluster_probability": 0.9},
                {"cluster_id": 1, "cluster_topic": "Topic B", "cluster_keywords": "k3", "cluster_probability": 0.8},
                {"cluster_id": -1, "cluster_probability": 0.0} # Outlier
            ],
            "embeddings": [[0.1], [0.2], [0.3]],
            "documents": ["d1", "d2", "d3"]
        }
        
        manager = ClusterManager(chroma_manager=mock_chroma)
        result = manager.load_clustering_results()
        
        assert result is not None
        assert isinstance(result, ClusteringResult)
        
        # Verify Labels
        np.testing.assert_array_equal(result.labels, np.array([0, 1, -1]))
        np.testing.assert_array_equal(result.probabilities, np.array([0.9, 0.8, 0.0]))
        
        # Verify Names
        assert len(result.cluster_names) == 2
        assert result.cluster_names[0] == "Topic A"
        assert result.cluster_names[1] == "Topic B"
        
        # Verify Topics (Keywords)
        assert len(result.topics) == 2
        assert result.topics[0][0][0] == "k1"
        assert result.topics[1][0][0] == "k3"
