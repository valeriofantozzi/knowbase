"""
Cluster Manager Module

Manages cluster storage, updates, and queries in ChromaDB.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np

from ..vector_store.chroma_manager import ChromaDBManager
from ..utils.logger import get_logger
from .pipeline import ClusteringResult


@dataclass
class ClusterMetadata:
    """
    Metadata for a cluster.
    
    Represents statistics and properties of a document cluster.
    """
    cluster_id: int
    size: int
    name: Optional[str] = None
    centroid: Optional[np.ndarray] = None
    keywords: Optional[List[str]] = None
    theme: Optional[str] = None
    source_ids: Optional[Set[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "name": self.name,
            "centroid": self.centroid.tolist() if self.centroid is not None else None,
            "keywords": self.keywords or [],
            "theme": self.theme,
            "source_ids": list(self.source_ids) if self.source_ids else []
        }


class ClusterManager:
    """
    Manages cluster storage and operations in ChromaDB.
    
    Handles:
    - Storing cluster labels in ChromaDB metadata
    - Updating cluster assignments
    - Querying chunks by cluster
    - Cluster statistics
    """
    
    def __init__(self, chroma_manager: Optional[ChromaDBManager] = None, collection_name: Optional[str] = None):
        """
        Initialize cluster manager.
        
        Args:
            chroma_manager: ChromaDBManager instance (creates default if None)
            collection_name: Optional name of the collection to manage
        """
        self.logger = get_logger(__name__)
        self.chroma_manager = chroma_manager or ChromaDBManager()
        self.collection = self.chroma_manager.get_or_create_collection(name=collection_name)
    
    def store_cluster_labels(
        self,
        chunk_ids: List[str],
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        cluster_names: Optional[Dict[int, str]] = None,
        topics: Optional[Dict[int, List[Tuple[str, float]]]] = None
    ) -> int:
        """
        Store cluster labels in ChromaDB metadata.
        
        Args:
            chunk_ids: List of chunk IDs
            labels: Array of cluster labels (-1 for outliers)
            probabilities: Optional array of cluster membership probabilities
            cluster_names: Optional mapping from cluster ID to name/topic
            topics: Optional mapping from cluster ID to list of (keyword, score) tuples
        
        Returns:
            Number of chunks updated
        """
        if len(chunk_ids) != len(labels):
            raise ValueError("chunk_ids and labels must have same length")
        
        if probabilities is not None and len(probabilities) != len(labels):
            raise ValueError("probabilities must have same length as labels")
        
        self.logger.info(f"Storing cluster labels for {len(chunk_ids)} chunks")
        
        self.logger.info(f"Storing cluster labels for {len(chunk_ids)} chunks")
        
        # Get existing metadata
        existing_data = self.collection.get(ids=chunk_ids, include=["metadatas"])
        fetched_ids = existing_data.get("ids", [])
        fetched_metadatas = existing_data.get("metadatas", [])
        
        # Create map of chunk_id -> existing_metadata
        id_to_meta = {}
        if fetched_ids and fetched_metadatas:
            # Chroma might return None for metadata if it's completely empty for an item
            # or it might return a list of Nones. Handle both.
            for cid, meta in zip(fetched_ids, fetched_metadatas):
                id_to_meta[cid] = meta if meta is not None else {}
        
        # Align metadata with input chunk_ids
        updated_metadatas = []
        for i, chunk_id in enumerate(chunk_ids):
            # Start with existing metadata or empty dict
            existing_meta = id_to_meta.get(chunk_id, {})
            
            # Update with cluster info
            updated_meta = existing_meta.copy()
            updated_meta["cluster_id"] = int(labels[i])
            
            if probabilities is not None:
                updated_meta["cluster_probability"] = float(probabilities[i])
            else:
                # Default probability: 1.0 for assigned clusters, 0.0 for outliers
                updated_meta["cluster_probability"] = 1.0 if labels[i] != -1 else 0.0
                
            if cluster_names and int(labels[i]) in cluster_names:
                updated_meta["cluster_topic"] = cluster_names[int(labels[i])]
            
            if topics and int(labels[i]) in topics:
                # Store top 10 keywords as comma-separated string
                keywords = topics[int(labels[i])]
                if keywords:
                    # extract just the words
                    keyword_strs = [k[0] for k in keywords[:10]]
                    updated_meta["cluster_keywords"] = ", ".join(keyword_strs)
            
            updated_metadatas.append(updated_meta)
        
        # Update in ChromaDB
        self.collection.update(
            ids=chunk_ids,
            metadatas=updated_metadatas
        )
        
        self.logger.info(f"Stored cluster labels for {len(chunk_ids)} chunks")
        return len(chunk_ids)
    
    def get_chunks_by_cluster(
        self,
        cluster_id: int,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get chunks belonging to a specific cluster.
        
        Args:
            cluster_id: Cluster ID to query
            limit: Maximum number of results
        
        Returns:
            Dictionary with ids, documents, metadatas, embeddings
        """
        where_clause = {"cluster_id": cluster_id}
        
        results = self.collection.get(
            where=where_clause,
            limit=limit,
            include=["documents", "metadatas", "embeddings"]
        )
        
        return results
    
    def get_cluster_statistics(self) -> Dict[int, ClusterMetadata]:
        """
        Get statistics for all clusters.
        
        Returns:
            Dictionary mapping cluster_id -> ClusterMetadata
        """
        # Get all chunks with cluster information
        all_data = self.collection.get(
            include=["metadatas", "embeddings"]
        )
        
        metadatas = all_data.get("metadatas", [])
        embeddings = all_data.get("embeddings", [])
        ids = all_data.get("ids", [])
        
        # Group by cluster_id
        clusters: Dict[int, List[int]] = {}  # cluster_id -> list of indices
        
        for i, metadata in enumerate(metadatas):
            if metadata and "cluster_id" in metadata:
                cluster_id = int(metadata["cluster_id"])
                if cluster_id != -1:  # Exclude outliers
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(i)
        
        # Calculate statistics for each cluster
        cluster_stats = {}
        
        for cluster_id, indices in clusters.items():
            cluster_embeddings = None
            if embeddings:
                cluster_embeddings = np.array([embeddings[i] for i in indices])
            
            # Calculate centroid
            centroid = None
            if cluster_embeddings is not None and len(cluster_embeddings) > 0:
                centroid = np.mean(cluster_embeddings, axis=0)
            
            # Get source IDs
            source_ids = set()
            for i in indices:
                if metadatas[i]:
                    sid = metadatas[i].get("source_id")
                    if sid:
                        source_ids.add(sid)
            
            cluster_stats[cluster_id] = ClusterMetadata(
                cluster_id=cluster_id,
                size=len(indices),
                centroid=centroid,
                source_ids=source_ids
            )
        
        return cluster_stats
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get sizes of all clusters.
        
        Returns:
            Dictionary mapping cluster_id -> size
        """
        stats = self.get_cluster_statistics()
        return {cluster_id: metadata.size for cluster_id, metadata in stats.items()}
    
    def get_outlier_count(self) -> int:
        """
        Get count of outliers (chunks with cluster_id == -1).
        
        Returns:
            Number of outliers
        """
        where_clause = {"cluster_id": -1}
        results = self.collection.get(where=where_clause)
        return len(results.get("ids", []))
    
    def update_cluster_metadata(
        self,
        cluster_id: int,
        keywords: Optional[List[str]] = None,
        theme: Optional[str] = None
    ):
        """
        Update cluster metadata (keywords, theme).
        
        Note: This stores metadata separately, not in ChromaDB.
        For persistent storage, consider using a separate JSON file or database.
        
        Args:
            cluster_id: Cluster ID
            keywords: List of keywords for the cluster
            theme: Theme description
        """
        # This is a placeholder - in a full implementation, you might store
        # cluster metadata in a separate collection or JSON file
        self.logger.info(
            f"Updating metadata for cluster {cluster_id}: "
            f"keywords={keywords}, theme={theme}"
        )
    
    def clear_cluster_labels(self, chunk_ids: Optional[List[str]] = None) -> int:
        """
        Clear cluster labels from metadata.
        
        Args:
            chunk_ids: List of chunk IDs to clear (None = all chunks)
        
        Returns:
            Number of chunks updated
        """
        if chunk_ids is None:
            # Get all chunk IDs
            all_data = self.collection.get(include=["metadatas"])
            chunk_ids = all_data.get("ids", [])
        
        if not chunk_ids:
            return 0
        
        # Get existing metadata
        existing_data = self.collection.get(ids=chunk_ids, include=["metadatas"])
        existing_metadatas = existing_data.get("metadatas", [])
        
        # Remove cluster fields
        updated_metadatas = []
        for existing_meta in existing_metadatas:
            if existing_meta is None:
                existing_meta = {}
            
            updated_meta = existing_meta.copy()
            updated_meta.pop("cluster_id", None)
            updated_meta.pop("cluster_probability", None)
            updated_metadatas.append(updated_meta)
        
        # Update in ChromaDB
        self.collection.update(
            ids=chunk_ids,
            metadatas=updated_metadatas
        )
        
        self.logger.info(f"Cleared cluster labels for {len(chunk_ids)} chunks")
        return len(chunk_ids)

    def store_clustering_results(self, chunk_ids: List[str], result: Any) -> int:
        """
        Store full clustering pipeline results.
        
        Args:
            chunk_ids: List of chunk IDs corresponding to the results
            result: ClusteringResult object from pipeline
            
        Returns:
            Number of chunks updated
        """
        return self.store_cluster_labels(
            chunk_ids=chunk_ids,
            labels=result.labels,
            probabilities=result.probabilities,
            cluster_names=result.cluster_names,
            topics=result.topics
        )

    def load_clustering_results(self, limit: int = 5000) -> Optional[ClusteringResult]:
        """
        Load clustering results from ChromaDB.
        
        Args:
            limit: Maximum number of documents to load to reconstruct state.
            
        Returns:
            ClusteringResult object if clusters found, else None
        """
        self.logger.info("Attempting to load clustering results from DB...")
        
        # Get data with cluster info
        # We need to filter for items that actually have a cluster_id
        # But Chroma doesn't support "IS NOT NULL" easily in where clause for metadata fields in all versions?
        # Let's just fetch a sample. If the user clicked "Save to DB", hopefully many chunks have it.
        
        # Try to find at least one chunk with cluster_id to verify existence
        # Check if we have any clusters
        # A quick check: count items with cluster_id != null ? 
        # Easier to just fetch.
        
        sample = self.collection.get(
            limit=limit,
            include=["metadatas", "embeddings", "documents"]
        )
        
        if not sample or not sample.get("ids"):
            self.logger.info("No data found in collection.")
            return None
            
        metadatas = sample.get("metadatas", [])
        ids = sample.get("ids", [])
        embeddings = sample.get("embeddings", [])
        
        if not metadatas:
            return None
            
        # Reconstruct state
        labels_list = []
        probs_list = []
        valid_indices = []
        
        topics: Dict[int, List[Tuple[str, float]]] = {}
        cluster_names: Dict[int, str] = {}
        
        found_clusters = False
        
        for i, meta in enumerate(metadatas):
            if meta and "cluster_id" in meta:
                cid = int(meta["cluster_id"])
                prob = float(meta.get("cluster_probability", 1.0))
                
                labels_list.append(cid)
                probs_list.append(prob)
                valid_indices.append(i)
                found_clusters = True
                
                # Reconstruct topic info
                if cid not in cluster_names and "cluster_topic" in meta:
                    cluster_names[cid] = meta["cluster_topic"]
                    
                if cid not in topics and "cluster_keywords" in meta:
                    # Parse back keywords
                    kws = meta["cluster_keywords"].split(", ")
                    # We lost the scores, so just assign dummy scores or descending
                    topics[cid] = [(k, 1.0) for k in kws]
                    
            else:
                # If we are loading partial state, what do we do with unclustered items?
                # Usually noise or just not part of the set we saved.
                # For visualization consistency, we should probably output arrays matching the FETCHED data size
                # OR we only return the clustered subset.
                # The ClusteringResult usually expects arrays matching the input embeddings.
                
                # Let's assume non-clustered items are outliers (-1) or just unassigned.
                # If we want to fully restore the "Cluster View", we usually run it on N items.
                labels_list.append(-1)
                probs_list.append(0.0)
        
        if not found_clusters:
            self.logger.info("No clustering metadata found in loaded sample.")
            return None
            
        # Convert to numpy
        labels = np.array(labels_list)
        probabilities = np.array(probs_list)
        
        # If we have embeddings, we can reconstruct that too (useful for viz)
        reduced_embeddings = None 
        # Note: We don't save reduced embeddings in DB. The pipeline usually re-calculates or we just don't have them.
        # But the Visualization tab calculates its own reduction usually. 
        # The Clustering Pipeline RESULT might have had them.
        
        self.logger.info(f"Loaded clustering results for {len(labels)} chunks. Found {len(cluster_names)} named clusters.")
        
        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            topics=topics,
            reduced_embeddings=reduced_embeddings,
            cluster_names=cluster_names
        )

