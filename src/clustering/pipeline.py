"""
Clustering Pipeline Module

Orchestrates the complete clustering workflow:
1. Dimensionality Reduction (UMAP/PCA)
2. Clustering (HDBSCAN)
3. Topic Modeling (c-TF-IDF)
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .dimensionality_reducer import DimensionalityReducer
from .hdbscan_clusterer import HDBSCANClusterer
from .topic_modeler import TopicModeler
from ..utils.logger import get_logger

@dataclass
class ClusteringResult:
    """Container for clustering results."""
    labels: np.ndarray
    probabilities: np.ndarray
    topics: Dict[int, List[Tuple[str, float]]]
    reduced_embeddings: Optional[np.ndarray] = None
    cluster_names: Optional[Dict[int, str]] = None

class ClusteringPipeline:
    """
    Unified pipeline for clustering documents.
    """
    
    def __init__(
        self,
        # Clustering params
        min_cluster_size: int = 15,
        min_samples: int = 5,
        metric: str = "cosine",
        # Reduction params
        use_reduction: bool = True,
        reduction_components: int = 10,  # Optimally 5-15 for HDBSCAN
        reduction_method: str = "umap",
        # Topic params
        extract_topics: bool = True,
        n_gram_range: Tuple[int, int] = (1, 2)
    ):
        self.logger = get_logger(__name__)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.use_reduction = use_reduction
        
        # Initialize components
        self.reducer = DimensionalityReducer(
            method=reduction_method,
            n_components=reduction_components,
            metric=metric
        ) if use_reduction else None
        
        self.clusterer = HDBSCANClusterer(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric if not use_reduction else "euclidean" 
            # Note: HDBSCAN works best with euclidean on UMAP reduced data
        )
        
        self.topic_modeler = TopicModeler(n_gram_range=n_gram_range) if extract_topics else None
        
    def fit_predict(
        self,
        embeddings: np.ndarray,
        documents: Optional[List[str]] = None
    ) -> ClusteringResult:
        """
        Run the full pipeline.
        
        Args:
            embeddings: Document embeddings (n_samples, n_features)
            documents: List of document texts (needed for topic modeling)
            
        Returns:
            ClusteringResult object
        """
        self.logger.info(f"Starting clustering pipeline on {len(embeddings)} documents...")
        
        # 1. Dimensionality Reduction
        working_embeddings = embeddings
        reduced_embeddings = None
        
        if self.use_reduction and self.reducer:
            self.logger.info("Step 1: Dimensionality Reduction")
            reduced_embeddings = self.reducer.fit_transform(embeddings)
            working_embeddings = reduced_embeddings
            
        # 2. Clustering
        self.logger.info("Step 2: Clustering with HDBSCAN")
        # Note: If we used UMAP, the metric is usually handled by UMAP's projection 
        # and we cluster in Euclidean space. If raw cosine, we trust HDBSCANClusterer 
        # to handle normalization.
        
        # Force the clusterer to see the data as it is. 
        # If reduced, we are likely in Euclidean space now.
        if self.use_reduction:
            # We temporarily override metric to euclidean for reduced space if needed, 
            # but HDBSCANClusterer handles 'cosine' by normalizing.
            # UMAP output is not necessarily normalized for cosine, but Euclidean is standard.
            # Let's trust the Clusterer wrapper logic or adapt it.
            # Ideally, UMAP preserves local structure such that Euclidean on output is good.
            pass

        labels, probabilities = self.clusterer.fit(working_embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        self.logger.info(f"Found {n_clusters} clusters")
        
        # 3. Topic Modeling
        topics = {}
        cluster_names = {}
        
        if self.topic_modeler and documents:
            if len(documents) != len(embeddings):
                self.logger.warning("Document count mismatch with embeddings. Skipping topic modeling.")
            else:
                self.logger.info("Step 3: Extracting Topics")
                topics = self.topic_modeler.extract_topics(documents, labels)
                
                # Generate simple names from top keywords
                for cid, keywords in topics.items():
                    # Take top 3 keywords join with underscore or space
                    if keywords:
                        name = "_".join([k[0] for k in keywords[:3]])
                        cluster_names[cid] = name
                    else:
                        cluster_names[cid] = f"Cluster {cid}"
                        
                if -1 in topics:
                    cluster_names[-1] = "Outliers"
        
        return ClusteringResult(
            labels=labels,
            probabilities=probabilities,
            topics=topics,
            reduced_embeddings=reduced_embeddings,
            cluster_names=cluster_names
        )
