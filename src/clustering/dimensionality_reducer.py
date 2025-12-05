"""
Dimensionality Reducer Module

Wraps UMAP and PCA for consistent usage in the clustering pipeline.
"""

from typing import Optional, Tuple, Literal
import numpy as np
from ..utils.logger import get_logger

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    umap = None

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class DimensionalityReducer:
    """
    Wrapper for dimensionality reduction algorithms.
    
    Supports:
    - UMAP (Uniform Manifold Approximation and Projection) - Recommended
    - PCA (Principal Component Analysis) - Fallback
    """
    
    def __init__(
        self,
        method: Literal["umap", "pca"] = "umap",
        n_components: int = 5,
        n_neighbors: int = 15,
        min_dist: float = 0.0,
        metric: str = "cosine",
        random_state: int = 42
    ):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: 'umap' or 'pca'
            n_components: Number of dimensions to reduce to
            n_neighbors: UMAP parameter - number of neighbors
            min_dist: UMAP parameter - minimum distance
            metric: Distance metric for UMAP ('cosine', 'euclidean')
            random_state: Random seed for reproducibility
        """
        self.logger = get_logger(__name__)
        self.method = method
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.reducer = None
        
        if method == "umap" and not UMAP_AVAILABLE:
            self.logger.warning("UMAP not available, falling back to PCA. Install umap-learn to use UMAP.")
            self.method = "pca"
            
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn not available. Dimensionality reduction will fail.")

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit reducer and transform embeddings.
        
        Args:
            embeddings: High-dimensional embeddings (n_samples, n_features)
            
        Returns:
            Reduced embeddings (n_samples, n_components)
        """
        if len(embeddings) == 0:
            return np.array([])
            
        # If n_components >= n_features, return original
        if self.n_components >= embeddings.shape[1]:
            self.logger.info("Target dimensions >= input dimensions, skipping reduction.")
            return embeddings
            
        # Adjust n_neighbors if we have few samples
        n_samples = embeddings.shape[0]
        adjusted_n_neighbors = min(self.n_neighbors, n_samples - 1)
        if adjusted_n_neighbors < 2:
            adjusted_n_neighbors = 2

        try:
            if self.method == "umap" and UMAP_AVAILABLE:
                self.reducer = umap.UMAP(
                    n_components=self.n_components,
                    n_neighbors=adjusted_n_neighbors,
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=self.random_state
                )
                self.logger.info(f"Reducing dimensions with UMAP to {self.n_components}")
                return self.reducer.fit_transform(embeddings)

            elif SKLEARN_AVAILABLE:
                self.reducer = PCA(
                    n_components=self.n_components,
                    random_state=self.random_state
                )
                self.logger.info(f"Reducing dimensions with PCA to {self.n_components}")
                return self.reducer.fit_transform(embeddings)
            else:
                raise ImportError("No dimensionality reduction libraries available.")
                
        except Exception as e:
            self.logger.error(f"Dimensionality reduction failed: {e}")
            # Return original embeddings as fallback
            return embeddings

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using fitted reducer.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            Reduced embeddings
        """
        if self.reducer is None:
            raise ValueError("Reducer has not been fitted.")
            
        return self.reducer.transform(embeddings)
