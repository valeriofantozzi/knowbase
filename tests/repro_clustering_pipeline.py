
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.clustering.pipeline import ClusteringPipeline

def test_pipeline():
    print("Generating mock data...")
    # Generate 100 embeddings of dim 1024
    # Create 3 distinct clusters + noise
    n_samples = 100
    dim = 1024
    
    embeddings = np.random.rand(n_samples, dim)
    
    # Create fake structure
    embeddings[:30] += 5.0 # Cluster 1
    embeddings[30:60] -= 5.0 # Cluster 2
    # Rest is noise/cluster 3
    
    documents = [f"doc_{i} keyword_a keyword_b" for i in range(30)] + \
                [f"doc_{i} keyword_c keyword_d" for i in range(30, 60)] + \
                [f"doc_{i} keyword_e" for i in range(60, 100)]
                
    print("Initializing pipeline...")
    # Use PCA because UMAP might not be installed or slow in test env
    pipeline = ClusteringPipeline(
        min_cluster_size=5,
        min_samples=2,
        use_reduction=True,
        reduction_method="pca", 
        reduction_components=5,
        extract_topics=True
    )
    
    print("Running fit_predict...")
    result = pipeline.fit_predict(embeddings, documents)
    
    print(f"Labels found: {set(result.labels)}")
    print(f"Topics found: {len(result.topics)}")
    
    for cid, topics in result.topics.items():
        print(f"Cluster {cid}: {[t[0] for t in topics[:3]]}")
        
    print("Verification successful if clusters and topics are printed above.")

if __name__ == "__main__":
    test_pipeline()
