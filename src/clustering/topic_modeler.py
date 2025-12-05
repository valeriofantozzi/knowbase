"""
Topic Modeler Module

Implements Class-based TF-IDF (c-TF-IDF) for extracting topics from clusters.
Based on the BERTopic architecture.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from ..utils.logger import get_logger

class TopicModeler:
    """
    Topic Modeler using Class-based TF-IDF (c-TF-IDF).
    
    Generates representative keywords for each cluster by treating all documents
    in a cluster as a single class.
    """
    
    def __init__(self, n_gram_range: Tuple[int, int] = (1, 1), stop_words: str = "english"):
        """
        Initialize TopicModeler.
        
        Args:
            n_gram_range: The lower and upper boundary of the range of n-values for different n-grams.
            stop_words: Language for stop words or list of stop words.
        """
        self.logger = get_logger(__name__)
        self.n_gram_range = n_gram_range
        self.stop_words = stop_words
        
    def extract_topics(
        self, 
        documents: List[str], 
        cluster_labels: np.ndarray, 
        top_k: int = 10
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract topics (keywords) for each cluster.
        
        Args:
            documents: List of document texts
            cluster_labels: Cluster label for each document
            top_k: Number of keywords to return per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of (keyword, score) tuples.
        """
        if len(documents) != len(cluster_labels):
            raise ValueError("Documents and labels must have same length")
            
        # Group documents by cluster
        docs_per_class = self._group_documents(documents, cluster_labels)
        
        # We need at least one cluster that is not an outlier to proceed
        valid_clusters = [c for c in docs_per_class.keys() if c != -1]
        
        if not valid_clusters:
            return {}

        # Prepare class contents
        class_contents = []
        cluster_ids = []
        
        # Sort to ensure consistent order, put outliers last or handle separately?
        # Typically we model outliers too if we want to know what they are about, 
        # but c-TF-IDF is usually for clusters. Let's include all.
        for cluster_id in sorted(docs_per_class.keys()):
            content = " ".join(docs_per_class[cluster_id])
            class_contents.append(content)
            cluster_ids.append(cluster_id)
            
        # Calculate c-TF-IDF
        try:
            # 1. Count Vectorizer (TF per class)
            count = CountVectorizer(ngram_range=self.n_gram_range, stop_words=self.stop_words).fit(class_contents)
            tfs = count.transform(class_contents)
            words = count.get_feature_names_out()
            
            # 2. Class-based TF-IDF calculation
            ctfidf = self._calculate_ctfidf(tfs)
            
            # 3. Extract top keywords per cluster
            topics = {}
            for i, cluster_id in enumerate(cluster_ids):
                # Get indices of top k scores
                indices = ctfidf[i].toarray()[0].argsort()[-top_k:][::-1]
                
                # Get words and scores
                keywords = [(words[idx], float(ctfidf[i, idx])) for idx in indices]
                topics[cluster_id] = keywords
                
            return topics
            
        except Exception as e:
            self.logger.error(f"Topic extraction failed: {e}")
            return {}
            
    def _group_documents(self, documents: List[str], labels: np.ndarray) -> Dict[int, List[str]]:
        """Group documents by their cluster label."""
        groups = {}
        for doc, label in zip(documents, labels):
            label = int(label)
            if label not in groups:
                groups[label] = []
            groups[label].append(doc)
        return groups
        
    def _calculate_ctfidf(self, tf_matrix):
        """
        Calculate Class-based TF-IDF.
        
        c-TF-IDF = tf_class * log(1 + avg_nr_samples / freq_global)
        """
        # Global frequency of words across all classes
        global_tf = np.sum(tf_matrix, axis=0)
        
        # Total number of documents (or in this case, classes)
        # Actually in c-TF-IDF formulation:
        # m = total number of documents (sum of all tf) ??? 
        # No, standard formula:
        # IDF = log(1 + (A / F))
        # A = average number of words per class
        # F = frequency of word across all classes
        
        # Average number of words per class
        avg_nr_samples = int(np.sum(tf_matrix) / tf_matrix.shape[0])
        
        # Inverse Document Frequency logic adapted for classes
        # Add epsilon to indices to avoid division by zero if necessary, though count vectorizer handles sparse
        idf = np.log((avg_nr_samples / (global_tf + 1)) + 1)
        
        # Compute c-TF-IDF
        ctfidf = tf_matrix.multiply(idf)
        
        return ctfidf.tocsr()
