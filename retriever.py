# ============================================================
# Part B: Retrieval Module
# ============================================================

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

class Retriever:
    """RAG retrieval module using cosine similarity."""
    
    def __init__(self, train_embeddings: np.ndarray,
                 train_texts: list, train_sentiments: list,
                 train_ratings: list, train_categories: list):
        """
        Initialize retriever with training embeddings and metadata.
        
        Args:
            train_embeddings: (N, d_model) array of CLS embeddings
            train_texts: List of review texts
            train_sentiments: List of sentiment labels
            train_ratings: List of ratings
            train_categories: List of product categories
        """
        self.train_embeddings = self.l2_normalize(train_embeddings)
        self.train_texts = train_texts
        self.train_sentiments = train_sentiments
        self.train_ratings = train_ratings
        self.train_categories = train_categories
    
    @staticmethod
    def l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """L2-normalize rows for cosine similarity via dot product."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        return matrix / norms
    
    def retrieve_top_k(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Retrieve top-k most similar training reviews.
        
        Args:
            query_embedding: (1, d_model) normalized query embedding
            k: Number of results to return
        
        Returns:
            List of dicts with retrieved reviews and metadata
        """
        query_embedding = self.l2_normalize(query_embedding)
        similarities = np.dot(self.train_embeddings, query_embedding.T).squeeze()
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            results.append({
                "rank": rank,
                "similarity": float(similarities[idx]),
                "text": self.train_texts[idx],
                "sentiment": self.train_sentiments[idx],
                "rating": self.train_ratings[idx],
                "category": self.train_categories[idx],
            })
        
        return results
    
    def retrieve_batch(self, query_embeddings: np.ndarray, k: int = 5) -> list[list[dict]]:
        """
        Retrieve top-k for multiple queries.
        
        Args:
            query_embeddings: (B, d_model) batch of embeddings
            k: Number of results per query
        
        Returns:
            List of lists (one list per query)
        """
        results = []
        for i in range(len(query_embeddings)):
            query_emb = query_embeddings[i:i+1]
            results.append(self.retrieve_top_k(query_emb, k))
        return results

class DenseRetriever:
    """Enhanced retriever with precomputed similarities."""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.precomputed_sims = None
    
    def precompute_similarities(self, query_embeddings: np.ndarray):
        """Precompute all similarities for efficiency."""
        query_normalized = self.retriever.l2_normalize(query_embeddings)
        self.precomputed_sims = np.dot(
            query_normalized, self.retriever.train_embeddings.T
        )  # (Q, N)
    
    def retrieve_top_k_fast(self, query_idx: int, k: int = 5) -> list[dict]:
        """Retrieve using precomputed similarities."""
        if self.precomputed_sims is None:
            raise RuntimeError("Must call precompute_similarities first")
        
        similarities = self.precomputed_sims[query_idx]
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_k_indices, 1):
            results.append({
                "rank": rank,
                "similarity": float(similarities[idx]),
                "text": self.retriever.train_texts[idx],
                "sentiment": self.retriever.train_sentiments[idx],
                "rating": self.retriever.train_ratings[idx],
                "category": self.retriever.train_categories[idx],
            })
        
        return results
