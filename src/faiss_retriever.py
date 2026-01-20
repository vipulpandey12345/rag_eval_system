"""
FAISS-based retriever for RAG experiments.

This module provides a unified interface for retrieving documents from the FAISS
vector database that was populated by the article ingestion pipeline.
"""
import pickle
from typing import List, Dict, Tuple, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from config import (
    FAISS_INDEX_PATH,
    FAISS_METADATA_PATH,
    EMBEDDING_MODEL,
    SPARSE_WEIGHT,
    DENSE_WEIGHT
)
from metrics import tokenize


class FAISSRetriever:
    """
    Retriever that uses FAISS vector database for dense retrieval
    and optionally combines with BM25 for hybrid search.
    """

    def __init__(
        self,
        sparse_weight: float = SPARSE_WEIGHT,
        dense_weight: float = DENSE_WEIGHT
    ):
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self.encoder: Optional[SentenceTransformer] = None
        self.bm25: Optional[BM25Okapi] = None
        self._loaded = False

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._loaded:
            return True

        if not FAISS_INDEX_PATH.exists():
            print(f"FAISS index not found at {FAISS_INDEX_PATH}")
            print("Run: python scripts/ingest_articles_to_faiss.py")
            return False

        if not FAISS_METADATA_PATH.exists():
            print(f"Metadata not found at {FAISS_METADATA_PATH}")
            return False

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("Loading metadata...")
        with open(FAISS_METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)

        print("Loading embedding model...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # Build BM25 index for hybrid search
        print("Building BM25 index for hybrid search...")
        doc_texts = [m.get('full_text', m.get('text_preview', '')) for m in self.metadata]
        tokenized_docs = [tokenize(text) for text in doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)

        print(f"Loaded {self.index.ntotal} vectors from FAISS index")
        self._loaded = True
        return True

    def is_loaded(self) -> bool:
        """Check if the retriever is loaded."""
        return self._loaded

    def dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Dense embedding-based search using FAISS.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (index, score) tuples
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # FAISS returns L2 distances, convert to similarity scores
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                # Convert L2 distance to similarity (smaller distance = higher similarity)
                # Using 1 / (1 + distance) for normalization
                similarity = 1 / (1 + distances[0][i])
                results.append((int(idx), float(similarity)))

        return results

    def sparse_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Sparse BM25-based search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (index, score) tuples
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def hybrid_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float, Dict]]:
        """
        Hybrid search combining dense and sparse results.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (index, combined_score, metadata) tuples
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        # Get more candidates from each method
        candidate_k = min(top_k * 3, len(self.metadata))

        dense_results = self.dense_search(query, candidate_k)
        sparse_results = self.sparse_search(query, candidate_k)

        # Normalize scores to [0, 1] range
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}

        max_dense = max(dense_scores.values()) if dense_scores else 1.0
        max_sparse = max(sparse_scores.values()) if sparse_scores else 1.0

        # Combine scores
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined = []

        for idx in all_indices:
            d_score = dense_scores.get(idx, 0) / max_dense if max_dense > 0 else 0
            s_score = sparse_scores.get(idx, 0) / max_sparse if max_sparse > 0 else 0

            combined_score = (
                self.dense_weight * d_score +
                self.sparse_weight * s_score
            )

            combined.append((idx, combined_score, {
                "dense_score": d_score,
                "sparse_score": s_score
            }))

        # Sort by combined score and return top-k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def get_document(self, idx: int) -> Optional[Dict]:
        """Get document metadata by index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None

    def retrieve(
        self,
        query: str,
        top_k: int,
        method: str = "hybrid"
    ) -> Tuple[List[str], List[Dict], str]:
        """
        Retrieve documents and return IDs, full documents, and combined context.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            method: "hybrid", "dense", or "sparse"

        Returns:
            Tuple of (doc_ids, documents, combined_context)
        """
        if not self._loaded:
            raise RuntimeError("Retriever not loaded. Call load() first.")

        if method == "hybrid":
            results = self.hybrid_search(query, top_k)
        elif method == "dense":
            results = [(idx, score, {"dense_score": score, "sparse_score": 0})
                      for idx, score in self.dense_search(query, top_k)]
        else:  # sparse
            results = [(idx, score, {"dense_score": 0, "sparse_score": score})
                      for idx, score in self.sparse_search(query, top_k)]

        retrieved_ids = []
        retrieved_docs = []
        contexts = []

        for idx, score, meta in results:
            doc = self.metadata[idx]
            # Use article URL as the document ID
            retrieved_ids.append(doc.get('article_id', doc.get('url', str(idx))))
            retrieved_docs.append({
                **doc,
                "retrieval_score": score,
                "retrieval_meta": meta
            })
            # Use full text for context, fall back to text_preview
            context_text = doc.get('full_text', doc.get('text_preview', ''))
            contexts.append(context_text)

        combined_context = "\n\n---\n\n".join(contexts)
        return retrieved_ids, retrieved_docs, combined_context


# Singleton instance for easy reuse
_retriever_instance: Optional[FAISSRetriever] = None


def get_faiss_retriever(
    sparse_weight: float = SPARSE_WEIGHT,
    dense_weight: float = DENSE_WEIGHT
) -> FAISSRetriever:
    """
    Get a shared FAISS retriever instance.

    Args:
        sparse_weight: Weight for sparse retrieval in hybrid search
        dense_weight: Weight for dense retrieval in hybrid search

    Returns:
        FAISSRetriever instance (loaded if possible)
    """
    global _retriever_instance

    if _retriever_instance is None:
        _retriever_instance = FAISSRetriever(
            sparse_weight=sparse_weight,
            dense_weight=dense_weight
        )
        _retriever_instance.load()

    return _retriever_instance
