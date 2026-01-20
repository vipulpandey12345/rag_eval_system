#!/usr/bin/env python3
"""
Query utility for the FAISS vector database.

Usage:
    python query_faiss.py "your search query"
    python query_faiss.py --interactive
"""

import argparse
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
METADATA_PATH = DATA_DIR / "article_metadata.pkl"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class FAISSSearcher:
    """Search interface for the FAISS vector database."""

    def __init__(self):
        self.index = None
        self.metadata = []
        self.embedding_model = None

    def load(self):
        """Load the FAISS index and metadata."""
        if not FAISS_INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {FAISS_INDEX_PATH}. "
                "Run ingest_articles_to_faiss.py first."
            )

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_INDEX_PATH))

        print("Loading metadata...")
        with open(METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"Loaded {self.index.ntotal} vectors")

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        """Search for similar documents."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(distances[0][i])))

        return results


def format_result(rank: int, metadata: dict, distance: float) -> str:
    """Format a search result for display."""
    lines = [
        f"\n{'='*60}",
        f"Rank {rank} (distance: {distance:.4f})",
        f"{'='*60}",
        f"Title: {metadata.get('title', 'N/A')}",
        f"Source: {metadata.get('source', 'N/A')}",
        f"URL: {metadata.get('url', 'N/A')}",
        f"\nSummary: {metadata.get('summary', 'N/A')}",
        f"\nText Preview: {metadata.get('text_preview', 'N/A')[:300]}...",
    ]
    return '\n'.join(lines)


def interactive_mode(searcher: FAISSSearcher):
    """Run interactive search mode."""
    print("\n" + "="*60)
    print("Interactive FAISS Search")
    print("Type 'quit' or 'exit' to stop")
    print("="*60)

    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not query:
                continue

            results = searcher.search(query, k=5)

            if not results:
                print("No results found.")
                continue

            print(f"\nFound {len(results)} results for: '{query}'")
            for i, (metadata, distance) in enumerate(results, 1):
                print(format_result(i, metadata, distance))

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="Query the FAISS vector database")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    parser.add_argument("--top-k", "-k", type=int, default=5,
                       help="Number of results to return (default: 5)")

    args = parser.parse_args()

    # Initialize searcher
    searcher = FAISSSearcher()
    searcher.load()

    if args.interactive:
        interactive_mode(searcher)
    elif args.query:
        results = searcher.search(args.query, k=args.top_k)

        if not results:
            print("No results found.")
            return

        print(f"\nSearch results for: '{args.query}'")
        for i, (metadata, distance) in enumerate(results, 1):
            print(format_result(i, metadata, distance))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
