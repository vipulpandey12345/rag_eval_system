"""Configuration for RAG evaluation system."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"

# FAISS vector database paths
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
FAISS_METADATA_PATH = DATA_DIR / "article_metadata.pkl"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model for dense embeddings
LLM_MODEL = "claude-sonnet-4-20250514"  # Anthropic model for generation

# Retrieval settings
DEFAULT_TOP_K = 3
SPARSE_WEIGHT = 0.3  # Weight for sparse (BM25) in hybrid search
DENSE_WEIGHT = 0.7   # Weight for dense (embedding) in hybrid search

# Context window experiment settings
CONTEXT_CHUNK_OPTIONS = [3]  # Only use top-k=3

# Anthropic API Key (set via environment variable)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
