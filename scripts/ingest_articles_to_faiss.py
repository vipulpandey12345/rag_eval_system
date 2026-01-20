#!/usr/bin/env python3
"""
Article Ingestion Pipeline with Claude Agent and FAISS Vector Database

This script:
1. Reads article URLs from golden_dataset.jsonl
2. Scrapes each article using httpx
3. Uses Claude to extract and clean article text
4. Generates embeddings using sentence-transformers
5. Stores everything in a FAISS vector database
"""

import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import httpx
import faiss
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import anthropic
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Configuration
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DATASET_PATH = DATA_DIR / "golden_dataset.jsonl"
ARTICLES_DIR = DATA_DIR / "articles"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
METADATA_PATH = DATA_DIR / "article_metadata.pkl"

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast and good quality
EMBEDDING_DIM = 384  # Dimension for all-MiniLM-L6-v2

# Rate limiting
REQUEST_DELAY = 1.0  # Seconds between requests
CLAUDE_DELAY = 0.5   # Seconds between Claude API calls


@dataclass
class ArticleDocument:
    """Represents a processed article document."""
    article_id: str
    url: str
    title: str
    source: str
    raw_html: str
    extracted_text: str
    claude_summary: str
    embedding_id: int


class ClaudeArticleAgent:
    """Agent that uses Claude to extract and process article content."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.model = "claude-sonnet-4-20250514"

    def extract_article_text(self, html_content: str, url: str) -> dict:
        """
        Use Claude to intelligently extract article text from HTML.
        Returns dict with 'text' and 'summary' keys.
        """
        # Truncate HTML if too long (Claude has context limits)
        max_html_length = 100000
        if len(html_content) > max_html_length:
            html_content = html_content[:max_html_length] + "... [truncated]"

        prompt = f"""You are an article text extraction agent. Given the HTML content of a news article, extract:

1. The main article text (the actual content, not navigation, ads, or sidebars)
2. A brief 2-3 sentence summary of the article

URL: {url}

HTML Content:
```html
{html_content}
```

Respond in this exact JSON format:
{{
    "extracted_text": "The full article text here...",
    "summary": "A 2-3 sentence summary of the article..."
}}

If you cannot extract meaningful content, set extracted_text to an empty string and explain in the summary."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response
            response_text = response.content[0].text

            # Try to extract JSON from the response
            # Handle case where Claude might wrap in markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            return {
                "text": result.get("extracted_text", ""),
                "summary": result.get("summary", "")
            }

        except json.JSONDecodeError as e:
            print(f"  Warning: Could not parse Claude response as JSON: {e}")
            # Fall back to using the raw response as text
            return {
                "text": response_text if 'response_text' in dir() else "",
                "summary": "Failed to parse structured response"
            }
        except Exception as e:
            print(f"  Error calling Claude API: {e}")
            return {"text": "", "summary": f"Error: {str(e)}"}


class ArticleScraper:
    """Scrapes articles from URLs."""

    def __init__(self):
        self.client = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

    def fetch_article(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            response = self.client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.HTTPStatusError as e:
            print(f"  HTTP error fetching {url}: {e.response.status_code}")
            return None
        except httpx.RequestError as e:
            print(f"  Request error fetching {url}: {e}")
            return None

    def basic_text_extraction(self, html: str) -> str:
        """
        Basic fallback text extraction using BeautifulSoup.
        Used when Claude extraction fails.
        """
        soup = BeautifulSoup(html, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()

        # Get text
        text = soup.get_text(separator='\n', strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)

    def close(self):
        self.client.close()


class FAISSVectorStore:
    """Manages FAISS vector database for article embeddings."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
        self.metadata: list[dict] = []
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.astype('float32')

    def add_document(self, text: str, metadata: dict) -> int:
        """Add a document to the vector store."""
        embedding = self.generate_embedding(text)
        embedding = embedding.reshape(1, -1)  # FAISS expects 2D array

        # Add to index
        idx = self.index.ntotal
        self.index.add(embedding)

        # Store metadata
        self.metadata.append(metadata)

        return idx

    def search(self, query: str, k: int = 5) -> list[tuple[dict, float]]:
        """Search for similar documents."""
        query_embedding = self.generate_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                results.append((self.metadata[idx], float(distances[0][i])))

        return results

    def save(self, index_path: Path, metadata_path: Path):
        """Save index and metadata to disk."""
        index_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

        print(f"Saved FAISS index to {index_path}")
        print(f"Saved metadata to {metadata_path}")

    def load(self, index_path: Path, metadata_path: Path):
        """Load index and metadata from disk."""
        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")


def get_unique_article_urls() -> list[dict]:
    """Extract unique article URLs from golden dataset."""
    urls_seen = set()
    articles = []

    with open(GOLDEN_DATASET_PATH, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            for source in entry.get('source_articles', []):
                url = source.get('article_id', '')
                if url and url not in urls_seen:
                    urls_seen.add(url)
                    articles.append({
                        'url': url,
                        'title': source.get('title', ''),
                        'source': source.get('source', '')
                    })

    return articles


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("Article Ingestion Pipeline with Claude Agent")
    print("=" * 60)

    # Initialize components
    print("\n[1/5] Initializing components...")
    scraper = ArticleScraper()
    claude_agent = ClaudeArticleAgent()
    vector_store = FAISSVectorStore()

    # Get unique article URLs
    print("\n[2/5] Loading article URLs from golden dataset...")
    articles = get_unique_article_urls()
    print(f"Found {len(articles)} unique articles to process")

    # Process each article
    print("\n[3/5] Scraping and processing articles...")
    processed_count = 0
    failed_count = 0

    for i, article in enumerate(articles):
        url = article['url']
        print(f"\n[{i+1}/{len(articles)}] Processing: {article['title'][:50]}...")

        # Scrape the article
        html_content = scraper.fetch_article(url)

        if not html_content:
            print(f"  Skipping - could not fetch HTML")
            failed_count += 1
            continue

        time.sleep(REQUEST_DELAY)

        # Use Claude to extract article text
        print(f"  Extracting text with Claude agent...")
        extraction = claude_agent.extract_article_text(html_content, url)

        extracted_text = extraction.get('text', '')
        summary = extraction.get('summary', '')

        # Fall back to basic extraction if Claude fails
        if not extracted_text:
            print(f"  Claude extraction empty, using basic extraction...")
            extracted_text = scraper.basic_text_extraction(html_content)

        if not extracted_text:
            print(f"  Skipping - no text extracted")
            failed_count += 1
            continue

        time.sleep(CLAUDE_DELAY)

        # Add to vector store
        metadata = {
            'article_id': url,
            'url': url,
            'title': article['title'],
            'source': article['source'],
            'summary': summary,
            'full_text': extracted_text,  # Store full text for retrieval
            'text_preview': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text
        }

        # Combine title, summary, and text for embedding
        text_for_embedding = f"{article['title']}\n\n{summary}\n\n{extracted_text}"

        embedding_id = vector_store.add_document(text_for_embedding, metadata)
        print(f"  Added to vector store (ID: {embedding_id})")

        processed_count += 1

    scraper.close()

    # Save the vector store
    print("\n[4/5] Saving FAISS index and metadata...")
    vector_store.save(FAISS_INDEX_PATH, METADATA_PATH)

    # Summary
    print("\n[5/5] Ingestion complete!")
    print("=" * 60)
    print(f"Total articles found: {len(articles)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"FAISS index size: {vector_store.index.ntotal} vectors")
    print(f"Index saved to: {FAISS_INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")
    print("=" * 60)

    # Test search
    print("\n[Bonus] Testing search functionality...")
    test_query = "Federal Reserve interest rates"
    results = vector_store.search(test_query, k=3)
    print(f"Search results for '{test_query}':")
    for i, (meta, distance) in enumerate(results):
        print(f"  {i+1}. {meta['title'][:60]}... (distance: {distance:.4f})")


if __name__ == "__main__":
    main()
