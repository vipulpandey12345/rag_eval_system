#!/usr/bin/env python3
"""
Generate a golden dataset of 100 Q&A pairs from news articles.
Each pair includes: question, answer, difficulty, question_type, source_articles
"""

import json
import random
from pathlib import Path

INPUT_FILE = Path(__file__).parent.parent / "data" / "articles" / "newsapi_backfill.jsonl"
OUTPUT_FILE = Path(__file__).parent.parent / "data" / "golden_dataset.jsonl"

DIFFICULTIES = ["easy", "medium", "hard"]
QUESTION_TYPES = ["factual", "conceptual", "procedural"]


def load_articles():
    """Load articles line by line from JSONL file."""
    articles = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def generate_qa_pairs(articles, num_pairs=100):
    """
    Generate Q&A pairs from articles.
    This creates a template structure - actual Q&A content needs manual curation
    or LLM generation based on article content.
    """
    qa_pairs = []

    for i in range(num_pairs):
        # Randomly select difficulty and question type
        difficulty = random.choice(DIFFICULTIES)
        question_type = random.choice(QUESTION_TYPES)

        # Select article(s) - for hard questions, might use multiple
        if difficulty == "hard" and random.random() > 0.5:
            selected_articles = random.sample(articles, min(2, len(articles)))
        else:
            selected_articles = [random.choice(articles)]

        qa_pair = {
            "id": i + 1,
            "difficulty": difficulty,
            "question_type": question_type,
            "question": "",  # To be filled
            "answer": "",    # To be filled with exact wording from article
            "source_articles": [
                {
                    "article_id": art["article_id"],
                    "title": art["title"],
                    "source": art["source"]
                }
                for art in selected_articles
            ],
            "article_content": [
                {
                    "title": art["title"],
                    "summary": art.get("summary", ""),
                    "content": art.get("content", "")[:1000]  # Truncate for review
                }
                for art in selected_articles
            ]
        }
        qa_pairs.append(qa_pair)

    return qa_pairs


def save_qa_pairs(qa_pairs):
    """Save Q&A pairs to JSONL file."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Saved {len(qa_pairs)} Q&A pairs to {OUTPUT_FILE}")


def main():
    print(f"Loading articles from {INPUT_FILE}...")
    articles = load_articles()
    print(f"Loaded {len(articles)} articles")

    print("Generating Q&A pair templates...")
    qa_pairs = generate_qa_pairs(articles, num_pairs=100)

    # Print distribution
    diff_counts = {d: 0 for d in DIFFICULTIES}
    type_counts = {t: 0 for t in QUESTION_TYPES}
    for pair in qa_pairs:
        diff_counts[pair["difficulty"]] += 1
        type_counts[pair["question_type"]] += 1

    print(f"\nDifficulty distribution: {diff_counts}")
    print(f"Question type distribution: {type_counts}")

    save_qa_pairs(qa_pairs)


if __name__ == "__main__":
    main()
