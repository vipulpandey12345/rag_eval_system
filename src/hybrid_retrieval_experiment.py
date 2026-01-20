"""
Hybrid Retrieval Experiment

This experiment combines dense (embedding-based) and sparse (BM25) retrieval
strategies using the FAISS vector database populated with scraped article content.
"""
import json
from typing import Dict
from datetime import datetime

import anthropic

from config import (
    GOLDEN_DATASET_PATH,
    RESULTS_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
    SPARSE_WEIGHT,
    DENSE_WEIGHT,
    DEFAULT_TOP_K,
    ANTHROPIC_API_KEY
)
from metrics import compute_all_metrics, aggregate_metrics
from faiss_retriever import FAISSRetriever


def generate_answer(
    client: anthropic.Anthropic,
    question: str,
    context: str,
    model: str = LLM_MODEL
) -> str:
    """Generate answer using Claude with retrieved context."""
    prompt = f"""Based on the following context, answer the question concisely and accurately.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[
                {"role": "user", "content": prompt}
            ],
            system="You are a helpful assistant that answers questions based on provided context."
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error: {str(e)}"


def run_hybrid_experiment(
    top_k: int = DEFAULT_TOP_K,
    sparse_weight: float = SPARSE_WEIGHT,
    dense_weight: float = DENSE_WEIGHT,
    use_llm: bool = True
) -> Dict:
    """
    Run the hybrid retrieval experiment using FAISS vector database.

    Args:
        top_k: Number of documents to retrieve
        sparse_weight: Weight for sparse retrieval
        dense_weight: Weight for dense retrieval
        use_llm: Whether to generate answers with LLM

    Returns:
        Dictionary containing experiment results
    """
    print("=" * 60)
    print("HYBRID RETRIEVAL EXPERIMENT (FAISS)")
    print("=" * 60)
    print(f"Top-K: {top_k}")
    print(f"Sparse Weight: {sparse_weight}, Dense Weight: {dense_weight}")
    print()

    # Load golden dataset (questions and ground truth)
    print("Loading golden dataset...")
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        dataset = [json.loads(line) for line in f]

    print(f"Loaded {len(dataset)} questions")

    # Initialize FAISS retriever
    print("\nInitializing FAISS retriever...")
    retriever = FAISSRetriever(
        sparse_weight=sparse_weight,
        dense_weight=dense_weight
    )

    if not retriever.load():
        print("\nERROR: Could not load FAISS index.")
        print("Please run the ingestion script first:")
        print("  python scripts/ingest_articles_to_faiss.py")
        return None

    # Initialize Anthropic client if using LLM
    client = None
    if use_llm and ANTHROPIC_API_KEY:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif use_llm:
        print("Warning: ANTHROPIC_API_KEY not set, skipping LLM generation")
        use_llm = False

    # Run experiment
    results = []
    all_metrics = []

    for i, item in enumerate(dataset):
        print(f"\nProcessing {i+1}/{len(dataset)}: {item['question'][:50]}...")

        question = item['question']
        ground_truth = item['answer']
        relevant_ids = [a['article_id'] for a in item.get('source_articles', [])]

        # Retrieve documents from FAISS
        retrieved_ids, retrieved_docs, context = retriever.retrieve(
            question, top_k, method="hybrid"
        )

        # Generate answer
        if use_llm:
            generated_answer = generate_answer(client, question, context)
        else:
            # Use retrieved context as a simple baseline
            generated_answer = context[:500]

        # Compute metrics
        metrics = compute_all_metrics(
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=relevant_ids,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth,
            source_context=context,
            k_values=[1, 3, 5]
        )

        # Add token_overlap for UI compatibility
        metrics['token_overlap'] = metrics.get('token_f1', 0)

        # Store result
        result = {
            "id": item['id'],
            "question": question,
            "difficulty": item.get('difficulty', 'unknown'),
            "question_type": item.get('question_type', 'unknown'),
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_doc_ids": retrieved_ids[:top_k],
            "relevant_doc_ids": relevant_ids,
            "metrics": metrics,
            "retrieval_method": "hybrid",
            "top_k": top_k
        }
        results.append(result)
        all_metrics.append(metrics)

    # Aggregate metrics
    aggregated = aggregate_metrics(all_metrics)

    # Aggregate by difficulty
    metrics_by_difficulty = {}
    for diff in ['easy', 'medium', 'hard']:
        diff_metrics = [r['metrics'] for r in results if r['difficulty'] == diff]
        if diff_metrics:
            metrics_by_difficulty[diff] = aggregate_metrics(diff_metrics)

    # Aggregate by question type
    metrics_by_type = {}
    for qtype in ['factual', 'conceptual', 'procedural']:
        type_metrics = [r['metrics'] for r in results if r['question_type'] == qtype]
        if type_metrics:
            metrics_by_type[qtype] = aggregate_metrics(type_metrics)

    # Final output
    experiment_output = {
        "experiment_name": "hybrid_retrieval",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": top_k,
            "sparse_weight": sparse_weight,
            "dense_weight": dense_weight,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL if use_llm else None,
            "use_llm": use_llm,
            "retrieval_source": "faiss_vector_db"
        },
        "overall_metrics": aggregated,
        "metrics_by_difficulty": metrics_by_difficulty,
        "metrics_by_question_type": metrics_by_type,
        "detailed_results": results,
        "num_questions": len(results)
    }

    # Save results
    output_path = RESULTS_DIR / "hybrid_retrieval_results.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_output, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")
    print("\nOverall Metrics:")
    for metric, values in aggregated.items():
        print(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})")

    return experiment_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run hybrid retrieval experiment")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                       help="Number of documents to retrieve")
    parser.add_argument("--sparse-weight", type=float, default=SPARSE_WEIGHT,
                       help="Weight for sparse retrieval")
    parser.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT,
                       help="Weight for dense retrieval")
    parser.add_argument("--no-llm", action="store_true",
                       help="Skip LLM generation")

    args = parser.parse_args()

    run_hybrid_experiment(
        top_k=args.top_k,
        sparse_weight=args.sparse_weight,
        dense_weight=args.dense_weight,
        use_llm=not args.no_llm
    )
