"""
Context Window Experiment

This experiment tests RAG performance using the FAISS vector database
with a fixed context window of k=3 documents.
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
    ANTHROPIC_API_KEY
)
from metrics import compute_all_metrics, aggregate_metrics
from faiss_retriever import FAISSRetriever

# Fixed top-k for this experiment
TOP_K = 3


def generate_answer(
    client: anthropic.Anthropic,
    question: str,
    context: str,
    model: str = LLM_MODEL
) -> str:
    """Generate answer using Claude with retrieved context."""
    prompt = f"""You are given {TOP_K} document chunks as context. Use them to answer the question accurately.
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
            system="You are a helpful assistant that answers questions based on provided context. Be concise and accurate."
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error: {str(e)}"


def run_context_window_experiment(
    use_llm: bool = True,
    **kwargs  # Accept but ignore extra args for compatibility
) -> Dict:
    """
    Run the context window experiment with k=3 using the FAISS vector database.

    Args:
        use_llm: Whether to generate answers with LLM

    Returns:
        Dictionary containing experiment results
    """
    print("=" * 60)
    print("CONTEXT WINDOW EXPERIMENT (FAISS)")
    print("=" * 60)
    print(f"Using top-k = {TOP_K}")
    print()

    # Load golden dataset (questions and ground truth)
    print("Loading golden dataset...")
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        dataset = [json.loads(line) for line in f]

    print(f"Loaded {len(dataset)} questions")

    # Initialize FAISS retriever
    print("\nInitializing FAISS retriever...")
    retriever = FAISSRetriever()

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
        print(f"\rProcessing {i+1}/{len(dataset)}...", end="", flush=True)

        question = item['question']
        ground_truth = item['answer']
        relevant_ids = [a['article_id'] for a in item.get('source_articles', [])]

        # Retrieve documents with k=3 using dense search
        retrieved_ids, retrieved_docs, context = retriever.retrieve(
            question, TOP_K, method="dense"
        )

        # Generate answer
        if use_llm:
            generated_answer = generate_answer(client, question, context)
        else:
            generated_answer = context[:500]

        # Compute metrics
        metrics = compute_all_metrics(
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=relevant_ids,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth,
            source_context=context,
            k_values=[1, 3]
        )

        # Add token_overlap for UI compatibility
        metrics['token_overlap'] = metrics.get('token_f1', 0)

        result = {
            "id": item['id'],
            "question": question,
            "difficulty": item.get('difficulty', 'unknown'),
            "question_type": item.get('question_type', 'unknown'),
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "retrieved_doc_ids": retrieved_ids,
            "relevant_doc_ids": relevant_ids,
            "metrics": metrics,
            "num_chunks": TOP_K
        }
        results.append(result)
        all_metrics.append(metrics)

    print()  # New line after progress

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

    # Final output - structured to match expected format in UI
    experiment_output = {
        "experiment_name": "context_window",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "top_k": TOP_K,
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL if use_llm else None,
            "use_llm": use_llm,
            "retrieval_source": "faiss_vector_db"
        },
        "results_by_chunk_count": {
            f"chunks_{TOP_K}": {
                "num_chunks": TOP_K,
                "overall_metrics": aggregated,
                "metrics_by_difficulty": metrics_by_difficulty,
                "metrics_by_question_type": metrics_by_type,
                "detailed_results": results,
                "num_questions": len(results)
            }
        }
    }

    # Save results
    output_path = RESULTS_DIR / "context_window_results.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_output, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {output_path}")

    print(f"\nMetrics (k={TOP_K}):")
    for metric, values in aggregated.items():
        print(f"  {metric}: {values['mean']:.4f} (+-{values['std']:.4f})")

    return experiment_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run context window experiment (k=3)")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM generation")

    args = parser.parse_args()

    run_context_window_experiment(use_llm=not args.no_llm)
