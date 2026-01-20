"""
Experiment Runner Script

This script runs both experiments:
1. Hybrid Retrieval Experiment (dense + sparse)
2. Context Window Experiment (configurable top-k chunks)

Results are saved to data/results/ for display in the Streamlit dashboard.
"""
import sys
import os
import argparse
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import RESULTS_DIR
from hybrid_retrieval_experiment import run_hybrid_experiment
from context_window_experiment import run_context_window_experiment


def run_all_experiments(
    use_llm: bool = True,
    hybrid_top_k: int = 5,
    context_chunks: list = None
):
    """
    Run all experiments and save results.

    Args:
        use_llm: Whether to use LLM for answer generation
        hybrid_top_k: Top-K for hybrid retrieval experiment
        context_chunks: List of chunk counts for context window experiment
    """
    print("=" * 70)
    print("RAG EVALUATION SYSTEM - RUNNING ALL EXPERIMENTS")
    print("=" * 70)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Results will be saved to: {RESULTS_DIR}")
    print()

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run Hybrid Retrieval Experiment
    print("\n" + "#" * 70)
    print("# EXPERIMENT 1: HYBRID RETRIEVAL")
    print("#" * 70 + "\n")

    try:
        hybrid_results = run_hybrid_experiment(
            top_k=hybrid_top_k,
            use_llm=use_llm
        )
        print("\nHybrid Retrieval Experiment completed successfully!")
    except Exception as e:
        print(f"\nError in Hybrid Retrieval Experiment: {e}")
        hybrid_results = None

    # Run Context Window Experiment
    print("\n" + "#" * 70)
    print("# EXPERIMENT 2: CONTEXT WINDOW")
    print("#" * 70 + "\n")

    try:
        context_results = run_context_window_experiment(
            chunk_counts=context_chunks,
            use_llm=use_llm
        )
        print("\nContext Window Experiment completed successfully!")
    except Exception as e:
        print(f"\nError in Context Window Experiment: {e}")
        context_results = None

    # Summary
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now().isoformat()}")
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - hybrid_retrieval_results.json")
    print("  - context_window_results.json")
    print("\nYou can now view results in the Streamlit dashboard:")
    print("  streamlit run ui/app.py")

    return {
        "hybrid_retrieval": hybrid_results,
        "context_window": context_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RAG evaluation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments with LLM generation
  python run_experiments.py

  # Run without LLM (faster, uses context as answer)
  python run_experiments.py --no-llm

  # Custom chunk counts for context window experiment
  python run_experiments.py --chunks 3 5 10

  # Custom top-k for hybrid retrieval
  python run_experiments.py --hybrid-top-k 10
        """
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip LLM generation (faster, uses retrieved context as baseline)"
    )

    parser.add_argument(
        "--hybrid-top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve in hybrid experiment (default: 5)"
    )

    parser.add_argument(
        "--chunks",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10],
        help="Chunk counts to test in context window experiment (default: 3 5 7 10)"
    )

    args = parser.parse_args()

    run_all_experiments(
        use_llm=not args.no_llm,
        hybrid_top_k=args.hybrid_top_k,
        context_chunks=args.chunks
    )
