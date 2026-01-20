"""Evaluation metrics for RAG system."""
import re
from collections import Counter
from typing import List, Dict, Set
import numpy as np


def tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase and split on non-alphanumeric."""
    return re.findall(r'\b\w+\b', text.lower())


def hit_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Hit@k: Did we retrieve at least one correct document in top-k?

    Args:
        retrieved_doc_ids: List of retrieved document IDs in ranked order
        relevant_doc_ids: List of ground truth relevant document IDs
        k: Number of top documents to consider

    Returns:
        1.0 if at least one relevant doc is in top-k, 0.0 otherwise
    """
    if not relevant_doc_ids:
        return 1.0  # No relevant docs needed, trivially satisfied

    top_k_retrieved = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)

    return 1.0 if len(top_k_retrieved & relevant_set) > 0 else 0.0


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """
    Recall@k: What fraction of all relevant documents did we retrieve in top-k?

    Args:
        retrieved_doc_ids: List of retrieved document IDs in ranked order
        relevant_doc_ids: List of ground truth relevant document IDs
        k: Number of top documents to consider

    Returns:
        Fraction of relevant documents found in top-k (0.0 to 1.0)
    """
    if not relevant_doc_ids:
        return 1.0  # No relevant docs needed, trivially satisfied

    top_k_retrieved = set(retrieved_doc_ids[:k])
    relevant_set = set(relevant_doc_ids)

    found = len(top_k_retrieved & relevant_set)
    return found / len(relevant_set)


def token_overlap_f1(predicted_answer: str, ground_truth_answer: str) -> Dict[str, float]:
    """
    Calculate token-level precision, recall, and F1 score.

    Args:
        predicted_answer: The model's generated answer
        ground_truth_answer: The expected correct answer

    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    pred_tokens = tokenize(predicted_answer)
    truth_tokens = tokenize(ground_truth_answer)

    if not pred_tokens and not truth_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not truth_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)

    # Calculate overlap
    overlap = sum((pred_counter & truth_counter).values())

    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(truth_tokens) if truth_tokens else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4)
    }


def factual_consistency_score(
    generated_answer: str,
    source_context: str,
    ground_truth: str
) -> Dict[str, float]:
    """
    Calculate factual consistency (anti-hallucination) score.

    This uses a simple heuristic: what fraction of the generated answer's
    key terms appear in either the source context or ground truth?

    A more sophisticated approach would use NLI models or LLM-as-judge.

    Args:
        generated_answer: The model's generated answer
        source_context: The retrieved context used for generation
        ground_truth: The expected correct answer

    Returns:
        Dictionary with consistency scores
    """
    gen_tokens = set(tokenize(generated_answer))
    context_tokens = set(tokenize(source_context))
    truth_tokens = set(tokenize(ground_truth))

    # Combined reference tokens
    reference_tokens = context_tokens | truth_tokens

    if not gen_tokens:
        return {
            "context_grounding": 1.0,
            "truth_alignment": 1.0,
            "overall_consistency": 1.0
        }

    # Remove common stopwords for more meaningful comparison
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'of', 'to', 'in',
        'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between', 'under',
        'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'he', 'she', 'his', 'her', 'him', 'we', 'us', 'our', 'you', 'your'
    }

    gen_content = gen_tokens - stopwords
    context_content = context_tokens - stopwords
    truth_content = truth_tokens - stopwords
    reference_content = reference_tokens - stopwords

    if not gen_content:
        return {
            "context_grounding": 1.0,
            "truth_alignment": 1.0,
            "overall_consistency": 1.0
        }

    # Context grounding: how much of the answer is grounded in retrieved context
    context_grounding = len(gen_content & context_content) / len(gen_content)

    # Truth alignment: how much of the answer aligns with ground truth
    truth_alignment = len(gen_content & truth_content) / len(gen_content)

    # Overall consistency: grounded in either context or truth
    overall = len(gen_content & reference_content) / len(gen_content)

    return {
        "context_grounding": round(context_grounding, 4),
        "truth_alignment": round(truth_alignment, 4),
        "overall_consistency": round(overall, 4)
    }


def compute_all_metrics(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: List[str],
    generated_answer: str,
    ground_truth_answer: str,
    source_context: str,
    k_values: List[int] = [1, 3, 5]
) -> Dict:
    """
    Compute all evaluation metrics for a single query.

    Args:
        retrieved_doc_ids: List of retrieved document IDs
        relevant_doc_ids: List of ground truth relevant document IDs
        generated_answer: The model's generated answer
        ground_truth_answer: The expected correct answer
        source_context: The retrieved context used for generation
        k_values: List of k values for Hit@k and Recall@k

    Returns:
        Dictionary containing all metrics
    """
    metrics = {}

    # Retrieval metrics at different k values
    for k in k_values:
        metrics[f"hit@{k}"] = hit_at_k(retrieved_doc_ids, relevant_doc_ids, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)

    # Token overlap F1
    f1_scores = token_overlap_f1(generated_answer, ground_truth_answer)
    metrics["token_precision"] = f1_scores["precision"]
    metrics["token_recall"] = f1_scores["recall"]
    metrics["token_f1"] = f1_scores["f1"]

    # Factual consistency
    consistency = factual_consistency_score(
        generated_answer, source_context, ground_truth_answer
    )
    metrics["context_grounding"] = consistency["context_grounding"]
    metrics["truth_alignment"] = consistency["truth_alignment"]
    metrics["factual_consistency"] = consistency["overall_consistency"]

    return metrics


def aggregate_metrics(all_metrics: List[Dict]) -> Dict:
    """
    Aggregate metrics across multiple queries.

    Args:
        all_metrics: List of metric dictionaries from compute_all_metrics

    Returns:
        Dictionary with mean and std for each metric
    """
    if not all_metrics:
        return {}

    aggregated = {}
    keys = all_metrics[0].keys()

    for key in keys:
        values = [m[key] for m in all_metrics]
        aggregated[key] = {
            "mean": round(np.mean(values), 4),
            "std": round(np.std(values), 4),
            "min": round(np.min(values), 4),
            "max": round(np.max(values), 4)
        }

    return aggregated
