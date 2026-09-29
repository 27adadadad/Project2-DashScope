"""确定性的检索质量指标。"""

from statistics import mean

METRIC_NAMES = ("recall_at_k", "hit_at_k", "all_evidence_at_k", "mrr", "id_precision_at_k")


def calculate_retrieval_metrics(retrieved_ids, reference_ids, *, k):
    if k <= 0:
        raise ValueError("k must be positive")
    ranked = list(retrieved_ids[:k])
    reference = set(reference_ids)
    matches = [position for position, item in enumerate(ranked, 1) if item in reference]
    found = reference.intersection(ranked)
    return {
        "recall_at_k": len(found) / len(reference) if reference else None,
        "hit_at_k": float(bool(found)) if reference else None,
        "all_evidence_at_k": float(found == reference) if reference else None,
        "mrr": 1 / matches[0] if matches else 0.0,
        "id_precision_at_k": len(found) / k,
    }


def aggregate_metrics(rows):
    return {key: mean(row[key] for row in rows) if rows else None for key in METRIC_NAMES}
