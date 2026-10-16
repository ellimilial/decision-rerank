"""Linear-gain nDCG and BioASQ's capped AP denominator; fixed evaluation cutoffs."""
from collections.abc import Mapping, Sequence
from math import isfinite, log2


def score_ranking(ranking: Sequence[str], qrels: Mapping[str, float]) -> dict[str, float]:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Duplicate document IDs in ranking")
    if any(not isfinite(g) or g < 0 for g in qrels.values()):
        raise ValueError("Relevance grades must be finite and nonnegative")
    positive = {doc: grade for doc, grade in qrels.items() if grade > 0}
    relevant = set(positive)
    hits = 0
    ap = rr = 0.0
    for rank, doc in enumerate(ranking[:10], 1):
        if doc in relevant:
            hits += 1
            ap += hits / rank
            if not rr:
                rr = 1 / rank
    ideal = sum(g / log2(i + 2) for i, g in enumerate(sorted(positive.values(), reverse=True)[:10]))
    dcg = sum(positive.get(doc, 0) / log2(i + 2) for i, doc in enumerate(ranking[:10]))
    return {
        "ndcg@10": dcg / ideal if ideal else 0.0,
        "bioasq_ap@10": ap / min(10, len(relevant)) if relevant else 0.0,
        "p@10": hits / 10,
        "mrr@10": rr,
        "recall@10": hits / len(relevant) if relevant else 0.0,
        "recall@100": len(set(ranking[:100]) & relevant) / len(relevant) if relevant else 0.0,
    }
