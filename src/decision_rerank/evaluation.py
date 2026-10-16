"""Evaluate complete runs separately from inference, retaining per-query metrics."""
from collections import defaultdict
from statistics import mean

from .contracts import Benchmark
from .metrics import score_ranking
from .results import Evaluation, RunResult
from .runner import validate_ranking


def evaluate(benchmark: Benchmark, result: RunResult) -> Evaluation:
    if (result.benchmark, result.benchmark_fingerprint) != (benchmark.name, benchmark.fingerprint):
        raise ValueError("Run belongs to a different benchmark package or query selection")
    items = tuple(benchmark.inputs())
    expected = {item.key: item for item in items}
    predictions = {row.key: row for row in result.results}
    if not expected or len(expected) != len(items):
        raise ValueError("Benchmark inputs must be nonempty with unique query keys")
    if len(predictions) != len(result.results):
        raise ValueError("Duplicate query in run")
    if set(expected) != set(predictions):
        raise ValueError("Run must cover every selected query exactly once; partial/extra runs are rejected")
    labels = benchmark.judgments()
    if set(labels) != set(expected):
        raise ValueError("Judgments must cover exactly the selected queries; use empty judgments when unjudged")
    per_query = []
    by_collection = defaultdict(list)
    for item in items:
        prediction = predictions[item.key]
        validate_ranking(prediction.ranking, item)
        metrics = score_ranking(prediction.ranking.doc_ids, labels[item.key])
        per_query.append({"collection": item.collection, "query_id": item.query_id, "metrics": metrics})
        by_collection[item.collection].append(metrics)
    collections = {
        name: {"queries": len(rows), "metrics": {metric: mean(row[metric] for row in rows)
                                                for metric in rows[0]}}
        for name, rows in by_collection.items()
    }
    if benchmark.aggregation == "collection_macro":
        score = mean(value["metrics"][benchmark.primary_metric] for value in collections.values())
    elif benchmark.aggregation == "query_mean":
        score = mean(row["metrics"][benchmark.primary_metric] for row in per_query)
    else:
        raise ValueError(f"Unknown aggregation: {benchmark.aggregation}")
    return Evaluation(benchmark.name, benchmark.fingerprint, result.method, result.parameters,
                      benchmark.primary_metric, score, benchmark.aggregation, collections, tuple(per_query))
