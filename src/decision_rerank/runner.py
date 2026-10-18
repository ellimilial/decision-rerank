"""Run a method without exposing labels; optionally checkpoint completed queries."""
import json
from math import isfinite
from pathlib import Path
from time import monotonic

from .contracts import Benchmark, Method, Ranking, RankingInput
from .results import QueryResult, RunResult


def validate_ranking(ranking: Ranking, item: RankingInput) -> None:
    expected = tuple(doc.id for doc in item.candidates)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("Input must contain unique candidates and be nonempty")
    if len(ranking.doc_ids) != len(expected) or set(ranking.doc_ids) != set(expected):
        raise ValueError("Ranking must be a complete permutation of the frozen candidates")
    if ranking.scores is not None:
        if len(ranking.scores) != len(expected) or not all(isfinite(s) for s in ranking.scores):
            raise ValueError("Ranking scores must be finite and aligned with all output IDs")


def run(benchmark: Benchmark, method: Method, *, checkpoint: Path | None = None) -> RunResult:
    """Reuse validated checkpoints; fail on mismatched inputs or method parameters.

    Interrupted runs keep completed queries; a successful return covers the selection.
    """
    parameters = json.loads(json.dumps(dict(method.parameters), allow_nan=False))
    result = RunResult(benchmark.name, benchmark.fingerprint, method.name, parameters, ())
    if checkpoint is not None and Path(checkpoint).exists():
        previous = RunResult.load(checkpoint)
        if (previous.benchmark, previous.benchmark_fingerprint, previous.method, previous.parameters) != (
            result.benchmark, result.benchmark_fingerprint, result.method, result.parameters
        ):
            raise ValueError("Checkpoint benchmark, method or parameters differ")
        result = previous
    items = tuple(benchmark.inputs())
    expected = {item.key: item for item in items}
    if not items or len(expected) != len(items):
        raise ValueError("Benchmark inputs must be nonempty with unique query keys")
    for item in items:
        validate_ranking(Ranking(tuple(doc.id for doc in item.candidates)), item)
    completed = {row.key: row for row in result.results}
    if len(completed) != len(result.results) or not set(completed) <= set(expected):
        raise ValueError("Checkpoint has duplicate or unknown queries")
    for key, row in completed.items():
        validate_ranking(row.ranking, expected[key])
    for item in items:
        if item.key in completed:
            continue
        before = monotonic()
        ranking = method.rank(item)
        validate_ranking(ranking, item)
        completed[item.key] = QueryResult(item.collection, item.query_id, ranking, monotonic() - before)
        result = RunResult(benchmark.name, benchmark.fingerprint, method.name, parameters,
                           tuple(completed[x.key] for x in items if x.key in completed))
        if checkpoint is not None:
            result.save(checkpoint)
        print(f'    {len(completed)}/{len(items)} queries: {item.collection}/{item.query_id} '
              f'({completed[item.key].seconds:.1f}s)', flush=True)
    return result
