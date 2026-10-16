"""Portable results, independent of the method implementation or serving provider."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .contracts import QueryKey, Ranking


def write_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class QueryResult:
    collection: str
    query_id: str
    ranking: Ranking
    seconds: float

    @property
    def key(self) -> QueryKey:
        return self.collection, self.query_id


@dataclass(frozen=True)
class RunResult:
    benchmark: str
    benchmark_fingerprint: str
    method: str
    parameters: dict[str, object]
    results: tuple[QueryResult, ...]

    def save(self, path: Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "RunResult":
        value = json.loads(Path(path).read_text())
        rows = []
        for row in value.pop("results"):
            ranking = row.pop("ranking")
            scores = ranking["scores"]
            rows.append(QueryResult(**row, ranking=Ranking(
                tuple(ranking["doc_ids"]), tuple(scores) if scores is not None else None,
            )))
        return cls(**value, results=tuple(rows))


@dataclass(frozen=True)
class Evaluation:
    benchmark: str
    benchmark_fingerprint: str
    method: str
    parameters: dict[str, object]
    primary_metric: str
    score: float
    aggregation: str
    collections: dict[str, dict[str, object]]
    per_query: tuple[dict[str, object], ...]

    def save(self, path: Path) -> None:
        write_json(path, asdict(self))
