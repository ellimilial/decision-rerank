"""Small, provider-independent contracts. Methods never receive relevance labels."""
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

QueryKey = tuple[str, str]  # Collection and query ID; IDs can repeat across collections.


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str


@dataclass(frozen=True)
class RankingInput:
    collection: str
    query_id: str
    query: str
    candidates: tuple[Candidate, ...]  # Frozen retrieval order, used for stable ties.

    @property
    def key(self) -> QueryKey:
        return self.collection, self.query_id


@dataclass(frozen=True)
class Ranking:
    doc_ids: tuple[str, ...]  # A complete permutation, including the unselected tail.
    scores: tuple[float, ...] | None = None  # Optional; aligned with output order.


class Benchmark(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def fingerprint(self) -> str: ...  # Inputs, candidate order, labels, selection and metric policy.

    @property
    def primary_metric(self) -> str: ...

    @property
    def aggregation(self) -> Literal["query_mean", "collection_macro"]: ...

    def inputs(self) -> Iterable[RankingInput]: ...

    def judgments(self) -> Mapping[QueryKey, Mapping[str, float]]: ...


class Method(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def parameters(self) -> Mapping[str, object]: ...  # JSON values: revision, precision, prompts, etc.

    def rank(self, item: RankingInput) -> Ranking: ...
