"""Load frozen benchmarks, acquiring missing data on first use."""
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from .contracts import Candidate, QueryKey, RankingInput


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class PreparedBenchmark:
    name: str
    primary_metric: str
    aggregation: Literal["query_mean", "collection_macro"]
    _root: Path
    _cases: tuple[RankingInput, ...]
    _label_files: tuple[tuple[str, dict], ...]
    _package_fingerprint: str

    @property
    def fingerprint(self) -> str:
        return digest([self._package_fingerprint, [item.key for item in self._cases]])

    def inputs(self) -> tuple[RankingInput, ...]:
        return self._cases

    def judgments(self) -> dict[QueryKey, dict[str, float]]:
        selected = {item.key for item in self._cases}
        labels = defaultdict(dict)
        for collection, descriptor in self._label_files:
            for row in read_jsonl(self._root / descriptor["path"]):
                key = collection, row["query_id"]
                if key in selected:
                    if row["doc_id"] in labels[key]:
                        raise ValueError(f"Duplicate relevance judgment: {key}, {row['doc_id']}")
                    labels[key][row["doc_id"]] = row["score"]
        return {key: labels[key] for key in selected}

    def select(self, keys: Iterable[QueryKey]) -> "PreparedBenchmark":
        """Select explicit IDs, preserving frozen order. Subsets get a new fingerprint."""
        keys = tuple(keys)
        selected = set(keys)
        available = {item.key for item in self._cases}
        if not keys or len(keys) != len(selected) or not selected <= available:
            raise ValueError("Selection must contain unique, known query keys and be nonempty")
        return replace(self, _cases=tuple(item for item in self._cases if item.key in selected))


def load_benchmark(root: Path, name: str, *, panel: str | None = None, limit: int | None = None,
                   offline: bool = False) -> PreparedBenchmark:
    if limit is not None and panel is None:
        raise ValueError("Use an explicit frozen panel when limiting query count")
    root = Path(root).resolve()
    directory = root / "research/benchmarks"
    prepared = json.loads((directory / "prepared-manifest.json").read_text())
    if name == "curev1_en_200":
        if panel is not None or limit is not None:
            raise ValueError("CURE-200 is a fixed 200-query benchmark; use the parent CURE pilot for smoke tests")
        spec = json.loads((directory / "cure200.json").read_text())
        source_collections = [c for c in prepared["collections"] if c["benchmark"] == "curev1_en"]
        expected = Counter({(c["name"], level): count for c in source_collections
                            for level, count in [("expert", 15), ("layman", 5)]})
        members = spec["members"]
        keys = [(r["collection"], r["query_id"]) for r in members]
        excluded = {(r["collection"], r["query_id"]) for r in spec["excluded_queries"]}
        origins = {(s["dataset"], s["revision"]) for c in source_collections for s in c["sources"]}
        if (spec["name"] != name or spec["parent"] != "curev1_en" or spec["schema_version"] != 1
            or len(source_collections) != 10 or len(keys) != 200 or len(set(keys)) != 200
            or len(excluded) != 30 or len(spec["excluded_queries"]) != 30 or set(keys) & excluded
            or Counter((r["collection"], r["expertise"]) for r in members) != expected
            or origins != {(spec["source"]["dataset"], spec["source"]["revision"])}):
            raise ValueError("CURE-200 source, strata or development exclusions differ from its protocol")
        parent = load_benchmark(root, "curev1_en", offline=offline)
        if not excluded <= {item.key for item in parent.inputs()}:
            raise ValueError("CURE-200 excludes unknown development queries")
        return replace(parent.select(keys), name=name,
                       _package_fingerprint=digest([parent._package_fingerprint, spec]))
    evidence = json.loads((directory / "evidence-manifest.json").read_text())
    collections = [c for c in prepared["collections"] if c["benchmark"] == name]
    if not collections:
        raise ValueError(f"Unknown benchmark: {name}")
    from .data import ensure_benchmark_data
    ensure_benchmark_data(root, name, offline=offline)
    policy = {
        "bioasq13b_curated": ("bioasq_ap@10", "query_mean"),
        "curev1_en": ("ndcg@10", "collection_macro"),
        "nfcorpus_dev": ("ndcg@10", "query_mean"),
    }
    primary, aggregation = policy[name]
    by_name = {c["name"]: c for c in evidence["collections"]}
    cases, labels, descriptors = [], [], []
    for collection in collections:
        shared = by_name[collection["name"]]["files"]
        files = {"candidates": collection["files"]["candidates"],
                 "labels": collection["files"]["qrels"],
                 "documents": shared["documents"], "queries": shared["queries"]}
        paths = {key: root / value["path"] for key, value in files.items()}
        files = {key: {**value, "sha256": sha256(paths[key].read_bytes()).hexdigest()}
                 for key, value in files.items()}
        descriptors.append([collection["name"], files])
        documents = {r["id"]: Candidate(r["id"], r["text"]) for r in read_jsonl(paths["documents"])}
        queries = {r["id"]: r["text"] for r in read_jsonl(paths["queries"])}
        candidate_rows = read_jsonl(paths["candidates"])
        if len(queries) != collection["queries"] or {r["query_id"] for r in candidate_rows} != set(queries):
            raise ValueError("Prepared query/candidate coverage differs")
        for row in candidate_rows:
            docs = tuple(documents[d["doc_id"]] for d in row["candidates"])
            if len(docs) != 100 or len({d.id for d in docs}) != 100:
                raise ValueError("Expected 100 unique frozen candidates per query")
            cases.append(RankingInput(collection["name"], row["query_id"], queries[row["query_id"]], docs))
        labels.append((collection["name"], files["labels"]))
    if len(cases) != len({item.key for item in cases}):
        raise ValueError("Duplicate query in prepared benchmark")
    benchmark = PreparedBenchmark(name, primary, aggregation, root, tuple(cases), tuple(labels),
                                  digest([name, primary, aggregation, descriptors]))
    if panel is not None:
        path = root / prepared["algorithm_panel_file"]["path"]
        members = json.loads(path.read_text())[panel]
        if limit is not None:
            if not isinstance(limit, int) or not 1 <= limit <= len(members):
                raise ValueError(f"Query limit must be between 1 and {len(members)}")
            members = members[:limit]
        benchmark = benchmark.select((r["collection"], r["query_id"]) for r in members)
    return benchmark
