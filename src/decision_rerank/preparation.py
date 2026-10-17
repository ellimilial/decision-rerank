"""Reproduce the original frozen normalization, BM25 and short-evidence bytes."""
import importlib.metadata
import json
from pathlib import Path
import re
import zipfile

def check_dependencies(versions: dict, tokenizer_version: str) -> None:
    for name, expected in {**versions, "tokenizers": tokenizer_version}.items():
        # NumPy follows the compatible installed runtime; other recipe dependencies stay pinned.
        if name == 'numpy':
            continue
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            raise RuntimeError(f"Data preparation needs {name}=={expected}, found {actual}. Run uv sync --locked.")


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize(root: Path, collection: dict):
    name = collection["benchmark"]
    sources = [root / source["path"] for source in collection["sources"]]
    if name == "bioasq13b_curated":
        import duckdb
        with duckdb.connect() as connection:
            docs = connection.execute("SELECT id,title,text FROM read_parquet(?)", [str(sources[0])]).fetchall()
            questions = connection.execute("SELECT id,question,documents FROM read_parquet(?)", [str(sources[1])]).fetchall()
        corpus = [{"id": str(i), "title": title or "", "text": text or ""} for i, title, text in docs]
        queries, qrels = [], []
        for qid, text, urls in questions:
            queries.append({"id": qid, "text": text.strip()})
            ids = set()
            for url in urls:
                match = re.search(r"(?:/pubmed/|pubmed\.ncbi\.nlm\.nih\.gov/)(\d+)", url)
                if match is None:
                    raise ValueError(f"Invalid BioASQ PMID URL: {url}")
                ids.add(match.group(1))
            qrels.extend({"query_id": qid, "doc_id": doc, "score": 1, "judge": "expert_13b_gold_mirror"}
                         for doc in sorted(ids))
    elif name == "curev1_en":
        corpus = [{"id": str(r["_id"]), "title": r.get("title") or "", "text": r["text"]}
                  for r in read_jsonl(sources[0])]
        raw = read_jsonl(sources[1])
        queries = [{"id": str(r["_id"]), "text": r["text"].strip()} for r in raw]
        qrels = [{"query_id": str(r["query-id"]), "doc_id": str(r["corpus-id"]),
                  "score": int(r["score"]), "judge": r["judge"]} for r in read_jsonl(sources[2])]
    elif name == "nfcorpus_dev":
        with zipfile.ZipFile(sources[0]) as archive:
            # Read known members directly; never extract untrusted archive paths.
            corpus = [json.loads(line) for line in archive.read("nfcorpus/corpus.jsonl").decode().splitlines()]
            queries = [json.loads(line) for line in archive.read("nfcorpus/queries.jsonl").decode().splitlines()]
            raw_labels = [line.split("\t") for line in archive.read("nfcorpus/qrels/dev.tsv").decode().splitlines()[1:]]
        ids = {row[0] for row in raw_labels}
        corpus = [{"id": r["_id"], "title": r.get("title") or "", "text": r["text"]} for r in corpus]
        queries = [{"id": r["_id"], "text": r["text"].strip()} for r in queries if r["_id"] in ids]
        qrels = [{"query_id": q, "doc_id": d, "score": int(g), "judge": "nfcorpus_native"}
                 for q, d, g in raw_labels if int(g) > 0]
    else:
        raise ValueError(f"No preparation recipe for {name}")
    corpus = sorted((r for r in corpus if (r["title"] + r["text"]).strip()), key=lambda r: r["id"])
    return corpus, sorted(queries, key=lambda r: r["id"]), qrels


def retrieve(corpus: list[dict], queries: list[dict], policy: dict) -> list[dict]:
    import bm25s
    from bm25s.tokenization import Tokenizer
    import numpy as np
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
    tokenizer = Tokenizer(lower=policy["tokenizer"]["lower"], splitter=policy["tokenizer"]["splitter"],
                          stopwords=policy["tokenizer"]["stopwords"], stemmer=stemmer.stemWord)
    tokens = tokenizer.tokenize([r["title"] + "\n" + r["text"] for r in corpus], show_progress=False)
    retriever = bm25s.BM25(k1=policy["k1"], b=policy["b"], method=policy["method"], dtype=policy["dtype"])
    retriever.index(tokens, show_progress=False)
    query_tokens = tokenizer.tokenize([r["text"] for r in queries], update_vocab=False, show_progress=False)
    result = []
    for query, tokens in zip(queries, query_tokens, strict=True):
        scores = retriever.get_scores(tokens) if tokens else np.zeros(len(corpus), dtype=np.float64)
        order = np.argsort(-scores, kind="stable")[:policy["depth"]]
        result.append({"query_id": query["id"], "candidates": [
            {"doc_id": corpus[int(i)]["id"], "rank": rank + 1, "retrieval_score": float(scores[int(i)])}
            for rank, i in enumerate(order)]})
    return result


def prefix(text, encoding, tokenizer, cap):
    if len(encoding.ids) <= cap:
        return text
    clipped = text[:encoding.offsets[cap - 1][1]]
    check = tokenizer.encode(clipped, add_special_tokens=False)
    while len(check.ids) > cap:
        end = check.offsets[cap - 2][1]
        if end >= len(clipped):
            raise ValueError("Non-decreasing tokenizer offsets")
        clipped = clipped[:end]
        check = tokenizer.encode(clipped, add_special_tokens=False)
    return clipped


def prepare_collection(root: Path, staged: Path, collection: dict, shared: dict) -> None:
    from tokenizers import Tokenizer
    corpus, queries, qrels = normalize(root, collection)
    candidates = retrieve(corpus, queries, collection["retriever"])
    files = collection["files"]
    for key, rows in [("qrels", sorted(qrels, key=lambda r: (r["query_id"], r["doc_id"]))),
                      ("candidates", candidates)]:
        write_jsonl(staged / files[key]["path"], rows)
    tokenizer = Tokenizer.from_file(str(root / "data/raw/medcpt-tokenizer/tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    used = {r["doc_id"] for row in candidates for r in row["candidates"]}
    for kind, records, cap in [("documents", [r for r in corpus if r["id"] in used], 384),
                                ("queries", queries, 96)]:
        result = []
        for start in range(0, len(records), 256):
            batch = records[start:start + 256]
            texts = [r["title"] + "\n" + r["text"] if kind == "documents" else r["text"] for r in batch]
            encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
            for record, text, encoding in zip(batch, texts, encodings, strict=True):
                result.append({"id": record["id"], "text": prefix(text, encoding, tokenizer, cap)})
        write_jsonl(staged / shared["files"][kind]["path"], result)
