"""Frozen retrieval order and stable pointwise ranking."""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import isfinite

from .contracts import Candidate, Ranking, RankingInput


@dataclass
class OriginalOrder:
    """The frozen BM25 baseline. Does not retrieve again or inspect labels."""
    name: str = "bm25"
    parameters: dict[str, object] = field(default_factory=dict)

    def rank(self, item: RankingInput) -> Ranking:
        return Ranking(tuple(doc.id for doc in item.candidates))


@dataclass
class Pointwise:
    """Score candidates independently; batch inside the injected scorer if desired."""
    name: str
    scorer: Callable[[str, tuple[Candidate, ...]], Sequence[float]]
    parameters: dict[str, object] = field(default_factory=dict)

    def rank(self, item: RankingInput) -> Ranking:
        scores = tuple(float(s) for s in self.scorer(item.query, item.candidates))
        if len(scores) != len(item.candidates) or not all(isfinite(s) for s in scores):
            raise ValueError("Scorer must return one finite score per candidate")
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return Ranking(tuple(item.candidates[i].id for i in order),
                       tuple(scores[i] for i in order))


class CandidateTfidf(Pointwise):
    """CPU lexical control: cosine TF-IDF fitted to the supplied candidate list."""
    def __init__(self):
        super().__init__('tfidf_candidates', self._score, {
            'tokenizer': 'Unicode word tokens, lowercase',
            'tf': '1 + log(count)', 'idf': 'log((N+1)/(df+1)) + 1',
            'idf_scope': 'the same 100 supplied candidates per query', 'score': 'cosine similarity',
        })

    @staticmethod
    def _score(query, candidates):
        from collections import Counter
        from math import log, sqrt
        import re
        words = lambda text: Counter(re.findall(r'\w+', text.lower()))
        documents = [words(doc.text) for doc in candidates]
        frequencies = Counter(word for doc in documents for word in doc)
        idf = {word: log((len(documents)+1)/(count+1))+1 for word,count in frequencies.items()}
        def vector(tokens):
            return {word:(1+log(count))*idf[word] for word,count in tokens.items() if word in idf}
        query_vector = vector(words(query))
        query_norm = sqrt(sum(x*x for x in query_vector.values()))
        scores = []
        for tokens in documents:
            doc = vector(tokens)
            denominator = query_norm * sqrt(sum(x*x for x in doc.values()))
            scores.append(sum(value*doc.get(word,0) for word,value in query_vector.items())/denominator if denominator else 0.)
        return scores
