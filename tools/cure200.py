"""Reproduce CURE-200's label-blind selection and audit its parent-data coverage.

Run from the checkout with: uv run python tools/cure200.py
Only missing source/prepared data is downloaded. No model inference is performed.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from decision_rerank.benchmarks import load_benchmark, read_jsonl
from decision_rerank.data import download_file
from decision_rerank.metrics import score_ranking

NAME = 'curev1_en_200'
PARENT = 'curev1_en'
REVISION = '072343ca2353c3dc832cf18cf0fef7c9a5a207b7'
QUOTAS = {'expert': 15, 'layman': 5}
SEED = 42


def source_queries(root):
    prepared = json.loads((root / 'research/benchmarks/prepared-manifest.json').read_text())
    collections = [c for c in prepared['collections'] if c['benchmark'] == PARENT]
    if len(collections) != 10:
        raise ValueError('Expected the ten original English CURE specialties')
    queries = {}
    for collection in collections:
        source = next(s for s in collection['sources'] if s['path'].endswith('/queries-en.jsonl'))
        if (source['dataset'], source['revision']) != ('clinia/CUREv1', REVISION):
            raise ValueError('CURE-200 uses the pinned CUREv1 source revision')
        rows = read_jsonl(download_file(root, source))
        if len(rows) != 200 or Counter(r['expertise'] for r in rows) != {'expert': 150, 'layman': 50}:
            raise ValueError('Source expertise distribution differs from CUREv1')
        for row in rows:
            key = collection['name'], row['_id']
            if key in queries:
                raise ValueError('Duplicate source query')
            queries[key] = row
    return collections, queries


def select_members(queries, excluded):
    """Use only original query IDs, specialty and expertise; never labels or scores."""
    if len(excluded) != 30 or not excluded <= queries.keys():
        raise ValueError('Expected 30 known pilot query exclusions')
    strata = defaultdict(list)
    for key, row in queries.items():
        if key not in excluded:
            strata[key[0], row['expertise']].append(key)
    selected = []
    for (collection, expertise), keys in sorted(strata.items()):
        def priority(key):
            value = f'{SEED}|cure200-v1|{collection}|{expertise}|{key[1]}'
            return sha256(value.encode('utf-8')).hexdigest(), key[1]
        count = QUOTAS[expertise]
        if len(keys) < count:
            raise ValueError('Too few eligible queries in a stratum')
        selected.extend({'collection': c, 'query_id': q, 'expertise': expertise}
                        for c, q in sorted(keys, key=priority)[:count])
    return sorted(selected, key=lambda r: (r['collection'], r['query_id']))


def manifest(root):
    collections, queries = source_queries(root)
    pilot = json.loads((root / 'research/benchmarks/query-panels.json').read_text())['cure']
    excluded = {(r['collection'], r['query_id']) for r in pilot}
    if len(pilot) != 30 or Counter(c for c, _ in excluded) != {c['name']: 3 for c in collections}:
        raise ValueError('Expected the existing three-per-specialty development pilot')
    return {
        'schema_version': 1, 'name': NAME, 'display_name': 'CURE-200', 'parent': PARENT,
        'source': {'dataset': 'clinia/CUREv1', 'revision': REVISION, 'language': 'en',
                   'url': f'https://huggingface.co/datasets/clinia/CUREv1/tree/{REVISION}'},
        'selection': {'seed': SEED, 'strata': ['specialty', 'expertise'],
                      'per_specialty': QUOTAS,
                      'algorithm': 'Sort eligible IDs by SHA256(UTF-8(seed|cure200-v1|collection|expertise|query_id)), then query_id; take the stratum quota',
                      'uses_relevance_labels_or_model_outputs': False,
                      'exclusion_reason': 'Existing 30-query CURE pilot reserved for development; conservative exclusion regardless of tuning history'},
        'evaluation': {'primary_metric': 'ndcg@10', 'gain': 'linear', 'aggregation': 'collection_macro',
                       'specialty_weight': 0.1, 'expertise_weights': {'expert': 0.75, 'layman': 0.25},
                       'candidate_depth': 100, 'candidate_source': 'Unmodified parent BM25 lists',
                       'evidence': 'Unmodified parent shared short evidence (96/384 MedCPT wordpieces for queries/documents)',
                       'estimand': 'Equal-specialty, 75:25 expertise-weighted performance on non-pilot English CURE queries; not an independent dataset or full-CURE leaderboard reproduction'},
        'excluded_queries': [{'collection': c, 'query_id': q, 'expertise': queries[c, q]['expertise']}
                             for c, q in sorted(excluded)],
        'members': select_members(queries, excluded),
    }


def audit(root, frozen):
    """Verify selection and identity first; compute descriptive diagnostics afterwards."""
    if manifest(root) != frozen:
        raise ValueError('Frozen CURE-200 manifest is not reproduced by the declared source and selection rule')
    collections, queries = source_queries(root)
    parent = load_benchmark(root, PARENT)
    subset = load_benchmark(root, NAME)
    parent_cases = {item.key: item for item in parent.inputs()}
    selected = {item.key for item in subset.inputs()}
    excluded = {(r['collection'], r['query_id']) for r in frozen['excluded_queries']}
    eligible = set(queries) - excluded
    labels, subset_labels = parent.judgments(), subset.judgments()
    if set(parent_cases) != set(queries) or not selected <= eligible:
        raise ValueError('Parent/subset query coverage differs from source')
    if subset.name == parent.name or subset.fingerprint == parent.fingerprint:
        raise ValueError('Subset must have a distinct benchmark identity')
    if subset.primary_metric != parent.primary_metric or subset.aggregation != parent.aggregation:
        raise ValueError('Subset evaluation policy differs from parent')
    if any(item != parent_cases[item.key] for item in subset.inputs()):
        raise ValueError('Subset changed query text, candidate IDs/order or passage evidence')
    if subset_labels != {key: labels[key] for key in selected}:
        raise ValueError('Subset changed parent judgments')
    for key, item in parent_cases.items():
        if not queries[key]['text'].strip().startswith(item.query):
            raise ValueError('Prepared query is not a prefix of the original query')
    coverage = []
    for collection in collections:
        name = collection['name']
        source = next(s for s in collection['sources'] if s['path'].endswith('/qrels.jsonl'))
        original_labels = defaultdict(dict)
        for row in read_jsonl(download_file(root, source)):
            key = name, str(row['query-id'])
            doc_id = str(row['corpus-id'])
            if doc_id in original_labels[key]:
                raise ValueError('Duplicate original relevance judgment')
            original_labels[key][doc_id] = int(row['score'])
        if {key: labels[key] for key in queries if key[0] == name} != dict(original_labels):
            raise ValueError('Prepared relevance grades differ from original CURE judgments')
        # Original passages remain identifiable; shared clipping is inherited, not reapplied.
        used = {doc.id: doc.text for key in selected if key[0] == name for doc in parent_cases[key].candidates}
        source = next(s for s in collection['sources'] if s['path'].endswith('/corpus.jsonl'))
        matched = set()
        for row in read_jsonl(download_file(root, source)):
            doc_id = str(row['_id'])
            if doc_id in used:
                original = (row.get('title') or '') + '\n' + row['text']
                if not original.startswith(used[doc_id]):
                    raise ValueError('Selected passage evidence differs from original CURE prefix')
                matched.add(doc_id)
        if matched != set(used):
            raise ValueError('Selected passage IDs missing from original CURE corpus')
        for expertise in QUOTAS:
            pool = {key for key, row in queries.items() if key[0] == name and row['expertise'] == expertise}
            coverage.append((name.split('/')[-1], expertise, len(pool), len(pool & excluded),
                             len(pool & eligible), len(pool & selected)))
    per_query = {key: score_ranking([doc.id for doc in item.candidates], labels[key])
                 for key, item in parent_cases.items()}
    def summary(keys):
        # Use original expertise weights even though exclusions slightly change the eligible mix.
        groups = defaultdict(list)
        for key in keys:
            groups[key[0], queries[key]['expertise']].append(key)
        def weighted(value):
            return sum(0.1 * {'expert': .75, 'layman': .25}[expertise] * mean(value(key) for key in group)
                       for (_, expertise), group in groups.items())
        return {
            'Queries': len(keys),
            'Expert / layman': f"{sum(queries[k]['expertise']=='expert' for k in keys)} / {sum(queries[k]['expertise']=='layman' for k in keys)}",
            'Mean query words (weighted)': weighted(lambda k: len(queries[k]['text'].split())),
            'Mean positive judgments (weighted)': weighted(lambda k: sum(v > 0 for v in labels[k].values())),
            'BM25 nDCG@10 (weighted)': weighted(lambda k: per_query[k]['ndcg@10']),
            'BM25 recall@100 (weighted)': weighted(lambda k: per_query[k]['recall@100']),
            'Queries with no positive top-100 candidate': sum(per_query[k]['recall@100'] == 0 for k in keys),
        }
    return coverage, {'Original CURE': summary(set(queries)), 'Eligible non-pilot': summary(eligible),
                      'CURE-200': summary(selected)}


def report(root, frozen):
    coverage, populations = audit(root, frozen)
    lines = ['# CURE-200 coverage audit', '',
             "Run `uv run python tools/cure200.py` to reproduce this report and verify the frozen selection.", '',
             f"Source: [clinia/CUREv1]({frozen['source']['url']}) at `{REVISION}`, English only.", '',
             'CURE-200 samples 200 queries from 1,970 eligible queries, excluding the fixed 30-query development pilot. '
             'Every specialty contributes 15 expert and 5 layman queries. Selection uses IDs and metadata only; '
             'the diagnostics below are computed after selection and are never used to choose or replace queries.', '',
             '| Specialty | Expertise | Original | Pilot excluded | Eligible | Selected |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    lines += ['| ' + ' | '.join(map(str, row)) + ' |' for row in coverage]
    lines += ['', '## Verified correspondence', '',
              '- All 200 IDs and expertise labels reproduce from the declared sampling rule; no pilot overlap.',
              '- Every selected query retains exactly its parent query text, 100 candidate IDs, ordering and passage evidence.',
              '- Every selected judgment equals its parent judgment; all parent relevance grades match original CURE qrels.',
              '- All selected passage IDs occur in original specialty corpora; shared evidence is an original-text prefix.',
              '- Primary metric and specialty aggregation match the parent; the subset has a distinct name and fingerprint.', '',
              '## Descriptive coverage', '',
              'Means use equal specialty weights and 75:25 expert/layman weights. Counts are unweighted. '
              'The eligible population is therefore standardized to the original mixture despite pilot exclusions.', '',
              '| Diagnostic | Original CURE | Eligible non-pilot | CURE-200 |', '| --- | ---: | ---: | ---: |']
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    for metric in populations['Original CURE']:
        lines.append('| ' + metric + ' | ' + ' | '.join(fmt(v[metric]) for v in populations.values()) + ' |')
    lines += ['', '## Interpretation', '',
              'Coverage establishes source correspondence and preserves the specialty/expertise design; it does not '
              'guarantee that every model performs identically on this subset and all 2,000 queries. The excluded pilot '
              'has zero inclusion probability. The primary estimand is the non-pilot population standardized to original '
              'specialty/expertise weights, not an unbiased estimate over all original queries.', '',
              'Report this as a frozen CUREv1 English subset under BM25 top-100 reranking and shared short evidence '
              '(96 query / 384 passage MedCPT wordpieces). The original full corpus and relevance grades are inherited, '
              'but clipping and fixed candidates mean this is not full-corpus retrieval or a full-CURE leaderboard result. '
              'Mixed human/Qwen2.5-generated judgments and training-overlap limitations remain.', '',
              'Use paired query-level comparisons and confidence intervals, resampling within the 20 specialty × expertise '
              'strata. Keep per-specialty conclusions descriptive at n=20. Do not treat 20,000 scored pairs as 20,000 '
              'independent test queries or enlarge the sample based on significance. Any additional queries already used '
              'for tuning must be disclosed before calling this a held-out evaluation.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Write the verified Markdown report to this path')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / 'research/benchmarks/cure200.json').read_text())
    text = report(root, frozen)
    if args.report:
        args.report.write_text(text)
    else:
        print(text, end='')
