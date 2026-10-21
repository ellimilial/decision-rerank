# CURE-200 coverage audit

Run `uv run python tools/cure200.py` to reproduce this report and verify the frozen selection.

Source: [clinia/CUREv1](https://huggingface.co/datasets/clinia/CUREv1/tree/072343ca2353c3dc832cf18cf0fef7c9a5a207b7) at `072343ca2353c3dc832cf18cf0fef7c9a5a207b7`, English only.

CURE-200 samples 200 queries from 1,970 eligible queries, excluding the fixed 30-query development pilot. Every specialty contributes 15 expert and 5 layman queries. Selection uses IDs and metadata only; the diagnostics below are computed after selection and are never used to choose or replace queries.

| Specialty | Expertise | Original | Pilot excluded | Eligible | Selected |
| --- | --- | ---: | ---: | ---: | ---: |
| dentistry_and_oral_health | expert | 150 | 2 | 148 | 15 |
| dentistry_and_oral_health | layman | 50 | 1 | 49 | 5 |
| dermatology | expert | 150 | 2 | 148 | 15 |
| dermatology | layman | 50 | 1 | 49 | 5 |
| gastroenterology | expert | 150 | 3 | 147 | 15 |
| gastroenterology | layman | 50 | 0 | 50 | 5 |
| genetics | expert | 150 | 3 | 147 | 15 |
| genetics | layman | 50 | 0 | 50 | 5 |
| neuroscience_and_neurology | expert | 150 | 3 | 147 | 15 |
| neuroscience_and_neurology | layman | 50 | 0 | 50 | 5 |
| orthopedic_surgery | expert | 150 | 2 | 148 | 15 |
| orthopedic_surgery | layman | 50 | 1 | 49 | 5 |
| otorhinolaryngology | expert | 150 | 2 | 148 | 15 |
| otorhinolaryngology | layman | 50 | 1 | 49 | 5 |
| plastic_surgery | expert | 150 | 3 | 147 | 15 |
| plastic_surgery | layman | 50 | 0 | 50 | 5 |
| psychiatry_and_psychology | expert | 150 | 0 | 150 | 15 |
| psychiatry_and_psychology | layman | 50 | 3 | 47 | 5 |
| pulmonology | expert | 150 | 2 | 148 | 15 |
| pulmonology | layman | 50 | 1 | 49 | 5 |

## Verified correspondence

- All 200 IDs and expertise labels reproduce from the declared sampling rule; no pilot overlap.
- Every selected query retains exactly its parent query text, 100 candidate IDs, ordering and passage evidence.
- Every selected judgment equals its parent judgment; all parent relevance grades match original CURE qrels.
- All selected passage IDs occur in original specialty corpora; shared evidence is an original-text prefix.
- Primary metric and specialty aggregation match the parent; the subset has a distinct name and fingerprint.

## Descriptive coverage

Means use equal specialty weights and 75:25 expert/layman weights. Counts are unweighted. The eligible population is therefore standardized to the original mixture despite pilot exclusions.

| Diagnostic | Original CURE | Eligible non-pilot | CURE-200 |
| --- | ---: | ---: | ---: |
| Queries | 2000 | 1970 | 200 |
| Expert / layman | 1500 / 500 | 1478 / 492 | 150 / 50 |
| Mean query words (weighted) | 11.998500 | 11.994841 | 11.555000 |
| Mean positive judgments (weighted) | 40.358000 | 40.324636 | 36.550000 |
| BM25 nDCG@10 (weighted) | 0.390536 | 0.390330 | 0.364097 |
| BM25 recall@100 (weighted) | 0.570464 | 0.569965 | 0.565030 |
| Queries with no positive top-100 candidate | 36 | 36 | 4 |

## Interpretation

Coverage establishes source correspondence and preserves the specialty/expertise design; it does not guarantee that every model performs identically on this subset and all 2,000 queries. The excluded pilot has zero inclusion probability. The primary estimand is the non-pilot population standardized to original specialty/expertise weights, not an unbiased estimate over all original queries.

Report this as a frozen CUREv1 English subset under BM25 top-100 reranking and shared short evidence (96 query / 384 passage MedCPT wordpieces). The original full corpus and relevance grades are inherited, but clipping and fixed candidates mean this is not full-corpus retrieval or a full-CURE leaderboard result. Mixed human/Qwen2.5-generated judgments and training-overlap limitations remain.

Use paired query-level comparisons and confidence intervals, resampling within the 20 specialty × expertise strata. Keep per-specialty conclusions descriptive at n=20. Do not treat 20,000 scored pairs as 20,000 independent test queries or enlarge the sample based on significance. Any additional queries already used for tuning must be disclosed before calling this a held-out evaluation.
