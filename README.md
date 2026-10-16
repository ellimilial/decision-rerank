# decision-rerank

Collect biomedical reranking results with a simple benchmark × method notebook.

[Open the collector in Google Colab](https://colab.research.google.com/github/ellimilial/decision-rerank/blob/main/notebooks/01_collect_results.ipynb).

1. Choose a GPU runtime for GPU methods, then run **Setup**. It clones this repository when needed and installs locked dependencies with uv. Automatic cloning requires public GitHub access. For a private repository, upload a `decision-rerank-colab*.tar.gz` bundle in the Files panel before Setup instead.
   If Setup replaces NumPy/SciPy already loaded by the kernel, it stops and asks for **Runtime → Restart session**. Restart, rerun Setup, then continue.
2. Uncomment the desired entries in `BENCHMARKS` and `METHODS`. The default is five NFCorpus development queries and BM25.
3. Run `collect_results(BENCHMARKS, METHODS, output='results')`, then download the ZIP from the final cell.

Each method runs across all selected benchmarks before the next method starts. Choices are listed roughly from heavier to lighter: Clef Flash 9B, Qwen3 reranker 8B/4B, BioDecision 4B and its matched Qwen3.5 backbone, Jina v3.5, MedCPT, then CPU controls and hosted JEV/OpenAI Decisions.

Missing datasets and weights download automatically from pinned sources. Every method receives the same natural BM25 top-100 candidates and frozen short evidence; labels are read only by the evaluator.

| Benchmark | Full queries | Frozen panel | Primary metric |
| --- | ---: | ---: | --- |
| BioASQ 13B curated | 340 | 30 | Capped AP@10, query mean |
| CUREv1 English | 2,000 | 30 | Linear-gain nDCG@10, specialty macro |
| NFCorpus dev | 324 | 20 | nDCG@10, development only |

BioASQ uses a curated corpus with some gold articles missing; this is not full-PubMed competition reproduction. CURE includes model-generated judgments. Keep benchmark/subset results separate. Full GPU inference and memory fit still need validation on the selected hardware.

Results contain `run.json` (rankings, settings and resume checkpoint), `run.evaluation.json` (aggregate and per-query metrics), optional hosted `calls.jsonl`, and one shared `session.json` with queries/statuses. Reuse the output folder to resume; use a mounted Drive folder to retain work across Colab resets. Hosted methods make paid API calls and cache successful responses. Before any selected model runs, the collector reads `OPENROUTER_API_KEY` (JEV) / `OPENAI_API_KEY` (OpenAI Decisions) from environment variables or Colab Secrets, or asks through a hidden prompt. CPU/GPU-only selections need no keys. Keys are not saved.

Use uv ≥0.13.1 and Python 3.12–3.14. For a local notebook session:

```sh
uv run --locked --group notebooks jupyter lab notebooks/01_collect_results.ipynb
```

This repository contains only the collector runtime and frozen manifests. Datasets, weights, results, papers, research notes and the local analysis workflow are excluded.
