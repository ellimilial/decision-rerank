# decision-rerank

Collect biomedical reranking results with a simple benchmark × method notebook.

[Open the collector in Google Colab](https://colab.research.google.com/github/ellimilial/decision-rerank/blob/main/notebooks/01_collect_results.ipynb).

1. Choose a GPU runtime for GPU methods, then run **Setup**. It clones this repository when needed and installs dependencies with uv, retaining the existing NumPy/SciPy pair and CUDA PyTorch. Other dependencies use the lockfile. Automatic cloning requires public GitHub access. For a private repository, upload a `decision-rerank-colab*.tar.gz` bundle in the Files panel before Setup instead.
   Fresh Colab sessions should not need a restart. If an earlier Setup already replaced loaded NumPy/SciPy, restart that session once, then rerun Setup.
2. Uncomment the desired entries in `BENCHMARKS` and `METHODS`. The default is five NFCorpus development queries and BM25.
3. Run `collect_results(BENCHMARKS, METHODS, output='results')`, then download the ZIP from the final cell.

Each local model loads once, runs across all selected benchmarks, then releases its weights before the next method starts. Hosted methods retain separate call logs and caches per benchmark. Choices are listed roughly from heavier to lighter: BioDecision 4B and the general Qwen3.5 relevance baseline, Kev-4B, Qwen3 reranker 4B, Jina v3.5, MedCPT, then lexical CPU controls and hosted JEV/OpenAI Decisions and GPT-6.1 Sol.

BioDecision, Qwen3.5, the Qwen3 reranker, Jina and MedCPT load weights directly onto the selected device, avoiding a full CPU model staging copy for GPU runs. After each local method, the collector explicitly clears its loaded model/tokenizer, runs garbage collection, synchronizes CUDA and releases its cache. Remaining allocated/reserved GPU memory is printed before the next method. Existing inference settings and checkpoints remain compatible.

Kev-4B uses the native pointer head and shipped calibration, with FP16 and batch size 1 by default. The backbone loads directly onto the GPU; LoRA layers visit CPU one at a time for merging, avoiding a full host copy of the model. Full-model Colab memory fit still needs validation. Setup removes optional torchao, which these unquantized runs do not use, to avoid conflicts with Colab’s preinstalled version.

Jina scores the full candidate list jointly, using query-token blocks for attention to avoid square attention tensors on small GPUs. This preserves cross-document attention and the native scoring rule; it can be slower than optimized fused kernels. The attention implementation is recorded in results.

Missing datasets and weights download automatically from pinned sources. Benchmark preparation keeps optional JAX on CPU so its BM25 import cannot reserve model GPU memory. Data SHA comparisons are not performed. Result fingerprints use the actual runtime files, so changed data cannot reuse an old checkpoint. Every method receives the same natural BM25 top-100 candidates and frozen short evidence; labels are read only by the evaluator.

| Benchmark | Full queries | Frozen panel | Primary metric |
| --- | ---: | ---: | --- |
| BioASQ 13B curated | 340 | 30 | Capped AP@10, query mean |
| CUREv1 English | 2,000 | 30 | Linear-gain nDCG@10, specialty macro |
| NFCorpus dev | 324 | 20 | nDCG@10, development only |

BioASQ uses a curated corpus with some gold articles missing; this is not full-PubMed competition reproduction. CURE includes model-generated judgments. Keep benchmark/subset results separate. Full GPU inference and memory fit still need validation on the selected hardware.

GPT-6.1 Sol is available as `sol-pointwise-low` (generated 0–3 relevance scores), `sol-listwise-low` (one response ranking every candidate), and optional `sol-listwise-high`. Compare the two low-effort methods first; the high-effort method measures the benefit of extra reasoning. Sol uses ordinary LLM generation, not a native decision head. Listwise requests use all frozen candidates in retrieval order and require a complete permutation; malformed, refused or truncated answers fail without repair. A 100-candidate query needs 100 pointwise requests or one listwise request before retries; input tokens and reasoning tokens still determine cost. The pinned model is `openai/gpt-6.1-sol-20260929`, using OpenRouter structured outputs.

Results contain `run.json` (rankings, settings and resume checkpoint), `run.evaluation.json` (aggregate and per-query metrics), optional hosted `calls.jsonl`, and one shared `session.json` with queries/statuses. Reuse the output folder to resume; use a mounted Drive folder to retain work across Colab resets. Hosted methods make paid API calls and cache successful responses. Before any selected model runs, the collector reads `OPENROUTER_API_KEY` (JEV, OpenAI Luna Decisions and Sol) from environment variables or Colab Secrets, or asks through a hidden prompt. CPU/GPU-only selections need no keys. Keys are not saved. OpenAI Luna uses the pinned `openai/gpt-6-luna-decisions-20261006` model through OpenRouter. Use a fresh output folder for this route; earlier direct-OpenAI checkpoints have different method parameters.

Use uv ≥0.13.1 and Python 3.12–3.14. For a local notebook session:

```sh
uv run --locked --group notebooks jupyter lab notebooks/01_collect_results.ipynb
```

This repository contains only the collector runtime and frozen manifests. Datasets, weights, results, papers, research notes and the local analysis workflow are excluded.
