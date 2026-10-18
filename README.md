# decision-rerank

Collect biomedical reranking results with a simple benchmark × method notebook.

[Open the collector in Google Colab](https://colab.research.google.com/github/ellimilial/decision-rerank/blob/main/notebooks/01_collect_results.ipynb).

1. Choose a GPU runtime for GPU methods, then run **Setup**. It clones this repository when needed and installs dependencies with uv, retaining the existing NumPy/SciPy pair and CUDA PyTorch. Other dependencies use the lockfile.
   Fresh Colab sessions should not need a restart. If an earlier Setup already replaced loaded NumPy/SciPy, restart that session once, then rerun Setup.
2. Uncomment the desired entries in `BENCHMARKS` and `METHODS`. The default smoke test is one query each from NFCorpus, BioASQ and CURE, using BM25. All 100 candidates are retained. A commented two-query NFCorpus selection is available to measure warmed-up model throughput before choosing larger panels.
3. Run `collect_results(BENCHMARKS, METHODS, output='results')`, then download the ZIP from the final cell.

Each local model loads once, runs across all selected benchmarks, then releases its weights before the next method starts. Hosted methods retain separate call logs and caches per benchmark.

BioDecision, Qwen3.5, the Qwen3 reranker, Jina and MedCPT load weights directly onto the selected device, avoiding a full CPU model staging copy for GPU runs. After each local method, the collector explicitly clears its loaded model/tokenizer, runs garbage collection, synchronizes CUDA and releases its cache. Remaining allocated/reserved GPU memory is printed before the next method. Existing inference settings and checkpoints remain compatible.

For an optional Qwen3.5 acceleration trial (BioDecision, `qwen35-base`, Kev), use `setup_notebook(linear_kernels=True)` instead of ordinary Setup in a fresh Linux NVIDIA session with compute capability 8.0+ (e.g. L4/A100). Current Triton excludes T4. The optional `linear-kernels` group installs pinned FLA while keeping the existing CUDA torch/Triton; it does not compile `causal-conv1d`. A partial fast-path warning may remain: the printed `delta_rule=fla` confirms the DeltaNet path is available. Results record the actual kernel paths and package versions. Use separate output folders for identical two-query before/after selections; compare the second query's time and rankings, since the first includes loading/compilation. Kernel metadata is newly part of method settings: use a new output folder for Qwen3.5-based runs with this version, and for each kernel trial; existing result files remain usable for analysis. GPU throughput and numerical agreement still need validation. See [Transformers](https://huggingface.co/docs/transformers/v5.10.2/model_doc/qwen3_5), [FLA installation](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md) and [Triton compatibility](https://github.com/triton-lang/triton#compatibility).

Kev-4B uses the native pointer head and shipped calibration, with FP16 and batch size 1 by default. The backbone loads directly onto the GPU; LoRA layers visit CPU one at a time for merging, avoiding a full host copy of the model. Full-model Colab memory fit still needs validation. Setup removes optional torchao, which these unquantized runs do not use, to avoid conflicts with Colab’s preinstalled version.

Jina scores the full candidate list jointly, using query-token blocks for attention to avoid square attention tensors on small GPUs. This preserves cross-document attention and the native scoring rule; it can be slower than optimized fused kernels. The attention implementation is recorded in results.

Missing datasets and weights download automatically from pinned sources. Benchmark preparation keeps optional JAX on CPU so its BM25 import cannot reserve model GPU memory. Data SHA comparisons are not performed. Result fingerprints use the actual runtime files, so changed data cannot reuse an old checkpoint. Every method receives the same natural BM25 top-100 candidates and frozen short evidence; labels are read only by the evaluator.

| Benchmark | Full queries | Frozen panel | Primary metric |
| --- | ---: | ---: | --- |
| BioASQ 13B curated | 340 | 30 | Capped AP@10, query mean |
| CUREv1 English | 2,000 | 30 | Linear-gain nDCG@10, specialty macro |
| NFCorpus dev | 324 | 20 | nDCG@10, development only |

BioASQ uses a curated corpus with some gold articles missing; this is not full-PubMed competition reproduction. CURE includes model-generated judgments. Keep benchmark/subset results separate. Full GPU inference and memory fit still need validation on the selected hardware.

GPT-6.1 Sol is a one-shot listwise baseline: `sol-listwise-low` for the main comparison and optional `sol-listwise-high` for additional reasoning. Each query uses one request containing all frozen candidates in retrieval order. The response must be a complete permutation; malformed, refused or truncated answers fail without repair. Input tokens and reasoning tokens determine cost. The pinned model is `openai/gpt-6.1-sol-20260929`, using OpenRouter structured outputs.

Results contain `run.json` (rankings, settings and resume checkpoint), `run.evaluation.json` (aggregate and per-query metrics), optional hosted `calls.jsonl`, and one shared `session.json` with method statuses. Reuse the output folder to resume; use a mounted Drive folder to retain work across Colab resets. Hosted methods make paid API calls and cache successful responses. Before any selected model runs, the collector reads `OPENROUTER_API_KEY` (JEV, OpenAI Luna Decisions and Sol) from environment variables or Colab Secrets, or asks through a hidden prompt. CPU/GPU-only selections need no keys. Keys are not saved. OpenAI Luna uses the pinned `openai/gpt-6-luna-decisions-20261006` model through OpenRouter.

Use uv ≥0.13.1 and Python 3.12–3.14. For a local notebook session:

```sh
uv run --locked --group notebooks jupyter lab notebooks/01_collect_results.ipynb
```

This repository contains only the collector runtime and frozen manifests. Datasets, weights, results, papers, research notes and the local analysis workflow are excluded.
