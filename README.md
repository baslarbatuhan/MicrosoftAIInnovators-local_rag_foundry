# Foundry Local Documentation Assistant

An offline RAG Q&A assistant that answers questions about **Microsoft Foundry Local** —
running **on** Foundry Local itself. Ask how to install the CLI, how tool calling works, or
what changed in SDK 0.8.0, and it answers from the current official documentation, cites its
sources, and refuses to guess when the docs don't contain the answer. No internet connection,
cloud account, or paid service required at runtime.

> Built as a summer school project, inspired by the Microsoft Tech Community guide
> [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968).

## Why RAG? A measurable answer

Foundry Local shipped after the base model's training cutoff, so the LLM literally cannot
know it. We measured this: asked without context, `phi-3.5-mini` confidently confused Foundry
Local with **four different real products** (an open-source initiative, an Ethereum toolchain,
ForgeRock identity management, and a VFX company's SDK) — 5 wrong answers out of 5
([evidence](knowledge_bases/foundry/eval/baseline_no_rag.txt)). The same questions with RAG
produce correct, source-grounded answers with **zero fabrications**
([evidence](knowledge_bases/foundry/eval/with_rag.txt)). The knowledge base is dated 2026 —
over two years past the model's cutoff — and can be refreshed any time by re-running ingestion,
without touching the model.

## Pipeline

Every component is **local and free** — no cloud, no paid API, no PyTorch.

```
                          Question (CLI / Streamlit)
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                  ▼
   Vector search (top-15)                            BM25 / FTS5 (top-20)
   qwen3-embedding-0.6b, cosine                      exact-term keyword match
            └────────────────────┬───────────────────────────┘
                                 ▼
                     Candidate pool (~30 chunks)
                                 ▼
                Cross-encoder reranker  →  top-3
                bge-reranker-base (ONNX, CPU)
                                 ▼
                Parent-document expansion
                (each chunk → its full source doc, windowed)
                                 ▼
                phi-3.5-mini  ◄── grounded system prompt + context + question
                                 ▼
                     Answer + deterministic source list
```

**Ingestion** (built once, refreshable): documents are split with **contextual, markdown-heading-aware
chunking** (each chunk is embedded as `heading path + content`, which bridges the gap between a user's
natural question and the docs' technical wording); pure-metadata lines (e.g. date stamps) are dropped.

| Component | Technology |
|---|---|
| LLM runtime | Microsoft Foundry Local (in-process SDK) |
| Chat model | `phi-3.5-mini` (3.8B, `temperature=0`, `max_tokens=256`; CUDA GPU variant when available, CPU fallback) |
| Embedding model | `qwen3-embedding-0.6b` (1024-dim vectors) |
| Reranker | `bge-reranker-base` cross-encoder, ONNX via `onnxruntime` (CPU, no PyTorch) |
| Vector + keyword store | SQLite — JSON-serialized embeddings (brute-force cosine) + FTS5/BM25 index |
| Knowledge base | 29 documents / **874 chunks** — official Foundry Local docs (MS Learn, CC-BY-4.0), product repo (MIT), installed-CLI help |

## Setup

**Requirements:** Windows, Python 3.10+ (tested with 3.14). Run all commands from the repo root.

```powershell
# 1. Install Foundry Local
winget install Microsoft.FoundryLocal --accept-source-agreements
foundry --version   # verify

# 2. Install Python dependencies
#    (if you have multiple Python installs, always use "python -m pip", not bare "pip")
python -m pip install -r requirements.txt

# 3. Build the knowledge base (downloads the official docs from GitHub — no scraping)
python scripts/prepare_dataset.py

# 4. Chunk, embed, and index the documents (data/foundry.db)
python -m rag.ingest
```

> On first run the models are downloaded automatically and cached — subsequent runs are fully
> offline: `qwen3-embedding-0.6b` (~600 MB), `phi-3.5-mini` (~2 GB), and the reranker
> `bge-reranker-base` ONNX (~280 MB). On machines with an NVIDIA GPU, the CUDA execution provider
> and the GPU chat-model variant (~2.1 GB) are also downloaded and used automatically.

## Usage

```powershell
python -m rag                     # CLI Q&A
python -m rag --verbose           # also logs retrieved chunks + stage timings

python -m streamlit run rag/app.py   # web UI (streaming answers, source captions,
                                     # retrieval-detail expander)
```

Example session:

```
Soru: How does tool calling work with Foundry Local?

Cevap: You prompt the model with definitions of available tools; the model decides which
tools to call and with what inputs, then your application runs them and feeds the results
back. Run `foundry model list` and look for the `tools` task to see which models support it.
Kaynaklar: how-to__how-to-use-tool-calling-with-foundry-local.txt

Soru: What is the internal Microsoft codename for the Foundry Local project?

Cevap: I don't have that information.
```

Type `exit`, `quit`, or `çık` to quit.

**To use your own documents:** drop `.txt` files into `knowledge_bases/foundry/documents/` and
re-run `python -m rag.ingest` (if the first line is `Kaynak: <url>`, it is parsed as the source URL).

## Evaluation

`python -m rag.evaluate` — runs a **37-question** set (26 answerable, each verified against the
corpus with `grep`, + 11 unanswerable traps such as pricing, codenames, a Java SDK, fine-tuning,
and image generation), prints the metrics, and writes details to
`knowledge_bases/foundry/eval/eval_results.csv`.

Current baseline (deterministic, `temperature=0`, GPU):

| Metric | Result |
|---|---|
| Retrieval hit rate | 26/26 (100%) |
| Answer rate | 26/26 (100%) |
| Answer correctness (max-sim recall ≥ 0.63) | **26/26 (100%)** |
| Correct refusals (no hallucination) | **11/11 (100%)** |
| Response time | avg. ~5.6 s (GPU, RTX 4060) |

**Answer correctness** is scored *reference-free of length*: the answer is split into sentences and,
for each reference sentence, the maximum cosine similarity over the answer's sentences is taken
(a recall-oriented, BERTScore-style match). This rewards complete, code-block-rich answers instead
of penalizing them the way a whole-answer cosine does, while still keeping genuinely off-target
answers below threshold.

## Project Structure

```
rag/                          core package (run modules with: python -m rag.<name>)
├── config.py                 paths, model aliases, retrieval flags (absolute paths, CWD-independent)
├── ingest.py                 contextual chunking + embedding → SQLite (+ FTS5 index)
├── retrieval.py              hybrid retrieval + cross-encoder rerank + parent-document expansion
├── reranker.py               bge-reranker-base ONNX cross-encoder (local, free, no PyTorch)
├── core.py                   answer_query() + grounded system prompt + CLI loop
├── evaluate.py               automated evaluation (recall-oriented correctness metric)
├── telemetry.py              per-query stage timings → data/telemetry.jsonl
├── app.py                    Streamlit web UI
└── __main__.py               `python -m rag` → CLI entry point
scripts/prepare_dataset.py    downloads the official docs from GitHub (raw + API, no scraping)
knowledge_bases/foundry/      documents/ (official docs) + eval/ (question set + before/after evidence)
examples/                     learning-phase demo scripts (embeddings, SQLite, prompting, baseline test)
tests/                        pytest unit tests (run in CI via .github/workflows/ci.yml)
data/                         generated database (gitignored; created by python -m rag.ingest)
```

## Known Limitations

- **Latency:** ~5–7 s per answer on GPU. Retrieval is fast (~0.5 s); the cost is the reranker
  (CPU), the parent-document context, and generating richer, code-block answers.
- **Single-turn:** the assistant answers each question independently — conversation history is not
  yet sent to the model, so follow-ups like "and on Linux?" aren't resolved against prior turns.
- **Small-model reading limits:** answers always come from the retrieved context, but a 3.8B model
  can occasionally misread which part of a large context answers the question.
- **Language:** the knowledge base is English, so the prompt and questions are English.
- **Scale:** brute-force cosine over 874 chunks is fine; thousands of documents would call for a
  dedicated vector database.

## Lessons Learned

This project is a log of **measure → diagnose → fix → re-measure**. What actually moved the numbers:

1. **Sequence beats technique.** Hybrid BM25 was measured and rejected *twice* on its own (it fed
   keyword-matched noise to the small model and induced fabrications). Re-introduced **behind a
   cross-encoder reranker** — which filters the noise — it became safe and fixed a long-standing
   retrieval miss. The reranker itself was the single biggest gain (answer rate 14→16 on the earlier set).
2. **More context is a double-edged sword.** Parent-document expansion made answers richer, but the
   extra context weakened the model's refusal discipline (it started to guess). The fix wasn't to
   shrink the context — it was to **harden the prompt** (allow inference, but refuse as a last resort
   and never deflect to a related term). A too-strict first draft over-refused; a *balanced* prompt
   restored both.
3. **The metric can be the bug.** A whole-answer cosine metric was *penalizing* correct, code-rich
   answers for diverging from terse references. Switching to a recall-oriented max-similarity metric
   (one small function, same embedding model) fixed it without any new dependency.
4. **Solve the class, not the instance.** A garbage "read this file" answer traced to date-stamp
   lines ranking #1 in retrieval; the fix was a general noise-line rule, not a one-off regex.
5. **You can't know without measuring.** Query rewriting, "possible questions", a larger `top_k`, a
   bigger chat model, and min-size chunk merging were each implemented, measured, found to hurt or add
   cost without benefit, and reverted — with the evidence kept.

## References

- [Microsoft Foundry Local documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/) (knowledge-base source, CC-BY-4.0 via [MicrosoftDocs/azure-ai-docs](https://github.com/MicrosoftDocs/azure-ai-docs))
- [microsoft/foundry-local](https://github.com/microsoft/foundry-local) product repository (MIT)
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) reranker (MIT); ONNX build via [Xenova/bge-reranker-base](https://huggingface.co/Xenova/bge-reranker-base)
- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Tech Community)
