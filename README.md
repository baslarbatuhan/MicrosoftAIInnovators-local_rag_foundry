# Local RAG Q&A Assistant

A fully **offline**, document-grounded question-answering assistant. It runs the LLM
on-device with Microsoft **Foundry Local** and uses the **RAG** (Retrieval-Augmented
Generation) pattern to ground every answer in a local knowledge base — no internet
connection, cloud account, or GPU required.

> Built as a summer school project, inspired by the Microsoft Tech Community guide
> [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968).

## Features

- 🔌 **100% local:** Zero network calls after the initial model download
- 📚 **Source attribution:** Every answer reports which document it came from, derived deterministically from retrieval
- 🚫 **No hallucinations:** If the answer is not in the knowledge base, the model says "I don't have that information" instead of guessing (10/10 correct refusals in evaluation)
- 🔍 **Transparent retrieval:** The `--verbose` flag shows exactly which chunks the model saw, with similarity scores

## Architecture

```
User Question (CLI)
      │
      ▼
answer_query()  ──►  get_top_chunks()  ──►  SQLite (chunk + embedding)
      │                    │ cosine similarity, top-3
      │              qwen3-embedding-0.6b (query embedding)
      ▼
phi-3.5-mini  ◄── system prompt + [Source]-tagged context + question
      │
      ▼
Answer + Source list
```

| Component | Technology |
|---|---|
| LLM runtime | Microsoft Foundry Local (in-process SDK, v1.2.3) |
| Chat model | `phi-3.5-mini` (3.8B, `temperature=0`, `max_tokens=256`) |
| Embedding model | `qwen3-embedding-0.6b` (1024-dimensional vectors) |
| Vector store | SQLite (JSON-serialized embeddings, brute-force cosine similarity) |
| Knowledge base | 5 documents / 114 chunks (paragraph-based + overlap, max 800 chars) |

## Setup

**Requirements:** Windows, Python 3.10+ (tested with 3.14)

```powershell
# 1. Install Foundry Local
winget install Microsoft.FoundryLocal --accept-source-agreements
foundry --version   # verify

# 2. Install Python dependencies
#    (if you have multiple Python installs, always use "python -m pip", not bare "pip")
python -m pip install -r requirements.txt

# 3. Prepare the knowledge base (downloads 5 documents + test question sets from Kaggle)
python prepare_dataset.py

# 4. Chunk, embed, and write the documents to the database (data/rag.db)
python ingest.py
```

> On first run the models are downloaded automatically (`qwen3-embedding-0.6b` ~600 MB,
> `phi-3.5-mini` ~2 GB) and cached — subsequent runs are fully offline.

## Usage

```powershell
python main.py            # normal mode
python main.py --verbose  # also shows retrieved chunks with similarity scores
```

Example session (the knowledge base is in English, so questions should be asked in English):

```
Soru: What do keybullet kin drop?

Cevap: Keybullet Kin drop keys upon death.
Kaynaklar: bullet_kin.txt

Soru: How much health does the Mutant Bullet Kin have?

Cevap: I don't have that information.
```

Type `exit`, `quit`, or `çık` to quit.

**To use your own documents:** drop your `.txt` files into `documents/` and run
`python ingest.py` (if the first line is in the `Kaynak: <url>` format, it is parsed
as the source URL).

## Evaluation

`python evaluate.py` — runs the 20 questions in `eval/` (10 answerable + 10 unanswerable),
prints the metrics, and writes the details to `eval/eval_results.csv`.

Final results (deterministic, `temperature=0`):

| Metric | Result |
|---|---|
| Retrieval hit rate | 9/10 (90%)* |
| Answer rate | 8/10 (80%) |
| Correct refusals (no hallucination) | **10/10 (100%)** |
| Response time | avg. 13.0 s (CPU) |

\* The answer passage for the single missed question is not present in the dataset's
document text at all (a dataset flaw) — refusing that question is the correct behavior.

## Project Structure

```
main.py              → entry point: answer_query() + CLI loop
ingest.py            → chunk + embed documents, write to SQLite
retrieval.py         → get_top_chunks(): top-k chunks via cosine similarity
evaluate.py          → automated 20-question evaluation + metric report
prepare_dataset.py   → builds the knowledge base and test sets from the Kaggle dataset
documents/           → knowledge base (.txt documents)
eval/                → test question sets + evaluation results
exercises/           → conceptual demo scripts from the learning phase (not part of the pipeline)
data/                → generated databases (created automatically by ingest.py)
```

## Known Limitations

- **Latency:** ~13 s per question on CPU (the reality of a 3.8B model). Could be reduced
  with a GPU variant or a smaller model on capable hardware.
- **Small-model reading limits:** The model can occasionally confuse a *similar* passage
  in the context with the one being asked about (one such case was observed with two
  similar changelog entries). This is not hallucination — answers always come from the
  context — but correctness is not guaranteed.
- **Over-caution:** Strict grounding can make the model refuse 1-2 borderline answerable
  questions (a precision/recall trade-off — deliberately chosen in favor of zero hallucination).
- **Language:** The knowledge base is in English, so the prompt and questions are English.
  If you add Turkish documents, the system prompt should be translated as well
  (`main.py` → `SYSTEM_PROMPT`).
- **Scale:** Brute-force cosine similarity is fine for small collections; thousands of
  documents would call for a dedicated vector database (ChromaDB, Qdrant, etc.).

## Lessons Learned

1. **Language consistency is critical:** A Turkish system prompt combined with English
   content confused the model; switching the prompt to English raised correct refusals
   from 30% to 100%.
2. **Chunk boundaries make or break retrieval:** Heading/date lines landing in a different
   chunk than their content lost us answerable questions; chunk overlap fixed it.
3. **More context ≠ better:** Raising `top_k` from 3 to 5 increased the answer rate but
   brought hallucinations back and made latency 2.4× worse — it was reverted.
4. **You can't know without measuring:** The first "seemingly working" system hid four
   separate issues; all of them surfaced only through systematic evaluation
   (7 measure→diagnose→fix rounds).

## References

- [Microsoft Foundry Local documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/)
- [Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968) (Tech Community)
- Knowledge base and test questions: [single-topic-rag-evaluation-dataset](https://www.kaggle.com/datasets/samuelmatsuoharris/single-topic-rag-evaluation-dataset) (Kaggle)
