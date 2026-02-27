# RAG Research Assistant

A production-quality RAG backend for querying research papers via CLI.

## Architecture

```
PDF → Text Extraction → Chunking → Embeddings → ChromaDB
                                                    ↓
User Query → Query Embedding → Vector Search → Reranking → LLM → Answer
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
```bash
# If using OpenAI
export OPENAI_API_KEY=sk-...

# Or use Ollama (free, local) — install from ollama.ai then:
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3.2
```

### 3. Ingest papers
```bash
python main.py ingest paper.pdf
python main.py ingest-folder ./papers/
```

### 4. Query
```bash
python main.py query "What is the attention mechanism?"
python main.py query "Compare transformer vs RNN architectures" --rerank --top-k 8
python main.py chat   # interactive mode
python main.py stats  # see what's in the DB
```

## File Structure

```
├── main.py          # CLI entrypoint
├── ingestion.py     # PDF → chunks → embeddings → DB
├── embeddings.py    # Text vectorization (local or OpenAI)
├── database.py      # ChromaDB vector store wrapper
├── retrieval.py     # Search + rerank + LLM generation
├── requirements.txt
└── chroma_db/       # Created automatically on first run
```

## Configuration (env vars)

| Variable | Default | Options |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence-transformers` | `openai` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | any HuggingFace model |
| `LLM_PROVIDER` | `openai` | `ollama` |
| `OPENAI_MODEL` | `gpt-4o-mini` | `gpt-4o`, etc. |
| `OLLAMA_MODEL` | `llama3.2` | any Ollama model |

## Key Design Decisions

**Why chunk_size=500 (not 1000)?**
Smaller chunks = more precise retrieval. A 1000-token chunk might contain
the answer plus 800 tokens of unrelated text, diluting relevance scores.
500 tokens is a balance: enough context for a complete thought, small
enough to be specific.

**Why chunk overlap?**
Facts at chunk boundaries get split. A 10% overlap ensures every sentence
appears fully in at least one chunk.

**Why rerank after vector search?**
Vector search is approximate (good recall, imperfect precision).
Cross-encoder reranking reads (query, document) together for precise
relevance scoring. Pattern: retrieve 20 candidates, rerank, keep top 5.

**Why temperature=0.1 for LLM?**
We want the model to faithfully report what's in the context, not
hallucinate. Low temperature reduces creative extrapolation.

## Intentionally Break It (Learning Exercise)

```bash
# Break 1: Chunk too large — retrieval becomes unfocused
python main.py ingest paper.pdf --chunk-size 2000

# Break 2: No overlap — facts at boundaries get lost
python main.py ingest paper.pdf --chunk-overlap 0

# Break 3: Ask about something not in any paper — should say "not found"
python main.py query "What is the GDP of France in 2024?"

# Break 4: top-k=1 — retrieval misses supporting context
python main.py query "Explain the method" --top-k 1
```
