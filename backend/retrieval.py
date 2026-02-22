"""
retrieval.py - The retrieval + generation pipeline.

This is where RAG quality lives or dies. Steps:
  1. Embed the query
  2. Vector search (top-k candidates)
  3. Reranking (optional but improves quality significantly)
  4. Context compression (keep only relevant parts)
  5. Prompt construction + LLM generation

Why reranking?
  Vector search finds "approximately similar" chunks — it's fast but
  can miss nuance. A cross-encoder reranker reads (query, chunk) pairs
  and scores them precisely. It's slower but far more accurate.
  The typical pattern: retrieve top-20 with vector search, rerank,
  keep top-5 for the LLM.
"""

import os
from typing import Optional

from database import search
from embeddings import embed_query


# ── Configuration ──────────────────────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "ollama"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None


# ── Reranking ──────────────────────────────────────────────────────────────────

def _get_reranker():
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANKER_MODEL)
            print(f"  Loaded reranker: {RERANKER_MODEL}")
        except ImportError:
            print("  ⚠️  sentence-transformers not available, skipping reranking")
            return None
    return _reranker


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder.

    Cross-encoders are more accurate than bi-encoders (used in vector search)
    because they see (query, document) together — not as independent vectors.
    Tradeoff: must compare query against each candidate = O(n) inference calls.
    That's why we only rerank top-20 candidates, not the full DB.

    Args:
        query: The user's question
        chunks: Candidate chunks from vector search
        top_k: How many to return after reranking

    Returns:
        Reranked list (most relevant first), truncated to top_k
    """
    reranker = _get_reranker()
    if reranker is None or not chunks:
        return chunks[:top_k]

    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    # Sort by reranker score (higher = more relevant)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    reranked = []
    for score, chunk in ranked[:top_k]:
        chunk = chunk.copy()
        chunk["rerank_score"] = float(score)
        reranked.append(chunk)

    return reranked


# ── Context Compression ────────────────────────────────────────────────────────

def compress_context(query: str, chunks: list[dict], max_chars: int = 8000) -> list[dict]:
    """
    Trim context to fit within LLM context window while keeping most relevant content.

    Simple strategy: keep chunks in relevance order until we hit max_chars.
    Advanced strategy (not implemented): sentence-level filtering within chunks.
    """
    compressed = []
    total_chars = 0

    for chunk in chunks:
        if total_chars + len(chunk["text"]) > max_chars:
            # Partially include the chunk if there's remaining space
            remaining = max_chars - total_chars
            if remaining > 200:  # Only add if meaningful amount remains
                partial_chunk = chunk.copy()
                partial_chunk["text"] = chunk["text"][:remaining] + "..."
                compressed.append(partial_chunk)
            break
        compressed.append(chunk)
        total_chars += len(chunk["text"])

    return compressed


# ── Prompt Engineering ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research assistant that answers questions based exclusively on provided research paper excerpts.

Rules:
1. Answer ONLY from the provided context. Do not use prior knowledge.
2. If the context doesn't contain enough information, say "The provided papers don't cover this."
3. Cite sources as [File, Page X] when making specific claims.
4. Be precise and academic in tone.
5. If multiple papers contradict each other, acknowledge the disagreement."""


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Build the grounded prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        source_label = f"[Source {i}: {meta.get('source_file', 'unknown')}, Page {meta.get('page_num', '?')}]"
        context_parts.append(f"{source_label}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""CONTEXT FROM RESEARCH PAPERS:
{context}

---

QUESTION: {query}

Answer based solely on the above context. Cite sources using [Source N] notation."""


# ── LLM Generation ─────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call the configured LLM with the grounded prompt."""
    if LLM_PROVIDER == "openai":
        return _call_openai(prompt)
    elif LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


def _call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,   # Low temp = more faithful to context
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _call_ollama(prompt: str) -> str:
    """Call a local Ollama model (free, private)."""
    import json
    try:
        import requests
    except ImportError:
        raise ImportError("requests not installed. Run: pip install requests")

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.1}
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ── Main Query Function ────────────────────────────────────────────────────────

def query_knowledge_base(
    question: str,
    top_k: int = 5,
    rerank: bool = True,
    show_citations: bool = True,
    filter_file: Optional[str] = None
) -> dict:
    """
    Full RAG query pipeline.

    Args:
        question: The user's question
        top_k: Number of chunks to include in context
        rerank: Whether to use cross-encoder reranking
        show_citations: Whether to return source metadata
        filter_file: Optional: only search in a specific PDF file

    Returns:
        Dict with: answer, sources, chunks_retrieved, chunks_used
    """
    if not question.strip():
        return {"answer": "Please ask a question.", "sources": []}

    # Step 1: Embed query
    query_vec = embed_query(question)

    # Step 2: Vector search — retrieve more candidates if reranking
    search_k = top_k * 4 if rerank else top_k
    filter_meta = {"source_file": filter_file} if filter_file else None

    raw_chunks = search(query_vec, top_k=search_k, filter_meta=filter_meta)

    if not raw_chunks:
        return {
            "answer": "No relevant documents found. Please ingest some PDFs first.",
            "sources": []
        }

    # Step 3: Rerank
    if rerank and len(raw_chunks) > 1:
        final_chunks = rerank_chunks(question, raw_chunks, top_k=top_k)
    else:
        final_chunks = raw_chunks[:top_k]

    # Step 4: Compress context (fit within LLM window)
    final_chunks = compress_context(question, final_chunks, max_chars=12000)

    # Step 5: Build prompt and call LLM
    prompt = build_prompt(question, final_chunks)
    answer = call_llm(prompt)

    # Step 6: Build source citations
    sources = []
    if show_citations:
        seen = set()
        for chunk in final_chunks:
            meta = chunk["metadata"]
            key = (meta.get("source_file"), meta.get("page_num"))
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": meta.get("source_file", "unknown"),
                    "page": meta.get("page_num", "?"),
                    "score": chunk.get("rerank_score", chunk.get("score", 0.0))
                })

    return {
        "answer": answer,
        "sources": sources,
        "chunks_retrieved": len(raw_chunks),
        "chunks_used": len(final_chunks)
    }
