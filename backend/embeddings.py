"""
embeddings.py - Text embedding generation.

Embeddings convert text into dense vectors so we can do semantic search.
"What is attention?" and "How does self-attention work?" will have
similar vectors even though they share few words.

Why does embedding model choice matter?
  - Dimension size: larger = more expressive but slower & more memory
  - Training data: a model trained on scientific text understands "ablation"
    differently than a general-purpose model
  - Max token length: most models cap at 512 tokens — chunks exceeding
    this get silently truncated
  - Speed: local models are free but slow; API models are fast but cost money

This module supports:
  - sentence-transformers (local, free, great quality)
  - OpenAI text-embedding-3-small (API, fast, strong)
"""

import os
from typing import Optional
from tqdm import tqdm


# ── Configuration ──────────────────────────────────────────────────────────────

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"  # 384-dim, fast, solid quality
    # Alternatives:
    # "all-mpnet-base-v2"           # 768-dim, higher quality, slower
    # "BAAI/bge-small-en-v1.5"      # state-of-art small model
    # "allenai-specter"             # trained on scientific papers!
)

_model = None  # Lazy-loaded


# ── Sentence Transformers (Local) ──────────────────────────────────────────────

def _get_local_model():
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "pip install sentence-transformers"
            )
        print(f"  Loading embedding model: {SENTENCE_TRANSFORMER_MODEL}")
        _model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _model


def _embed_local(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> list[list[float]]:
    model = _get_local_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2 normalize → cosine sim = dot product
        convert_to_numpy=True
    )
    return embeddings.tolist()


# ── OpenAI Embeddings (API) ────────────────────────────────────────────────────

def _embed_openai(texts: list[str], batch_size: int = 100, show_progress: bool = False) -> list[list[float]]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai not installed. Run: pip install openai")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    client = OpenAI(api_key=OPENAI_API_KEY)
    all_embeddings = []

    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    iterator = tqdm(batches, desc="Embedding") if show_progress else batches

    for batch in iterator:
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"  # 1536-dim, fast and cheap
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> list[list[float]]:
    """
    Embed a list of texts into vectors.

    Args:
        texts: List of strings to embed
        batch_size: How many texts to embed at once
        show_progress: Show tqdm progress bar

    Returns:
        List of embedding vectors (each a list of floats)
    """
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts, batch_size=batch_size, show_progress=show_progress)
    else:
        return _embed_local(texts, batch_size=batch_size, show_progress=show_progress)


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.
    Some models (like bge) use a special query prefix for better retrieval.
    """
    # BGE models use a query prefix for better asymmetric retrieval
    if "bge" in SENTENCE_TRANSFORMER_MODEL.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"

    embeddings = embed_texts([query])
    return embeddings[0]


def get_embedding_dimension() -> int:
    """Return the dimension of the embedding vectors."""
    test = embed_texts(["test"])
    return len(test[0])
