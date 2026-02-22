"""
database.py - Vector database management using ChromaDB (persistent).

Why ChromaDB?
- Persistent storage (survives restarts)
- Built-in embedding support
- Fast cosine similarity search
- Simple API
"""

import os
import chromadb
from chromadb.config import Settings
from pathlib import Path

# Directory for persistent vector DB
DB_PATH = "./chroma_db"

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection(name: str = "research_papers"):
    """Get or create the main collection."""
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
    return _collection


def init_db():
    """Initialize the database. Call this at startup."""
    os.makedirs(DB_PATH, exist_ok=True)
    get_collection()
    print(f"📦 Vector DB initialized at {DB_PATH}")


def get_stats() -> dict:
    """Return statistics about the knowledge base."""
    collection = get_collection()
    count = collection.count()

    # Get unique files from metadata
    files_info = []
    if count > 0:
        all_items = collection.get(include=["metadatas"])
        file_chunks = {}
        for meta in all_items["metadatas"]:
            fname = meta.get("source_file", "unknown")
            file_chunks[fname] = file_chunks.get(fname, 0) + 1
        files_info = [{"name": k, "chunks": v} for k, v in file_chunks.items()]

    # DB size on disk
    db_size = sum(
        f.stat().st_size for f in Path(DB_PATH).rglob("*") if f.is_file()
    ) / (1024 * 1024) if Path(DB_PATH).exists() else 0

    return {
        "documents": len(files_info),
        "chunks": count,
        "db_size_mb": db_size,
        "files": files_info
    }


def add_chunks(chunks: list[dict], embeddings: list[list[float]]):
    """
    Add chunks to the vector database.

    Args:
        chunks: List of dicts with keys: id, text, metadata
        embeddings: Parallel list of embedding vectors
    """
    collection = get_collection()

    ids = [c["id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # ChromaDB upsert handles duplicates gracefully
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def search(query_embedding: list[float], top_k: int = 5, filter_meta: dict = None) -> list[dict]:
    """
    Semantic search in the vector DB.

    Args:
        query_embedding: Query vector
        top_k: Number of results to return
        filter_meta: Optional metadata filter (e.g. {"source_file": "paper.pdf"})

    Returns:
        List of dicts with: text, metadata, score, id
    """
    collection = get_collection()

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=filter_meta,
        include=["documents", "metadatas", "distances"]
    )

    output = []
    for i in range(len(results["ids"][0])):
        # ChromaDB returns distance (lower = better for cosine)
        # Convert to similarity score (higher = better)
        distance = results["distances"][0][i]
        score = 1 - distance  # cosine similarity

        output.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": score
        })

    return output


def file_already_ingested(file_path: str) -> bool:
    """Check if a file has already been ingested (by filename)."""
    collection = get_collection()
    if collection.count() == 0:
        return False

    filename = Path(file_path).name
    results = collection.get(
        where={"source_file": filename},
        limit=1
    )
    return len(results["ids"]) > 0
