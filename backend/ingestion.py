"""
ingestion.py - PDF ingestion pipeline.

Pipeline:
  PDF → Text Extraction → Chunking → Embedding → Vector DB

Key concepts implemented here:
- Chunking with overlap (prevents context loss at boundaries)
- Metadata tagging (source file, page number, chunk index)
- Deduplication (skip already-ingested files)
"""

import hashlib
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from tqdm import tqdm

from database import add_chunks, file_already_ingested
from embeddings import embed_texts


# ── Text Extraction ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from PDF, page by page.

    Returns:
        List of dicts: {page_num, text}
    """
    pages = []
    doc = fitz.open(pdf_path)

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        # Clean up common PDF artifacts
        text = _clean_text(text)
        if text.strip():
            pages.append({
                "page_num": page_num,
                "text": text
            })

    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Remove PDF extraction noise."""
    # Remove excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove hyphenation at line breaks (common in academic PDFs)
    text = re.sub(r'-\n(\w)', r'\1', text)
    # Normalize spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


# ── Chunking ───────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[dict]:
    """
    Split pages into overlapping chunks.

    Why overlap?
        A fact might span two chunks. Without overlap, a query hitting
        chunk boundary would get half the context. Overlap ensures
        every sentence has full surrounding context in at least one chunk.

    Why not just use chunk_size=10000?
        LLM context windows have limits. More importantly, large chunks
        dilute the relevance signal — a 3000-token chunk retrieved for
        a narrow query brings lots of irrelevant text as noise.

    Args:
        pages: Output from extract_text_from_pdf()
        chunk_size: Target chunk size in characters (≈ tokens * 4)
        chunk_overlap: Overlap between consecutive chunks in characters

    Returns:
        List of chunk dicts with text and metadata
    """
    # Convert token-ish sizes to character counts (rough: 1 token ≈ 4 chars)
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4

    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        page_num = page["page_num"]

        start = 0
        while start < len(text):
            end = start + char_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end near the chunk boundary
                boundary = text.rfind('. ', start + char_size // 2, end)
                if boundary != -1:
                    end = boundary + 1

            chunk_text = text[start:end].strip()

            if len(chunk_text) > 50:  # Skip tiny fragments
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "page_num": page_num,
                        "chunk_index": chunk_index,
                        "start_char": start,
                        "chunk_size_chars": len(chunk_text),
                    }
                })
                chunk_index += 1

            # Move forward with overlap
            start = end - char_overlap
            if start >= len(text):
                break

    return chunks


# ── Ingestion Pipeline ─────────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force: bool = False
) -> int:
    """
    Full ingestion pipeline for a single PDF.

    Args:
        pdf_path: Path to the PDF file
        chunk_size: Chunk size in tokens (approximate)
        chunk_overlap: Overlap between chunks in tokens
        force: Re-ingest even if file was already processed

    Returns:
        Number of chunks ingested
    """
    path = Path(pdf_path)
    filename = path.name

    # Skip if already ingested
    if not force and file_already_ingested(pdf_path):
        print(f"  ⏭️  {filename} already ingested (use --force to re-ingest)")
        return 0

    # 1. Extract text
    print(f"  📄 Extracting text from {filename}...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        print(f"  ⚠️  No text extracted from {filename} (scanned PDF?)")
        return 0

    # 2. Chunk
    print(f"  ✂️  Chunking {len(pages)} pages...")
    raw_chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"  📦 Created {len(raw_chunks)} chunks")

    # 3. Prepare chunks with full metadata and unique IDs
    prepared_chunks = []
    for chunk in raw_chunks:
        # Create deterministic ID from content hash
        content_hash = hashlib.md5(
            (filename + chunk["text"]).encode()
        ).hexdigest()[:12]

        chunk_id = f"{filename}_{chunk['metadata']['chunk_index']}_{content_hash}"

        prepared_chunks.append({
            "id": chunk_id,
            "text": chunk["text"],
            "metadata": {
                **chunk["metadata"],
                "source_file": filename,
                "source_path": str(path.resolve()),
            }
        })

    # 4. Embed in batches (API rate limit friendly)
    print(f"  🧠 Generating embeddings...")
    texts = [c["text"] for c in prepared_chunks]
    embeddings = embed_texts(texts, batch_size=32, show_progress=True)

    # 5. Store in vector DB
    print(f"  💾 Storing in vector DB...")
    add_chunks(prepared_chunks, embeddings)

    return len(prepared_chunks)


def ingest_folder(
    folder_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    force: bool = False
) -> dict[str, int]:
    """
    Ingest all PDFs in a folder.

    Returns:
        Dict mapping filename → chunk count
    """
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {folder_path}")
        return {}

    print(f"Found {len(pdf_files)} PDF(s) in {folder_path}\n")

    results = {}
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")
        count = ingest_pdf(
            str(pdf_path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            force=force
        )
        results[pdf_path.name] = count
        print()

    return results
