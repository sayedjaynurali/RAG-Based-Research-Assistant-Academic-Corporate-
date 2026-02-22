"""
RAG Research Assistant - Main Entry Point
CLI-based interface for ingesting PDFs and querying the knowledge base.
"""

import argparse
import sys
from pathlib import Path
from ingestion import ingest_pdf, ingest_folder
from retrieval import query_knowledge_base
from database import init_db, get_stats


def main():
    parser = argparse.ArgumentParser(
        description="RAG Research Assistant - Chat with your research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py ingest paper.pdf
  python main.py ingest-folder ./papers/
  python main.py query "What is the transformer architecture?"
  python main.py stats
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Ingest single PDF
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a single PDF")
    ingest_parser.add_argument("pdf_path", help="Path to the PDF file")
    ingest_parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in tokens (default: 500)")
    ingest_parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap in tokens (default: 50)")

    # Ingest folder
    folder_parser = subparsers.add_parser("ingest-folder", help="Ingest all PDFs in a folder")
    folder_parser.add_argument("folder_path", help="Path to folder containing PDFs")
    folder_parser.add_argument("--chunk-size", type=int, default=500)
    folder_parser.add_argument("--chunk-overlap", type=int, default=50)

    # Query
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("question", help="Your question")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    query_parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    query_parser.add_argument("--no-citations", action="store_false", dest="citations", help="Disable source citations")

    # Interactive mode
    subparsers.add_parser("chat", help="Start interactive chat mode")

    # Stats
    subparsers.add_parser("stats", help="Show knowledge base statistics")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Initialize database
    init_db()

    if args.command == "ingest":
        path = Path(args.pdf_path)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
        print(f"Ingesting {path.name}...")
        count = ingest_pdf(str(path), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        print(f"✅ Ingested {count} chunks from {path.name}")

    elif args.command == "ingest-folder":
        folder = Path(args.folder_path)
        if not folder.exists():
            print(f"Error: Folder not found: {folder}")
            sys.exit(1)
        results = ingest_folder(str(folder), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        total = sum(results.values())
        print(f"\n✅ Ingested {len(results)} files, {total} total chunks")
        for fname, count in results.items():
            print(f"   {fname}: {count} chunks")

    elif args.command == "query":
        response = query_knowledge_base(
            args.question,
            top_k=args.top_k,
            rerank=args.rerank,
            show_citations=args.citations
        )
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(response["answer"])
        if args.citations and response.get("sources"):
            print("\n" + "-"*60)
            print("SOURCES:")
            for i, src in enumerate(response["sources"], 1):
                print(f"  [{i}] {src['file']} (page {src['page']}, score: {src['score']:.3f})")

    elif args.command == "chat":
        print("🔬 RAG Research Assistant - Interactive Mode")
        print("Type 'quit' to exit, 'stats' to see DB info\n")
        while True:
            try:
                question = input("You: ").strip()
                if not question:
                    continue
                if question.lower() in ("quit", "exit", "q"):
                    break
                if question.lower() == "stats":
                    stats = get_stats()
                    print(f"  Documents: {stats['documents']}, Chunks: {stats['chunks']}")
                    continue
                response = query_knowledge_base(question, top_k=5, rerank=True)
                print(f"\nAssistant: {response['answer']}")
                if response.get("sources"):
                    print("Sources:", ", ".join(f"{s['file']}:p{s['page']}" for s in response["sources"][:3]))
                print()
            except (KeyboardInterrupt, EOFError):
                break
        print("Goodbye!")

    elif args.command == "stats":
        stats = get_stats()
        print("\n📊 Knowledge Base Statistics")
        print(f"  Documents : {stats['documents']}")
        print(f"  Chunks    : {stats['chunks']}")
        print(f"  DB size   : {stats['db_size_mb']:.2f} MB")
        if stats.get("files"):
            print("\n  Files:")
            for f in stats["files"]:
                print(f"    - {f['name']}: {f['chunks']} chunks")


if __name__ == "__main__":
    main()
