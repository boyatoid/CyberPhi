"""
ChromaDB Vector Store
----------------------
Indexes CVE descriptions, OWASP docs, and other security text into a
persistent ChromaDB collection so the RAG retriever can do fast similarity
search at inference time.

Each document is chunked, embedded via Ollama's embedding endpoint, and
stored with metadata (source, vuln_type, cve_id, etc.).

Usage:
    python inference/rag/vector_store.py --index data/raw/nvd_cves.jsonl
    python inference/rag/vector_store.py --index docs/owasp_top10.txt
    python inference/rag/vector_store.py --query "SQL injection bypass WAF"
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ChromaDB dependency — install: pip install chromadb
# from chromadb import PersistentClient
# from chromadb.config import Settings

# Ollama client for embeddings
# from inference.ollama_client import OllamaClient

CHROMA_PERSIST_DIR = Path("./chroma_db")
COLLECTION_NAME    = "cyberphi_security_docs"
CHUNK_SIZE         = 800    # characters per chunk
CHUNK_OVERLAP      = 100    # character overlap between consecutive chunks


class SecurityVectorStore:
    """
    Wraps a ChromaDB persistent collection.

    Responsibilities:
      - add_documents(texts, metadatas): chunk, embed, and upsert
      - query(text, n_results):         embed query and return top-k chunks
      - delete_collection():            wipe and start fresh

    TODO:
        1. Initialize PersistentClient with CHROMA_PERSIST_DIR
        2. Get or create collection COLLECTION_NAME with cosine distance
        3. In add_documents: chunk text with overlap, embed each chunk via
           OllamaClient.embed(), upsert into the collection
        4. In query: embed the query, call collection.query(), return
           (documents, metadatas, distances) tuples

    Example (once implemented):
        store = SecurityVectorStore()
        store.add_documents(
            texts=["CVE-2023-44487 is an HTTP/2 Rapid Reset vulnerability…"],
            metadatas=[{"cve_id": "CVE-2023-44487", "severity": "high"}]
        )
        results = store.query("rapid reset attack", n_results=5)
    """

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        # TODO: self.client     = PersistentClient(path=str(persist_dir))
        # TODO: self.collection = self.client.get_or_create_collection(
        #           COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        raise NotImplementedError("Wire up ChromaDB — see TODOs in this class")

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        """Chunk, embed, and upsert documents into the collection."""
        raise NotImplementedError

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most similar chunks with their metadata."""
        raise NotImplementedError

    @staticmethod
    def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        """Split text into overlapping character-level chunks."""
        chunks = []
        start  = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or query the security vector store")
    parser.add_argument("--index",  default=None,
                        help="Path to .jsonl or .txt file to index")
    parser.add_argument("--query",  default=None,
                        help="Query string to search the vector store")
    parser.add_argument("--top-k",  type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s")

    store = SecurityVectorStore()
    if args.index:
        logger.info("Indexing %s …", args.index)
        # TODO: load file, extract text fields, call store.add_documents()
    if args.query:
        results = store.query(args.query, args.top_k)
        for r in results:
            print(r)


if __name__ == "__main__":
    main()
