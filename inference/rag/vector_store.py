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
import json
import logging
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from inference.ollama_client import OllamaClient

# chromadb is imported lazily inside __init__ — it requires sqlite3 which is broken
# on some macOS/conda environments. Deferring avoids breaking module-level imports.

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
    """

    def __init__(self, persist_dir: Path = CHROMA_PERSIST_DIR):
        import chromadb  # noqa: PLC0415 — deferred to avoid breaking module imports on some envs
        self.persist_dir  = persist_dir
        self._ollama      = OllamaClient()
        self.client       = chromadb.PersistentClient(path=str(persist_dir))
        self.collection   = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        """Chunk, embed, and upsert documents into the collection."""
        ids        = []
        embeddings = []
        docs       = []
        metas      = []

        for text, meta in zip(texts, metadatas):
            for chunk in self._chunk(text):
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk))
                embedding = self._ollama.embed(chunk)
                ids.append(chunk_id)
                embeddings.append(embedding)
                docs.append(chunk)
                metas.append(meta)

        if ids:
            self.collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
            logger.info("Upserted %d chunks into '%s'", len(ids), COLLECTION_NAME)

    def query(self, text: str, n_results: int = 5) -> list[dict]:
        """Return the top-n most similar chunks with their metadata."""
        embedding = self._ollama.embed(text)
        results   = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({"text": doc, "metadata": meta, "distance": dist})
        return output

    def delete_collection(self) -> None:
        """Wipe the collection and start fresh."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection '%s' reset.", COLLECTION_NAME)

    @staticmethod
    def _chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
        """Split text into overlapping character-level chunks."""
        chunks = []
        start  = 0
        while start < len(text):
            end = start + size
            chunks.append(text[start:end])
            if end >= len(text):
                break
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
        index_path = Path(args.index)
        logger.info("Indexing %s …", index_path)
        texts     = []
        metadatas = []

        if index_path.suffix == ".jsonl":
            with open(index_path) as f:
                for line in f:
                    entry = json.loads(line)
                    text = entry.get("description") or entry.get("input") or entry.get("output", "")
                    if text:
                        texts.append(text)
                        metadatas.append({
                            "source":    entry.get("source", "unknown"),
                            "vuln_type": entry.get("vuln_type", "unknown"),
                            "id":        entry.get("id", ""),
                        })
        else:
            text = index_path.read_text()
            texts.append(text)
            metadatas.append({"source": index_path.name})

        store.add_documents(texts, metadatas)

    if args.query:
        results = store.query(args.query, args.top_k)
        for r in results:
            print(f"[dist={r['distance']:.3f}] [{r['metadata'].get('source', '')}] {r['text'][:200]}")


if __name__ == "__main__":
    main()
