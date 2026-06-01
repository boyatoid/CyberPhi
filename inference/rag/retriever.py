"""
RAG Retriever
-------------
Fetches the top-k relevant chunks from the vector store and formats them
as a context block to prepend to the LLM prompt.

The retriever is called by thinking_loop.py before the THINK step so that
the model has up-to-date CVE descriptions, OWASP guidance, or tool docs
even if the information post-dates its training cut-off.

Usage (programmatic):
    retriever = Retriever()
    context   = retriever.retrieve("HTTP/2 rapid reset attack")
    prompt    = f"Context:\n{context}\n\nQuestion: How does CVE-2023-44487 work?"
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

from inference.rag.vector_store import SecurityVectorStore

TOP_K           = 5
MAX_CONTEXT_LEN = 2000   # characters — trim to fit in prompt window


class Retriever:
    """Wraps SecurityVectorStore to produce a formatted context string."""

    def __init__(self):
        self.store = SecurityVectorStore()

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Return a formatted context string of the top-k relevant chunks.

        Args:
            query:  natural language security question
            top_k:  number of chunks to retrieve

        Returns:
            Formatted context string, or "" if no results.
        """
        try:
            results = self.store.query(query, top_k)
        except Exception as exc:
            logger.warning("Vector store query failed: %s", exc)
            return ""

        if not results:
            return ""

        parts = [
            f"[Source: {r['metadata'].get('source', 'unknown')}] {r['text']}"
            for r in results
        ]
        context = "\n\n---\n\n".join(parts)
        return context[:MAX_CONTEXT_LEN]
