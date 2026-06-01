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

# from inference.rag.vector_store import SecurityVectorStore

TOP_K           = 5
MAX_CONTEXT_LEN = 2000   # characters — trim to fit in prompt window


class Retriever:
    """
    Wraps SecurityVectorStore to produce a formatted context string.

    TODO:
        1. Instantiate SecurityVectorStore in __init__
        2. In retrieve(query, top_k):
           a. Call store.query(query, top_k)
           b. Format each chunk as:
              "[Source: {source}] {chunk_text}"
           c. Join with "\\n\\n---\\n\\n"
           d. Truncate to MAX_CONTEXT_LEN
           e. Return formatted string
        3. Handle empty results gracefully (return "")

    Example (once implemented):
        retriever = Retriever()
        ctx = retriever.retrieve("heap overflow exploitation")
        # ctx = "[Source: nvd] CVE-2023-1234: A heap-based buffer overflow in..."
    """

    def __init__(self):
        # TODO: self.store = SecurityVectorStore()
        raise NotImplementedError("Wire up SecurityVectorStore — see TODOs")

    def retrieve(self, query: str, top_k: int = TOP_K) -> str:
        """
        Return a formatted context string of the top-k relevant chunks.

        Args:
            query:  natural language security question
            top_k:  number of chunks to retrieve

        Returns:
            Formatted context string, or "" if no results.
        """
        raise NotImplementedError
