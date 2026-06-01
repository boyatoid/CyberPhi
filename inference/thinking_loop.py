"""
3-Step Thinking Loop
---------------------
Implements a multi-turn reasoning loop that mirrors the CoT training format:

  Step 1 — THINK
    The model generates a <think>…</think> block with internal reasoning:
    attack surface mapping, vulnerability identification, exploitation path.

  Step 2 — REFLECT
    A second pass critiques the initial thinking:
    "Is the exploitation path correct? Did I miss edge cases? Is the CVSS accurate?"

  Step 3 — ANSWER
    A final pass produces the clean, user-facing answer informed by the think
    and reflect steps.

The fine-tuned Phi-3.5-mini model may naturally produce <think> blocks in a
single generation (matching training format). This loop adds an explicit
reflect step for higher-stakes queries where extra scrutiny is valuable.

Usage:
    python inference/thinking_loop.py --query "How does CVE-2023-44487 work?"
    python inference/thinking_loop.py --query "..." --use-rag
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)

from inference.ollama_client import OllamaClient


THINK_SYSTEM = (
    "You are a senior penetration tester. "
    "Think step-by-step inside a <think>…</think> block. "
    "Do not produce a final answer yet."
)

REFLECT_SYSTEM = (
    "You are a security peer reviewer. "
    "Given an initial analysis, identify any errors, missing attack paths, "
    "or inaccurate CVSS justifications. Be concise and precise."
)

ANSWER_SYSTEM = (
    "You are a senior penetration tester providing a final, definitive security analysis. "
    "Use the internal reasoning and reflection to produce a clear, accurate answer."
)


def thinking_loop(query: str, use_rag: bool = False) -> str:
    """
    Run the 3-step think → reflect → answer loop.

    Args:
        query:   the user's security question
        use_rag: if True, fetch relevant context chunks before thinking

    Returns:
        final answer string
    """
    client = OllamaClient()

    augmented_query = query
    if use_rag:
        from inference.rag.retriever import Retriever  # noqa: PLC0415 — lazy: pulls in chromadb
        retriever = Retriever()
        context = retriever.retrieve(query)
        if context:
            augmented_query = f"Context:\n{context}\n\nQuestion: {query}"

    logger.debug("Step 1: THINK")
    think_output = client.generate(prompt=augmented_query, system=THINK_SYSTEM)

    logger.debug("Step 2: REFLECT")
    reflection = client.generate(
        prompt=f"Initial analysis:\n{think_output}\n\nReflect on any errors.",
        system=REFLECT_SYSTEM,
    )

    logger.debug("Step 3: ANSWER")
    final_answer = client.generate(
        prompt=(
            f"Query: {query}\n\n"
            f"Reasoning:\n{think_output}\n\n"
            f"Reflection:\n{reflection}"
        ),
        system=ANSWER_SYSTEM,
    )

    return final_answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CyberPhi 3-step reasoning loop")
    parser.add_argument("--query",   required=True, help="Security question to answer")
    parser.add_argument("--use-rag", action="store_true",
                        help="Retrieve relevant context from the CVE vector store")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    answer = thinking_loop(args.query, args.use_rag)
    print(answer)


if __name__ == "__main__":
    main()
