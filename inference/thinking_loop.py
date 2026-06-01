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

# Import deferred until implementation is complete to avoid crashing on missing deps
# from inference.ollama_client import OllamaClient
# from inference.rag.retriever import Retriever


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

    TODO:
        1. Instantiate OllamaClient and Retriever
        2. If use_rag: context = retriever.retrieve(query)
           Prepend context to the query string
        3. Step 1: call client.generate(prompt=query, system=THINK_SYSTEM)
           → think_output (should contain <think>…</think>)
        4. Step 2: call client.generate(
               prompt=f"Initial analysis:\n{think_output}\n\nReflect on any errors.",
               system=REFLECT_SYSTEM)
           → reflection
        5. Step 3: call client.generate(
               prompt=f"Query: {query}\n\nReasoning:\n{think_output}\n\nReflection:\n{reflection}",
               system=ANSWER_SYSTEM)
           → final_answer
        6. Return final_answer
    """
    raise NotImplementedError(
        "thinking_loop requires a trained GGUF model loaded in Ollama. "
        "See inference/README.md for setup."
    )


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
