"""
Q&A Pair Generator
-------------------
Reads a text file or JSONL and generates 3–5 question-answer pairs per chunk
using the Anthropic API. Covers five question types:

    factual      – "What is X vulnerability?"
    reasoning    – "Why is this code vulnerable?"
    procedural   – "How would you exploit X?"
    defensive    – "How do you prevent X?"
    detection    – "How would you detect X in a codebase?"

Input:  any .txt or .jsonl file (pass via --input-file)
Output: data/enriched/qa_pairs.jsonl

Usage:
    python dataset/enrichment/qa_generator.py --input-file docs/owasp_top10.txt
    python dataset/enrichment/qa_generator.py --input-file data/raw/nvd_cves.jsonl --limit 50
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from textwrap import wrap

import anthropic
import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_INPUT_COST_PER_MTOK,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL,
    CLAUDE_OUTPUT_COST_PER_MTOK,
    ENRICHED_DIR,
)

logger = logging.getLogger(__name__)
OUTPUT_FILE = ENRICHED_DIR / "qa_pairs.jsonl"

CHUNK_SIZE   = 1200   # characters per text chunk
QA_PER_CHUNK = 4      # how many Q&A pairs to request per chunk

SYSTEM_PROMPT = (
    "You are a cybersecurity instructor creating training data for a security LLM. "
    "Generate precise, technically accurate question-answer pairs from the provided text."
)

QA_TYPES = ["factual", "reasoning", "procedural", "defensive", "detection"]


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into overlapping chunks of ~size characters."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > size and current:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text[:size]]


def _build_prompt(chunk: str) -> str:
    qa_type_list = ", ".join(QA_TYPES)
    return (
        f"Generate exactly {QA_PER_CHUNK} question-answer pairs from the following "
        f"security text. Use these question types (vary them): {qa_type_list}.\n\n"
        "For each pair, respond in JSON lines format:\n"
        '{"question_type": "...", "instruction": "<question>", "input": "", "output": "<answer>"}\n\n'
        f"Text:\n{chunk}\n\n"
        "Output only the JSON lines, one per line. Each answer must be technically accurate "
        "and at least 3 sentences long."
    )


def _call_claude(client: anthropic.Anthropic, prompt: str) -> tuple[str, int, int]:
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return text, response.usage.input_tokens, response.usage.output_tokens
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 10
            logger.warning("Rate limited. Waiting %ds", wait)
            time.sleep(wait)
        except anthropic.APIConnectionError as exc:
            wait = 2 ** attempt * 5
            logger.warning("Connection error: %s. Waiting %ds", exc, wait)
            time.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                time.sleep(2 ** attempt * 8)
            else:
                raise
    raise RuntimeError("Claude call failed after 5 attempts")


def _parse_qa_response(raw: str, chunk_id: str, source_doc: str) -> list[dict]:
    """Parse JSON-lines output from Claude into structured Q&A entries."""
    pairs: list[dict] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip code-fence markers Claude sometimes adds
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
            if "instruction" not in obj or "output" not in obj:
                continue
            obj["entry_id"]     = f"qa_{chunk_id}_{len(pairs)}"
            obj["source_type"]  = "synthetic"
            obj["source_doc"]   = source_doc
            obj.setdefault("input",         "")
            obj.setdefault("question_type", "factual")
            pairs.append(obj)
        except json.JSONDecodeError:
            logger.debug("Could not parse Q&A line: %s", line[:80])
    return pairs


def _load_existing_ids() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    ids: set = set()
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            ids.add(entry.get("entry_id", ""))
    return ids


def _iter_chunks(input_path: Path) -> list[tuple[str, str]]:
    """
    Yield (chunk_id, chunk_text) pairs from either a .txt or .jsonl file.
    """
    pairs: list[tuple[str, str]] = []
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        with jsonlines.open(input_path) as r:
            for entry in r:
                # Use description / summary / content field if available
                text = (
                    entry.get("description")
                    or entry.get("summary")
                    or entry.get("raw_content")
                    or entry.get("output")
                    or ""
                )
                if len(text) < 100:
                    continue
                chunk_id = hashlib.md5(text[:200].encode()).hexdigest()[:10]
                pairs.append((chunk_id, text[:CHUNK_SIZE * 3]))

    else:  # plain text
        text   = input_path.read_text(encoding="utf-8", errors="replace")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()[:10]
            pairs.append((chunk_id, chunk))

    return pairs


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_qa(input_file: str | Path, limit: int | None = None) -> None:
    """
    Generate Q&A pairs from a text or JSONL file.

    Args:
        input_file: path to the source document
        limit:      maximum total Q&A pairs to generate
    """
    input_path   = Path(input_file)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return

    existing_ids  = _load_existing_ids()
    client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    chunks        = _iter_chunks(input_path)
    total_saved   = 0
    total_in_tok  = 0
    total_out_tok = 0

    logger.info("Generating Q&A pairs from %d chunks in %s", len(chunks), input_path.name)

    with tqdm(total=limit or len(chunks), desc="Q&A generation", unit="chunk") as pbar:
        for chunk_id, chunk_text in chunks:
            prompt = _build_prompt(chunk_text)
            try:
                raw, in_t, out_t = _call_claude(client, prompt)
            except Exception as exc:
                logger.warning("Skipping chunk %s: %s", chunk_id, exc)
                pbar.update(1)
                continue

            total_in_tok  += in_t
            total_out_tok += out_t

            pairs = _parse_qa_response(raw, chunk_id, input_path.name)

            with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                for pair in pairs:
                    if pair["entry_id"] in existing_ids:
                        continue
                    writer.write(pair)
                    existing_ids.add(pair["entry_id"])
                    total_saved += 1
                    if limit and total_saved >= limit:
                        pbar.update(1)
                        cost = (total_in_tok / 1e6 * CLAUDE_INPUT_COST_PER_MTOK +
                                total_out_tok / 1e6 * CLAUDE_OUTPUT_COST_PER_MTOK)
                        logger.info("Reached limit=%d | est. cost $%.4f", limit, cost)
                        return

            pbar.update(1)

    cost = (total_in_tok / 1e6 * CLAUDE_INPUT_COST_PER_MTOK +
            total_out_tok / 1e6 * CLAUDE_OUTPUT_COST_PER_MTOK)
    logger.info("Done. Saved %d Q&A pairs → %s | est. cost $%.4f",
                total_saved, OUTPUT_FILE, cost)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from security documents")
    parser.add_argument("--input-file", required=True,
                        help="Path to .txt or .jsonl source document")
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generate_qa(args.input_file, args.limit)


if __name__ == "__main__":
    main()
