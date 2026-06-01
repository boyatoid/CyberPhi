"""
Quality Filter
--------------
Applies rule-based quality gates to data/cleaned/deduped.jsonl and writes
passing entries to data/cleaned/filtered.jsonl.

Filter-OUT rules (any one fails → drop):
  • output length < 200 chars
  • <think> block missing or < 100 chars
  • output contains placeholder tokens (TODO, [INSERT], example.com …)
  • instruction word count < 10
  • technical accuracy score < 0.6

Filter-IN requirements (all must pass):
  • output contains at least one security keyword
  • if input is non-empty it contains a code indicator (def, function, class …)
  • CoT reasoning has >= 3 distinct steps

Usage:
    python dataset/cleaning/quality_filter.py
    python dataset/cleaning/quality_filter.py --min-output 300
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    CLEANED_DIR,
    MIN_INSTRUCTION_WORDS,
    MIN_OUTPUT_LENGTH,
    MIN_TECHNICAL_SCORE,
    MIN_THINK_LENGTH,
)

logger   = logging.getLogger(__name__)
IN_FILE  = CLEANED_DIR / "deduped.jsonl"
OUT_FILE = CLEANED_DIR / "filtered.jsonl"

# Tokens that indicate placeholder / boilerplate content
PLACEHOLDER_TOKENS = [
    "todo", "[insert]", "[your_", "example.com", "yourdomain",
    "placeholder", "fixme", "<your", "fill in", "n/a",
]

SECURITY_KEYWORDS = [
    "vulnerability", "exploit", "injection", "overflow", "bypass",
    "payload", "cve-", "cwe-", "rce", "xss", "sqli", "ssrf", "xxe",
    "deserialization", "traversal", "race condition", "idor", "csrf",
    "privilege", "escalation", "shellcode", "gadget", "heap", "stack",
    "sanitize", "sanitize", "remediation", "patch", "mitigate",
]

CODE_INDICATORS = [
    "def ", "function ", "class ", "<?php", "import ", "#include",
    "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "var ", "const ",
    "func ", "public ", "private ",
]

THINK_STEP_PATTERN = re.compile(
    r"(?m)^(?:\d+[.\)]\s|\-\s|\*\s|step\s+\d|phase\s+\d)", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _technical_score(text: str) -> float:
    """Return fraction of security keywords present (0.0 – 1.0)."""
    lower = text.lower()
    hits  = sum(1 for kw in SECURITY_KEYWORDS if kw in lower)
    return min(hits / max(len(SECURITY_KEYWORDS) * 0.15, 1), 1.0)


def _think_block(output: str) -> str:
    """Extract content of <think>…</think> block."""
    m = re.search(r"<think>(.*?)</think>", output, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _count_cot_steps(think_text: str) -> int:
    """Count distinct reasoning steps in the think block."""
    steps = THINK_STEP_PATTERN.findall(think_text)
    if len(steps) >= 3:
        return len(steps)
    # Fallback: count non-empty paragraphs
    paragraphs = [p.strip() for p in think_text.split("\n\n") if p.strip()]
    return len(paragraphs)


def _has_placeholder(text: str) -> bool:
    lower = text.lower()
    return any(tok in lower for tok in PLACEHOLDER_TOKENS)


def _has_security_content(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in SECURITY_KEYWORDS)


def _has_code_indicator(text: str) -> bool:
    return any(ind in text for ind in CODE_INDICATORS)


# ---------------------------------------------------------------------------
# Main filter function
# ---------------------------------------------------------------------------

def filter_dataset(
    min_output: int  = MIN_OUTPUT_LENGTH,
    min_think:  int  = MIN_THINK_LENGTH,
    min_words:  int  = MIN_INSTRUCTION_WORDS,
    min_score:  float = MIN_TECHNICAL_SCORE,
) -> None:
    """
    Apply quality gates and write passing entries.

    Args:
        min_output: minimum output character length
        min_think:  minimum <think> block character length
        min_words:  minimum instruction word count
        min_score:  minimum technical keyword score (0–1)
    """
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run deduplicator first", IN_FILE)
        return

    stats = {
        "total":          0,
        "too_short":      0,
        "missing_think":  0,
        "placeholder":    0,
        "low_word_count": 0,
        "low_tech_score": 0,
        "no_security_kw": 0,
        "few_cot_steps":  0,
        "passed":         0,
    }

    with jsonlines.open(IN_FILE) as reader, jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in tqdm(reader, desc="Quality filter", unit="entry"):
            stats["total"] += 1

            output      = entry.get("output", "")
            instruction = entry.get("instruction", "")
            input_ctx   = entry.get("input", "")
            is_general  = entry.get("source_type") == "general"

            # --- Filter-OUT rules ---
            if len(output) < min_output:
                stats["too_short"] += 1
                continue

            think = _think_block(output)
            # General instruction examples are allowed to have no think block
            if not is_general and len(think) < min_think:
                stats["missing_think"] += 1
                continue

            if _has_placeholder(output) or _has_placeholder(instruction):
                stats["placeholder"] += 1
                continue

            if len(instruction.split()) < min_words:
                stats["low_word_count"] += 1
                continue

            score = _technical_score(output + " " + instruction)
            if not is_general and score < min_score:
                stats["low_tech_score"] += 1
                continue

            # --- Filter-IN requirements ---
            if not is_general and not _has_security_content(output):
                stats["no_security_kw"] += 1
                continue

            if not is_general and think and _count_cot_steps(think) < 3:
                stats["few_cot_steps"] += 1
                continue

            if input_ctx and len(input_ctx) > 50 and not _has_code_indicator(input_ctx):
                # Inputs that claim to be code but have no code indicators are suspicious;
                # only warn, don't drop, since natural language inputs are also valid.
                logger.debug("Non-code input for entry %s", entry.get("entry_id"))

            entry["technical_score"] = round(score, 3)
            writer.write(entry)
            stats["passed"] += 1

    logger.info(
        "Quality filter results:\n"
        "  Total input:        %d\n"
        "  Too short output:   %d\n"
        "  Missing <think>:    %d\n"
        "  Placeholder text:   %d\n"
        "  Low word count:     %d\n"
        "  Low tech score:     %d\n"
        "  No security terms:  %d\n"
        "  Too few CoT steps:  %d\n"
        "  Passed:             %d  (%.1f%%)\n"
        "  Output:             %s",
        stats["total"],
        stats["too_short"],
        stats["missing_think"],
        stats["placeholder"],
        stats["low_word_count"],
        stats["low_tech_score"],
        stats["no_security_kw"],
        stats["few_cot_steps"],
        stats["passed"],
        100 * stats["passed"] / max(stats["total"], 1),
        OUT_FILE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter low-quality dataset entries")
    parser.add_argument("--min-output", type=int,   default=MIN_OUTPUT_LENGTH)
    parser.add_argument("--min-think",  type=int,   default=MIN_THINK_LENGTH)
    parser.add_argument("--min-words",  type=int,   default=MIN_INSTRUCTION_WORDS)
    parser.add_argument("--min-score",  type=float, default=MIN_TECHNICAL_SCORE)
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    filter_dataset(args.min_output, args.min_think, args.min_words, args.min_score)


if __name__ == "__main__":
    main()
