"""
Sample Reviewer
---------------
Prints N random entries from data/final/validated.jsonl in a human-readable
format for quick sanity-checking before training.

Usage:
    python dataset/validation/sample_reviewer.py
    python dataset/validation/sample_reviewer.py --samples 20
    python dataset/validation/sample_reviewer.py --vuln-type sqli
    python dataset/validation/sample_reviewer.py --source hackerone --samples 5
"""

from __future__ import annotations
import argparse
import logging
import random
import re
import sys
import textwrap
from pathlib import Path

import jsonlines

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import FINAL_DIR

logger  = logging.getLogger(__name__)
IN_FILE = FINAL_DIR / "validated.jsonl"

SEPARATOR = "=" * 72
SUBSEP    = "-" * 72


def _wrap(text: str, width: int = 80, indent: str = "  ") -> str:
    lines = text.splitlines()
    wrapped = []
    for line in lines:
        if len(line) > width:
            wrapped.extend(textwrap.wrap(line, width=width, initial_indent=indent,
                                          subsequent_indent=indent))
        else:
            wrapped.append(indent + line)
    return "\n".join(wrapped)


def _extract_think(output: str) -> tuple[str, str]:
    """Split output into (think_block, final_answer)."""
    m = re.search(r"<think>(.*?)</think>(.*)", output, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", output.strip()


def _print_entry(entry: dict, index: int) -> None:
    think, answer = _extract_think(entry.get("output", ""))
    tags = entry.get("metadata", {}).get("tags", [])

    print(f"\n{SEPARATOR}")
    print(f"  Entry #{index + 1}")
    print(f"  ID:         {entry.get('id', '')[:24]}…")
    print(f"  Source:     {entry.get('source', '?'):<15}  "
          f"Vuln:  {entry.get('vuln_type', '?')}")
    print(f"  Language:   {entry.get('language', '?'):<15}  "
          f"Severity: {entry.get('severity', '?')}")
    print(f"  Difficulty: {entry.get('difficulty', '?'):<15}  "
          f"Tags: {', '.join(tags)}")
    print(SUBSEP)
    print("  INSTRUCTION:")
    print(_wrap(entry.get("instruction", ""), indent="    "))

    if entry.get("input", "").strip():
        print(SUBSEP)
        print("  INPUT (code/context):")
        snippet = entry["input"][:800]
        if len(entry["input"]) > 800:
            snippet += "\n  …[truncated]"
        print(_wrap(snippet, indent="    "))

    if think:
        print(SUBSEP)
        print("  <THINK> BLOCK (first 600 chars):")
        snippet = think[:600]
        if len(think) > 600:
            snippet += "\n  …[truncated]"
        print(_wrap(snippet, indent="    "))

    print(SUBSEP)
    print("  FINAL ANSWER (first 800 chars):")
    snippet = answer[:800]
    if len(answer) > 800:
        snippet += "\n  …[truncated]"
    print(_wrap(snippet, indent="    "))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def review(
    n_samples:  int = 10,
    vuln_type:  str | None = None,
    source:     str | None = None,
) -> None:
    """
    Print random samples from the validated dataset.

    Args:
        n_samples: number of entries to display
        vuln_type: filter to this vulnerability type
        source:    filter to this data source
    """
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run schema_validator first", IN_FILE)
        return

    all_entries: list[dict] = []
    with jsonlines.open(IN_FILE) as r:
        for entry in r:
            if vuln_type and entry.get("vuln_type") != vuln_type:
                continue
            if source and entry.get("source") != source:
                continue
            all_entries.append(entry)

    if not all_entries:
        print(f"No entries found (vuln_type={vuln_type!r}, source={source!r})")
        return

    print(f"\nTotal matching entries: {len(all_entries)}")
    print(f"Showing {min(n_samples, len(all_entries))} random samples")

    samples = random.sample(all_entries, min(n_samples, len(all_entries)))
    for i, entry in enumerate(samples):
        _print_entry(entry, i)

    print(f"\n{SEPARATOR}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Print random samples from validated dataset")
    parser.add_argument("--samples",   type=int, default=10,
                        help="Number of samples to display")
    parser.add_argument("--vuln-type", default=None,
                        help="Filter to this vulnerability type")
    parser.add_argument("--source",    default=None,
                        help="Filter to this data source (nvd, ctf, hackerone…)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    review(args.samples, args.vuln_type, args.source)


if __name__ == "__main__":
    main()
