"""
Schema Validator
----------------
Validates every entry in data/final/mixed.jsonl against the unified schema,
checks <think> block integrity, and writes only valid entries to
data/final/validated.jsonl.

Usage:
    python dataset/validation/schema_validator.py
    python dataset/validation/schema_validator.py --strict
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
    FINAL_DIR,
    LANGUAGES,
    MIN_OUTPUT_LENGTH,
    MIN_THINK_LENGTH,
    VULN_TYPES,
)

logger   = logging.getLogger(__name__)
IN_FILE  = FINAL_DIR / "mixed.jsonl"
OUT_FILE = FINAL_DIR / "validated.jsonl"

REQUIRED_FIELDS = [
    "id", "source", "vuln_type", "language", "severity",
    "difficulty", "instruction", "input", "output", "metadata",
]
REQUIRED_METADATA_FIELDS = ["source_id", "date", "tags"]

VALID_SOURCES     = {"nvd", "exploitdb", "ctf", "hackerone", "synthetic", "general"}
VALID_SEVERITIES  = {"critical", "high", "medium", "low", "unknown"}
VALID_DIFFICULTIES = {"beginner", "intermediate", "advanced"}
VALID_VULN_TYPES  = set(VULN_TYPES) | {"unknown"}
VALID_LANGUAGES   = set(LANGUAGES)  | {"unknown"}

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _validate_entry(entry: dict, strict: bool = False) -> list[str]:
    """
    Return a list of error strings. An empty list means valid.
    In strict mode the <think> block is required for all sources.
    """
    errors: list[str] = []

    # Required fields present
    for field in REQUIRED_FIELDS:
        if field not in entry:
            errors.append(f"missing field: {field}")

    if errors:
        return errors

    # Type checks
    for text_field in ("id", "source", "vuln_type", "language",
                       "severity", "difficulty", "instruction", "input", "output"):
        if not isinstance(entry[text_field], str):
            errors.append(f"field '{text_field}' must be str, got {type(entry[text_field])}")

    if not isinstance(entry["metadata"], dict):
        errors.append("field 'metadata' must be dict")
    else:
        for mf in REQUIRED_METADATA_FIELDS:
            if mf not in entry["metadata"]:
                errors.append(f"metadata missing field: {mf}")
        if not isinstance(entry["metadata"].get("tags"), list):
            errors.append("metadata.tags must be a list")

    # Vocabulary checks
    if entry.get("source") not in VALID_SOURCES:
        errors.append(f"invalid source: {entry.get('source')!r}")
    if entry.get("severity") not in VALID_SEVERITIES:
        errors.append(f"invalid severity: {entry.get('severity')!r}")
    if entry.get("difficulty") not in VALID_DIFFICULTIES:
        errors.append(f"invalid difficulty: {entry.get('difficulty')!r}")
    if entry.get("vuln_type") not in VALID_VULN_TYPES:
        errors.append(f"invalid vuln_type: {entry.get('vuln_type')!r}")
    if entry.get("language") not in VALID_LANGUAGES:
        errors.append(f"invalid language: {entry.get('language')!r}")

    output      = entry.get("output", "")
    instruction = entry.get("instruction", "")

    # Content length
    if len(output) < MIN_OUTPUT_LENGTH:
        errors.append(f"output too short: {len(output)} < {MIN_OUTPUT_LENGTH}")

    if not instruction.strip():
        errors.append("instruction is empty")

    # <think> block check
    is_general = entry.get("source") == "general"
    if not is_general or strict:
        think_match = THINK_RE.search(output)
        if not think_match:
            errors.append("<think> block missing from output")
        elif len(think_match.group(1).strip()) < MIN_THINK_LENGTH:
            errors.append(
                f"<think> block too short: {len(think_match.group(1).strip())} < {MIN_THINK_LENGTH}"
            )

    # UUID format (loose check)
    uid = entry.get("id", "")
    if len(uid) < 32 or not re.match(r"[0-9a-f\-]{32,}", uid):
        errors.append(f"id looks malformed: {uid!r}")

    return errors


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def validate(strict: bool = False) -> None:
    """
    Validate mixed.jsonl and write passing entries to validated.jsonl.

    Args:
        strict: if True, require <think> block even for general entries
    """
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run general_mixer first", IN_FILE)
        return

    stats: dict[str, int] = {"total": 0, "valid": 0, "invalid": 0}
    error_counts: dict[str, int] = {}

    with jsonlines.open(IN_FILE) as reader, jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in tqdm(reader, desc="Validating", unit="entry"):
            stats["total"] += 1
            errs = _validate_entry(entry, strict)

            if errs:
                stats["invalid"] += 1
                for e in errs:
                    error_counts[e] = error_counts.get(e, 0) + 1
                logger.debug("Invalid entry %s: %s", entry.get("id"), errs)
            else:
                writer.write(entry)
                stats["valid"] += 1

    # Summary
    error_summary = "\n".join(
        f"  {k}: {v}"
        for k, v in sorted(error_counts.items(), key=lambda x: -x[1])
    )
    logger.info(
        "Validation complete.\n"
        "  Total:   %d\n"
        "  Valid:   %d  (%.1f%%)\n"
        "  Invalid: %d\n"
        "  Error breakdown:\n%s\n"
        "  Output:  %s",
        stats["total"],
        stats["valid"],
        100 * stats["valid"] / max(stats["total"], 1),
        stats["invalid"],
        error_summary or "  (none)",
        OUT_FILE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate final dataset against unified schema")
    parser.add_argument("--strict",    action="store_true",
                        help="Require <think> block for all entries including general")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    validate(args.strict)


if __name__ == "__main__":
    main()
