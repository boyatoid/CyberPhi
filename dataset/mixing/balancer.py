"""
Dataset Balancer
----------------
Analyses the vuln_type and language distribution in data/cleaned/formatted.jsonl,
downsamples over-represented classes, and flags under-represented ones.

Rules:
  • Any vuln_type with > 30% share is downsampled to exactly 30%.
  • Any vuln_type with < 3%  share is printed as "underrepresented"
    (flag it for variant generation to top up).

Output: data/final/balanced.jsonl

Usage:
    python dataset/mixing/balancer.py
    python dataset/mixing/balancer.py --max-fraction 0.25
"""

import argparse
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    CLEANED_DIR,
    FINAL_DIR,
    MAX_VULN_TYPE_FRACTION,
    MIN_VULN_TYPE_FRACTION,
    VULN_TYPES,
)

logger   = logging.getLogger(__name__)
IN_FILE  = CLEANED_DIR / "formatted.jsonl"
OUT_FILE = FINAL_DIR   / "balanced.jsonl"


def _ascii_table(counts: Counter, total: int, title: str = "") -> str:
    if title:
        lines = [title, "-" * 52]
    else:
        lines = ["-" * 52]
    lines.append(f"{'Category':<25} {'Count':>7}  {'%':>6}")
    lines.append("-" * 52)
    for key, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(total, 1)
        lines.append(f"{key:<25} {cnt:>7}  {pct:>5.1f}%")
    lines.append("-" * 52)
    lines.append(f"{'TOTAL':<25} {total:>7}  100.0%")
    return "\n".join(lines)


def balance(
    max_fraction: float = MAX_VULN_TYPE_FRACTION,
    min_fraction: float = MIN_VULN_TYPE_FRACTION,
) -> None:
    """
    Balance the formatted dataset across vulnerability types.

    Args:
        max_fraction: downsample any vuln_type that exceeds this share
        min_fraction: flag as underrepresented below this share
    """
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run formatter first", IN_FILE)
        return

    # --- Load all entries ---
    entries: list[dict] = []
    with jsonlines.open(IN_FILE) as r:
        for entry in tqdm(r, desc="Loading", unit="entry"):
            entries.append(entry)

    if not entries:
        logger.warning("No entries in %s", IN_FILE)
        return

    total = len(entries)
    logger.info("Loaded %d entries", total)

    # --- Analyse distribution before balancing ---
    type_counts = Counter(e.get("vuln_type", "unknown") for e in entries)
    lang_counts = Counter(e.get("language", "unknown") for e in entries)

    logger.info("\nVuln-type distribution BEFORE balancing:\n%s",
                _ascii_table(type_counts, total, "Vuln Types"))
    logger.info("\nLanguage distribution:\n%s",
                _ascii_table(lang_counts, total, "Languages"))

    # --- Downsample over-represented types ---
    max_entries_per_type = max(1, int(total * max_fraction))
    by_type: dict[str, list[dict]] = {}
    for entry in entries:
        vt = entry.get("vuln_type", "unknown")
        by_type.setdefault(vt, []).append(entry)

    balanced: list[dict] = []
    for vtype, bucket in by_type.items():
        if len(bucket) > max_entries_per_type:
            sampled = random.sample(bucket, max_entries_per_type)
            logger.info("Downsampled %s: %d → %d entries", vtype, len(bucket), max_entries_per_type)
        else:
            sampled = bucket
        balanced.extend(sampled)

    random.shuffle(balanced)
    new_total = len(balanced)

    # --- Report after balancing ---
    new_counts = Counter(e.get("vuln_type", "unknown") for e in balanced)
    logger.info("\nVuln-type distribution AFTER balancing:\n%s",
                _ascii_table(new_counts, new_total, "Vuln Types (balanced)"))

    # --- Flag underrepresented ---
    underrepresented = []
    for vtype in VULN_TYPES:
        fraction = new_counts.get(vtype, 0) / max(new_total, 1)
        if fraction < min_fraction:
            underrepresented.append((vtype, new_counts.get(vtype, 0), round(fraction * 100, 2)))

    if underrepresented:
        logger.warning(
            "Underrepresented vuln types (< %.0f%%) — consider running variant_generator:\n%s",
            min_fraction * 100,
            "\n".join(f"  {vt}: {cnt} entries ({pct}%)" for vt, cnt, pct in underrepresented),
        )

    # --- Write output ---
    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in balanced:
            writer.write(entry)

    logger.info(
        "Balancing complete. Input: %d → Output: %d | Saved → %s",
        total, new_total, OUT_FILE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance dataset across vulnerability classes")
    parser.add_argument("--max-fraction", type=float, default=MAX_VULN_TYPE_FRACTION,
                        help="Max fraction a single vuln_type may occupy (0–1)")
    parser.add_argument("--min-fraction", type=float, default=MIN_VULN_TYPE_FRACTION,
                        help="Flag as underrepresented below this fraction")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    balance(args.max_fraction, args.min_fraction)


if __name__ == "__main__":
    main()
