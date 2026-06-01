"""
Deduplicator
------------
Loads all JSONL files from data/enriched/, removes near-duplicate entries
using MinHash LSH (Jaccard similarity > threshold), and also performs exact
deduplication on source IDs (cve_id, exploit_id, etc.).

Input:  data/enriched/*.jsonl
Output: data/cleaned/deduped.jsonl

Usage:
    python dataset/cleaning/deduplicator.py
    python dataset/cleaning/deduplicator.py --threshold 0.9 --num-perm 256
"""

from __future__ import annotations
import argparse
import glob
import logging
import re
import sys
from pathlib import Path

import jsonlines
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    CLEANED_DIR,
    DEDUP_THRESHOLD,
    ENRICHED_DIR,
    MINHASH_NUM_PERM,
)

logger    = logging.getLogger(__name__)
OUT_FILE  = CLEANED_DIR / "deduped.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shingles(text: str, k: int = 5) -> set[bytes]:
    """Return the set of k-character shingles from text as bytes."""
    text  = re.sub(r"\s+", " ", text.lower()).strip()
    return {text[i : i + k].encode("utf-8") for i in range(max(1, len(text) - k + 1))}


def _make_minhash(text: str, num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        m.update(shingle)
    return m


def _fingerprint(entry: dict) -> str:
    """Concatenate instruction + output for similarity comparison."""
    return (entry.get("instruction", "") + " " + entry.get("output", "")).strip()


def _exact_id(entry: dict) -> str | None:
    """Return a canonical source ID for exact deduplication."""
    for field in ("cve_id", "exploit_id", "report_id", "url", "entry_id"):
        val = entry.get(field)
        if val:
            return str(val)
    return None


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def deduplicate(threshold: float = DEDUP_THRESHOLD,
                num_perm: int = MINHASH_NUM_PERM) -> None:
    """
    Deduplicate all enriched entries and write cleaned output.

    Args:
        threshold: Jaccard similarity above which entries are considered duplicates
        num_perm:  MinHash permutation count (higher = more accurate, slower)
    """
    # Collect all entries
    all_entries: list[dict] = []
    for fpath in sorted(glob.glob(str(ENRICHED_DIR / "*.jsonl"))):
        with jsonlines.open(fpath) as r:
            file_entries = list(r)
        logger.info("Loaded %d entries from %s", len(file_entries), Path(fpath).name)
        all_entries.extend(file_entries)

    if not all_entries:
        logger.warning("No enriched entries found in %s", ENRICHED_DIR)
        return

    logger.info("Total entries before dedup: %d", len(all_entries))

    # --- Pass 1: exact deduplication on source IDs ---
    seen_exact: set = set()
    after_exact: list[dict] = []
    for entry in all_entries:
        eid = _exact_id(entry)
        if eid and eid in seen_exact:
            continue
        if eid:
            seen_exact.add(eid)
        after_exact.append(entry)

    exact_removed = len(all_entries) - len(after_exact)
    logger.info("Exact dedup removed %d entries (%d remaining)", exact_removed, len(after_exact))

    # --- Pass 2: MinHash LSH near-dedup ---
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict]    = []
    near_removed: int   = 0

    for i, entry in enumerate(tqdm(after_exact, desc="MinHash LSH", unit="entry")):
        fp = _fingerprint(entry)
        if not fp:
            kept.append(entry)
            continue

        mh  = _make_minhash(fp, num_perm)
        key = f"entry_{i}"

        if lsh.query(mh):
            near_removed += 1
            continue

        lsh.insert(key, mh)
        kept.append(entry)

    logger.info(
        "Near-dedup removed %d entries (threshold=%.2f, %d remaining)",
        near_removed, threshold, len(kept),
    )

    # Write output
    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in kept:
            writer.write(entry)

    logger.info(
        "Deduplication complete.\n"
        "  Total input:      %d\n"
        "  Exact duplicates: %d\n"
        "  Near duplicates:  %d\n"
        "  Remaining:        %d\n"
        "  Output:           %s",
        len(all_entries), exact_removed, near_removed, len(kept), OUT_FILE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate enriched dataset entries")
    parser.add_argument("--threshold", type=float, default=DEDUP_THRESHOLD,
                        help="Jaccard similarity threshold (0–1)")
    parser.add_argument("--num-perm",  type=int,   default=MINHASH_NUM_PERM,
                        help="MinHash permutation count")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    deduplicate(args.threshold, args.num_perm)


if __name__ == "__main__":
    main()
