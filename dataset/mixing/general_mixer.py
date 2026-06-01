"""
General Instruction Mixer
--------------------------
Downloads a small general-instruction dataset from HuggingFace and mixes
~15% of the cybersecurity dataset's size into the final dataset.

This prevents catastrophic forgetting of general language capabilities
during fine-tuning.

Input:  data/final/balanced.jsonl
Output: data/final/mixed.jsonl

Usage:
    python dataset/mixing/general_mixer.py
    python dataset/mixing/general_mixer.py --ratio 0.10 --dataset tatsu-lab/alpaca
"""

from __future__ import annotations
import argparse
import logging
import random
import sys
import uuid
from pathlib import Path

import jsonlines
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    FINAL_DIR,
    GENERAL_DATASET_NAME,
    GENERAL_MIX_RATIO,
    HUGGINGFACE_TOKEN,
)

logger   = logging.getLogger(__name__)
IN_FILE  = FINAL_DIR / "balanced.jsonl"
OUT_FILE = FINAL_DIR / "mixed.jsonl"


def _convert_alpaca(row: dict) -> dict:
    """Convert an Alpaca-style row to our unified schema."""
    return {
        "id":         str(uuid.uuid4()),
        "source":     "general",
        "vuln_type":  "unknown",
        "language":   "unknown",
        "severity":   "unknown",
        "difficulty": "beginner",
        "instruction": row.get("instruction", ""),
        "input":       row.get("input", ""),
        "output":      row.get("output", ""),
        "metadata": {
            "source_id": "",
            "date":      "",
            "tags":      ["general"],
        },
    }


def _convert_ultrachat(row: dict) -> dict | None:
    """Convert an UltraChat-style row (conversation list) to our schema."""
    messages = row.get("messages", [])
    if len(messages) < 2:
        return None
    instruction = messages[0].get("content", "") if messages[0].get("role") == "user" else ""
    output      = messages[1].get("content", "") if messages[1].get("role") == "assistant" else ""
    if not instruction or not output:
        return None
    return {
        "id":         str(uuid.uuid4()),
        "source":     "general",
        "vuln_type":  "unknown",
        "language":   "unknown",
        "severity":   "unknown",
        "difficulty": "beginner",
        "instruction": instruction,
        "input":       "",
        "output":      output,
        "metadata": {
            "source_id": "",
            "date":      "",
            "tags":      ["general"],
        },
    }


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def mix(ratio: float = GENERAL_MIX_RATIO, dataset_name: str = GENERAL_DATASET_NAME) -> None:
    """
    Mix general instruction data into the balanced cybersecurity dataset.

    Args:
        ratio:        fraction of general examples relative to cybersec dataset size
        dataset_name: HuggingFace dataset to sample from
    """
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run balancer first", IN_FILE)
        return

    # --- Load cybersecurity entries ---
    cybersec: list[dict] = []
    with jsonlines.open(IN_FILE) as r:
        for entry in tqdm(r, desc="Loading balanced dataset", unit="entry"):
            cybersec.append(entry)

    if not cybersec:
        logger.warning("No entries in %s", IN_FILE)
        return

    n_general = max(1, int(len(cybersec) * ratio))
    logger.info(
        "Cybersec entries: %d | General target: %d (ratio=%.0f%%)",
        len(cybersec), n_general, ratio * 100,
    )

    # --- Download & sample general dataset ---
    logger.info("Downloading %s from HuggingFace…", dataset_name)
    kwargs: dict = {}
    if HUGGINGFACE_TOKEN:
        kwargs["token"] = HUGGINGFACE_TOKEN

    try:
        ds = load_dataset(dataset_name, split="train", **kwargs)
    except Exception as exc:
        logger.error("Failed to load dataset '%s': %s", dataset_name, exc)
        return

    logger.info("Downloaded %d rows from %s", len(ds), dataset_name)

    # Pick converter based on dataset structure
    sample_row = ds[0]
    if "messages" in sample_row:
        converter = _convert_ultrachat
    else:
        converter = _convert_alpaca

    # Sample up to n_general rows
    indices = random.sample(range(len(ds)), min(n_general * 3, len(ds)))
    general_entries: list[dict] = []
    for idx in indices:
        row   = ds[idx]
        entry = converter(row)
        if entry and entry["instruction"] and entry["output"]:
            general_entries.append(entry)
        if len(general_entries) >= n_general:
            break

    logger.info("Sampled %d general entries", len(general_entries))

    # --- Interleave and write ---
    combined = cybersec + general_entries
    random.shuffle(combined)

    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in combined:
            writer.write(entry)

    logger.info(
        "Mix complete. Total: %d (%d cybersec + %d general) → %s",
        len(combined), len(cybersec), len(general_entries), OUT_FILE,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Mix general instruction data into cybersec dataset")
    parser.add_argument("--ratio",   type=float, default=GENERAL_MIX_RATIO,
                        help="Fraction of general examples (default 0.15)")
    parser.add_argument("--dataset", default=GENERAL_DATASET_NAME,
                        help="HuggingFace dataset name")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    mix(args.ratio, args.dataset)


if __name__ == "__main__":
    main()
