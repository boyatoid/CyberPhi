"""
Evaluation Metrics
------------------
Utility functions for scoring model outputs against reference answers.

Metrics implemented (TODO):
  - rouge_l_score(hypothesis, reference)
  - exact_match(hypothesis, reference)
  - flag_extraction_accuracy(outputs, expected_flags)
  - vuln_class_accuracy(outputs, expected_classes)
  - think_block_presence_rate(outputs)
  - cot_step_count_distribution(outputs)

Usage:
    python evaluation/metrics.py --results outputs/eval_results.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric stubs — implement after the model is available
# ---------------------------------------------------------------------------

def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis and reference.

    TODO: implement LCS-based ROUGE-L or use the `rouge-score` package:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'])
        return scorer.score(reference, hypothesis)['rougeL'].fmeasure
    """
    raise NotImplementedError


def exact_match(hypothesis: str, reference: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return hypothesis.strip().lower() == reference.strip().lower()


def flag_extraction_accuracy(outputs: list[str], expected_flags: list[str]) -> float:
    """
    Fraction of outputs that contain the expected CTF flag.

    TODO: handle common CTF flag patterns: flag{...}, CTF{...}, etc.
    """
    raise NotImplementedError


def vuln_class_accuracy(outputs: list[str], expected_classes: list[str]) -> float:
    """
    Fraction of outputs that correctly identify the vulnerability class.
    Checks for class name in output text (case-insensitive).
    """
    correct = 0
    for out, cls in zip(outputs, expected_classes):
        if cls.lower() in out.lower():
            correct += 1
    return correct / max(len(outputs), 1)


def think_block_presence_rate(outputs: list[str]) -> float:
    """Fraction of outputs that contain a non-empty <think>…</think> block."""
    pattern = re.compile(r"<think>\s*.+?\s*</think>", re.DOTALL | re.IGNORECASE)
    hits    = sum(1 for o in outputs if pattern.search(o))
    return hits / max(len(outputs), 1)


def cot_step_count_distribution(outputs: list[str]) -> dict:
    """
    Return a dict mapping step-count bucket → number of outputs.
    Useful for checking that the model produces multi-step reasoning.
    """
    step_re  = re.compile(r"(?m)^(?:\d+[.\)]\s|\-\s|\*\s)", re.IGNORECASE)
    buckets: dict[str, int] = {"0": 0, "1-2": 0, "3-5": 0, "6+": 0}
    for out in outputs:
        n = len(step_re.findall(out))
        if n == 0:
            buckets["0"] += 1
        elif n <= 2:
            buckets["1-2"] += 1
        elif n <= 5:
            buckets["3-5"] += 1
        else:
            buckets["6+"] += 1
    return buckets


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(results_file: Path) -> None:
    """Load a results JSON and print a formatted evaluation report."""
    with open(results_file) as f:
        results = json.load(f)

    outputs  = [r.get("output", "") for r in results]
    expected = [r.get("expected", "") for r in results]
    classes  = [r.get("vuln_type", "") for r in results]

    print("\n=== CyberPhi Evaluation Report ===")
    print(f"Total samples:         {len(results)}")
    print(f"<think> block rate:    {think_block_presence_rate(outputs):.1%}")
    print(f"Vuln-class accuracy:   {vuln_class_accuracy(outputs, classes):.1%}")
    print(f"CoT step distribution: {cot_step_count_distribution(outputs)}")
    print("ROUGE-L and flag accuracy: TODO — requires full implementation")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute evaluation metrics from results JSON")
    parser.add_argument("--results",   required=True,
                        help="Path to JSON file with {output, expected, vuln_type} entries")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    print_report(Path(args.results))


if __name__ == "__main__":
    main()
