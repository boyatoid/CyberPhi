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

def _lcs_len(x: list, y: list) -> int:
    """Length of the longest common subsequence of two token lists."""
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            curr[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """
    Compute ROUGE-L F1 between hypothesis and reference.
    Uses rouge_score package when available; falls back to a native LCS implementation
    (the package's NLTK dependency has a broken sqlite3 on some macOS/conda environments).
    """
    try:
        from rouge_score import rouge_scorer  # noqa: PLC0415
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return scorer.score(reference, hypothesis)["rougeL"].fmeasure
    except Exception:
        pass

    h_tokens = hypothesis.lower().split()
    r_tokens  = reference.lower().split()
    if not h_tokens or not r_tokens:
        return 0.0
    lcs       = _lcs_len(h_tokens, r_tokens)
    precision = lcs / len(h_tokens)
    recall    = lcs / len(r_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(hypothesis: str, reference: str) -> bool:
    """Case-insensitive exact match after stripping whitespace."""
    return hypothesis.strip().lower() == reference.strip().lower()


_FLAG_RE = re.compile(r"(?:flag|ctf|htb|picoctf|pico)\{[^}]+\}", re.IGNORECASE)


def flag_extraction_accuracy(outputs: list[str], expected_flags: list[str]) -> float:
    """Fraction of outputs that contain the expected CTF flag.

    Pairs with an empty expected flag are excluded from the denominator —
    they can never be scored correct, so counting them would deflate accuracy.
    """
    correct = 0
    scored  = 0
    for out, expected in zip(outputs, expected_flags):
        if not expected:
            continue
        scored += 1
        if expected.lower() in out.lower():
            correct += 1
            continue
        # Also accept if any extracted flag token matches
        for match in _FLAG_RE.findall(out):
            if match.lower() == expected.lower():
                correct += 1
                break
    return correct / max(scored, 1)


def vuln_class_accuracy(outputs: list[str], expected_classes: list[str]) -> float:
    """
    Fraction of outputs that correctly identify the vulnerability class.
    Checks for class name in output text (case-insensitive).

    Pairs with an empty/unknown class are excluded — an empty string is a
    substring of everything and would count as a free correct answer.
    """
    correct = 0
    scored  = 0
    for out, cls in zip(outputs, expected_classes):
        if not cls or cls == "unknown":
            continue
        scored += 1
        if cls.lower() in out.lower():
            correct += 1
    return correct / max(scored, 1)


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

    # run_benchmarks.py writes {"benchmark": ..., "raw": [...]}; also accept
    # a bare list of {output, expected, ...} entries.
    if isinstance(results, dict):
        results = results.get("raw", [])

    outputs  = [r.get("output", "") for r in results]
    expected = [r.get("expected", "") for r in results]
    classes  = [r.get("vuln_type", "") for r in results]

    flags = [r.get("expected_flag", "") for r in results]
    mean_rouge = (
        sum(rouge_l_score(o, e) for o, e in zip(outputs, expected) if e) /
        max(sum(1 for e in expected if e), 1)
    )

    print("\n=== CyberPhi Evaluation Report ===")
    print(f"Total samples:         {len(results)}")
    print(f"<think> block rate:    {think_block_presence_rate(outputs):.1%}")
    print(f"Vuln-class accuracy:   {vuln_class_accuracy(outputs, classes):.1%}")
    print(f"Mean ROUGE-L:          {mean_rouge:.3f}")
    if any(flags):
        print(f"Flag extraction acc:   {flag_extraction_accuracy(outputs, flags):.1%}")
    print(f"CoT step distribution: {cot_step_count_distribution(outputs)}")


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
