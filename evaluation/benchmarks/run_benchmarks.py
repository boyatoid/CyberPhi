"""
Benchmark Runner
----------------
Evaluates the fine-tuned CyberPhi model against:

  1. ctf-held-out   – CTF challenges withheld from training (flag accuracy)
  2. securityeval   – SecurityEval 130-task Python coding benchmark
  3. cwe-qa         – CWE knowledge Q&A (ROUGE-L / BERTScore)

Results are written to outputs/eval_results_{benchmark}.json.

Usage:
    python evaluation/benchmarks/run_benchmarks.py --model cyberphi
    python evaluation/benchmarks/run_benchmarks.py --benchmark securityeval --limit 20
    python evaluation/benchmarks/run_benchmarks.py --model cyberphi --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# from inference.ollama_client import OllamaClient
# from inference.thinking_loop import thinking_loop
# from evaluation.metrics import (
#     flag_extraction_accuracy, vuln_class_accuracy,
#     think_block_presence_rate, rouge_l_score,
# )

OUTPUT_DIR = Path("outputs")


# ---------------------------------------------------------------------------
# Benchmark loaders (stubs)
# ---------------------------------------------------------------------------

def load_ctf_held_out() -> list[dict]:
    """
    Load the held-out CTF challenge set.

    TODO:
        - At dataset build time, reserve ~10% of CTF writeups by setting
          aside their entries before training (add a 'split': 'test' field).
        - Load from data/final/validated.jsonl where source='ctf' and split='test'.
        - Return list of {instruction, input, expected_flag, vuln_type} dicts.
    """
    raise NotImplementedError


def load_securityeval() -> list[dict]:
    """
    Load SecurityEval benchmark tasks.

    TODO:
        - Clone https://github.com/s3c2/SecurityEval
        - Parse its JSON tasks into {instruction, input, expected_pattern} dicts.
        - expected_pattern: Bandit / Semgrep rule that should fire on correct output.
    """
    raise NotImplementedError


def load_cwe_qa() -> list[dict]:
    """
    Load the CWE Q&A evaluation set.

    TODO:
        - Build a 200-question set from the CWE corpus (mitre.org) or
          manually curate from training data held-out examples.
        - Return list of {instruction, input, expected_output} dicts.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Benchmark runners (stubs)
# ---------------------------------------------------------------------------

def run_ctf_held_out(model: str, limit: int | None = None) -> dict:
    """
    Run the CTF held-out benchmark.

    TODO:
        1. Load tasks via load_ctf_held_out()
        2. For each task, call thinking_loop(task['instruction']) or
           OllamaClient().generate(...)
        3. Check if the model output contains the expected flag
        4. Compute flag_extraction_accuracy and vuln_class_accuracy
        5. Save and return results dict
    """
    raise NotImplementedError


def run_securityeval(model: str, limit: int | None = None) -> dict:
    """
    Run the SecurityEval benchmark.

    TODO:
        1. Load tasks via load_securityeval()
        2. For each task, generate Python code with the model
        3. Run Bandit on the generated code:
               subprocess.run(["bandit", "-r", "-", ...])
           Check if the expected vulnerability pattern is detected.
        4. Compute pass rate and vuln detection accuracy
    """
    raise NotImplementedError


def run_cwe_qa(model: str, limit: int | None = None) -> dict:
    """
    Run the CWE Q&A benchmark.

    TODO:
        1. Load tasks via load_cwe_qa()
        2. Generate answers with the model
        3. Score with rouge_l_score against reference answers
        4. Report mean ROUGE-L and think_block_presence_rate
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "ctf-held-out":  run_ctf_held_out,
    "securityeval":  run_securityeval,
    "cwe-qa":        run_cwe_qa,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CyberPhi evaluation benchmarks")
    parser.add_argument("--model",     default="cyberphi",
                        help="Ollama model name to evaluate")
    parser.add_argument("--benchmark", choices=list(BENCHMARKS), default=None,
                        help="Which benchmark to run (omit to use --all)")
    parser.add_argument("--all",       action="store_true",
                        help="Run all benchmarks")
    parser.add_argument("--limit",     type=int, default=None,
                        help="Max tasks per benchmark (for quick smoke tests)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)

    to_run = list(BENCHMARKS.keys()) if args.all else [args.benchmark]
    if not to_run or to_run == [None]:
        parser.error("Specify --benchmark <name> or --all")

    for name in to_run:
        logger.info("Running benchmark: %s", name)
        try:
            results = BENCHMARKS[name](args.model, args.limit)
            out_file = OUTPUT_DIR / f"eval_results_{name}.json"
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info("Results saved → %s", out_file)
        except NotImplementedError:
            logger.warning("Benchmark '%s' is not yet implemented.", name)


if __name__ == "__main__":
    main()
