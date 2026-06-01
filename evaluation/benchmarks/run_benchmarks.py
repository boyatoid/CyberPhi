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

Note: securityeval runner requires `bandit` on PATH:
    pip install bandit
"""

import argparse
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Heavy inference imports are deferred inside runner functions so data loaders
# can be used without Ollama or ChromaDB being available.
from evaluation.metrics import (
    flag_extraction_accuracy,
    vuln_class_accuracy,
    think_block_presence_rate,
    rouge_l_score,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR       = Path("outputs")
VALIDATED_JSONL  = Path("data/final/validated.jsonl")
EXTERNAL_DIR     = Path("data/external")
SECURITYEVAL_URL = (
    "https://raw.githubusercontent.com/s3c2/SecurityEval/main/SecurityEval.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_validated(source_filter: Optional[str] = None) -> list[dict]:
    entries = []
    with open(VALIDATED_JSONL) as f:
        for line in f:
            entry = json.loads(line)
            if source_filter is None or entry.get("source") == source_filter:
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_ctf_held_out() -> list[dict]:
    """
    Load the held-out CTF challenge set (last 10% of CTF entries, by id sort).
    """
    ctf_entries = _load_validated(source_filter="ctf")
    ctf_entries.sort(key=lambda e: e.get("id", ""))
    held_out_n  = max(1, len(ctf_entries) // 10)
    held_out    = ctf_entries[-held_out_n:]
    return [
        {
            "instruction":   e["instruction"],
            "input":         e.get("input", ""),
            "expected_output": e["output"],
            "vuln_type":     e.get("vuln_type", "unknown"),
        }
        for e in held_out
    ]


def load_securityeval() -> list[dict]:
    """
    Load SecurityEval benchmark tasks from a local cache or GitHub download.

    To use this benchmark:
      1. Obtain SecurityEval.json from the SecurityEval project.
      2. Place it at data/external/securityeval.json.

    Expected format: list of {id, prompt, cwe} dicts.
    Returns an empty list if the file is unavailable.
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EXTERNAL_DIR / "securityeval.json"

    if not cache_path.exists():
        import requests  # noqa: PLC0415
        logger.info("Attempting to download SecurityEval …")
        try:
            resp = requests.get(SECURITYEVAL_URL, timeout=30)
            resp.raise_for_status()
            cache_path.write_text(resp.text)
            logger.info("Cached → %s", cache_path)
        except Exception as exc:
            logger.warning(
                "Could not download SecurityEval (%s). "
                "Place SecurityEval.json at %s to enable this benchmark.",
                exc, cache_path,
            )
            return []

    raw = json.loads(cache_path.read_text())
    tasks = []
    for item in raw:
        tasks.append({
            "instruction":      item.get("prompt", ""),
            "input":            "",
            "expected_pattern": item.get("cwe", ""),
            "task_id":          item.get("id", ""),
        })
    logger.info("Loaded %d SecurityEval tasks from %s", len(tasks), cache_path)
    return tasks


def load_cwe_qa() -> list[dict]:
    """
    Load a Q&A eval set derived from NVD/HackerOne entries in validated.jsonl.
    Takes 10% of entries (up to 200) as the eval set.
    """
    entries = []
    for entry in _load_validated():
        if entry.get("source") in ("nvd", "hackerone"):
            entries.append(entry)

    entries.sort(key=lambda e: e.get("id", ""))
    sample_n = min(200, max(1, len(entries) // 10))
    sample   = entries[:sample_n]

    return [
        {
            "instruction":   e["instruction"],
            "input":         e.get("input", ""),
            "expected_output": e["output"],
            "vuln_type":     e.get("vuln_type", "unknown"),
        }
        for e in sample
    ]


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def run_ctf_held_out(model: str, limit: Optional[int] = None) -> dict:
    """Run the CTF held-out benchmark using the 3-step thinking loop."""
    from inference.thinking_loop import thinking_loop  # noqa: PLC0415
    tasks = load_ctf_held_out()
    if limit:
        tasks = tasks[:limit]

    outputs       = []
    expected_flags = []
    vuln_classes  = []

    for i, task in enumerate(tasks):
        logger.info("[ctf-held-out] %d/%d", i + 1, len(tasks))
        try:
            out = thinking_loop(task["instruction"])
        except Exception as exc:
            logger.warning("thinking_loop failed: %s", exc)
            out = ""
        outputs.append(out)

        # Extract flag from expected output if present
        import re
        flag_match = re.search(r"(?:flag|ctf|htb|picoctf|pico)\{[^}]+\}", task["expected_output"], re.I)
        expected_flags.append(flag_match.group(0) if flag_match else "")
        vuln_classes.append(task["vuln_type"])

    results = {
        "benchmark":      "ctf-held-out",
        "model":          model,
        "samples":        len(tasks),
        "flag_accuracy":  flag_extraction_accuracy(outputs, expected_flags),
        "vuln_accuracy":  vuln_class_accuracy(outputs, vuln_classes),
        "think_rate":     think_block_presence_rate(outputs),
        "raw":            [
            {"output": o, "expected_flag": f, "vuln_type": v}
            for o, f, v in zip(outputs, expected_flags, vuln_classes)
        ],
    }
    logger.info("ctf-held-out: flag_acc=%.1f%% vuln_acc=%.1f%%",
                results["flag_accuracy"] * 100, results["vuln_accuracy"] * 100)
    return results


def run_securityeval(model: str, limit: Optional[int] = None) -> dict:
    """
    Run SecurityEval: generate Python code, check for CWE via bandit.
    Requires: bandit on PATH (pip install bandit).
    """
    from inference.ollama_client import OllamaClient  # noqa: PLC0415
    tasks  = load_securityeval()
    if limit:
        tasks = tasks[:limit]

    client    = OllamaClient(model=model)
    passed    = 0
    raw       = []

    for i, task in enumerate(tasks):
        logger.info("[securityeval] %d/%d", i + 1, len(tasks))
        try:
            code = client.generate(
                prompt=task["instruction"],
                system=(
                    "You are a security-focused Python developer. "
                    "Write secure, correct Python code only. "
                    "Output only valid Python, no explanation."
                ),
                temperature=0.2,
            )
        except Exception as exc:
            logger.warning("generate failed: %s", exc)
            code = ""

        # Run bandit on generated code
        bandit_issues: list[str] = []
        if code.strip():
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name
            try:
                proc = subprocess.run(
                    ["bandit", "-r", tmp_path, "-f", "json", "-q"],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.stdout:
                    bandit_out = json.loads(proc.stdout)
                    bandit_issues = [
                        r["test_id"] for r in bandit_out.get("results", [])
                    ]
            except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                logger.warning("bandit error: %s", exc)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        expected = task.get("expected_pattern", "")
        hit = expected == "" or any(expected in issue for issue in bandit_issues)
        if hit:
            passed += 1

        raw.append({
            "task_id":        task.get("task_id", ""),
            "output":         code[:500],
            "bandit_issues":  bandit_issues,
            "expected":       expected,
            "pass":           hit,
        })

    results = {
        "benchmark":  "securityeval",
        "model":      model,
        "samples":    len(tasks),
        "pass_rate":  passed / max(len(tasks), 1),
        "raw":        raw,
    }
    logger.info("securityeval: pass_rate=%.1f%%", results["pass_rate"] * 100)
    return results


def run_cwe_qa(model: str, limit: Optional[int] = None) -> dict:
    """Run the CWE Q&A benchmark, scoring with ROUGE-L."""
    from inference.ollama_client import OllamaClient  # noqa: PLC0415
    tasks = load_cwe_qa()
    if limit:
        tasks = tasks[:limit]

    client  = OllamaClient(model=model)
    outputs = []
    refs    = []

    for i, task in enumerate(tasks):
        logger.info("[cwe-qa] %d/%d", i + 1, len(tasks))
        prompt = task["instruction"]
        if task.get("input"):
            prompt = f"{prompt}\n\n{task['input']}"
        try:
            out = client.generate(prompt=prompt, temperature=0.3)
        except Exception as exc:
            logger.warning("generate failed: %s", exc)
            out = ""
        outputs.append(out)
        refs.append(task["expected_output"])

    rouge_scores = [rouge_l_score(o, r) for o, r in zip(outputs, refs) if r]
    mean_rouge   = sum(rouge_scores) / max(len(rouge_scores), 1)

    results = {
        "benchmark":       "cwe-qa",
        "model":           model,
        "samples":         len(tasks),
        "mean_rouge_l":    mean_rouge,
        "think_rate":      think_block_presence_rate(outputs),
        "raw":             [
            {"output": o, "expected": r, "rouge_l": rouge_l_score(o, r) if r else None}
            for o, r in zip(outputs, refs)
        ],
    }
    logger.info("cwe-qa: mean_rouge_l=%.3f think_rate=%.1f%%",
                results["mean_rouge_l"], results["think_rate"] * 100)
    return results


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
        results  = BENCHMARKS[name](args.model, args.limit)
        out_file = OUTPUT_DIR / f"eval_results_{name}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved → %s", out_file)


if __name__ == "__main__":
    main()
