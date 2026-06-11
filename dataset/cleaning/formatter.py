"""
Formatter
---------
Normalises all entries from data/cleaned/filtered.jsonl into the unified
training schema and writes to data/cleaned/formatted.jsonl.

Unified schema:
    {
        "id":         "uuid",
        "source":     "nvd|exploitdb|ctf|hackerone|synthetic",
        "vuln_type":  "sqli|xss|rce|...|unknown",
        "language":   "python|php|...|unknown",
        "severity":   "critical|high|medium|low|unknown",
        "difficulty": "beginner|intermediate|advanced",
        "instruction":"string",
        "input":      "string",
        "output":     "<think>…</think>\\n\\n[final answer]",
        "metadata": {
            "source_id": "...",
            "date":      "...",
            "tags":      []
        }
    }

Usage:
    python dataset/cleaning/formatter.py
"""

import argparse
import logging
import re
import sys
import uuid
from pathlib import Path

import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import CLEANED_DIR, LANGUAGES, VULN_TYPES

logger   = logging.getLogger(__name__)
IN_FILE  = CLEANED_DIR / "filtered.jsonl"
OUT_FILE = CLEANED_DIR / "formatted.jsonl"

# ---------------------------------------------------------------------------
# Keyword maps for auto-detection
# ---------------------------------------------------------------------------

VULN_KEYWORDS: dict[str, list[str]] = {
    "sqli":            ["sql injection", "sqli", "union select", "' or 1=1", "sqlmap",
                        "mysql_query", "pg_query", "database injection"],
    "xss":             ["cross-site scripting", "xss", "<script>", "document.cookie",
                        "innerhtml", "alert(1)", "javascript:", "onclick="],
    "rce":             ["remote code execution", "rce", "command injection", "os.system(",
                        "eval(", "system(", "shell_exec", "subprocess.call", "arbitrary code"],
    "ssrf":            ["server-side request forgery", "ssrf", "169.254.169.254",
                        "internal endpoint", "metadata service", "request forwarding"],
    "xxe":             ["xml external entity", "xxe", "<!entity", "system file",
                        "dtd injection", "external entity"],
    "deserialization": ["deserialization", "pickle.loads", "unserialize(",
                        "objectinputstream", "yaml.load(", "marshal.loads"],
    "buffer_overflow": ["buffer overflow", "stack overflow", "heap overflow",
                        "out-of-bounds write", "memory corruption", " bof "],
    "race_condition":  ["race condition", "toctou", "time-of-check", "time-of-use",
                        "concurrent access", "double-fetch"],
    "idor":            ["insecure direct object", "idor", "broken access control",
                        "unauthorized object", "object reference manipulation"],
    "path_traversal":  ["path traversal", "directory traversal", "../", "..\\",
                        "local file inclusion", " lfi ", "file inclusion"],
}

LANG_KEYWORDS: dict[str, list[str]] = {
    "python":     ["def ", "import ", "print(", ".py\n", "pip install",
                   "flask", "django", "fastapi", "requests.get(", "os.system("],
    "php":        ["<?php", "echo ", "$_get", "$_post", "include(", "require(",
                   ".php", "mysql_query(", "pdo->"],
    "javascript": ["function(", "const ", "let ", "var ", "=>", "node.js",
                   "express", "document.", "window.", "require('"],
    "java":       ["public class", "static void main", "system.out", "import java.",
                   "throws ", "new scanner(", "servlet", "spring"],
    "c":          ["#include <stdio.h>", "int main(", "malloc(", "printf(",
                   "scanf(", "#include <stdlib.h>", "char *"],
    "cpp":        ["#include <iostream>", "std::", "cout <<", "cin >>",
                   "nullptr", "vector<", "new int["],
    "go":         ["package main", "func main()", "import \"fmt\"", "fmt.println",
                   " := ", "go func", "goroutine"],
    "ruby":       [".rb\n", "gem install", "puts ", "require '", "class ",
                   " do\n", "rails", "sinatra", ".each do"],
}

ADVANCED_VULN_TYPES = {"deserialization", "buffer_overflow", "race_condition", "ssrf", "xxe"}
BEGINNER_VULN_TYPES = {"sqli", "xss", "path_traversal", "idor"}

SEVERITY_MAP = {
    "critical": "critical",
    "high":     "high",
    "medium":   "medium",
    "moderate": "medium",
    "low":      "low",
    "info":     "low",
    "unknown":  "unknown",
}


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------

def detect_vuln_type(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for vtype, keywords in VULN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[vtype] = hits
    if not scores:
        return "unknown"
    return max(scores, key=lambda k: scores[k])


def detect_language(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for lang, keywords in LANG_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lower)
        if hits:
            scores[lang] = hits
    if not scores:
        return "unknown"
    return max(scores, key=lambda k: scores[k])


def classify_difficulty(vuln_type: str, output: str) -> str:
    output_len = len(output)
    if vuln_type in ADVANCED_VULN_TYPES or output_len > 2500:
        return "advanced"
    if vuln_type in BEGINNER_VULN_TYPES and output_len < 900:
        return "beginner"
    return "intermediate"


def normalize_severity(raw: str) -> str:
    if not raw:
        return "unknown"
    return SEVERITY_MAP.get(raw.lower().strip(), "unknown")


def detect_source(entry: dict) -> str:
    if "cve_id" in entry:
        return "nvd"
    if "exploit_id" in entry:
        return "exploitdb"
    if "challenge_name" in entry:
        return "ctf"
    if "report_id" in entry:
        return "hackerone"
    st = entry.get("source_type", "")
    if st in ("nvd", "exploitdb", "ctf", "hackerone", "synthetic", "general"):
        return st
    return "synthetic"


def get_source_id(entry: dict) -> str:
    for field in ("cve_id", "exploit_id", "report_id", "entry_id", "url"):
        val = entry.get(field)
        if val:
            return str(val)
    return ""


def get_date(entry: dict) -> str:
    for field in ("published_date", "disclosed_at", "date", "event_year"):
        val = entry.get(field)
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def format_dataset() -> None:
    """Normalise filtered entries to the unified training schema."""
    if not IN_FILE.exists():
        logger.error("Input not found: %s — run quality_filter first", IN_FILE)
        return

    total = 0
    with jsonlines.open(IN_FILE) as reader, jsonlines.open(OUT_FILE, mode="w") as writer:
        for entry in tqdm(reader, desc="Formatting", unit="entry"):
            instruction = entry.get("instruction", "")
            input_ctx   = entry.get("input", "")
            output      = entry.get("output", "")

            combined_text = f"{instruction} {input_ctx} {output}"

            source    = detect_source(entry)
            vuln_type = entry.get("vuln_type") or detect_vuln_type(combined_text)
            language  = entry.get("language")  or detect_language(combined_text)

            raw_severity = (
                entry.get("severity")
                or entry.get("severity_rating")
                or "unknown"
            )
            severity   = normalize_severity(str(raw_severity))
            difficulty = classify_difficulty(vuln_type, output)

            tags: list[str] = []
            if vuln_type != "unknown":
                tags.append(vuln_type)
            if language != "unknown":
                tags.append(language)
            if entry.get("variant_type"):
                tags.append(entry["variant_type"])
            if entry.get("question_type"):
                tags.append(entry["question_type"])

            formatted = {
                "id":         str(uuid.uuid4()),
                "source":     source,
                "vuln_type":  vuln_type,
                "language":   language,
                "severity":   severity,
                "difficulty": difficulty,
                "instruction": instruction,
                "input":      input_ctx,
                "output":     output,
                "metadata": {
                    "source_id": get_source_id(entry),
                    "date":      get_date(entry),
                    "tags":      tags,
                },
            }

            writer.write(formatted)
            total += 1

    logger.info("Formatted %d entries → %s", total, OUT_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Normalise entries to unified training schema")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    format_dataset()


if __name__ == "__main__":
    main()
