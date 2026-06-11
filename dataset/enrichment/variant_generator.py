"""
Vulnerability Variant Generator
---------------------------------
Reads enriched vulnerability examples and uses Claude to synthesise N
variants per entry, diversifying language, obfuscation level, framework
context, and mitigation completeness.

Input:  data/enriched/{source}_enriched.jsonl   (any source)
Output: data/enriched/variants.jsonl

Each output entry includes its own CoT reasoning trace and a reference to
the parent entry.

Variant types
-------------
language_variant        – same bug rewritten in a different language
obfuscated_variant      – more indirect / subtle version of the same flaw
framework_variant       – bug embedded inside a real framework (Django, Express…)
partial_mitigation_variant – patched version that remains bypassable

Usage:
    python dataset/enrichment/variant_generator.py
    python dataset/enrichment/variant_generator.py --vuln-type sqli --limit 200 --variants 2
"""

from __future__ import annotations
import argparse
import glob
import logging
import random
import sys
import time
from pathlib import Path

import anthropic
import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL_VARIANT,
    ENRICHED_DIR,
    LANGUAGES,
    SONNET_CACHE_READ_COST_PER_MTOK,
    SONNET_CACHE_WRITE_COST_PER_MTOK,
    SONNET_INPUT_COST_PER_MTOK,
    SONNET_OUTPUT_COST_PER_MTOK,
)

logger = logging.getLogger(__name__)
OUTPUT_FILE = ENRICHED_DIR / "variants.jsonl"

VARIANT_TYPES = [
    "language_variant",
    "obfuscated_variant",
    "framework_variant",
    "partial_mitigation_variant",
]

FRAMEWORKS = {
    "python":     ["Django", "Flask", "FastAPI"],
    "javascript": ["Express", "Next.js", "Fastify"],
    "java":       ["Spring Boot", "Servlet/JSP"],
    "php":        ["Laravel", "WordPress plugin"],
    "ruby":       ["Rails", "Sinatra"],
    "go":         ["Gin", "Echo"],
}

SYSTEM_PROMPT = (
    "You are a senior penetration tester and security educator. "
    "Generate realistic, diverse vulnerability variants for security training datasets."
)


def _build_variant_prompt(entry: dict, variant_type: str, target_lang: str | None,
                          target_framework: str | None) -> str:
    base_instruction = entry.get("instruction", "")
    base_input       = entry.get("input", "")
    base_output      = entry.get("output", "")

    extras = ""
    if variant_type == "language_variant":
        extras = f"Rewrite the vulnerable code in **{target_lang}** (keeping the same flaw)."
    elif variant_type == "obfuscated_variant":
        extras = (
            "Make the vulnerability **harder to spot**: use indirect function calls, "
            "rename variables to look benign, or split the exploit across multiple steps."
        )
    elif variant_type == "framework_variant":
        extras = (
            f"Embed this vulnerability inside a realistic **{target_framework}** application "
            "(use framework-idiomatic patterns — models, views, routes, etc.)."
        )
    elif variant_type == "partial_mitigation_variant":
        extras = (
            "Apply a **partial fix** that looks correct but is still exploitable "
            "(e.g. missing sanitisation path, wrong allowlist, bypassable regex)."
        )

    return (
        f"Original vulnerability context:\n{base_instruction}\n\n"
        f"Original code:\n{base_input}\n\n"
        f"Task: {extras}\n\n"
        "Respond with a complete training pair:\n"
        "1. A <think> block with step-by-step reasoning about the variant.\n"
        "2. A final answer with: variant code, explanation of the flaw, "
        "and exploitation/bypass notes.\n\n"
        "Format strictly as:\n<think>\n[reasoning]\n</think>\n\n[final answer]"
    )


def _call_claude(client: anthropic.Anthropic, prompt: str) -> tuple[str, int, int, int, int]:
    for attempt in range(5):
        try:
            response = client.messages.create(
                model=CLAUDE_MODEL_VARIANT,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": prompt}],
            )
            if not response.content:
                raise ValueError("Empty response content from Claude")
            cache_write = getattr(response.usage, "cache_creation_input_tokens", 0)
            cache_read  = getattr(response.usage, "cache_read_input_tokens",      0)
            return (
                response.content[0].text,
                response.usage.input_tokens,
                response.usage.output_tokens,
                cache_write,
                cache_read,
            )
        except anthropic.RateLimitError:
            wait = 2 ** attempt * 10
            logger.warning("Rate limited. Waiting %ds", wait)
            time.sleep(wait)
        except anthropic.APIConnectionError as exc:
            wait = 2 ** attempt * 5
            logger.warning("Connection error: %s. Waiting %ds", exc, wait)
            time.sleep(wait)
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                wait = 2 ** attempt * 8
                logger.warning("Server error %s. Waiting %ds", exc.status_code, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Claude call failed after 5 attempts")


def _load_existing_parent_variant_ids() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    ids: set = set()
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            vid = entry.get("variant_id")
            if vid:
                ids.add(vid)
    logger.info("Loaded %d existing variant IDs", len(ids))
    return ids


def _load_enriched_entries(vuln_type_filter: str | None = None) -> list[dict]:
    """Load all enriched entries from ENRICHED_DIR (excluding variants.jsonl itself)."""
    entries: list[dict] = []
    for fpath in glob.glob(str(ENRICHED_DIR / "*_enriched.jsonl")):
        with jsonlines.open(fpath) as r:
            for entry in r:
                if vuln_type_filter:
                    detected = entry.get("vuln_type", "") or ""
                    if vuln_type_filter.lower() not in detected.lower():
                        # Also check instruction / output text
                        text = (entry.get("instruction", "") + " " + entry.get("output", "")).lower()
                        if vuln_type_filter.lower() not in text:
                            continue
                entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_variants(
    vuln_type: str | None = None,
    limit: int | None = None,
    n_variants: int = 3,
) -> None:
    """
    Generate vulnerability variants for enriched entries.

    Args:
        vuln_type:  optional filter to only process entries of this type
        limit:      max variants to generate total
        n_variants: how many variants per entry
    """
    entries         = _load_enriched_entries(vuln_type)
    existing_ids    = _load_existing_parent_variant_ids()
    client          = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    total_saved       = 0
    total_in_tok      = 0
    total_out_tok     = 0
    total_cache_write = 0
    total_cache_read  = 0

    logger.info("Generating up to %d variants for %d entries", n_variants, len(entries))

    with tqdm(total=limit or (len(entries) * n_variants), desc="Variants", unit="variant") as pbar:
        for entry in entries:
            parent_id = entry.get("entry_id", "unknown")

            for vtype in random.sample(VARIANT_TYPES, min(n_variants, len(VARIANT_TYPES))):
                variant_id = f"variant_{parent_id}_{vtype}"
                if variant_id in existing_ids:
                    pbar.update(1)
                    continue

                # Pick a target language / framework
                current_lang    = entry.get("language", random.choice(LANGUAGES))
                available_langs = [l for l in LANGUAGES if l != current_lang]
                target_lang     = random.choice(available_langs) if available_langs else current_lang
                target_fw       = random.choice(FRAMEWORKS.get(target_lang, ["generic"]))

                prompt = _build_variant_prompt(entry, vtype, target_lang, target_fw)

                try:
                    output_text, in_t, out_t, cw_t, cr_t = _call_claude(client, prompt)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", variant_id, exc)
                    pbar.update(1)
                    continue

                total_in_tok      += in_t
                total_out_tok     += out_t
                total_cache_write += cw_t
                total_cache_read  += cr_t

                variant_entry = {
                    "entry_id":       variant_id,
                    "variant_id":     variant_id,
                    "parent_id":      parent_id,
                    "source_type":    "synthetic",
                    "variant_type":   vtype,
                    "target_language": target_lang,
                    "target_framework": target_fw,
                    "instruction":    entry.get("instruction", ""),
                    "input":          "",  # Claude will generate new code in the output
                    "output":         output_text,
                }

                with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                    writer.write(variant_entry)

                existing_ids.add(variant_id)
                total_saved += 1
                pbar.update(1)

                if limit and total_saved >= limit:
                    logger.info("Reached limit=%d", limit)
                    cost = (
                        total_in_tok      / 1e6 * SONNET_INPUT_COST_PER_MTOK        +
                        total_cache_write / 1e6 * SONNET_CACHE_WRITE_COST_PER_MTOK  +
                        total_cache_read  / 1e6 * SONNET_CACHE_READ_COST_PER_MTOK   +
                        total_out_tok     / 1e6 * SONNET_OUTPUT_COST_PER_MTOK
                    )
                    logger.info(
                        "Est. cost so far: $%.4f | cache_write=%d cache_read=%d",
                        cost, total_cache_write, total_cache_read,
                    )
                    return

    cost = (
        total_in_tok      / 1e6 * SONNET_INPUT_COST_PER_MTOK        +
        total_cache_write / 1e6 * SONNET_CACHE_WRITE_COST_PER_MTOK  +
        total_cache_read  / 1e6 * SONNET_CACHE_READ_COST_PER_MTOK   +
        total_out_tok     / 1e6 * SONNET_OUTPUT_COST_PER_MTOK
    )
    logger.info(
        "Done. Generated %d variants → %s | tokens in=%d cache_write=%d"
        " cache_read=%d out=%d | est. cost $%.4f",
        total_saved, OUTPUT_FILE, total_in_tok, total_cache_write,
        total_cache_read, total_out_tok, cost,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vulnerability variants via Claude")
    parser.add_argument("--vuln-type", default=None,
                        help="Only generate variants for this vulnerability class")
    parser.add_argument("--limit",    type=int, default=None)
    parser.add_argument("--variants", type=int, default=3,
                        help="Number of variants per entry (max 4)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    generate_variants(args.vuln_type, args.limit, min(args.variants, 4))


if __name__ == "__main__":
    main()
