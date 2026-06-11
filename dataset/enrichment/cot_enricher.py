"""
Chain-of-Thought Enricher
--------------------------
Reads raw scraper output and calls the Anthropic API to generate a training
pair with an explicit <think>…</think> reasoning trace followed by a final
answer.

Input:  data/raw/{source}.jsonl
Output: data/enriched/{source}_enriched.jsonl

Each output entry schema:
    {
        "entry_id":    "{source}_{original_id}",
        "source_type": "nvd|exploitdb|ctf|hackerone",
        "source_id":   "original identifier",
        "instruction": "string",
        "input":       "string (code or context, may be empty)",
        "output":      "<think>…</think>\\n\\n[final answer]",
        ...original fields preserved...
    }

Usage:
    python dataset/enrichment/cot_enricher.py --source nvd
    python dataset/enrichment/cot_enricher.py --source exploitdb --limit 100 --batch-size 5
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import anthropic
import jsonlines
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_BATCH_SIZE,
    CLAUDE_MAX_TOKENS,
    CLAUDE_MODEL_COT,
    CLAUDE_MODEL_NVD,
    ENRICHED_DIR,
    HAIKU_CACHE_READ_COST_PER_MTOK,
    HAIKU_CACHE_WRITE_COST_PER_MTOK,
    HAIKU_INPUT_COST_PER_MTOK,
    HAIKU_OUTPUT_COST_PER_MTOK,
    RAW_DIR,
    SONNET_CACHE_READ_COST_PER_MTOK,
    SONNET_CACHE_WRITE_COST_PER_MTOK,
    SONNET_INPUT_COST_PER_MTOK,
    SONNET_OUTPUT_COST_PER_MTOK,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior penetration tester and security researcher. "
    "Given a vulnerability, generate a detailed, technically accurate security "
    "analysis training example."
)

# Maps source name → (raw file stem, function that builds instruction+input from entry)
SOURCE_RAW_FILES = {
    "nvd":        RAW_DIR / "nvd_cves.jsonl",
    "exploitdb":  RAW_DIR / "exploitdb_entries.jsonl",
    "ctf":        RAW_DIR / "ctf_writeups.jsonl",
    "hackerone":  RAW_DIR / "hackerone_reports.jsonl",
}


# ---------------------------------------------------------------------------
# Per-source prompt builders
# ---------------------------------------------------------------------------

def _build_nvd_prompt(entry: dict) -> tuple[str, str]:
    instruction = (
        f"Analyze CVE {entry.get('cve_id', 'UNKNOWN')} "
        f"(CVSS {entry.get('cvss_score', '?')} / {entry.get('severity', '?')}). "
        "Provide a step-by-step security analysis covering: vulnerability class, "
        "root cause, exploitation path, real-world impact, and remediation."
    )
    input_ctx = entry.get("description", "")
    return instruction, input_ctx


def _build_exploitdb_prompt(entry: dict) -> tuple[str, str]:
    instruction = (
        f"Analyze exploit '{entry.get('title', 'Unknown')}' "
        f"(platform: {entry.get('platform', '?')}, type: {entry.get('type', '?')}). "
        "Identify the vulnerability, explain the exploit technique, trace the execution "
        "flow, assess impact, and describe the patch."
    )
    input_ctx = (entry.get("code") or "")[:3000]   # Truncate very long exploit files
    return instruction, input_ctx


def _build_ctf_prompt(entry: dict) -> tuple[str, str]:
    instruction = (
        f"Solve CTF challenge '{entry.get('challenge_name', 'Unknown')}' "
        f"(category: {entry.get('category', '?')}). "
        "Explain your approach, tools, exploit steps, and what the flag demonstrates."
    )
    input_ctx = (entry.get("raw_content") or "")[:3000]
    return instruction, input_ctx


def _build_hackerone_prompt(entry: dict) -> tuple[str, str]:
    instruction = (
        f"Analyze this HackerOne report: '{entry.get('title', 'Unknown')}' "
        f"(severity: {entry.get('severity', '?')}, "
        f"type: {entry.get('vulnerability_type', '?')}). "
        "Explain the vulnerability, reproduction steps, business impact, "
        "and recommended fix."
    )
    input_ctx = (entry.get("summary") or "")[:3000]
    return instruction, input_ctx


PROMPT_BUILDERS = {
    "nvd":       _build_nvd_prompt,
    "exploitdb": _build_exploitdb_prompt,
    "ctf":       _build_ctf_prompt,
    "hackerone": _build_hackerone_prompt,
}


def _source_id(source: str, entry: dict) -> str:
    mapping = {
        "nvd":       "cve_id",
        "exploitdb": "exploit_id",
        "ctf":       "url",
        "hackerone": "report_id",
    }
    return str(entry.get(mapping.get(source, ""), "unknown"))


# ---------------------------------------------------------------------------
# Claude API call with retry
# ---------------------------------------------------------------------------

def _call_claude(
    client: anthropic.Anthropic,
    model: str,
    instruction: str,
    input_ctx: str,
) -> tuple[str, int, int, int, int]:
    """
    Call Claude and return (output_text, input_tokens, output_tokens, cache_write, cache_read).
    The system prompt is sent with cache_control so repeated calls in the same session
    pay the cheap cache-read rate instead of the full input rate.
    Retries up to 5 times with exponential backoff on rate-limit / server errors.
    """
    user_msg = (
        f"Instruction: {instruction}\n\n"
        f"Context / Code:\n{input_ctx}\n\n"
        "Respond with:\n"
        "1. A <think> block containing step-by-step reasoning: input tracing, "
        "vulnerability identification, exploitation path, impact assessment.\n"
        "2. A final answer (after the </think> tag) with: vulnerability explanation, "
        "PoC walkthrough, CVSS justification, and remediation steps.\n\n"
        "Format strictly as:\n<think>\n[reasoning]\n</think>\n\n[final answer]"
    )

    for attempt in range(5):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=CLAUDE_MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
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
            logger.warning("Rate limited. Waiting %ds (attempt %d/5)", wait, attempt + 1)
            time.sleep(wait)

        except anthropic.APIConnectionError as exc:
            wait = 2 ** attempt * 5
            logger.warning("Connection error: %s. Waiting %ds", exc, wait)
            time.sleep(wait)

        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                wait = 2 ** attempt * 8
                logger.warning("Server error %d. Waiting %ds", exc.status_code, wait)
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Claude API call failed after 5 attempts")


# ---------------------------------------------------------------------------
# Cost helper
# ---------------------------------------------------------------------------

def _calc_cost(model: str, in_t: int, out_t: int, cw_t: int, cr_t: int) -> float:
    if model == CLAUDE_MODEL_NVD:
        return (
            in_t  / 1e6 * HAIKU_INPUT_COST_PER_MTOK  +
            cw_t  / 1e6 * HAIKU_CACHE_WRITE_COST_PER_MTOK +
            cr_t  / 1e6 * HAIKU_CACHE_READ_COST_PER_MTOK  +
            out_t / 1e6 * HAIKU_OUTPUT_COST_PER_MTOK
        )
    return (
        in_t  / 1e6 * SONNET_INPUT_COST_PER_MTOK  +
        cw_t  / 1e6 * SONNET_CACHE_WRITE_COST_PER_MTOK +
        cr_t  / 1e6 * SONNET_CACHE_READ_COST_PER_MTOK  +
        out_t / 1e6 * SONNET_OUTPUT_COST_PER_MTOK
    )


# ---------------------------------------------------------------------------
# Main enrich function
# ---------------------------------------------------------------------------

def enrich(source: str, limit: int | None = None, batch_size: int = CLAUDE_BATCH_SIZE) -> None:
    """
    Enrich raw entries from `source` with CoT reasoning traces.

    Args:
        source:     one of nvd / exploitdb / ctf / hackerone
        limit:      stop after this many new enriched entries
        batch_size: save checkpoint every N entries
    """
    if source not in SOURCE_RAW_FILES:
        raise ValueError(f"Unknown source '{source}'. Choose from {list(SOURCE_RAW_FILES)}")

    raw_file = SOURCE_RAW_FILES[source]
    if not raw_file.exists():
        logger.error("Raw file not found: %s — run the scraper first", raw_file)
        return

    out_file                = ENRICHED_DIR / f"{source}_enriched.jsonl"
    existing_ids, existing_count = _load_existing_ids(out_file)
    client       = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    builder      = PROMPT_BUILDERS[source]
    model        = CLAUDE_MODEL_NVD if source == "nvd" else CLAUDE_MODEL_COT

    logger.info("Using model %s for source=%s", model, source)

    total_saved        = 0
    total_input_t      = 0
    total_output_t     = 0
    total_cache_write  = 0
    total_cache_read   = 0

    raw_entries: list[dict] = []
    with jsonlines.open(raw_file) as r:
        for entry in r:
            raw_entries.append(entry)

    total_raw = len(raw_entries)
    remaining = min(limit, total_raw - existing_count) if limit else (total_raw - existing_count)

    logger.info(
        "source=%s: have %d/%d enriched, need %d more",
        source, existing_count, total_raw, max(remaining, 0),
    )

    if remaining <= 0:
        logger.info("source=%s already fully enriched — skipping", source)
        return

    with tqdm(total=remaining, desc=f"Enrich {source}", unit="entry") as pbar:
        for entry in raw_entries:
            sid = _source_id(source, entry)
            entry_id = f"{source}_{sid}"

            if entry_id in existing_ids:
                continue

            try:
                instruction, input_ctx = builder(entry)
                output_text, in_t, out_t, cw_t, cr_t = _call_claude(
                    client, model, instruction, input_ctx
                )
            except Exception as exc:
                logger.warning("Skipping %s: %s", entry_id, exc)
                pbar.update(1)
                continue

            total_input_t     += in_t
            total_output_t    += out_t
            total_cache_write += cw_t
            total_cache_read  += cr_t

            enriched = {
                **entry,
                "entry_id":    entry_id,
                "source_type": source,
                "source_id":   sid,
                "instruction": instruction,
                "input":       input_ctx,
                "output":      output_text,
            }

            with jsonlines.open(out_file, mode="a") as writer:
                writer.write(enriched)

            existing_ids.add(entry_id)
            total_saved += 1
            pbar.update(1)

            if total_saved % batch_size == 0:
                cost = _calc_cost(model, total_input_t, total_output_t,
                                  total_cache_write, total_cache_read)
                logger.info(
                    "Batch checkpoint: %d saved | tokens in=%d cache_write=%d"
                    " cache_read=%d out=%d | est. cost $%.4f",
                    total_saved, total_input_t, total_cache_write,
                    total_cache_read, total_output_t, cost,
                )

            if limit and total_saved >= limit:
                logger.info("Reached limit=%d", limit)
                break

    total_cost = _calc_cost(model, total_input_t, total_output_t,
                            total_cache_write, total_cache_read)
    logger.info(
        "Done. Saved %d entries → %s | tokens in=%d cache_write=%d"
        " cache_read=%d out=%d | est. cost $%.4f",
        total_saved, out_file, total_input_t, total_cache_write,
        total_cache_read, total_output_t, total_cost,
    )


def _load_existing_ids(out_file: Path) -> tuple[set, int]:
    if not out_file.exists():
        return set(), 0
    ids: set = set()
    with jsonlines.open(out_file) as r:
        for entry in r:
            eid = entry.get("entry_id")
            if eid:
                ids.add(eid)
    count = len(ids)
    logger.info("Loaded %d existing enriched IDs from %s (resume mode)", count, out_file.name)
    return ids, count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CoT reasoning traces via Claude API")
    parser.add_argument("--source", required=True,
                        choices=list(SOURCE_RAW_FILES),
                        help="Which raw source file to enrich")
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=CLAUDE_BATCH_SIZE)
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    enrich(args.source, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
