# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full pipeline (test run)
python dataset/pipeline.py all --limit 100

# Individual stages
python dataset/pipeline.py scrape --source nvd --limit 200
python dataset/pipeline.py enrich --type cot --source nvd --limit 50
python dataset/pipeline.py clean
python dataset/pipeline.py mix
python dataset/pipeline.py validate --samples 15

# Run any script standalone
python dataset/scrapers/nvd_scraper.py --limit 100
python dataset/enrichment/cot_enricher.py --source nvd --limit 20
python dataset/cleaning/deduplicator.py
python dataset/validation/sample_reviewer.py --vuln-type sqli

# Install dependencies
pip install -r requirements.txt
```

## Architecture

The project is a **linear data pipeline** that produces `data/final/validated.jsonl` for fine-tuning:

```
scrapers/ → enrichment/ → cleaning/ → mixing/ → validation/
```

**Key file: `dataset/config.py`** — single source of truth for all paths, API keys (from env), taxonomies, and thresholds. Every other script imports from it; never hardcode paths elsewhere.

**Pipeline orchestrator: `dataset/pipeline.py`** — imports each module's top-level function directly (e.g. `nvd_scrape`, `enrich`, `deduplicate`) and calls them. Each module is also runnable as a standalone CLI.

**Every script** follows this pattern:
- Top-level entry-point function (e.g. `scrape(limit, ...)`) — called by pipeline.py
- `main()` that parses argparse and calls the entry-point — for standalone use
- Resume capability via loading existing IDs from the output file before starting

## Data flow

| Stage | Input | Output file |
|-------|-------|-------------|
| Scraper | External APIs | `data/raw/{source}.jsonl` |
| `cot_enricher` | raw JSONL | `data/enriched/{source}_enriched.jsonl` |
| `variant_generator` | enriched/ | `data/enriched/variants.jsonl` |
| `qa_generator` | any file | `data/enriched/qa_pairs.jsonl` |
| `deduplicator` | all of enriched/ | `data/cleaned/deduped.jsonl` |
| `quality_filter` | deduped | `data/cleaned/filtered.jsonl` |
| `formatter` | filtered | `data/cleaned/formatted.jsonl` |
| `balancer` | formatted | `data/final/balanced.jsonl` |
| `general_mixer` | balanced | `data/final/mixed.jsonl` |
| `schema_validator` | mixed | `data/final/validated.jsonl` |

## Unified schema

All entries in `data/final/` conform to:
```json
{
  "id": "uuid", "source": "nvd|exploitdb|ctf|hackerone|synthetic|general",
  "vuln_type": "sqli|xss|rce|ssrf|xxe|deserialization|buffer_overflow|race_condition|idor|path_traversal",
  "language": "python|php|javascript|c|cpp|java|go|ruby|unknown",
  "severity": "critical|high|medium|low|unknown",
  "difficulty": "beginner|intermediate|advanced",
  "instruction": "string", "input": "string", "output": "<think>…</think>\n\n[final]",
  "metadata": {"source_id": "…", "date": "…", "tags": []}
}
```

## Anthropic API usage

`cot_enricher.py`, `variant_generator.py`, and `qa_generator.py` all call `claude-opus-4-5`. Each has retry logic with exponential backoff for `RateLimitError`, `APIConnectionError`, and 5xx `APIStatusError`. Cost is logged per batch using `response.usage.input_tokens` / `output_tokens`.

## What is and isn't implemented

- `dataset/` — **fully implemented**, all scripts runnable
- `training/`, `inference/`, `evaluation/` — **scaffold only**; functions raise `NotImplementedError` with TODO comments explaining what to build

## Environment variables

Required in `.env`:
- `ANTHROPIC_API_KEY` — for all enrichment scripts
- `NVD_API_KEY` — optional; without it NVD rate limit is 5 req/30s instead of 50
- `HUGGINGFACE_TOKEN` — for downloading the Alpaca dataset in `general_mixer.py`
