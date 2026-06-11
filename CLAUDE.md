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

# End-to-end (data pipeline + training; requires CUDA GPU for training)
./quickstart.sh --limit 200
./quickstart.sh --skip-data --run-name exp_lr2e4   # train only, named run → outputs/exp_lr2e4/

# Training (thin wrapper around Axolotl)
python training/train.py                      # = axolotl train training/axolotl_config.yaml
python training/train.py --merge-and-export   # post-training only: merge LoRA + export GGUF (does NOT retrain)

# Evaluation (requires Ollama running the exported model)
python evaluation/benchmarks/run_benchmarks.py --model cyberphi --all
python evaluation/benchmarks/run_benchmarks.py --benchmark securityeval --limit 20
python evaluation/metrics.py --results outputs/eval_results_cwe-qa.json

# Inference
python inference/thinking_loop.py --query "How does CVE-2023-44487 work?" [--use-rag]
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

## Training (`training/`)

All QLoRA hyperparameters live in `training/axolotl_config.yaml` (Phi-3.5-mini base, ChatML template, 4-bit NF4). `train.py` is a thin subprocess wrapper around the Axolotl CLI:
- No flags → runs training. `--merge-and-export` → merge-only post-training step (merges the LoRA adapter, converts to GGUF via llama.cpp's `convert_hf_to_gguf.py`, writes an Ollama Modelfile). The convert script only supports `f32/f16/bf16/q8_0`; we export `q8_0` — K-quants like q4_k_m need a separate `llama-quantize` pass.
- The `OUTPUT_DIR` env var overrides the output path for both training and merge; `quickstart.sh` uses it for per-run directories (`outputs/<run-name>/`).

## Evaluation (`evaluation/`)

`benchmarks/run_benchmarks.py` runs three benchmarks against an Ollama model and writes `outputs/eval_results_{benchmark}.json` as `{"benchmark", "model", …, "raw": [...]}`:
- `ctf-held-out` — last 10% of CTF entries by id, scored on flag extraction via the 3-step thinking loop
- `securityeval` — downloads `s2e-lab/SecurityEval` `dataset.jsonl` (cached at `data/external/securityeval.jsonl`; JSONL with `ID`/`Prompt` keys, expected CWE parsed from the ID). A task **passes when bandit does NOT find the task's CWE** in the generated code; empty generations fail.
- `cwe-qa` — first 10% (≤200) of NVD/HackerOne entries, scored with ROUGE-L

`metrics.py` holds the scoring functions and accepts either a bare list or the benchmark dict format via `--results`. Accuracy metrics exclude pairs with an empty reference (empty expected flag / empty or `unknown` vuln class) from the denominator.

## Inference (`inference/`)

`ollama_client.py` (REST wrapper for generate/embed, Ollama must be running), `thinking_loop.py` (think → reflect → answer loop used by the CTF benchmark), `rag/` (ChromaDB vector store + retriever, imported lazily).

## Environment variables

Required in `.env`:
- `ANTHROPIC_API_KEY` — for all enrichment scripts
- `NVD_API_KEY` — optional; without it NVD rate limit is 5 req/30s instead of 50
- `HUGGINGFACE_TOKEN` — for downloading the Alpaca dataset in `general_mixer.py`
