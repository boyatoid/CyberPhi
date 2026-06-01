# CyberPhi — Cybersecurity Fine-Tune for Phi-3.5-mini

Fine-tunes Microsoft Phi-3.5-mini-instruct on a custom cybersecurity dataset
featuring Chain-of-Thought (`<think>`) reasoning traces, producing a compact
local model capable of vulnerability analysis, exploit explanation, and CTF
challenge solving.

## Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │                   Data Sources                          │
  │  NVD API   Exploit-DB   CTFtime   HackerOne Hacktivity  │
  └─────────────────────┬───────────────────────────────────┘
                        │  dataset/scrapers/
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Claude API Enrichment                      │
  │   CoT traces (cot_enricher)  Variants  Q&A pairs        │
  └─────────────────────┬───────────────────────────────────┘
                        │  dataset/enrichment/
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │              Cleaning & Mixing                          │
  │   MinHash dedup → quality filter → format → balance     │
  │   + 15% general instruction data (Alpaca)               │
  └─────────────────────┬───────────────────────────────────┘
                        │  data/final/validated.jsonl
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │           QLoRA Fine-Tuning (Axolotl)                   │
  │   Phi-3.5-mini-instruct  |  rank=16  |  4-bit quant     │
  └─────────────────────┬───────────────────────────────────┘
                        │  training/
                        ▼
  ┌─────────────────────────────────────────────────────────┐
  │               Local Inference                           │
  │  Ollama  +  3-step thinking loop  +  RAG (ChromaDB)     │
  └─────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11+
- API keys: `ANTHROPIC_API_KEY` (required for enrichment), `NVD_API_KEY` (optional)
- HuggingFace token: `HUGGINGFACE_TOKEN` (for general dataset download)
- For training: GPU with ≥24 GB VRAM (A100 recommended)
- For inference: [Ollama](https://ollama.com)

## Quickstart

```bash
# 1. Clone & install
git clone <repo-url>
cd cybersec-finetune
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 3. Run the full dataset pipeline (small test: --limit 100)
python dataset/pipeline.py all --limit 100

# 4. Review samples before training
python dataset/validation/sample_reviewer.py --samples 20

# 5. Train  (requires GPU — see training/README.md)
python training/train.py

# 6. Run inference
ollama run cyberphi "Explain how CVE-2021-44228 (Log4Shell) works."
```

## Folder structure

| Folder | Purpose |
|--------|---------|
| `dataset/` | Scrapers, enrichment, cleaning, mixing, validation — **fully implemented** |
| `training/` | Axolotl QLoRA config and training launcher — scaffold |
| `inference/` | Ollama client, 3-step thinking loop, RAG pipeline — scaffold |
| `evaluation/` | Benchmark runner and metrics — scaffold |
| `data/`      | Pipeline outputs (gitignored) |
| `logs/`      | Pipeline run logs |

See each subfolder's README for details.

## Estimated costs

| Step | Cost |
|------|------|
| NVD scraping | Free (rate-limited) |
| Claude API enrichment (~5K entries @ 2K tokens/entry) | ~$5–15 |
| RunPod A100 training (3 epochs, ~10K entries) | ~$8–18 |
| **Total** | **~$13–33** |

Using `--limit` on each stage during development keeps API costs near zero
until you're ready for a full production run.
