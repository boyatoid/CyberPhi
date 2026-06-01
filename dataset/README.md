# Dataset

Builds a cybersecurity instruction-tuning dataset with CoT reasoning traces.

## Pipeline stages

```
scrapers/  →  enrichment/  →  cleaning/  →  mixing/  →  validation/
```

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| Scrape NVD | `scrapers/nvd_scraper.py` | NVD API | `raw/nvd_cves.jsonl` |
| Scrape Exploit-DB | `scrapers/exploitdb_scraper.py` | GitLab CSV | `raw/exploitdb_entries.jsonl` |
| Scrape CTF writeups | `scrapers/ctf_scraper.py` | CTFtime API | `raw/ctf_writeups.jsonl` |
| Scrape HackerOne | `scrapers/hackerone_scraper.py` | GraphQL | `raw/hackerone_reports.jsonl` |
| CoT enrichment | `enrichment/cot_enricher.py` | raw/*.jsonl | `enriched/*_enriched.jsonl` |
| Variants | `enrichment/variant_generator.py` | enriched/ | `enriched/variants.jsonl` |
| Q&A pairs | `enrichment/qa_generator.py` | any text | `enriched/qa_pairs.jsonl` |
| Dedup | `cleaning/deduplicator.py` | enriched/ | `cleaned/deduped.jsonl` |
| Quality filter | `cleaning/quality_filter.py` | deduped | `cleaned/filtered.jsonl` |
| Normalize schema | `cleaning/formatter.py` | filtered | `cleaned/formatted.jsonl` |
| Balance classes | `mixing/balancer.py` | formatted | `final/balanced.jsonl` |
| Mix general data | `mixing/general_mixer.py` | balanced | `final/mixed.jsonl` |
| Validate schema | `validation/schema_validator.py` | mixed | `final/validated.jsonl` |
| Human review | `validation/sample_reviewer.py` | validated | stdout |

## Quickstart

```bash
# Full pipeline (small test run)
python dataset/pipeline.py all --limit 100

# Individual stages
python dataset/pipeline.py scrape --source nvd --limit 200
python dataset/pipeline.py enrich --type cot --source nvd --limit 50
python dataset/pipeline.py clean
python dataset/pipeline.py mix
python dataset/pipeline.py validate --samples 15
```

## Unified entry schema

```json
{
  "id":         "uuid",
  "source":     "nvd|exploitdb|ctf|hackerone|synthetic|general",
  "vuln_type":  "sqli|xss|rce|ssrf|xxe|deserialization|...",
  "language":   "python|php|javascript|c|cpp|java|go|ruby|unknown",
  "severity":   "critical|high|medium|low|unknown",
  "difficulty": "beginner|intermediate|advanced",
  "instruction":"string",
  "input":      "string (code or context, may be empty)",
  "output":     "<think>…</think>\n\n[final answer]",
  "metadata":   { "source_id": "…", "date": "…", "tags": [] }
}
```
