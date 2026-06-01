"""
Dataset Pipeline Orchestrator
------------------------------
Runs the full dataset build end-to-end or any individual stage.

Subcommands:
    scrape    – run one or all scrapers
    enrich    – run CoT enrichment / variant / Q&A generation
    clean     – dedup → quality filter → format  (in sequence)
    mix       – balance → general mix  (in sequence)
    validate  – schema validation + human sample review
    all       – run everything in order

Usage:
    python dataset/pipeline.py all --limit 1000
    python dataset/pipeline.py scrape --source nvd --limit 500
    python dataset/pipeline.py enrich --type cot --source nvd
    python dataset/pipeline.py clean
    python dataset/pipeline.py validate --samples 20
    python dataset/pipeline.py all --dry-run
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stage imports — each module exposes a clean entry-point function
from dataset.cleaning.deduplicator import deduplicate
from dataset.cleaning.formatter import format_dataset
from dataset.cleaning.quality_filter import filter_dataset
from dataset.config import LOGS_DIR
from dataset.enrichment.cot_enricher import enrich
from dataset.enrichment.qa_generator import generate_qa
from dataset.enrichment.variant_generator import generate_variants
from dataset.mixing.balancer import balance
from dataset.mixing.general_mixer import mix
from dataset.scrapers.ctf_scraper import scrape as ctf_scrape
from dataset.scrapers.exploitdb_scraper import scrape as exploitdb_scrape
from dataset.scrapers.hackerone_scraper import scrape as hackerone_scrape
from dataset.scrapers.nvd_scraper import scrape as nvd_scrape
from dataset.validation.sample_reviewer import review
from dataset.validation.schema_validator import validate

LOG_FILE = LOGS_DIR / "pipeline.log"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        ],
    )


logger = logging.getLogger("pipeline")


def _stage(name: str, fn, dry_run: bool, *args, **kwargs) -> float:
    """
    Run one pipeline stage.
    Returns elapsed seconds (0 if dry_run).
    """
    if dry_run:
        logger.info("[DRY RUN] Would run: %s  args=%s  kwargs=%s", name, args, kwargs)
        return 0.0

    logger.info("╔══ START  %s", name)
    t0 = time.perf_counter()
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.error("Stage '%s' failed: %s", name, exc, exc_info=True)
        raise
    elapsed = time.perf_counter() - t0
    logger.info("╚══ DONE   %s  (%.1fs)", name, elapsed)
    return elapsed


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_scrape(args) -> None:
    sources = (
        ["nvd", "exploitdb", "ctf", "hackerone"]
        if args.source == "all"
        else [args.source]
    )
    scrapers = {
        "nvd":       lambda: nvd_scrape(["HIGH", "CRITICAL"], args.limit),
        "exploitdb": lambda: exploitdb_scrape(args.limit),
        "ctf":       lambda: ctf_scrape(args.limit),
        "hackerone": lambda: hackerone_scrape(args.limit),
    }
    for src in sources:
        _stage(f"scrape/{src}", scrapers[src], args.dry_run)


def run_enrich(args) -> None:
    etype   = getattr(args, "type", "all")
    sources = ["nvd", "exploitdb", "ctf", "hackerone"]
    limit   = getattr(args, "limit", None)

    if etype in ("cot", "all"):
        for src in sources:
            _stage(f"enrich/cot/{src}", enrich, args.dry_run, src, limit)

    if etype in ("variants", "all"):
        _stage("enrich/variants", generate_variants, args.dry_run, None, limit)

    if etype in ("qa", "all"):
        # Q&A generation requires an --input-file; skip if not specified
        input_file = getattr(args, "input_file", None)
        if input_file:
            _stage("enrich/qa", generate_qa, args.dry_run, input_file, limit)
        else:
            logger.info("Skipping Q&A enrichment (no --input-file provided)")


def run_clean(args) -> None:
    _stage("clean/deduplicate",    deduplicate,    args.dry_run)
    _stage("clean/quality_filter", filter_dataset, args.dry_run)
    _stage("clean/format",         format_dataset, args.dry_run)


def run_mix(args) -> None:
    _stage("mix/balance", balance, args.dry_run)
    _stage("mix/general", mix,     args.dry_run)


def run_validate(args) -> None:
    n_samples = getattr(args, "samples", 10)
    _stage("validate/schema",  validate, args.dry_run)
    _stage("validate/samples", review,   args.dry_run, n_samples)


def run_all(args) -> None:
    t_start = time.perf_counter()

    run_scrape(args)
    run_enrich(args)
    run_clean(args)
    run_mix(args)
    run_validate(args)

    total = time.perf_counter() - t_start
    logger.info("Pipeline complete in %.1fs", total)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CyberPhi dataset pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without executing anything")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    sub = parser.add_subparsers(dest="command", required=True)

    # scrape
    p_scrape = sub.add_parser("scrape", help="Run data scrapers")
    p_scrape.add_argument("--source", default="all",
                          choices=["all", "nvd", "exploitdb", "ctf", "hackerone"])
    p_scrape.add_argument("--limit", type=int, default=None)

    # enrich
    p_enrich = sub.add_parser("enrich", help="Run enrichment (CoT / variants / Q&A)")
    p_enrich.add_argument("--type", default="all",
                          choices=["all", "cot", "variants", "qa"])
    p_enrich.add_argument("--source", default="all",
                          choices=["all", "nvd", "exploitdb", "ctf", "hackerone"])
    p_enrich.add_argument("--input-file", default=None,
                          help="Text/JSONL file for Q&A generation")
    p_enrich.add_argument("--limit", type=int, default=None)

    # clean
    sub.add_parser("clean", help="Run dedup → filter → format")

    # mix
    sub.add_parser("mix", help="Run balance → general mix")

    # validate
    p_val = sub.add_parser("validate", help="Validate and review samples")
    p_val.add_argument("--samples", type=int, default=10)

    # all
    p_all = sub.add_parser("all", help="Run full pipeline end-to-end")
    p_all.add_argument("--limit",      type=int, default=None,
                       help="Per-source scrape / enrich limit (useful for testing)")
    p_all.add_argument("--input-file", default=None)
    p_all.add_argument("--samples",    type=int, default=10)
    p_all.add_argument("--source",     default="all")

    args = parser.parse_args()

    # Ensure common attributes exist regardless of subcommand
    for attr, default in [("limit", None), ("source", "all"), ("samples", 10),
                          ("input_file", None)]:
        if not hasattr(args, attr):
            setattr(args, attr, default)

    _setup_logging(args.log_level)

    dispatch = {
        "scrape":   run_scrape,
        "enrich":   run_enrich,
        "clean":    run_clean,
        "mix":      run_mix,
        "validate": run_validate,
        "all":      run_all,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
