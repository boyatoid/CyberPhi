"""
HackerOne Public Hacktivity Scraper
------------------------------------
Fetches publicly disclosed bug bounty reports via the HackerOne GraphQL API
and saves them to data/raw/hackerone_reports.jsonl.

Outputs (per entry):
    report_id, title, vulnerability_type, severity, bounty,
    disclosed_at, program_name, summary, url

The HackerOne GraphQL endpoint is the same one their web app uses.
If the schema drifts, check network requests on https://hackerone.com/hacktivity/overview
and update H1_QUERY / the variable names accordingly.

Usage:
    python dataset/scrapers/hackerone_scraper.py
    python dataset/scrapers/hackerone_scraper.py --limit 500
"""

from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import jsonlines
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import H1_GRAPHQL_URL, H1_SCRAPE_DELAY, RAW_DIR

logger = logging.getLogger(__name__)
OUTPUT_FILE = RAW_DIR / "hackerone_reports.jsonl"

MIN_SUMMARY_LENGTH = 20  # vulnerability_information is auth-gated; use title+weakness

# Correct GraphQL query verified against live HackerOne schema (June 2026).
# Note: vulnerability_information is null for unauthenticated requests — we fall
# back to composing a summary from title + weakness + severity.
H1_QUERY = """
query GetDisclosedReports($cursor: String) {
  reports(first: 25, after: $cursor, substate: resolved) {
    pageInfo {
      hasNextPage
      endCursor
    }
    edges {
      node {
        id
        title
        substate
        disclosed_at
        url
        vulnerability_information
        severity { rating score }
        weakness { name }
        team { name url }
        reporter { username }
      }
    }
  }
}
"""

HEADERS = {
    "User-Agent":   "Mozilla/5.0 (compatible; CyberPhiDatasetBuilder/1.0; educational research)",
    "Content-Type": "application/json",
    "Accept":       "application/json",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_existing_ids() -> tuple[set, int]:
    if not OUTPUT_FILE.exists():
        return set(), 0
    ids: set = set()
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            ids.add(str(entry.get("report_id", "")))
    logger.info("Loaded %d existing report IDs (resume mode)", len(ids))
    return ids, len(ids)


def _fetch_page(session: requests.Session, cursor: str | None, page_size: int = 25) -> dict:
    """Send a GraphQL request. Returns the raw JSON response."""
    payload: dict = {
        "query":     H1_QUERY,
        "variables": {"cursor": cursor},
    }
    for attempt in range(5):
        try:
            resp = session.post(H1_GRAPHQL_URL, json=payload, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            wait = 2 ** attempt * 4
            logger.warning("GraphQL request failed (attempt %d/5): %s — retry in %ds",
                           attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError("HackerOne GraphQL request failed after 5 attempts")


def _extract(node: dict) -> dict | None:
    """Extract clean fields from a Report node."""
    title      = node.get("title", "")
    if not title:
        return None

    weakness   = node.get("weakness") or {}
    severity   = node.get("severity") or {}
    team       = node.get("team") or {}
    reporter   = node.get("reporter") or {}

    vuln_type  = weakness.get("name", "")
    sev_rating = severity.get("rating", "") or ""
    sev_score  = severity.get("score")

    # vulnerability_information is null without auth; compose a summary from available fields
    vuln_info  = node.get("vulnerability_information") or ""
    summary    = vuln_info if vuln_info else (
        f"{title}. Vulnerability type: {vuln_type}. "
        f"Severity: {sev_rating} (score: {sev_score}). "
        f"Program: {team.get('name', 'Unknown')}."
    )

    if len(summary) < MIN_SUMMARY_LENGTH:
        return None

    disclosed  = node.get("disclosed_at", "")
    # Only include reports that have actually been disclosed
    if not disclosed:
        return None

    return {
        "report_id":          str(node.get("id", "")),
        "title":              title,
        "vulnerability_type": vuln_type,
        "severity":           sev_rating.lower(),
        "cvss_score":         sev_score,
        "disclosed_at":       disclosed,
        "program_name":       team.get("name", ""),
        "program_url":        team.get("url", ""),
        "reporter":           reporter.get("username", ""),
        "summary":            summary,
        "url":                node.get("url", ""),
    }


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def scrape(limit: int | None = None) -> None:
    """
    Fetch public disclosed HackerOne reports.

    Args:
        limit: stop after this many saved entries (None = unlimited)
    """
    existing_ids, existing_count = _load_existing_ids()

    if limit and existing_count >= limit:
        logger.info("Skipping HackerOne — already have %d/%d entries", existing_count, limit)
        return

    remaining   = (limit - existing_count) if limit else None
    logger.info("HackerOne: have %d, need %d more", existing_count, remaining or 0)

    cursor      = None
    total_saved = 0

    with requests.Session() as session, tqdm(desc="HackerOne reports", unit="report") as pbar:
        while True:
            time.sleep(H1_SCRAPE_DELAY)

            try:
                data = _fetch_page(session, cursor)
            except RuntimeError as exc:
                logger.error("Stopping early: %s", exc)
                break

            errors = data.get("errors")
            if errors:
                logger.error("GraphQL errors: %s", errors)
                break

            reports_data = (
                data.get("data", {})
                    .get("reports", {})
            )
            if not reports_data:
                logger.warning("Unexpected response shape; check H1_QUERY")
                break

            page_info = reports_data.get("pageInfo", {})
            edges     = reports_data.get("edges", [])

            for edge in edges:
                node  = edge.get("node", {})
                entry = _extract(node)
                if entry is None or entry["report_id"] in existing_ids:
                    continue

                with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                    writer.write(entry)

                existing_ids.add(entry["report_id"])
                total_saved += 1
                pbar.update(1)

                if remaining and total_saved >= remaining:
                    logger.info("Reached target: %d new + %d existing = %d/%d",
                                total_saved, existing_count, total_saved + existing_count, limit)
                    return

            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

    logger.info("Done. Saved %d new reports → %s", total_saved, OUTPUT_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape HackerOne public disclosed reports")
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape(args.limit)


if __name__ == "__main__":
    main()
