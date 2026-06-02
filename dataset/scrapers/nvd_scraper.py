"""
NVD CVE Scraper
---------------
Fetches HIGH and CRITICAL CVEs from the NVD 2.0 REST API and writes them to
data/raw/nvd_cves.jsonl.

Outputs (per entry):
    cve_id, description, published_date, last_modified, severity,
    cvss_score, cvss_vector, cwe_ids, references

Usage:
    python dataset/scrapers/nvd_scraper.py
    python dataset/scrapers/nvd_scraper.py --limit 200 --severity CRITICAL
    python dataset/scrapers/nvd_scraper.py --log-level DEBUG
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
from dataset.config import (
    ANTHROPIC_API_KEY,
    NVD_API_BASE,
    NVD_API_KEY,
    NVD_PAGE_SIZE,
    NVD_RATE_LIMIT_WITH_KEY,
    NVD_RATE_LIMIT_WITHOUT_KEY,
    RAW_DIR,
)

logger = logging.getLogger(__name__)
OUTPUT_FILE = RAW_DIR / "nvd_cves.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sleep_per_request() -> float:
    if NVD_API_KEY:
        return 30.0 / NVD_RATE_LIMIT_WITH_KEY      # ~0.6 s
    return 30.0 / NVD_RATE_LIMIT_WITHOUT_KEY        # 6.0 s


def _load_existing_ids() -> set:
    if not OUTPUT_FILE.exists():
        return set()
    ids: set = set()
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            ids.add(entry.get("cve_id", ""))
    logger.info("Loaded %d existing CVE IDs (resume mode)", len(ids))
    return ids


def _fetch_page(session: requests.Session, start_index: int, severity: str) -> dict:
    """Fetch one page from NVD with exponential-backoff retry."""
    params: dict = {
        "resultsPerPage": NVD_PAGE_SIZE,
        "startIndex":     start_index,
        "cvssV3Severity": severity,
    }
    headers: dict = {}
    if NVD_API_KEY:
        headers["apiKey"] = NVD_API_KEY

    for attempt in range(8):
        try:
            resp = session.get(NVD_API_BASE, params=params, headers=headers, timeout=90)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            code = exc.response.status_code
            # 4xx from NVD (including 404) is a soft throttle/overload signal — wait longer
            wait = 2 ** attempt * 8 if code >= 500 else 2 ** attempt * 15
            logger.warning("HTTP %d (attempt %d/8): %s — retry in %ds", code, attempt + 1, exc, wait)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = 2 ** attempt * 5
            logger.warning("Timeout (attempt %d/8) — retry in %ds", attempt + 1, wait)
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = 2 ** attempt * 5
            logger.warning("Request error (attempt %d/8): %s — retry in %ds", attempt + 1, exc, wait)
            time.sleep(wait)

    raise RuntimeError(f"Could not fetch NVD page startIndex={start_index} after 8 attempts")


def _extract(vuln: dict) -> dict | None:
    """Return a clean dict from a raw NVD vulnerability node, or None if it should be skipped."""
    cve = vuln.get("cve", {})
    cve_id = cve.get("id", "")

    descriptions = cve.get("descriptions", [])
    description = next((d["value"] for d in descriptions if d.get("lang") == "en"), "")
    if len(description) < 100:
        return None

    metrics     = cve.get("metrics", {})
    cvss_data   = {}
    severity    = "UNKNOWN"
    cvss_score  = None

    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        bucket = metrics.get(key, [])
        if bucket:
            primary = next((m for m in bucket if m.get("type") == "Primary"), bucket[0])
            cvss_data  = primary.get("cvssData", {})
            severity   = cvss_data.get("baseSeverity", "UNKNOWN").upper()
            cvss_score = cvss_data.get("baseScore")
            break

    cwe_ids: list = []
    for w in cve.get("weaknesses", []):
        for d in w.get("description", []):
            val = d.get("value", "")
            if d.get("lang") == "en" and val.startswith("CWE-"):
                cwe_ids.append(val)

    references = [r.get("url", "") for r in cve.get("references", [])[:5]]

    return {
        "cve_id":         cve_id,
        "description":    description,
        "published_date": cve.get("published", ""),
        "last_modified":  cve.get("lastModified", ""),
        "severity":       severity,
        "cvss_score":     cvss_score,
        "cvss_vector":    cvss_data.get("vectorString", ""),
        "cwe_ids":        cwe_ids,
        "references":     references,
    }


# ---------------------------------------------------------------------------
# Main scrape function (also importable by pipeline.py)
# ---------------------------------------------------------------------------

def scrape(severities: list[str] = None, limit: int | None = None) -> None:
    """
    Scrape NVD CVEs for the given severity levels and save to OUTPUT_FILE.

    Args:
        severities: list of CVSS v3 severity strings, e.g. ["HIGH", "CRITICAL"]
        limit:      stop after this many new entries (None = unlimited)
    """
    if severities is None:
        severities = ["HIGH", "CRITICAL"]

    existing_ids  = _load_existing_ids()
    sleep_secs    = _sleep_per_request()
    session       = requests.Session()
    total_saved   = 0

    for severity in severities:
        logger.info("Scraping severity=%s", severity)
        data          = _fetch_page(session, 0, severity)
        total_results = data.get("totalResults", 0)
        logger.info("Total available for %s: %d", severity, total_results)

        cap = min(total_results, limit - total_saved if limit else total_results)

        with tqdm(total=cap, desc=f"NVD {severity}", unit="CVE") as pbar:
            start_index = 0
            while True:
                vulns = data.get("vulnerabilities", [])
                if not vulns:
                    break

                with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                    for vuln in vulns:
                        entry = _extract(vuln)
                        if entry is None or entry["cve_id"] in existing_ids:
                            continue
                        writer.write(entry)
                        existing_ids.add(entry["cve_id"])
                        total_saved += 1
                        pbar.update(1)
                        if limit and total_saved >= limit:
                            logger.info("Reached limit=%d", limit)
                            return

                start_index += len(vulns)
                if start_index >= total_results:
                    break

                time.sleep(sleep_secs)
                data = _fetch_page(session, start_index, severity)

    logger.info("Done. Saved %d new CVEs → %s", total_saved, OUTPUT_FILE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape CVEs from NVD 2.0 API")
    parser.add_argument("--limit",    type=int, default=None,
                        help="Maximum number of new CVEs to download")
    parser.add_argument("--severity", nargs="+",
                        choices=["HIGH", "CRITICAL", "MEDIUM", "LOW"],
                        default=["HIGH", "CRITICAL"])
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape(args.severity, args.limit)


if __name__ == "__main__":
    main()
