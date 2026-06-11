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

_UUID_RE = __import__("re").compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    __import__("re").IGNORECASE,
)


def _validate_api_key() -> None:
    if NVD_API_KEY and not _UUID_RE.match(NVD_API_KEY):
        raise ValueError(
            f"NVD_API_KEY looks malformed (got {len(NVD_API_KEY)} chars, expected 36). "
            "Check your .env or environment variable for accidental duplication."
        )


def _sleep_per_request() -> float:
    if NVD_API_KEY:
        return 30.0 / NVD_RATE_LIMIT_WITH_KEY      # ~0.6 s
    return 30.0 / NVD_RATE_LIMIT_WITHOUT_KEY        # 6.0 s


def _load_existing_ids() -> tuple[set, dict[str, int]]:
    """Return (id_set, per_severity_counts) in a single pass over the output file."""
    if not OUTPUT_FILE.exists():
        return set(), {}
    ids: set = set()
    counts: dict[str, int] = {}
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            ids.add(entry.get("cve_id", ""))
            sev = entry.get("severity", "UNKNOWN").upper()
            counts[sev] = counts.get(sev, 0) + 1
    logger.info("Loaded %d existing CVE IDs (resume mode) | per-severity: %s", len(ids), counts)
    return ids, counts


def _fetch_page(session: requests.Session, start_index: int, severity: str) -> dict:
    """Fetch one page from NVD with exponential-backoff retry."""
    params: dict = {
        "resultsPerPage": NVD_PAGE_SIZE,
        "startIndex":     start_index,
        "cvssV3Severity": severity,
    }
    headers: dict = {
        "User-Agent":      "CyberPhi/1.0 (security dataset builder)",
        "Accept-Encoding": "identity",
        "Accept":          "application/json",
    }
    if NVD_API_KEY:
        headers["apiKey"] = NVD_API_KEY   # v2.0 API: key goes in header, not query string

    for attempt in range(8):
        try:
            req      = requests.Request("GET", NVD_API_BASE, params=params, headers=headers)
            prepared = session.prepare_request(req)

            # Mask API key in log but keep everything else visible
            safe_url = prepared.url.replace(NVD_API_KEY, "***") if NVD_API_KEY else prepared.url
            logger.debug("REQUEST  url:     %s", safe_url)
            logger.debug("REQUEST  headers: %s", dict(prepared.headers))

            resp = session.send(prepared, timeout=90)

            logger.debug("RESPONSE status:  %d", resp.status_code)
            logger.debug("RESPONSE headers: %s", dict(resp.headers))
            logger.debug("RESPONSE body:    %s", resp.content[:500])

            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            r    = exc.response
            code = r.status_code
            wait = 2 ** attempt * 8 if code >= 500 else 2 ** attempt * 15
            logger.warning("HTTP %d (attempt %d/8) — retry in %ds", code, attempt + 1, wait)
            logger.warning("  url:             %s", r.url)
            logger.warning("  response headers:%s", dict(r.headers))
            logger.warning("  raw body bytes:  %s", r.content[:500])
            logger.warning("  decoded text:    %s", r.content.decode("utf-8", errors="replace")[:300])
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

    _validate_api_key()
    existing_ids, existing_counts = _load_existing_ids()
    sleep_secs    = _sleep_per_request()
    total_saved   = 0

    # Per-severity target: split the limit evenly across all requested severities.
    # Severities that already meet their target are skipped so the run fills gaps.
    per_sev_target = (limit // len(severities)) if limit else None

    if per_sev_target:
        logger.info(
            "Per-severity target: %d (limit=%d across %d severities)",
            per_sev_target, limit, len(severities),
        )
        for sev in severities:
            have = existing_counts.get(sev, 0)
            status = "DONE" if have >= per_sev_target else f"need {per_sev_target - have} more"
            logger.info("  %-10s existing=%-6d target=%-6d  %s", sev, have, per_sev_target, status)

    with requests.Session() as session:
        for severity in severities:
            already = existing_counts.get(severity, 0)

            if per_sev_target and already >= per_sev_target:
                logger.info(
                    "Skipping severity=%s — already have %d/%d entries",
                    severity, already, per_sev_target,
                )
                continue

            sev_need = (per_sev_target - already) if per_sev_target else None
            logger.info(
                "Scraping severity=%s — have %d, need %d more",
                severity, already, sev_need if sev_need else 0,
            )

            data          = _fetch_page(session, 0, severity)
            total_results = data.get("totalResults", 0)
            logger.info("Total available for %s on NVD: %d", severity, total_results)

            cap       = min(total_results, sev_need if sev_need else total_results)
            sev_saved = 0

            with tqdm(total=cap, desc=f"NVD {severity}", unit="CVE") as pbar:
                start_index  = 0
                sev_complete = False
                while not sev_complete:
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
                            sev_saved   += 1
                            pbar.update(1)
                            if sev_need and sev_saved >= sev_need:
                                logger.info(
                                    "Reached target for %s: %d new + %d existing = %d/%d",
                                    severity, sev_saved, already,
                                    sev_saved + already, per_sev_target,
                                )
                                sev_complete = True
                                break

                    if sev_complete:
                        break

                    start_index += len(vulns)
                    if start_index >= total_results:
                        break

                    time.sleep(sleep_secs)
                    data = _fetch_page(session, start_index, severity)

            logger.info("Finished %s: saved %d new entries this run", severity, sev_saved)

    logger.info("Done. Saved %d new CVEs total → %s", total_saved, OUTPUT_FILE)


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
