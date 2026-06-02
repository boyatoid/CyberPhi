"""
CTFtime Writeup Scraper
-----------------------
Uses the CTFtime REST API to discover recent events, then scrapes the event
pages and follows writeup links (GitHub, blogs) to extract challenge write-ups.

Outputs (per entry):
    challenge_name, category, event_name, event_year, description,
    solution_steps, tools_used, raw_content, url

Usage:
    python dataset/scrapers/ctf_scraper.py
    python dataset/scrapers/ctf_scraper.py --limit 300 --year 2024
"""

from __future__ import annotations
import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

import jsonlines
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataset.config import CTF_CATEGORIES, CTF_SCRAPE_DELAY, CTFTIME_API_BASE, RAW_DIR

logger = logging.getLogger(__name__)
OUTPUT_FILE = RAW_DIR / "ctf_writeups.jsonl"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CyberPhiDatasetBuilder/1.0; educational research)"
}

MIN_CONTENT_LENGTH = 300


# ---------------------------------------------------------------------------
# CTFtime API helpers
# ---------------------------------------------------------------------------

def _get_events(year: int, session: requests.Session) -> list[dict]:
    """Return a list of CTFtime events for the given year."""
    start  = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    finish = int(datetime(year, 12, 31, tzinfo=timezone.utc).timestamp())
    url    = f"{CTFTIME_API_BASE}/events/"
    params = {"limit": 200, "start": start, "finish": finish}

    try:
        resp = session.get(url, params=params, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch events for %d: %s", year, exc)
        return []


def _list_ctftime_writeups(page: int, session: requests.Session) -> list[dict]:
    """
    Scrape a page of the global CTFtime writeups listing.
    Returns list of dicts with keys: writeup_id, task_name, event_name, category.
    CTFtime structure: /writeups lists rows: [event, task, team, Read→/writeup/{id}]
    """
    url  = f"https://ctftime.org/writeups?page={page}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        soup  = BeautifulSoup(resp.text, "lxml")
        table = soup.find("table")
        if not table:
            return []
        rows = []
        for tr in table.find_all("tr")[1:]:  # skip header
            cells = tr.find_all("td")
            links = {a.get_text(strip=True): a["href"] for a in tr.find_all("a")}
            # Read link → /writeup/{id}
            read_href = links.get("Read", "")
            if not read_href or "/writeup/" not in read_href:
                continue
            wid = read_href.split("/writeup/")[-1].rstrip("/")
            task_link = next((v for k, v in links.items() if "/task/" in v), "")
            rows.append({
                "writeup_id": wid,
                "task_name":  cells[1].get_text(strip=True) if len(cells) > 1 else "",
                "event_name": cells[0].get_text(strip=True) if cells else "",
                "task_url":   f"https://ctftime.org{task_link}" if task_link else "",
            })
        return rows
    except Exception as exc:
        logger.debug("Failed to list writeups page %d: %s", page, exc)
        return []


def _get_external_link(writeup_id: str, session: requests.Session) -> tuple[str, str]:
    """
    Fetch https://ctftime.org/writeup/{id} and return (external_url, category).
    The page contains a link to the actual writeup content (GitHub, blog, etc.).
    """
    url = f"https://ctftime.org/writeup/{writeup_id}"
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return "", "misc"
        soup = BeautifulSoup(resp.text, "lxml")
        # External writeup link
        ext_links = [
            a["href"] for a in soup.find_all("a")
            if a.get("href","").startswith("http")
            and "ctftime.org" not in a["href"]
            and "transdata.no" not in a["href"]      # CTFtime sponsor ad
        ]
        ext_url = ext_links[0] if ext_links else ""
        # Category from task info block
        cat = "misc"
        for tag in soup.find_all(["td", "th", "span", "p"]):
            text = tag.get_text(strip=True).lower()
            for c in CTF_CATEGORIES:
                if c in text:
                    cat = c
                    break
        return ext_url, cat
    except Exception as exc:
        logger.debug("Failed to get external link for writeup %s: %s", writeup_id, exc)
        return "", "misc"


# ---------------------------------------------------------------------------
# Content extraction
# ---------------------------------------------------------------------------

def _extract_github_raw(url: str, session: requests.Session) -> str | None:
    """Convert a GitHub blob URL to raw content and fetch it."""
    raw_url = (
        url.replace("github.com", "raw.githubusercontent.com")
           .replace("/blob/", "/")
    )
    try:
        resp = session.get(raw_url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def _extract_blog_content(url: str, session: requests.Session) -> str | None:
    """Generic HTML → plain-text extraction for blog-style writeup pages."""
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove nav / header / footer / script / style noise
    for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
        tag.decompose()

    # Prefer <article> or <main> blocks
    for selector in ("article", "main", ".post-content", ".entry-content", "#content"):
        block = soup.select_one(selector)
        if block:
            return block.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


def _fetch_writeup(url: str, session: requests.Session) -> str | None:
    if "github.com" in url and "/blob/" in url:
        return _extract_github_raw(url, session)
    return _extract_blog_content(url, session)


def _parse_solution_steps(text: str) -> list[str]:
    """Heuristically split writeup text into numbered steps."""
    steps = re.split(r"\n(?=\d+[\.\)]\s)", text)
    if len(steps) < 2:
        steps = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 20]
    return [s.strip() for s in steps if s.strip()]


TOOL_NAMES = [
    "pwntools", "gdb", "ghidra", "ida", "binwalk", "wireshark", "burpsuite",
    "sqlmap", "nmap", "hashcat", "john", "z3", "angr", "radare2", "objdump",
    "strace", "ltrace", "cycript", "frida", "volatility", "strings", "file",
]


def _extract_tools(text: str) -> list[str]:
    lower = text.lower()
    return [t for t in TOOL_NAMES if t in lower]


# ---------------------------------------------------------------------------
# Main scrape function
# ---------------------------------------------------------------------------

def scrape(limit: int | None = None, years: list[int] | None = None) -> None:
    """
    Scrape CTF writeups from the CTFtime global writeups listing.

    Flow: /writeups (paginated) → /writeup/{id} (get external link)
          → fetch external content (GitHub raw / blog HTML)

    Args:
        limit: maximum entries to save (None = unlimited)
        years: unused (kept for CLI compatibility; CTFtime listing is global)
    """
    existing_urls, existing_count = _load_existing_urls()

    if limit and existing_count >= limit:
        logger.info("Skipping CTF — already have %d/%d entries", existing_count, limit)
        return

    remaining   = (limit - existing_count) if limit else None
    logger.info("CTF writeups: have %d, need %d more", existing_count, remaining or 0)

    session     = requests.Session()
    total_saved = 0
    page        = 1

    with tqdm(desc="CTF writeups", unit="writeup") as pbar:
        while True:
            time.sleep(CTF_SCRAPE_DELAY)
            rows = _list_ctftime_writeups(page, session)
            if not rows:
                logger.info("No more writeups at page %d", page)
                break

            for row in rows:
                time.sleep(CTF_SCRAPE_DELAY)
                ext_url, cat = _get_external_link(row["writeup_id"], session)

                if not ext_url or ext_url in existing_urls:
                    continue

                time.sleep(CTF_SCRAPE_DELAY)
                content = _fetch_writeup(ext_url, session)
                if not content or len(content) < MIN_CONTENT_LENGTH:
                    continue

                # Refine category using content if the page-level guess was generic
                if cat == "misc":
                    cat = _infer_category(ext_url, content)

                entry = {
                    "challenge_name":  row["task_name"] or _infer_challenge_name(ext_url, content),
                    "category":        cat,
                    "event_name":      row["event_name"],
                    "event_year":      datetime.now().year,
                    "description":     content[:500],
                    "solution_steps":  _parse_solution_steps(content),
                    "tools_used":      _extract_tools(content),
                    "raw_content":     content,
                    "url":             ext_url,
                    "ctftime_writeup": f"https://ctftime.org/writeup/{row['writeup_id']}",
                }

                with jsonlines.open(OUTPUT_FILE, mode="a") as writer:
                    writer.write(entry)

                existing_urls.add(ext_url)
                total_saved += 1
                pbar.update(1)

                if remaining and total_saved >= remaining:
                    logger.info("Reached target: %d new + %d existing = %d/%d",
                                total_saved, existing_count, total_saved + existing_count, limit)
                    return

            page += 1

    logger.info("Done. Saved %d writeups → %s", total_saved, OUTPUT_FILE)


def _load_existing_urls() -> tuple[set, int]:
    if not OUTPUT_FILE.exists():
        return set(), 0
    urls: set = set()
    with jsonlines.open(OUTPUT_FILE) as r:
        for entry in r:
            urls.add(entry.get("url", ""))
    logger.info("Loaded %d existing URLs (resume mode)", len(urls))
    return urls, len(urls)


def _infer_category(url: str, content: str) -> str:
    combined = (url + " " + content[:1000]).lower()
    order    = ["pwn", "web", "crypto", "reverse", "forensics", "misc"]
    for cat in order:
        if cat in combined:
            return cat
    return "misc"


def _infer_challenge_name(url: str, content: str) -> str:
    # Try to pull a title from the first heading in content
    match = re.search(r"(?m)^#\s+(.+)$", content)
    if match:
        return match.group(1).strip()
    # Fall back to last URL path component
    return urlparse(url).path.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape CTF writeups from CTFtime")
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--year",      type=int, nargs="+", default=None,
                        help="Years to scrape (default: last 3 years)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    scrape(args.limit, args.year)


if __name__ == "__main__":
    main()
