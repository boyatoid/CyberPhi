"""
Dataset pipeline configuration.
Loads API keys from environment and defines all shared constants.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

RAW_DIR      = DATA_DIR / "raw"
ENRICHED_DIR = DATA_DIR / "enriched"
CLEANED_DIR  = DATA_DIR / "cleaned"
FINAL_DIR    = DATA_DIR / "final"

for _d in [RAW_DIR, ENRICHED_DIR, CLEANED_DIR, FINAL_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY")
NVD_API_KEY        = os.getenv("NVD_API_KEY")
HUGGINGFACE_TOKEN  = os.getenv("HUGGINGFACE_TOKEN")

# ---------------------------------------------------------------------------
# Target dataset sizes  (min, max)
# ---------------------------------------------------------------------------
TARGET_SIZES = {
    "nvd":        (3000, 5000),
    "exploitdb":  (1000, 2000),
    "ctf":        (2000, 3000),
    "hackerone":  (1000, 2000),
    "variants":   (2000, 4000),
}

# ---------------------------------------------------------------------------
# Taxonomies
# ---------------------------------------------------------------------------
VULN_TYPES = [
    "sqli", "xss", "rce", "ssrf", "xxe",
    "deserialization", "buffer_overflow", "race_condition",
    "idor", "path_traversal",
]

LANGUAGES = ["python", "php", "javascript", "c", "cpp", "java", "go", "ruby"]

# ---------------------------------------------------------------------------
# NVD API
# ---------------------------------------------------------------------------
NVD_API_BASE               = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_PAGE_SIZE              = 500   # smaller pages reduce NVD 404 throttle risk
NVD_RATE_LIMIT_WITHOUT_KEY = 5   # requests / 30 s
NVD_RATE_LIMIT_WITH_KEY    = 50  # requests / 30 s

# ---------------------------------------------------------------------------
# Exploit-DB
# ---------------------------------------------------------------------------
EXPLOITDB_CSV_URL  = (
    "https://gitlab.com/exploit-database/exploitdb/-/raw/main/files_exploits.csv"
)
EXPLOITDB_FILE_URL = (
    "https://gitlab.com/exploit-database/exploitdb/-/raw/main/{file}"
)

# ---------------------------------------------------------------------------
# CTFtime
# ---------------------------------------------------------------------------
CTFTIME_API_BASE = "https://ctftime.org/api/v1"
CTF_CATEGORIES   = ["pwn", "web", "crypto", "reverse", "forensics", "misc"]
CTF_SCRAPE_DELAY = 1.5  # seconds between requests

# ---------------------------------------------------------------------------
# HackerOne
# ---------------------------------------------------------------------------
H1_GRAPHQL_URL  = "https://hackerone.com/graphql"
H1_SCRAPE_DELAY = 2.0  # seconds between requests

# ---------------------------------------------------------------------------
# Claude / Anthropic
# ---------------------------------------------------------------------------
# Model routing — cheapest model that meets the quality bar for each task
CLAUDE_MODEL_COT     = "claude-sonnet-4-6"           # CoT enrichment
CLAUDE_MODEL_VARIANT = "claude-sonnet-4-6"           # Variant generation
CLAUDE_MODEL_QA      = "claude-haiku-4-5-20251001"   # Q&A pairs (structured, simpler)
CLAUDE_MODEL_NVD     = "claude-haiku-4-5-20251001"   # Bulk NVD CVE processing

CLAUDE_MODEL      = "claude-sonnet-4-6"   # generic fallback
CLAUDE_BATCH_SIZE = 10
CLAUDE_MAX_TOKENS = 4096

# Pricing per million tokens — update if Anthropic changes rates
# Sonnet 4.6
SONNET_INPUT_COST_PER_MTOK        =  3.00
SONNET_OUTPUT_COST_PER_MTOK       = 15.00
SONNET_CACHE_WRITE_COST_PER_MTOK  =  3.75   # 125% of input (cache creation)
SONNET_CACHE_READ_COST_PER_MTOK   =  0.30   # 10%  of input (cache hit)

# Haiku 4.5
HAIKU_INPUT_COST_PER_MTOK         =  0.80
HAIKU_OUTPUT_COST_PER_MTOK        =  4.00
HAIKU_CACHE_WRITE_COST_PER_MTOK   =  1.00   # 125% of input
HAIKU_CACHE_READ_COST_PER_MTOK    =  0.08   # 10%  of input

# ---------------------------------------------------------------------------
# General instruction mixing
# ---------------------------------------------------------------------------
GENERAL_DATASET_NAME = "tatsu-lab/alpaca"
GENERAL_MIX_RATIO    = 0.15   # 15 % of cybersec dataset size

# ---------------------------------------------------------------------------
# Quality / dedup thresholds
# ---------------------------------------------------------------------------
MIN_DESCRIPTION_LENGTH = 100
MIN_OUTPUT_LENGTH      = 200
MIN_THINK_LENGTH       = 100
MIN_INSTRUCTION_WORDS  = 10
MIN_TECHNICAL_SCORE    = 0.6

DEDUP_THRESHOLD  = 0.85   # Jaccard similarity above this → duplicate
MINHASH_NUM_PERM = 128

MAX_VULN_TYPE_FRACTION = 0.30   # downsample if a class exceeds this
MIN_VULN_TYPE_FRACTION = 0.03   # flag as underrepresented below this
