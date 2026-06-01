#!/usr/bin/env bash
# quickstart.sh — CyberPhi end-to-end runner
#
# Usage:
#   ./quickstart.sh                                  # full run: data + train
#   ./quickstart.sh --skip-data                      # train only (data already built)
#   ./quickstart.sh --skip-train                     # data pipeline only
#   ./quickstart.sh --run-name exp_lr2e4             # named run → outputs/exp_lr2e4/
#   ./quickstart.sh --limit 200                      # cap scraper + enrichment at 200 rows
#   ./quickstart.sh --merge-export --llama-cpp-dir /opt/llama.cpp
#
# Multiple runs example:
#   ./quickstart.sh --skip-data --run-name baseline
#   ./quickstart.sh --skip-data --run-name higher_lr   # tweak axolotl_config.yaml first

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
RUN_NAME=""
SKIP_DATA=false
SKIP_TRAIN=false
MERGE_EXPORT=false
DATA_LIMIT=""
LLAMA_CPP_DIR=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-name)       RUN_NAME="$2";      shift 2 ;;
        --skip-data)      SKIP_DATA=true;     shift   ;;
        --skip-train)     SKIP_TRAIN=true;    shift   ;;
        --merge-export)   MERGE_EXPORT=true;  shift   ;;
        --limit)          DATA_LIMIT="$2";    shift 2 ;;
        --llama-cpp-dir)  LLAMA_CPP_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Auto-name runs by timestamp if not specified
if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="run_$(date +%Y%m%d_%H%M%S)"
fi

OUTPUT_DIR="outputs/${RUN_NAME}"
LOG_FILE="logs/quickstart_${RUN_NAME}.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"; }
die() { echo "ERROR: $*" >&2; exit 1; }

mkdir -p logs "$OUTPUT_DIR"

log "=== CyberPhi quickstart — run: $RUN_NAME ==="

# ---------------------------------------------------------------------------
# 1. Environment checks
# ---------------------------------------------------------------------------
log "Checking environment …"

[[ -f .env ]] || die ".env not found — copy .env.example and fill in your API keys"
# shellcheck disable=SC1091
source .env

[[ -n "${ANTHROPIC_API_KEY:-}" ]] || die "ANTHROPIC_API_KEY not set in .env"

# GPU info (non-fatal if unavailable — needed only for training)
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    log "GPU: $GPU_INFO"
else
    log "WARNING: nvidia-smi not found — training requires a CUDA GPU"
fi

# ---------------------------------------------------------------------------
# 2. Python dependencies
# ---------------------------------------------------------------------------
log "Installing Python dependencies …"
pip install -r requirements.txt -q

# ---------------------------------------------------------------------------
# 3. Axolotl (training only)
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "false" ]]; then
    if ! python3 -c "import axolotl" &>/dev/null 2>&1; then
        log "Installing Axolotl (this takes a few minutes) …"
        pip install axolotl[flash-attn] -q
    else
        log "Axolotl already installed ✓"
    fi
fi

# ---------------------------------------------------------------------------
# 4. Data pipeline
# ---------------------------------------------------------------------------
if [[ "$SKIP_DATA" == "false" ]]; then
    log "--- Data pipeline ---"

    PIPELINE_ARGS="all"
    if [[ -n "$DATA_LIMIT" ]]; then
        PIPELINE_ARGS="all --limit $DATA_LIMIT"
        log "Row limit: $DATA_LIMIT"
    fi

    log "Running: python dataset/pipeline.py $PIPELINE_ARGS"
    python dataset/pipeline.py $PIPELINE_ARGS 2>&1 | tee -a "$LOG_FILE"

    VALIDATED="data/final/validated.jsonl"
    if [[ ! -s "$VALIDATED" ]]; then
        die "Pipeline produced no output at $VALIDATED"
    fi
    ROWS=$(wc -l < "$VALIDATED")
    log "Dataset ready: $ROWS rows in $VALIDATED"
else
    log "Skipping data pipeline (--skip-data)"
    VALIDATED="data/final/validated.jsonl"
    [[ -s "$VALIDATED" ]] || die "$VALIDATED missing or empty — run without --skip-data first"
fi

# ---------------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRAIN" == "false" ]]; then
    log "--- Training: $RUN_NAME ---"
    log "Output dir: $OUTPUT_DIR"

    # Axolotl accepts --output_dir to override the config value
    TRAIN_CMD="axolotl train training/axolotl_config.yaml --output_dir $OUTPUT_DIR"
    log "Running: $TRAIN_CMD"
    $TRAIN_CMD 2>&1 | tee -a "$LOG_FILE"

    log "Training complete → $OUTPUT_DIR"
else
    log "Skipping training (--skip-train)"
fi

# ---------------------------------------------------------------------------
# 6. Merge + GGUF export (optional)
# ---------------------------------------------------------------------------
if [[ "$MERGE_EXPORT" == "true" ]]; then
    log "--- Merge & export ---"

    EXPORT_ARGS="--merge-and-export"
    [[ -n "$LLAMA_CPP_DIR" ]] && EXPORT_ARGS="$EXPORT_ARGS --llama-cpp-dir $LLAMA_CPP_DIR"

    # Pass the run-specific output dir via env so train.py finds the right adapter
    log "Running: python training/train.py $EXPORT_ARGS"
    OUTPUT_DIR="$OUTPUT_DIR" python training/train.py $EXPORT_ARGS 2>&1 | tee -a "$LOG_FILE"

    GGUF="${OUTPUT_DIR}/cyberphi.gguf"
    MODELFILE="${OUTPUT_DIR}/Modelfile"
    [[ -f "$GGUF" ]]      && log "GGUF:      $GGUF"
    [[ -f "$MODELFILE" ]] && log "Modelfile: $MODELFILE"
    log "Next: ollama create cyberphi-${RUN_NAME} -f $MODELFILE"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
log "=== Run '$RUN_NAME' complete ==="
log "Log: $LOG_FILE"
