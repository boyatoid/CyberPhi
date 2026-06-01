"""
Training configuration.

This file is intentionally minimal — Axolotl reads its own config from
axolotl_config.yaml. Add any Python-side training helpers here.
"""

# Path to the validated dataset produced by the dataset pipeline
DATASET_PATH = "../data/final/validated.jsonl"

# HuggingFace Hub repository (set to None to use local path)
HF_DATASET_REPO = None   # e.g. "your-username/cyberphi-dataset"

# Output directory for LoRA adapter checkpoints
OUTPUT_DIR = "./outputs/cyberphi-lora"

# Base model
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
