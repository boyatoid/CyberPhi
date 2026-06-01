"""
Training entry point for CyberPhi QLoRA fine-tuning.

This script is a thin wrapper around Axolotl. The heavy lifting (LoRA setup,
quantization, gradient checkpointing) is declared in axolotl_config.yaml.

Setup on a fresh RunPod instance
---------------------------------
1. Start a RunPod pod with:
   - Image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
   - At least 1× A100 40GB (80GB recommended for sequence_len=4096)

2. Install Axolotl:
   pip install axolotl[flash-attn]

3. (Optional) Authenticate with HuggingFace to pull gated models:
   huggingface-cli login

4. (Optional) Upload the dataset to HuggingFace Hub before training so you
   don't copy large files into the pod:
   huggingface-cli upload <username>/cyberphi-dataset data/final/validated.jsonl

5. Clone / upload this repo to the pod:
   git clone <your-repo-url>
   cd cybersec-finetune

Running training
-----------------
Option A — direct Axolotl CLI (recommended):
    axolotl train training/axolotl_config.yaml

Option B — via this script:
    python training/train.py
    python training/train.py --merge-and-export   # merge adapter + export GGUF

Saving the LoRA adapter
------------------------
Axolotl saves adapter weights to outputs/cyberphi-lora/ after each checkpoint
and at the end of training. The adapter is a small directory (~200–600 MB for
rank=16) that can be loaded on top of the base model.

Exporting to GGUF for Ollama
------------------------------
After training:
    # 1. Merge LoRA into base weights
    python training/train.py --merge-and-export

    # 2. This calls llama.cpp's convert script internally.
    #    The result is outputs/cyberphi-lora/cyberphi.gguf

    # 3. Create an Ollama Modelfile pointing at the GGUF:
    #    FROM ./outputs/cyberphi-lora/cyberphi.gguf
    #    PARAMETER stop "<|im_end|>"
    ollama create cyberphi -f Modelfile
    ollama run cyberphi
"""

import argparse
import subprocess
import sys
from pathlib import Path

CONFIG_YAML = Path(__file__).parent / "axolotl_config.yaml"
OUTPUT_DIR  = Path("outputs/cyberphi-lora")


def run_training() -> None:
    """Launch Axolotl training via subprocess."""
    cmd = ["axolotl", "train", str(CONFIG_YAML)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def merge_and_export() -> None:
    """
    Merge the LoRA adapter into the base model weights and convert to GGUF.

    Requires:
        - Training complete (adapter in OUTPUT_DIR)
        - llama.cpp installed and on PATH  (or: pip install llama-cpp-python)
    """
    # Step 1 – merge adapter
    merge_cmd = [
        "axolotl", "merge-lora", str(CONFIG_YAML),
        "--lora-model-dir", str(OUTPUT_DIR),
    ]
    print(f"Merging: {' '.join(merge_cmd)}")
    subprocess.run(merge_cmd, check=True)

    # Step 2 – convert to GGUF  (requires llama.cpp convert_hf_to_gguf.py)
    merged_dir = OUTPUT_DIR / "merged"
    gguf_out   = OUTPUT_DIR / "cyberphi.gguf"
    convert_cmd = [
        sys.executable, "-m", "llama_cpp.server",   # placeholder — adjust to your llama.cpp path
        "--model", str(merged_dir),
        "--outfile", str(gguf_out),
        "--outtype", "q4_k_m",
    ]
    print(f"Converting to GGUF: {gguf_out}")
    # subprocess.run(convert_cmd, check=True)   # uncomment when llama.cpp is available
    print("GGUF export placeholder — wire up the actual llama.cpp convert_hf_to_gguf.py path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CyberPhi training launcher")
    parser.add_argument("--merge-and-export", action="store_true",
                        help="Merge LoRA adapter and export to GGUF after training")
    args = parser.parse_args()

    run_training()
    if args.merge_and_export:
        merge_and_export()


if __name__ == "__main__":
    main()
