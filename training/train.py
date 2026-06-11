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
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

CONFIG_YAML = Path(__file__).parent / "axolotl_config.yaml"
# Respect OUTPUT_DIR env var so quickstart.sh can set per-run output directories
OUTPUT_DIR  = Path(os.environ.get("OUTPUT_DIR", "outputs/cyberphi-lora"))


def run_training() -> None:
    """Launch Axolotl training via subprocess."""
    cmd = ["axolotl", "train", str(CONFIG_YAML)]
    if "OUTPUT_DIR" in os.environ:
        # Override the yaml's output_dir so training and merge use the same path
        cmd += ["--output_dir", str(OUTPUT_DIR)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _find_convert_script(llama_cpp_dir: Optional[str]) -> Path:
    """Locate convert_hf_to_gguf.py from an explicit dir, env var, or llama-cpp-python."""
    candidates: list[Path] = []

    if llama_cpp_dir:
        candidates.append(Path(llama_cpp_dir) / "convert_hf_to_gguf.py")

    env_dir = os.environ.get("LLAMA_CPP_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / "convert_hf_to_gguf.py")

    # llama-cpp-python bundles the convert script alongside its package
    try:
        import llama_cpp
        pkg_dir = Path(llama_cpp.__file__).parent
        candidates.append(pkg_dir / "convert_hf_to_gguf.py")
        candidates.append(pkg_dir.parent / "convert_hf_to_gguf.py")
    except ImportError:
        pass

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "convert_hf_to_gguf.py not found. Set LLAMA_CPP_DIR, pass --llama-cpp-dir, "
        "or install llama-cpp-python."
    )


def generate_modelfile(gguf_path: Path) -> Path:
    """Write an Ollama Modelfile next to the GGUF and return its path."""
    modelfile_path = gguf_path.parent / "Modelfile"
    modelfile_path.write_text(
        f'FROM ./{gguf_path.name}\n'
        'PARAMETER stop "<|im_end|>"\n'
        'SYSTEM "You are a senior cybersecurity expert and penetration tester."\n'
    )
    print(f"Modelfile written → {modelfile_path}")
    return modelfile_path


def merge_and_export(llama_cpp_dir: Optional[str] = None) -> None:
    """
    Merge the LoRA adapter into the base model weights and convert to GGUF.

    Requires:
        - Training complete (adapter in OUTPUT_DIR)
        - llama.cpp convert_hf_to_gguf.py accessible (set LLAMA_CPP_DIR or pass --llama-cpp-dir)
    """
    # Step 1 – merge adapter
    merge_cmd = [
        "axolotl", "merge-lora", str(CONFIG_YAML),
        "--lora-model-dir", str(OUTPUT_DIR),
    ]
    print(f"Merging: {' '.join(merge_cmd)}")
    subprocess.run(merge_cmd, check=True)

    # Step 2 – convert merged weights to GGUF
    merged_dir = OUTPUT_DIR / "merged"
    gguf_out   = OUTPUT_DIR / "cyberphi.gguf"
    convert_script = _find_convert_script(llama_cpp_dir)
    # convert_hf_to_gguf.py only supports f32/f16/bf16/q8_0; K-quants like
    # q4_k_m require a separate llama-quantize pass on the q8_0/f16 GGUF.
    convert_cmd = [
        sys.executable, str(convert_script),
        str(merged_dir),
        "--outfile", str(gguf_out),
        "--outtype", "q8_0",
    ]
    print(f"Converting to GGUF: {' '.join(str(c) for c in convert_cmd)}")
    subprocess.run(convert_cmd, check=True)
    print(f"GGUF written → {gguf_out}")

    # Step 3 – write Ollama Modelfile
    generate_modelfile(gguf_out)
    print("Next step: ollama create cyberphi -f outputs/cyberphi-lora/Modelfile")


def main() -> None:
    parser = argparse.ArgumentParser(description="CyberPhi training launcher")
    parser.add_argument("--merge-and-export", action="store_true",
                        help="Merge LoRA adapter and export to GGUF after training")
    parser.add_argument("--llama-cpp-dir", default=None,
                        help="Path to llama.cpp directory containing convert_hf_to_gguf.py")
    args = parser.parse_args()

    # --merge-and-export is a post-training step (quickstart.sh invokes it
    # after `axolotl train` has already run) — don't retrain.
    if args.merge_and_export:
        merge_and_export(args.llama_cpp_dir)
    else:
        run_training()


if __name__ == "__main__":
    main()
