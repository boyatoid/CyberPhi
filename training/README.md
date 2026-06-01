# Training

QLoRA fine-tuning of Phi-3.5-mini-instruct using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl).

## Hardware recommendations

| Config | VRAM | Est. time (3 epochs) | Est. cost (RunPod) |
|--------|------|---------------------|-------------------|
| 1× A100 80GB | 80 GB | ~4–6 h | ~$12–18 |
| 1× A100 40GB | 40 GB | ~6–9 h | ~$8–12 |
| 1× RTX 4090 | 24 GB | ~10–14 h | ~$3–5 |

## Prerequisites

1. Dataset at `data/final/validated.jsonl` (run `python dataset/pipeline.py all`)
2. Upload dataset to HuggingFace Hub (or keep local — update `axolotl_config.yaml`):
   ```bash
   huggingface-cli upload your-username/cyberphi-dataset data/final/validated.jsonl
   ```
3. RunPod instance with PyTorch 2.x + CUDA 12.x image

## Setup on RunPod

```bash
# Install Axolotl
pip install axolotl[flash-attn]
pip install -e '.[deepspeed]'   # optional, for multi-GPU

# Clone this repo or upload files
git clone <your-repo>
cd cybersec-finetune
```

## Running training

```bash
python training/train.py
# or directly via Axolotl
axolotl train training/axolotl_config.yaml
```

## Output

- LoRA adapter saved to `outputs/cyberphi-lora/`
- Merge and export to GGUF for Ollama inference:
  ```bash
  python training/train.py --merge-and-export
  ```
