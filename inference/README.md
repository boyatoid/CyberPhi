# Inference

Local inference via [Ollama](https://ollama.com) with an optional RAG pipeline
for injecting fresh vulnerability context at query time.

## Setup

```bash
# 1. Install Ollama  (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Create the Modelfile pointing at your exported GGUF
cat > Modelfile <<'EOF'
FROM ./outputs/cyberphi-lora/cyberphi.gguf
SYSTEM "You are a senior cybersecurity expert and penetration tester."
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096
EOF

# 3. Register the model with Ollama
ollama create cyberphi -f Modelfile

# 4. Test
ollama run cyberphi "Explain SQL injection and provide a Python PoC."
```

## Components

| File | Purpose |
|------|---------|
| `ollama_client.py` | Thin wrapper around the Ollama REST API |
| `thinking_loop.py` | 3-step reasoning loop (think → reflect → answer) |
| `rag/vector_store.py` | ChromaDB-backed CVE / security-doc index |
| `rag/retriever.py` | Fetch top-k chunks and inject into the prompt |

## RAG setup

```bash
# Index your CVE / OWASP docs
python inference/rag/vector_store.py --index data/raw/nvd_cves.jsonl

# Run inference with RAG
python inference/thinking_loop.py --query "How does CVE-2024-1234 work?" --use-rag
```
