"""
Ollama Client
-------------
Thin wrapper around the Ollama REST API for text generation and embeddings.

Ollama must be running locally:  ollama serve
Default base URL:                http://localhost:11434

TODO: implement the methods below after the model is exported to GGUF
      and registered with Ollama.  See inference/README.md for setup.
"""

# Standard library
import json
import logging
from typing import Generator

# Third-party
import requests

logger = logging.getLogger(__name__)

# Default model name — created via `ollama create cyberphi -f Modelfile`
DEFAULT_MODEL    = "cyberphi"
OLLAMA_BASE_URL  = "http://localhost:11434"


class OllamaClient:
    """
    Minimal Ollama client that supports:
      - generate()       : streaming or blocking text generation
      - embed()          : single-string embedding for RAG
      - list_models()    : list registered models

    Usage:
        client = OllamaClient()
        response = client.generate("Explain XSS.")
        print(response)
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = DEFAULT_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.session  = requests.Session()

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """
        Generate text from a prompt.

        Args:
            prompt:      user message
            system:      optional system prompt override
            temperature: sampling temperature (0 = deterministic)
            max_tokens:  maximum tokens to generate
            stream:      if True, yield tokens as they arrive

        Returns:
            full response string (stream=False) or token generator (stream=True)

        TODO: implement request to POST /api/generate
              parse streaming NDJSON when stream=True
        """
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        """
        Return an embedding vector for text using the current model.
        Used by the RAG retriever to index and query documents.

        TODO: implement request to POST /api/embeddings
        """
        raise NotImplementedError

    def list_models(self) -> list[str]:
        """
        Return names of locally available Ollama models.

        TODO: implement request to GET /api/tags
        """
        raise NotImplementedError

    def health(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            resp = self.session.get(f"{self.base_url}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            return False
