"""
Adapters for external LLM servers (Ollama, vLLM, OpenAI-compatible APIs).
Allows running early-exit reasoning when models are hosted on local servers.
"""

from zelt.adapters import OllamaAdapter, VLLMAdapter

__all__ = ["OllamaAdapter", "VLLMAdapter"]
