"""Ports (abstractions) layer - defines interfaces for external systems."""
from .llm_provider import LLMProvider
from .paper_source import PaperSource

__all__ = ["LLMProvider", "PaperSource"]
