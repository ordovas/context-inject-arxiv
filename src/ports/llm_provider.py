"""
Port (abstraction) for LLM providers.

This is the interface that all LLM providers must implement.
It represents a "port" in hexagonal architecture - the contract
between the domain and external LLM systems.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """Abstract port for LLM providers."""
    
    @abstractmethod
    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Get a response from the LLM.
        
        Args:
            prompt: Input prompt/query
            
        Returns:
            Dict with keys:
            - content (str): Generated response
            - latency_ms (float): Response time in milliseconds
            - tokens (int | None): Token count (provider-specific)
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model identifier."""
        pass
