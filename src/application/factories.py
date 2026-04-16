"""
Adapter factory - instantiates appropriate adapters based on configuration.

This is part of the application layer, responsible for wiring adapters.
"""
from src.ports.llm_provider import LLMProvider
from src.ports.paper_source import PaperSource
from src.adapters.outbound.llm import LMStudioAdapter, ClaudeAdapter, OllamaAdapter
from src.adapters.outbound.arxiv.arxiv_adapter import ArxivAdapter


class LLMProviderFactory:
    """Factory for creating LLM provider adapters."""
    
    @staticmethod
    def create(provider: str, model_id: str, **kwargs) -> LLMProvider:
        """
        Create an LLM provider adapter.
        
        Args:
            provider: "lmstudio", "claude", or "ollama"
            model_id: Model identifier
            **kwargs: Additional provider-specific arguments
                      - For Ollama: base_url, auto_pull, auto_unload
                      
        Returns:
            LLMProvider implementation
        """
        if provider == "lmstudio":
            return LMStudioAdapter(model_id)
        
        elif provider == "claude":
            return ClaudeAdapter(model_id)
        
        elif provider == "ollama":
            base_url = kwargs.get('base_url', 'http://localhost:11434')
            auto_pull = kwargs.get('auto_pull', False)
            auto_unload = kwargs.get('auto_unload', False)
            return OllamaAdapter(
                model_id, 
                base_url=base_url,
                auto_pull=auto_pull,
                auto_unload=auto_unload
            )
        
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")


class PaperSourceFactory:
    """Factory for creating paper source adapters."""
    
    @staticmethod
    def create(source: str) -> PaperSource:
        """
        Create a paper source adapter.
        
        Args:
            source: "arxiv" or other source
            
        Returns:
            PaperSource implementation
        """
        if source == "arxiv":
            return ArxivAdapter()
        
        else:
            raise ValueError(f"Unknown paper source: {source}")
