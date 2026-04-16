"""
LMStudio LLM provider adapter.

Concrete implementation of LLMProvider port for LMStudio local models.
"""
import time
from typing import Dict, Any
from src.ports.llm_provider import LLMProvider


class LMStudioAdapter(LLMProvider):
    """Adapter for LMStudio local LLM models."""
    
    def __init__(self, model_id: str):
        """
        Initialize LMStudio adapter.
        
        Args:
            model_id: Model identifier (for reference; actual model must be 
                     loaded in LMStudio UI)
        """
        self._model_id = model_id
        self.model = None
        self._init_connection()
    
    def _init_connection(self) -> None:
        """Connect to LMStudio (assumes model is already loaded in UI)."""
        try:
            import lmstudio as lms
            print(f"Connecting to LMStudio model: {self._model_id}")
            self.model = lms.llm()
            print("Connected to LMStudio model")
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to LMStudio. Error: {e}\n"
                "Make sure LMStudio is running with a model loaded."
            )
    
    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Get response from LMStudio model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict with content, latency_ms, and tokens
        """
        start_time = time.time()
        
        try:
            import lmstudio as lms
            
            # Use Chat API to properly collect full response
            chat = lms.Chat()
            chat.add_user_message(prompt)
            
            config = {
                "temperature": 0.1,  # Low temperature for deterministic results
                "maxTokens": 2048,
            }
            
            # respond() returns the content directly
            response = self.model.respond(chat, config=config)
            
            # Extract content (respond() returns string directly)
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            content = content.strip()
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "content": content,
                "latency_ms": latency_ms,
                "tokens": None,
            }
            
        except Exception as e:
            raise RuntimeError(f"LMStudio response error: {e}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "lmstudio"
    
    @property
    def model_id(self) -> str:
        """Return model identifier."""
        return self._model_id
