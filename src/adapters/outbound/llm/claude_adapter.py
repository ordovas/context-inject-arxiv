"""
Claude LLM provider adapter.

Concrete implementation of LLMProvider port for Anthropic Claude API.
"""
import time
from typing import Dict, Any
from src.ports.llm_provider import LLMProvider


class ClaudeAdapter(LLMProvider):
    """Adapter for Anthropic Claude API."""
    
    def __init__(self, model_id: str):
        """
        Initialize Claude adapter.
        
        Args:
            model_id: Claude model identifier 
                     (e.g., "claude-3-5-sonnet-20241022")
        """
        self._model_id = model_id
        self.client = None
        self._init_connection()
    
    def _init_connection(self) -> None:
        """Initialize Anthropic client."""
        try:
            import anthropic
            self.client = anthropic.Anthropic()
            # API key should be in ANTHROPIC_API_KEY environment variable
        except ImportError:
            raise RuntimeError(
                "Anthropic SDK not installed. "
                "Install with: pip install anthropic"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Claude client: {e}")
    
    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Get response from Claude model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict with content, latency_ms, and tokens
        """
        start_time = time.time()
        
        try:
            message = self.client.messages.create(
                model=self._model_id,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            latency_ms = (time.time() - start_time) * 1000
            tokens = message.usage.input_tokens + message.usage.output_tokens
            
            return {
                "content": message.content[0].text,
                "latency_ms": latency_ms,
                "tokens": tokens,
            }
            
        except Exception as e:
            raise RuntimeError(f"Claude API error: {e}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "claude"
    
    @property
    def model_id(self) -> str:
        """Return model identifier."""
        return self._model_id
