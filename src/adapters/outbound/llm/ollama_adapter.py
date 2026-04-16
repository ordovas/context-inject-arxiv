"""
Ollama LLM provider adapter.

Concrete implementation of LLMProvider port for Ollama local models.
"""
import time
from typing import Dict, Any
from src.ports.llm_provider import LLMProvider


class OllamaAdapter(LLMProvider):
    """Adapter for Ollama local LLM models."""
    
    def __init__(
        self, 
        model_id: str, 
        base_url: str = "http://localhost:11434",
        auto_pull: bool = False,
        auto_unload: bool = False
    ):
        """
        Initialize Ollama adapter.
        
        Args:
            model_id: Model identifier (e.g., "llama2", "mistral", "neural-chat")
            base_url: Ollama server base URL (default: http://localhost:11434)
            auto_pull: If True, automatically pull model if not available (default: False)
            auto_unload: If True, unload model after each response (default: False)
        """
        self._model_id = model_id
        self._base_url = base_url
        self._auto_pull = auto_pull
        self._auto_unload = auto_unload
        self.client = None
        self._init_connection()
    
    def _init_connection(self) -> None:
        """Connect to Ollama server and optionally pull model if missing."""
        try:
            import ollama
            print(f"Connecting to Ollama at {self._base_url}")
            print(f"Using model: {self._model_id}")
            
            # Ollama client connects to the server
            self.client = ollama.Client(host=self._base_url)
            
            # Check available models
            try:
                models = self.client.list()
                available_model_names = [m['name'].split(':')[0] for m in models.get('models', [])]
                model_base_name = self._model_id.split(':')[0]
                
                model_available = any(
                    m.startswith(model_base_name) 
                    for m in [model['name'] for model in models.get('models', [])]
                )
                
                if not model_available and self._auto_pull:
                    print(f"Model '{self._model_id}' not found locally. Auto-pulling...")
                    self._pull_model()
                    print(f"Model '{self._model_id}' pulled successfully")
                    
                elif not model_available:
                    print(f"Warning: Model '{self._model_id}' not found locally.")
                    print(f"Available models: {', '.join(available_model_names) if available_model_names else 'none'}")
                    print(f"Tip: Run 'ollama pull {self._model_id}' or set auto_pull=True")
                else:
                    print(f"Model '{self._model_id}' found locally")
                    
            except Exception as e:
                if self._auto_pull:
                    print(f"Could not verify model availability: {e}")
                    print(f"Attempting to pull model '{self._model_id}'...")
                    try:
                        self._pull_model()
                        print(f"Model '{self._model_id}' pulled successfully")
                    except Exception as pull_error:
                        print(f"Warning: Could not pull model: {pull_error}")
                else:
                    print(f"Warning: Could not list models: {e}")
            
        except ImportError:
            raise RuntimeError(
                "Ollama SDK not installed. "
                "Install with: pip install ollama"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Ollama. Error: {e}\n"
                "Make sure Ollama is running (ollama serve) and accessible at {self._base_url}"
            )
    
    def _pull_model(self) -> None:
        """Pull a model from ollama.com if it's not available."""
        try:
            import ollama
            self.client.pull(self._model_id)
        except Exception as e:
            raise RuntimeError(
                f"Failed to pull model '{self._model_id}'. Error: {e}\n"
                "Make sure the model name is valid and you have internet connection.\n"
                f"Valid models: https://ollama.ai/library"
            )
    
    def _unload_model(self) -> None:
        """Unload/stop the model from memory."""
        try:
            import ollama
            # Generate empty prompt to unload model
            self.client.generate(
                model=self._model_id,
                prompt="",
                stream=False,
                keep_alive=0  # 0 means unload immediately
            )
        except Exception as e:
            print(f"Warning: Could not unload model: {e}")
    
    def respond(self, prompt: str) -> Dict[str, Any]:
        """
        Get response from Ollama model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict with content, latency_ms, and tokens
        """
        start_time = time.time()
        
        try:
            import ollama
            
            # Call Ollama generate endpoint
            response = self.client.generate(
                model=self._model_id,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.1,  # Low temperature for deterministic results
                    "num_predict": 2048,  # Max tokens to generate
                }
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract content and token counts
            content = response.get('response', '').strip()
            
            # Ollama doesn't provide token counts in the response by default
            # but we can estimate from the response
            tokens = None
            
            result = {
                "content": content,
                "latency_ms": latency_ms,
                "tokens": tokens,
            }
            
            # Unload model if auto_unload is enabled
            if self._auto_unload:
                self._unload_model()
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {e}")
    
    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "ollama"
    
    @property
    def model_id(self) -> str:
        """Return model identifier."""
        return self._model_id
