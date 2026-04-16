"""
Prompt template loader for arXiv query generation.

Loads prompt templates from prompts.yaml and exports them as module-level variables.
"""

# ---------------------------------------------------------------------------
# Load prompts from YAML configuration
# ---------------------------------------------------------------------------

import yaml
import sys
from pathlib import Path

_prompts_file = Path(__file__).parent / "prompts.yaml"

def _load_prompts():
    """Load all prompts from YAML configuration."""
    try:
        with open(_prompts_file, "r") as f:
            config = yaml.safe_load(f)
        
        if config is None or "prompts" not in config:
            raise ValueError(f"Invalid YAML structure: missing 'prompts' key in {_prompts_file}")
        
        prompts_dict = config["prompts"]
        
        # Main prompts
        result = {
            "PROMPT_QUERY_ARXIV": prompts_dict.get("query", {}).get("content", ""),
            "AUTHOR_QUERY_ARXIV": prompts_dict.get("author", {}).get("content", ""),
            "PROMPT_ARXIV_CATEGORY": prompts_dict.get("category", {}).get("content", ""),
            "CATEGORY_DOMAIN_PROMPT": prompts_dict.get("category_domain", {}).get("content", ""),
        }
        
        # Domain-specific category prompts dictionary
        domain_prompts = {}
        for domain_key in ["category_cs", "category_physics", "category_math", 
                           "category_stats", "category_biology", "category_econ", 
                           "category_engineering", "category_nonlinear"]:
            if domain_key in prompts_dict:
                domain_name = domain_key.replace("category_", "")
                domain_prompts[domain_name] = prompts_dict[domain_key].get("content", "")
        
        result["CATEGORY_DOMAIN_PROMPTS"] = domain_prompts
        return result
    
    except Exception as e:
        print(f"ERROR: Failed to load prompts from {_prompts_file}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        raise ImportError(f"Cannot load prompts: {e}")

# Load and inject into module namespace
_loaded = _load_prompts()
PROMPT_QUERY_ARXIV = _loaded["PROMPT_QUERY_ARXIV"]
AUTHOR_QUERY_ARXIV = _loaded["AUTHOR_QUERY_ARXIV"]
PROMPT_ARXIV_CATEGORY = _loaded["PROMPT_ARXIV_CATEGORY"]
CATEGORY_DOMAIN_PROMPT = _loaded["CATEGORY_DOMAIN_PROMPT"]
CATEGORY_DOMAIN_PROMPTS = _loaded["CATEGORY_DOMAIN_PROMPTS"]
