"""
Port (abstraction) for paper/document sources.

This represents a "port" for retrieving papers from external systems.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class PaperSource(ABC):
    """Abstract port for paper/document retrieval systems."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 200) -> pd.DataFrame:
        """
        Search for papers using a query.
        
        Args:
            query: Search query string (format depends on implementation)
            max_results: Maximum number of results to return
            
        Returns:
            DataFrame with paper metadata
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of this paper source."""
        pass
