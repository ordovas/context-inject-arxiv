"""
ArXiv paper source adapter.

Concrete implementation of PaperSource port for ArXiv API.
"""
import pandas as pd
from src.ports.paper_source import PaperSource
from src.adapters.outbound.arxiv.arxiv_client import get_papers_from_query


class ArxivAdapter(PaperSource):
    """Adapter for ArXiv paper retrieval."""
    
    def search(self, query: str, max_results: int = 200) -> pd.DataFrame:
        """
        Search for papers on ArXiv.
        
        Args:
            query: ArXiv query string (e.g., "cat:cs.LG AND submittedDate:[...]")
            max_results: Maximum number of papers to retrieve
            
        Returns:
            DataFrame with paper metadata
        """
        return get_papers_from_query(query, max_results)
    
    @property
    def source_name(self) -> str:
        """Return source name."""
        return "arxiv"
