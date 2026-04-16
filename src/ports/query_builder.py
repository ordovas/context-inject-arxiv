"""
Port (abstraction) for query builders.

This is the interface that all query builders must implement.
It represents a "port" in hexagonal architecture - the contract
between the domain and query generation logic.
"""
from abc import ABC, abstractmethod


class QueryBuilderPort(ABC):
    """Abstract port for query builders."""
    
    @abstractmethod
    async def build_query_async(self, user_input: str) -> str:
        """
        Build a query asynchronously from user input.
        
        Args:
            user_input: Natural language description of papers to find
            
        Returns:
            Formatted query string (implementation-specific format)
            
        Raises:
            ValueError: If query generation fails
        """
        pass
    
    @property
    @abstractmethod
    def builder_type(self) -> str:
        """Return the type of query builder (e.g., 'arxiv', 'scholar')."""
        pass
