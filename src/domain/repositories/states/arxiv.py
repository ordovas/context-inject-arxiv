"""
State definitions for the LangGraph agent.
"""
from typing import TypedDict, Optional, List, Annotated
import pandas as pd
from operator import add


class ArxivState(TypedDict):
    """
    State for the arXiv paper retrieval agent.

    Attributes:
        user_input: Natural language query from the user
        arxiv_query: Final combined arXiv API query string
        query_content: Raw LLM output for the content component (all:/ti: etc.)
        query_author: Raw LLM output for the author component (au:)
        query_category: Raw LLM output for the category component (cat:)
        papers_df: DataFrame containing retrieved papers
        error: Error message if something goes wrong
        messages: List of messages for conversation history
    """
    user_input: str
    arxiv_query: str
    query_content: Optional[str]
    query_author: Optional[str]
    query_category: Optional[str]

    papers_df: Optional[pd.DataFrame]
    error: Optional[str]
    messages: Annotated[List[str], add]