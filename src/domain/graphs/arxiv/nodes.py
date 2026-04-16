"""
Node functions for the LangGraph agent workflow.
"""
from typing import Dict, Any
from src.domain.repositories.states.arxiv import ArxivState
from src.ports.paper_source import PaperSource


def paper_retrieval_node(state: ArxivState, paper_source: PaperSource) -> Dict[str, Any]:
    """Retrieve papers from a paper source using the generated query."""
    arxiv_query = state.get("arxiv_query")

    if not arxiv_query:
        error_msg = "No arXiv query available for retrieval"
        print(f"[Paper Retrieval] Error: {error_msg}")
        return {
            "error": error_msg,
            "messages": state.get("messages", []) + [error_msg],
        }

    print(f"[Paper Retrieval] Fetching papers for query: {arxiv_query}")

    try:
        papers_df = paper_source.search(arxiv_query)
        num_papers = len(papers_df)
        print(f"[Paper Retrieval] Retrieved {num_papers} papers")
        return {
            "papers_df": papers_df,
            "messages": state.get("messages", []) + [f"Retrieved {num_papers} papers from arXiv"],
        }
    except Exception as e:
        error_msg = f"Failed to retrieve papers: {str(e)}"
        print(f"[Paper Retrieval] Error: {error_msg}")
        return {
            "error": error_msg,
            "messages": state.get("messages", []) + [error_msg],
        }


async def query_generation_node(state: ArxivState, query_builder) -> Dict[str, Any]:
    """
    Generate the arXiv query and store raw per-component outputs in state.

    The individual components (query_content, query_author, query_category)
    are stored separately so the benchmark can validate each one without
    relying solely on whether papers were retrieved.
    """
    print(f"[Query Generation] Processing user input: {state['user_input']}")

    try:
        combined, raw_content, raw_author, raw_category = (
            await query_builder.build_query_with_components_async(state["user_input"])
        )
        print(f"[Query Generation] Generated query: {combined}")

        return {
            "arxiv_query": combined,
            "query_content":  raw_content,
            "query_author":   raw_author,
            "query_category": raw_category,
            "messages": state.get("messages", []) + [f"Generated query: {combined}"],
        }
    except Exception as e:
        error_msg = f"Failed to generate query: {str(e)}"
        print(f"[Query Generation] Error: {error_msg}")
        return {
            "error": error_msg,
            "messages": state.get("messages", []) + [error_msg],
        }