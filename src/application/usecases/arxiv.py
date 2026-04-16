"""
ArXiv paper retrieval use case.

High-level orchestration of the ArXiv agent workflow.
This represents the primary use case: "retrieve papers based on natural language query"

Extended with optional context_yaml_path to enable domain-specific
knowledge injection into the query generation pipeline.
"""
from typing import Optional

from src.application.agents.arxiv_agent import create_arxiv_agent
import langgraph


async def run_arxiv_agent(
    user_prompt: str,
    llm_provider: str = "lmstudio",
    model_id: str = None,
    paper_source: str = "arxiv",
    category_mode: str = "two_step",
    context_yaml_path: Optional[str] = None,
    **llm_kwargs
) -> langgraph.pregel.io.AddableValuesDict:
    """
    Run the ArxivPaperAgent with the given user prompt.

    This is the main entry point for the ArXiv paper retrieval use case.

    Args:
        user_prompt: Natural language description of papers to find
        llm_provider: LLM provider to use ("lmstudio", "claude", or "ollama")
        model_id: Model identifier. If None, uses provider defaults
        paper_source: Paper source to use (default "arxiv")
        category_mode: "single_step" or "two_step"
        context_yaml_path: Optional path to external context YAML file.
                           When provided, domain-specific knowledge is
                           injected into LLM prompts for enriched queries.
        **llm_kwargs: Provider-specific keyword arguments

    Returns:
        Final state after running the agent containing papers_df and metadata
    """
    agent = create_arxiv_agent(
        llm_provider=llm_provider,
        model_id=model_id,
        paper_source=paper_source,
        category_mode=category_mode,
        context_yaml_path=context_yaml_path,
        **llm_kwargs
    )
    return await agent.run(user_prompt)