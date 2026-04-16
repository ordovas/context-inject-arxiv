"""
LangGraph agent for arXiv paper retrieval.

This is orchestration logic (application layer), not domain logic.
It wires together domain services with adapters via dependency injection.

Extended to support optional context injection — when a context_yaml_path
is provided, the agent uses ContextQueryBuilder to inject domain-specific
knowledge into LLM prompts.
"""
from langgraph.graph import StateGraph, END
from functools import partial
from typing import Optional

from src.application.factories import LLMProviderFactory
from src.domain.query_builder import QueryBuilder
from src.domain.context_query_builder import ContextQueryBuilder
from src.domain.context.context_resolver import ContextResolver
from src.domain.repositories.states.arxiv import ArxivState
from src.domain.graphs.arxiv.nodes import query_generation_node, paper_retrieval_node
from src.config import LLM_MODEL
from src.ports.llm_provider import LLMProvider
from src.ports.paper_source import PaperSource
from src.application.factories import PaperSourceFactory


class ArxivPaperAgent:
    """
    LangGraph agent for retrieving arXiv papers based on natural language queries.

    This agent orchestrates the workflow:
    1. Takes user input and generates an arXiv query using LLM
    2. Retrieves papers from arXiv using the generated query
    3. Returns results as a pandas DataFrame

    Dependencies are injected, maintaining separation of concerns:
    - LLM provider for query generation
    - Paper source for paper retrieval
    - Optional ContextResolver for domain-specific knowledge injection
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        paper_source: PaperSource,
        category_mode: str = "single_step",
        context_resolver: Optional[ContextResolver] = None,
    ):
        """
        Initialize the agent with dependencies.

        Args:
            llm_provider: LLMProvider implementation
            paper_source: PaperSource implementation
            category_mode: "single_step" or "two_step"
            context_resolver: Optional ContextResolver for domain knowledge injection
        """
        self.llm_provider = llm_provider
        self.paper_source = paper_source
        self.context_resolver = context_resolver

        # Use context-aware builder when resolver is provided
        if context_resolver is not None:
            self.query_builder = ContextQueryBuilder(
                self.llm_provider,
                category_mode=category_mode,
                context_resolver=context_resolver,
            )
        else:
            self.query_builder = QueryBuilder(
                self.llm_provider,
                category_mode=category_mode,
            )

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ArxivState)

        workflow.add_node(
            "query_generation",
            partial(query_generation_node, query_builder=self.query_builder)
        )
        workflow.add_node(
            "paper_retrieval",
            partial(paper_retrieval_node, paper_source=self.paper_source)
        )

        workflow.set_entry_point("query_generation")

        def should_retrieve(state: ArxivState) -> str:
            if state.get("error"):
                return "end"
            return "retrieve"

        workflow.add_conditional_edges(
            "query_generation",
            should_retrieve,
            {"retrieve": "paper_retrieval", "end": END}
        )

        workflow.add_edge("paper_retrieval", END)

        return workflow.compile()

    async def run(self, user_input: str) -> dict:
        """Run the agent to retrieve papers."""
        initial_state = {
            "user_input": user_input,
            "arxiv_query": None,
            "papers_df": None,
            "error": None,
            "messages": []
        }

        final_state = await self.graph.ainvoke(initial_state)

        if final_state.get("error"):
            raise ValueError(final_state["error"])

        papers_df = final_state.get("papers_df")
        if papers_df is None:
            raise ValueError("No papers were retrieved")

        return final_state

    def get_state_info(self, user_input: str) -> dict:
        """Run the agent and return full state information (synchronous)."""
        initial_state = {
            "user_input": user_input,
            "arxiv_query": None,
            "papers_df": None,
            "error": None,
            "messages": []
        }
        return self.graph.invoke(initial_state)


def create_arxiv_agent(
    llm_provider: str = "lmstudio",
    model_id: str = None,
    paper_source: str = "arxiv",
    category_mode: str = "single_step",
    context_yaml_path: Optional[str] = None,
    **llm_kwargs
) -> ArxivPaperAgent:
    """
    Factory function to create an ArxivPaperAgent with dependencies.

    Args:
        llm_provider: Type of LLM provider ("lmstudio", "claude", or "ollama")
        model_id: Model identifier. If None, uses default from config
        paper_source: Type of paper source ("arxiv")
        category_mode: "single_step" or "two_step"
        context_yaml_path: Optional path to external context YAML.
                           If provided, enables context injection.
        **llm_kwargs: Provider-specific keyword arguments

    Returns:
        An instance of ArxivPaperAgent with all dependencies injected
    """
    if model_id is None:
        model_id = LLM_MODEL

    llm_adapter = LLMProviderFactory.create(llm_provider, model_id, **llm_kwargs)
    source_adapter = PaperSourceFactory.create(paper_source)

    # Create context resolver if path is provided
    context_resolver = None
    if context_yaml_path is not None:
        context_resolver = ContextResolver(context_yaml_path)

    return ArxivPaperAgent(
        llm_adapter,
        source_adapter,
        category_mode=category_mode,
        context_resolver=context_resolver,
    )
