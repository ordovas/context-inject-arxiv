"""
Context-aware query builder — extends QueryBuilder with external context injection.

When a ContextResolver is provided, the builder injects a formatted context block
into each LLM prompt before the user request, allowing the LLM to use domain-specific
knowledge (glossary, teams, project phases, instruments, datasets).

When no ContextResolver is provided, behaviour is identical to the base QueryBuilder.
"""
import re
from typing import Optional

from src.domain.query_builder import QueryBuilder
from src.domain.context.context_resolver import ContextResolver, ResolvedContext


# Regex to find the injection point in prompts (before "USER REQUEST:")
_INJECTION_RE = re.compile(
    r"(\n\s*USER REQUEST:\s*\{user_input\})",
    re.IGNORECASE,
)


def _inject_context_into_prompt(
    prompt_template: str,
    context_block: str,
) -> str:
    """
    Insert a context block into a prompt template just before the
    USER REQUEST line.

    If the context block is empty or the injection point is not found,
    the original template is returned unchanged.
    """
    if not context_block.strip():
        return prompt_template

    match = _INJECTION_RE.search(prompt_template)
    if match:
        insertion_point = match.start()
        return (
            prompt_template[:insertion_point]
            + "\n\n" + context_block.strip() + "\n"
            + prompt_template[insertion_point:]
        )

    # Fallback: append context before the last line
    return prompt_template + "\n\n" + context_block.strip() + "\n"


class ContextQueryBuilder(QueryBuilder):
    """
    Query builder with optional external context injection.

    Extends QueryBuilder by supplying a prompt_enricher that injects resolved
    context into each prompt template before it is formatted and sent to the LLM.

    Args:
        model: LLMProvider instance
        category_mode: "single_step" or "two_step"
        context_resolver: Optional ContextResolver instance. If None,
                          behaviour is identical to base QueryBuilder.
    """

    def __init__(
        self,
        model,
        category_mode: str = "single_step",
        context_resolver: Optional[ContextResolver] = None,
    ):
        def enricher(template, prompt_type, user_input):
            if context_resolver is None:
                return template
            resolved = context_resolver.resolve(user_input)
            if not resolved.has_matches:
                return template
            return _inject_context_into_prompt(
                template, resolved.format_for_prompt(prompt_type)
            )

        super().__init__(model, category_mode=category_mode, prompt_enricher=enricher)
        self.context_resolver = context_resolver

    # ------------------------------------------------------------------
    # Introspection helpers (useful for benchmark)
    # ------------------------------------------------------------------

    def resolve_context(self, user_input: str) -> Optional[ResolvedContext]:
        """
        Resolve context for a query without running LLM calls.
        Useful for debugging and the benchmark notebook.
        """
        if self.context_resolver is None:
            return None
        return self.context_resolver.resolve(user_input)

    def get_enriched_prompt(
        self, prompt_template: str, user_input: str, prompt_type: str
    ) -> str:
        """
        Return the full prompt that would be sent to the LLM, with
        context injected. Useful for debugging.
        """
        if self.context_resolver is None:
            return prompt_template.format(user_input=user_input)

        resolved = self.context_resolver.resolve(user_input)
        enriched = _inject_context_into_prompt(
            prompt_template, resolved.format_for_prompt(prompt_type)
        )
        return enriched.format(user_input=user_input)
