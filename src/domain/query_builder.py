"""
Query builder — converts natural language to arXiv API queries via parallel LLM calls.

Raw LLM outputs are cleaned by output_cleaner.clean_component()
before the components are combined or stored in state.  This removes the most common
contamination patterns (outer quotes, prose prefixes, markdown fences) so they never
propagate to the arXiv API or the validator.

Category classification supports two modes (controlled by `category_mode`):
  - "single_step" : one LLM call with the full 150+ code list (original behaviour)
  - "two_step"    : two sequential LLM calls — domain classification then
                    domain-specific code selection with a shorter list
"""
from typing import Optional, Tuple, List
import asyncio
import re

from src.ports.query_builder import QueryBuilderPort
from src.domain.prompts.prompts import (
    PROMPT_QUERY_ARXIV,
    AUTHOR_QUERY_ARXIV,
    PROMPT_ARXIV_CATEGORY,
    CATEGORY_DOMAIN_PROMPT,
    CATEGORY_DOMAIN_PROMPTS,
)
from src.domain.output_cleaner import clean_component


# Recognised domain labels for the two-step classifier
_VALID_DOMAINS = frozenset(CATEGORY_DOMAIN_PROMPTS.keys())


class QueryBuilder(QueryBuilderPort):
    """
    Query builder for converting natural language to arXiv API queries using LLM.

    Args:
        model: LLMProvider instance (any adapter implementing LLMProvider port)
        category_mode: "single_step" (default, original 1-call category) or
                       "two_step" (2-call domain→codes classification)
    """

    def __init__(self, model, category_mode: str = "single_step", prompt_enricher=None):
        self.model = model
        if category_mode not in ("single_step", "two_step"):
            raise ValueError(f"Unknown category_mode: {category_mode!r}. Use 'single_step' or 'two_step'.")
        self.category_mode = category_mode
        self._enrich = prompt_enricher or (lambda template, ptype, user_input: template)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def build_query_async(self, user_input: str) -> str:
        """
        Build a combined arXiv query string from user input.

        Backward-compatible entry point — returns only the final string.
        Call build_query_with_components_async() to also get the per-component outputs.

        Raises:
            ValueError: If all three components returned NONE after cleaning.
        """
        combined, _, _, _ = await self.build_query_with_components_async(user_input)
        return combined

    async def build_query_with_components_async(
        self, user_input: str
    ) -> Tuple[str, str, str, str]:
        """
        Build an arXiv query and return the cleaned per-component strings.

        Content and author LLM calls always run in parallel.  The category
        component uses either a single parallel call (single_step) or a
        two-stage sequential pipeline (two_step) depending on self.category_mode.

        Returns:
            (combined_query, cleaned_content, cleaned_author, cleaned_category)

        Raises:
            ValueError: If all three components returned NONE after cleaning.
        """
        if self.category_mode == "two_step":
            return await self._build_two_step(user_input)
        else:
            return await self._build_single_step(user_input)

    # ------------------------------------------------------------------
    # Single-step
    # ------------------------------------------------------------------

    async def _build_single_step(self, user_input: str) -> Tuple[str, str, str, str]:
        """All three LLM calls in parallel, category uses the full 150+ code list."""
        tasks = [
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(PROMPT_QUERY_ARXIV, "content", user_input).format(user_input=user_input),
            )),
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(AUTHOR_QUERY_ARXIV, "author", user_input).format(user_input=user_input),
            )),
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(PROMPT_ARXIV_CATEGORY, "category", user_input).format(user_input=user_input),
            )),
        ]
        results = await asyncio.gather(*tasks)

        raw_content  = self._extract_text(results[0])
        raw_author   = self._extract_text(results[1])
        raw_category = self._extract_text(results[2])

        return self._combine(raw_content, raw_author, raw_category)

    # ------------------------------------------------------------------
    # Two-step category classification
    # ------------------------------------------------------------------

    async def _build_two_step(self, user_input: str) -> Tuple[str, str, str, str]:
        """
        Content and author run in parallel with domain classification (step 1).
        Then step 2 calls the domain-specific category prompt(s) sequentially.

        Latency ≈ max(content, author, domain) + category_step2
        """
        # Phase 1: content + author + domain classification — all in parallel
        tasks = [
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(PROMPT_QUERY_ARXIV, "content", user_input).format(user_input=user_input),
            )),
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(AUTHOR_QUERY_ARXIV, "author", user_input).format(user_input=user_input),
            )),
            asyncio.create_task(asyncio.to_thread(
                self.model.respond,
                self._enrich(CATEGORY_DOMAIN_PROMPT, "category", user_input).format(user_input=user_input),
            )),
        ]
        results = await asyncio.gather(*tasks)

        raw_content = self._extract_text(results[0])
        raw_author  = self._extract_text(results[1])
        raw_domain  = self._extract_text(results[2])

        # Phase 2: resolve domain(s) → specific category codes
        raw_category = await self._resolve_domain_category(raw_domain, user_input)

        return self._combine(raw_content, raw_author, raw_category)

    async def _resolve_domain_category(
        self, raw_domain: str, user_input: str
    ) -> str:
        """
        Parse the domain label(s) from step 1 and call the matching
        domain-specific prompt(s) for step 2.

        If the domain is NONE or unrecognised, returns "NONE".
        If multiple domains are returned (e.g. "CS, MATH"), both prompts
        are called in parallel and their results OR-combined.
        """
        cleaned_domain = clean_component(raw_domain).strip().upper()

        if cleaned_domain == "NONE" or not cleaned_domain:
            return "NONE"

        # Parse comma-separated domain labels
        domains = self._parse_domains(cleaned_domain)

        if not domains:
            return "NONE"

        # Call domain-specific prompt(s) — parallel if multiple
        if len(domains) == 1:
            prompt = self._enrich(CATEGORY_DOMAIN_PROMPTS[domains[0]], "category", user_input)
            result = await asyncio.to_thread(
                self.model.respond,
                prompt.format(user_input=user_input),
            )
            return self._extract_text(result)
        else:
            # Multiple domains: call each in parallel, combine results
            tasks = [
                asyncio.create_task(asyncio.to_thread(
                    self.model.respond,
                    self._enrich(CATEGORY_DOMAIN_PROMPTS[d], "category", user_input).format(user_input=user_input),
                ))
                for d in domains
            ]
            results = await asyncio.gather(*tasks)

            parts = []
            for r in results:
                text = clean_component(self._extract_text(r))
                if text.upper() != "NONE":
                    parts.append(text)

            return " OR ".join(parts) if parts else "NONE"

    @staticmethod
    def _parse_domains(text: str) -> List[str]:
        """
        Extract valid domain labels from a cleaned domain string.

        Handles formats like:
          "CS"
          "CS, MATH"
          "PHYSICS, MATH"
          "CS,PHYSICS"   (no space)

        Returns at most 2 valid domain labels.
        """
        # Split on comma, clean each token
        tokens = [t.strip().lower() for t in re.split(r"[,\s]+", text) if t.strip()]

        # Keep only recognised domain labels, deduplicate, limit to 2
        seen = set()
        valid = []
        for t in tokens:
            if t in _VALID_DOMAINS and t not in seen:
                seen.add(t)
                valid.append(t)
                if len(valid) >= 2:
                    break
        return valid

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _combine(
        self, raw_content: str, raw_author: str, raw_category: str
    ) -> Tuple[str, str, str, str]:
        """Clean components and combine non-NONE parts into a query string."""
        content  = clean_component(raw_content)
        author   = clean_component(raw_author)
        category = clean_component(raw_category)

        parts = [p for p in [content, author, category] if p.strip().upper() != "NONE"]

        if not parts:
            raise ValueError("INVALID QUERY: All query components returned NONE after cleaning")

        combined = "(" + ") AND (".join(parts) + ")" if len(parts) > 1 else parts[0]

        return combined, content, author, category

    @staticmethod
    def _extract_text(result) -> str:
        """Extract the content string from a LLM response dict or object."""
        if isinstance(result, dict):
            return result.get("content", "")
        return getattr(result, "content", str(result))

    @property
    def builder_type(self) -> str:
        return "arxiv"