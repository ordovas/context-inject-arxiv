"""
Context resolver v2 — strict per-prompt-type injection.

CHANGES FROM v1:
================

The original format_for_prompt() injected nearly everything into every prompt
type. This caused cross-component leakage: models saw date ranges in the
category prompt and stuffed submittedDate into their cat: output; glossary
expansions in the author prompt caused hallucinated au: names; etc.

v2 applies the principle of MINIMAL INJECTION — each prompt type receives
ONLY the context it needs to do its specific job:

  CONTENT prompt receives:
    - Glossary: full expansion + description (for search term construction)
    - Phases:   full date arithmetic (submittedDate format, before/after)
    - Instruments: name expansion only (one line — no description, no categories)
    - Datasets: name expansion only (one line — no description, no categories)
    - Teams: brief note (do NOT include authors), NO member names

  AUTHOR prompt receives:
    - Teams: full member list with OR-join instructions
    - Nothing else. No glossary, no dates, no instruments, no datasets.

  CATEGORY prompt receives:
    - Glossary: category hints (suggestive language)
    - Instruments: REQUIRED category directives ("MUST include at least one")
    - Datasets: REQUIRED category directives ("MUST include at least one")
    - No dates. No team info. No full descriptions.

This eliminates the three main failure modes:
  1. submittedDate leaking into category output (8/12 models affected)
  2. Glossary causing hallucinated authors (Phi-3 Mini)
  3. Context noise overwhelming tiny models
"""

import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


# ---------------------------------------------------------------------------
# Data classes (unchanged from v1)
# ---------------------------------------------------------------------------

@dataclass
class GlossaryMatch:
    term: str
    expansion: str
    description: str
    categories: List[str] = field(default_factory=list)


@dataclass
class TeamMatch:
    code: str
    full_name: str
    description: str
    members: List[str] = field(default_factory=list)


@dataclass
class PhaseMatch:
    code: str
    project: str
    phase_label: str
    description: str
    start_date: str
    end_date: str


@dataclass
class InstrumentMatch:
    code: str
    full_name: str
    description: str
    categories: List[str] = field(default_factory=list)


@dataclass
class DatasetMatch:
    code: str
    full_name: str
    description: str
    categories: List[str] = field(default_factory=list)


@dataclass
class ResolvedContext:
    """All context entries matched for a given user query."""
    glossary: List[GlossaryMatch] = field(default_factory=list)
    teams: List[TeamMatch] = field(default_factory=list)
    phases: List[PhaseMatch] = field(default_factory=list)
    instruments: List[InstrumentMatch] = field(default_factory=list)
    datasets: List[DatasetMatch] = field(default_factory=list)

    @property
    def has_matches(self) -> bool:
        return bool(
            self.glossary or self.teams or self.phases
            or self.instruments or self.datasets
        )

    def format_for_prompt(self, prompt_type: str) -> str:
        """
        Format matched context as a text block for injection into a prompt.

        v2: Each prompt type receives ONLY the context it needs.

        Args:
            prompt_type: One of "content", "author", "category".

        Returns:
            Formatted context block, or empty string if nothing matched.
        """
        if not self.has_matches:
            return ""

        if prompt_type == "content":
            return self._format_for_content()
        elif prompt_type == "author":
            return self._format_for_author()
        elif prompt_type == "category":
            return self._format_for_category()
        else:
            return ""

    # ------------------------------------------------------------------
    # CONTENT prompt: glossary + phases + instrument/dataset names
    # ------------------------------------------------------------------

    def _format_for_content(self) -> str:
        sections = []

        # Glossary: full expansion + description (helps build search terms)
        if self.glossary:
            lines = [
                "## Domain-specific context",
                "The following terms appear in the user request. "
                "Use these definitions when building the content query:",
            ]
            for g in self.glossary:
                lines.append(
                    f"- **{g.term}** = {g.expansion}. "
                    f"{g.description.strip()}"
                )
            sections.append("\n".join(lines))

        # Teams: minimal note — do NOT leak member names into content
        if self.teams:
            lines = ["## Team references"]
            for t in self.teams:
                lines.append(
                    f"- **{t.code}** is a research team. "
                    f"Author handling is done separately — do NOT include "
                    f"author names or au: fields in your output."
                )
            sections.append("\n".join(lines))

        # Project phases: full date arithmetic (this is WHERE dates belong)
        if self.phases:
            lines = ["## Project phase dates"]
            for p in self.phases:
                start = p.start_date.replace("-", "")
                end = p.end_date.replace("-", "")
                lines.append(
                    f"- **{p.project} {p.phase_label}**: "
                    f"{p.start_date} to {p.end_date}"
                )
                lines.append(
                    f'  If the user refers to "{p.phase_label}" of '
                    f'"{p.project}", use submittedDate:[{start} TO {end}].'
                )
                lines.append(
                    f'  "Before {p.phase_label}" means '
                    f"submittedDate:[17000101 TO {start}]."
                )
                lines.append(
                    f'  "After {p.phase_label}" means '
                    f"submittedDate:[{end} TO 24000101]."
                )
            sections.append("\n".join(lines))

        # Instruments: name expansion only (one line — no description, no categories)
        if self.instruments:
            lines = ["## Instruments mentioned"]
            for inst in self.instruments:
                lines.append(f"- **{inst.code}** = {inst.full_name}.")
            sections.append("\n".join(lines))

        # Datasets: name expansion only (one line — no description, no categories)
        if self.datasets:
            lines = ["## Datasets mentioned"]
            for ds in self.datasets:
                lines.append(f"- **{ds.code}** = {ds.full_name}.")
            sections.append("\n".join(lines))

        if not sections:
            return ""
        return "\n\n".join(sections) + "\n"

    # ------------------------------------------------------------------
    # AUTHOR prompt: ONLY team member lists
    # ------------------------------------------------------------------

    def _format_for_author(self) -> str:
        """
        Author prompt receives ONLY team member information.
        No glossary, no dates, no instruments, no datasets.
        """
        if not self.teams:
            return ""

        lines = [
            "## Team definitions",
            "The user's request references a named team. "
            "Generate author queries for the team members listed below.",
        ]
        for t in self.teams:
            members_str = ", ".join(t.members)
            lines.append(
                f"- **{t.code}** ({t.full_name}): "
                f"members are {members_str}."
            )
            lines.append(
                f"  Generate: au:Member1 OR au:Member2 OR ... "
                f"(use OR, not AND — papers may have any subset)."
            )

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # CATEGORY prompt: ONLY category hints (no dates, no descriptions)
    # ------------------------------------------------------------------

    def _format_for_category(self) -> str:
        """
        Category prompt receives ONLY category code hints.
        No dates. No team info. No full descriptions. Just codes.
        """
        hints = []

        # Glossary → just the category hint
        if self.glossary:
            for g in self.glossary:
                if g.categories:
                    cats = ", ".join(g.categories)
                    hints.append(
                        f"- {g.term} ({g.expansion}): "
                        f"typically in {cats}"
                    )

        if not hints:
            # No glossary hints — check if we have instrument/dataset directives
            pass

        instrument_directives = []
        dataset_directives = []

        # Instruments → directive language
        if self.instruments:
            for inst in self.instruments:
                if inst.categories:
                    cats = " OR ".join(f"cat:{c}" for c in inst.categories)
                    instrument_directives.append(
                        f"- **{inst.code}** ({inst.full_name}): USE {cats}"
                    )

        # Datasets → directive language
        if self.datasets:
            for ds in self.datasets:
                if ds.categories:
                    cats = " OR ".join(f"cat:{c}" for c in ds.categories)
                    dataset_directives.append(
                        f"- **{ds.code}** ({ds.full_name}): USE {cats}"
                    )

        if not hints and not instrument_directives and not dataset_directives:
            return ""

        lines = [
            "## Category hints for terms in this request",
            "Use these as guidance for selecting arXiv category codes.",
            "Return ONLY cat: codes. Do NOT include dates or author fields.",
        ]
        lines.extend(hints)

        if instrument_directives:
            lines.append(
                "\n## Instruments — REQUIRED category hints\n"
                "The user's query mentions specific instruments. "
                "You MUST include at least one of the listed categories "
                "for each instrument, because these instruments only "
                "produce data in those specific research areas."
            )
            lines.extend(instrument_directives)

        if dataset_directives:
            lines.append(
                "\n## Datasets — REQUIRED category hints\n"
                "The user's query mentions specific datasets. "
                "You MUST include at least one of the listed categories "
                "for each dataset, because these datasets only cover "
                "those specific research areas."
            )
            lines.extend(dataset_directives)

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# ContextResolver (unchanged from v1 — only ResolvedContext changed)
# ---------------------------------------------------------------------------

class ContextResolver:
    """
    Loads external context from YAML and resolves matching entries for
    a given user query.

    Usage:
        resolver = ContextResolver()
        resolved = resolver.resolve("Papers about LINER using SDSS data")
        block = resolved.format_for_prompt("content")
    """

    def __init__(self, yaml_path: Optional[str] = None):
        if yaml_path is None:
            yaml_path = str(Path(__file__).parent / "context.yaml")

        with open(yaml_path, "r", encoding="utf-8") as f:
            self._raw = yaml.safe_load(f)

        self._glossary = self._raw.get("glossary", {})
        self._teams = self._raw.get("teams", {})
        self._phases = self._raw.get("project_phases", {})
        self._instruments = self._raw.get("instruments", {})
        self._datasets = self._raw.get("datasets", {})

        self._patterns: List[tuple] = []
        self._build_patterns()

    def _build_patterns(self) -> None:
        def _add(term: str, section: str, key: str):
            escaped = re.escape(term)
            pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            self._patterns.append((pattern, section, key))

        for key, entry in self._glossary.items():
            _add(key, "glossary", key)
            for alias in entry.get("aliases", []):
                if alias:
                    _add(alias, "glossary", key)

        for key, entry in self._teams.items():
            _add(key, "teams", key)
            full_name = entry.get("full_name", "")
            if full_name:
                _add(full_name, "teams", key)

        for key, entry in self._phases.items():
            _add(key, "phases", key)
            phase_label = entry.get("phase_label", "")
            project = entry.get("project", "")
            if phase_label and project:
                escaped_phase = re.escape(phase_label)
                escaped_project = re.escape(project)
                combined_re = re.compile(
                    rf"\b(?:{escaped_project}\s+{escaped_phase}|"
                    rf"{escaped_phase}\s+(?:of\s+(?:the\s+)?)?{escaped_project})\b",
                    re.IGNORECASE,
                )
                self._patterns.append((combined_re, "phases", key))
                self._patterns.append(
                    (re.compile(rf"\b{escaped_phase}\b", re.IGNORECASE),
                     "phases_weak", key)
                )

        for key, entry in self._instruments.items():
            _add(key, "instruments", key)

        for key, entry in self._datasets.items():
            _add(key, "datasets", key)
            for alias in entry.get("aliases", []):
                if alias:
                    _add(alias, "datasets", key)

    def resolve(self, user_query: str) -> ResolvedContext:
        ctx = ResolvedContext()
        seen = set()

        project_names_in_query = set()
        for key, entry in self._phases.items():
            project = entry.get("project", "")
            if project and re.search(
                rf"\b{re.escape(project)}\b", user_query, re.IGNORECASE
            ):
                project_names_in_query.add(project)

        for pattern, section, key in self._patterns:
            effective_section = "phases" if section == "phases_weak" else section

            if (effective_section, key) in seen:
                continue

            if section == "phases_weak":
                phase_entry = self._phases[key]
                if phase_entry.get("project", "") not in project_names_in_query:
                    continue

            if pattern.search(user_query):
                seen.add((effective_section, key))
                self._add_match(ctx, effective_section, key)

        return ctx

    def _add_match(self, ctx: ResolvedContext, section: str, key: str) -> None:
        if section == "glossary":
            entry = self._glossary[key]
            ctx.glossary.append(GlossaryMatch(
                term=key,
                expansion=entry.get("expansion", ""),
                description=entry.get("description", ""),
                categories=entry.get("categories", []),
            ))
        elif section == "teams":
            entry = self._teams[key]
            ctx.teams.append(TeamMatch(
                code=key,
                full_name=entry.get("full_name", ""),
                description=entry.get("description", ""),
                members=entry.get("members", []),
            ))
        elif section == "phases":
            entry = self._phases[key]
            ctx.phases.append(PhaseMatch(
                code=key,
                project=entry.get("project", ""),
                phase_label=entry.get("phase_label", ""),
                description=entry.get("description", ""),
                start_date=entry.get("start_date", ""),
                end_date=entry.get("end_date", ""),
            ))
        elif section == "instruments":
            entry = self._instruments[key]
            ctx.instruments.append(InstrumentMatch(
                code=key,
                full_name=entry.get("full_name", ""),
                description=entry.get("description", ""),
                categories=entry.get("categories", []),
            ))
        elif section == "datasets":
            entry = self._datasets[key]
            ctx.datasets.append(DatasetMatch(
                code=key,
                full_name=entry.get("full_name", ""),
                description=entry.get("description", ""),
                categories=entry.get("categories", []),
            ))

    def get_all_terms(self) -> Dict[str, List[str]]:
        return {
            "glossary": list(self._glossary.keys()),
            "teams": list(self._teams.keys()),
            "project_phases": list(self._phases.keys()),
            "instruments": list(self._instruments.keys()),
            "datasets": list(self._datasets.keys()),
        }

    @property
    def num_entries(self) -> int:
        return (
            len(self._glossary)
            + len(self._teams)
            + len(self._phases)
            + len(self._instruments)
            + len(self._datasets)
        )