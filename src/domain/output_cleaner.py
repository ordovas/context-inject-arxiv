"""
LLM output cleanup for arXiv query components

Design principle: clean_component is CONSERVATIVE.  It only removes noise it
can positively identify.  If nothing matches, the string is returned unchanged
so the downstream validator can report the failure with full context.
"""

import re
from typing import Optional


_KNOWN_PREFIXES = [
    r"^the (?:arxiv )?query (?:is|would be|should be):?\s*",
    r"^arxiv query:?\s*",
    r"^query:?\s*",
    r"^answer:?\s*",
    r"^result:?\s*",
    r"^here(?:'s| is) the (?:arxiv )?query:?\s*",
    r"^output:?\s*",
    r"^response:?\s*",
    r"^author arxiv query:?\s*",
    r"^author query:?\s*",
    r"^category query:?\s*",
]
_PREFIX_RE = re.compile("|".join(_KNOWN_PREFIXES), re.IGNORECASE)

# Inline "ARXIV QUERY:" label that some models add mid-output
_LABEL_RE = re.compile(r"^ARXIV\s+QUERY:\s*", re.IGNORECASE)

# Any arXiv field operator — used to find the query line inside prose
_ARXIV_OP_RE = re.compile(
    r"\b(cat:|au:|all:|ti:|abs:|co:|jr:|rn:|id:|submittedDate:|lastUpdatedDate:)",
    re.IGNORECASE,
)

# Markdown code fences (triple backticks)
_FENCE_OPEN_RE  = re.compile(r"^```[a-z]*\s*\n?", re.MULTILINE)
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$", re.MULTILINE)

# Inline backtick wrapping (single backticks around entire output)
# Matches: `all:"LINER classification methods"` → all:"LINER classification methods"
_INLINE_BACKTICK_RE = re.compile(r"^`([^`]+)`$")

# au:NONE pattern — "au:NONE" instead of "NONE"
_AU_NONE_RE = re.compile(r"^au:NONE$", re.IGNORECASE)

# Category-specific normalization patterns (two-step regressions in smaller models)
_CAT_LOWERCASE_OR_RE = re.compile(r"(cat:\S+)\s+or\s+(\S+)")
_CAT_COMMA_RE = re.compile(r"(cat:\S+?)\s*,\s*(\S+)")
_CAT_TRAILING_NONE_RE = re.compile(r"\s+(?:or|OR)\s+(?:none|NONE)\b.*$")
_CAT_BARE_CODE_RE = re.compile(r"\bOR\s+(?!cat:)([a-z][\w-]*(?:\.[A-Za-z][\w-]*))\b")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_markdown(text: str) -> str:
    """Remove opening and closing markdown code fences AND inline backtick wrapping."""
    # Step 1: Triple-backtick fences
    text = _FENCE_OPEN_RE.sub("", text)
    text = _FENCE_CLOSE_RE.sub("", text)
    text = text.strip()

    # Step 2: Inline single-backtick wrapping
    # Only strip if the entire string is wrapped in single backticks
    m = _INLINE_BACKTICK_RE.match(text)
    if m:
        text = m.group(1).strip()

    return text


def _strip_wrapper_quotes(text: str) -> str:
    """
    Strip outer quote characters only when the ENTIRE string is a quoted wrapper.

    Correctly handles the Mistral pattern:
        "cat:cs.LG OR cat:cs.AI"   →   cat:cs.LG OR cat:cs.AI

    Leaves valid quoted phrases untouched:
        all:"machine learning"     →   all:"machine learning"   (quotes are internal)
    """
    for quote_char in ('"', "'"):
        if (
            len(text) >= 2
            and text.startswith(quote_char)
            and text.endswith(quote_char)
        ):
            inner = text[1:-1]
            # Only strip if the quote character does not appear inside —
            # that would indicate the quotes are structural, not a wrapper.
            if quote_char not in inner:
                return inner.strip()
    return text


def _extract_operator_line(text: str) -> str:
    """
    Scan a multi-line string and return the first line that contains an
    arXiv operator or is exactly 'NONE'.

    Handles the Llama pattern:
        Based on the user request, I would generate...

        cat:cs.LG OR cat:stat.ML

        This is because...

    Returns 'NONE' if no operator line is found.
    """
    for line in text.splitlines():
        line = line.strip()
        # Strip inline label before checking
        line = _LABEL_RE.sub("", line).strip()

        if not line:
            continue
        if line.upper() == "NONE":
            return "NONE"
        if _ARXIV_OP_RE.search(line):
            return line

    return "NONE"


def _normalize_category_syntax(text: str) -> str:
    """
    Fix common category-specific syntax issues that small models produce:
      - Trailing parenthetical prose: cat:cs.LG (computer science) → cat:cs.LG
      - Trailing "or none":          cat:cs.LG OR none → cat:cs.LG
      - Comma-separated codes:       cat:cs.LG, cat:cs.AI → cat:cs.LG OR cat:cs.AI
      - Lowercase or:                cat:cs.LG or cat:cs.AI → cat:cs.LG OR cat:cs.AI
      - Bare codes after OR:         cat:cs.LG OR cs.AI → cat:cs.LG OR cat:cs.AI
    """
    if "cat:" not in text.lower():
        return text

    result = text

    # Strip trailing parenthetical prose
    result = re.sub(r"\s*\([^)]*\)\s*$", "", result).strip()

    # Remove trailing "or none"
    result = _CAT_TRAILING_NONE_RE.sub("", result).strip()

    # Normalize commas → OR
    prev = None
    while prev != result:
        prev = result
        result = _CAT_COMMA_RE.sub(r"\1 OR \2", result)

    # Normalize lowercase "or" → OR
    prev = None
    while prev != result:
        prev = result
        result = _CAT_LOWERCASE_OR_RE.sub(r"\1 OR \2", result)

    # Add missing cat: prefix after OR
    result = _CAT_BARE_CODE_RE.sub(r"OR cat:\1", result)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_component(raw: Optional[str]) -> str:
    """
    Normalize a single LLM component output to a bare arXiv query string.

    Applies cleanup steps in order:
      1. Strip markdown code fences AND inline backtick wrapping
      2. Strip outer wrapper quotes  
      3a. Single-line: strip known explanation prefixes and inline labels
      3b. Multi-line:  extract the first line that contains an arXiv operator
      4. Normalize category syntax (lowercase or, commas, trailing none, bare codes)
      5. Strip au:NONE → NONE  

    Returns 'NONE' if the input is None, empty, or nothing salvageable is found.

    Args:
        raw: Raw string returned by the LLM for one query component.

    Returns:
        Cleaned query string suitable for passing to validate_query_components().
    """
    if not raw:
        return "NONE"

    text = raw.strip()

    # Step 1 — markdown fences + inline backticks
    text = _strip_markdown(text)

    # Step 2 — outer wrapper quotes
    text = _strip_wrapper_quotes(text)

    # Step 3 — single-line vs multi-line handling
    if "\n" in text:
        text = _extract_operator_line(text)
    else:
        text = _PREFIX_RE.sub("", text).strip()
        text = _LABEL_RE.sub("", text).strip()

    # Step 4 — category-specific normalization
    text = _normalize_category_syntax(text)

    # Step 5 (FIX 3) — au:NONE → NONE
    if _AU_NONE_RE.match(text.strip()):
        text = "NONE"

    return text.strip() if text.strip() else "NONE"


def clean_components(
    raw_content:  Optional[str],
    raw_author:   Optional[str],
    raw_category: Optional[str],
) -> tuple:
    """
    Clean all three query components in one call.

    Returns:
        (cleaned_content, cleaned_author, cleaned_category)
    """
    return (
        clean_component(raw_content),
        clean_component(raw_author),
        clean_component(raw_category),
    )
