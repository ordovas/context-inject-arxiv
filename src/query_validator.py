"""
Structural validation of arXiv query components.

Replaces the old "did we get papers back?" heuristic with per-component
checks that catch contaminated outputs (explanation text, wrong quoting,
invalid category codes) independently of whether papers were retrieved.
"""
import re
from typing import TypedDict, Optional


_VALID_CATEGORIES = {
    "cs.AI","cs.AR","cs.CC","cs.CE","cs.CG","cs.CL","cs.CR","cs.CV",
    "cs.CY","cs.DB","cs.DC","cs.DL","cs.DM","cs.DS","cs.ET","cs.FL",
    "cs.GL","cs.GR","cs.GT","cs.HC","cs.IR","cs.IT","cs.LG","cs.LO",
    "cs.MA","cs.MM","cs.MS","cs.NA","cs.NE","cs.NI","cs.OH","cs.OS",
    "cs.PF","cs.PL","cs.RO","cs.SC","cs.SD","cs.SE","cs.SI","cs.SY",
    "astro-ph.CO","astro-ph.EP","astro-ph.GA","astro-ph.HE","astro-ph.IM","astro-ph.SR",
    "cond-mat.dis-nn","cond-mat.mes-hall","cond-mat.mtrl-sci","cond-mat.other",
    "cond-mat.quant-gas","cond-mat.soft","cond-mat.stat-mech","cond-mat.str-el","cond-mat.supr-con",
    "gr-qc","hep-ex","hep-lat","hep-ph","hep-th","math-ph","nucl-ex","nucl-th","quant-ph",
    "physics.acc-ph","physics.ao-ph","physics.app-ph","physics.atm-clus","physics.atom-ph",
    "physics.bio-ph","physics.chem-ph","physics.class-ph","physics.comp-ph","physics.data-an",
    "physics.ed-ph","physics.flu-dyn","physics.gen-ph","physics.geo-ph","physics.hist-ph",
    "physics.ins-det","physics.med-ph","physics.optics","physics.plasm-ph","physics.pop-ph",
    "physics.soc-ph","physics.space-ph",
    "math.AC","math.AG","math.AP","math.AT","math.CA","math.CO","math.CT","math.CV",
    "math.DG","math.DS","math.FA","math.GM","math.GN","math.GR","math.GT","math.HO",
    "math.IT","math.KT","math.LO","math.MG","math.MP","math.NA","math.NT","math.OA",
    "math.OC","math.PR","math.QA","math.RA","math.RT","math.SG","math.SP","math.ST",
    "econ.EM","econ.GN","econ.TH",
    "eess.AS","eess.IV","eess.SP","eess.SY",
    "stat.AP","stat.CO","stat.ME","stat.ML","stat.OT","stat.TH",
    "q-bio.BM","q-bio.CB","q-bio.GN","q-bio.MN","q-bio.NC","q-bio.OT",
    "q-bio.PE","q-bio.QM","q-bio.SC","q-bio.TO",
    "q-fin.CP","q-fin.EC","q-fin.GN","q-fin.MF","q-fin.PM","q-fin.PR",
    "q-fin.RM","q-fin.ST","q-fin.TR",
    "nlin.AO","nlin.CD","nlin.CG","nlin.PS","nlin.SI",
}
_VALID_TOP_LEVEL = {
    "cs","econ","eess","math","astro-ph","cond-mat","physics",
    "quant-ph","gr-qc","hep-ex","hep-lat","hep-ph","hep-th",
    "math-ph","nucl-ex","nucl-th","nlin","q-bio","q-fin","stat",
}

_PROSE_PATTERN = re.compile(
    r"\b(the|is|are|was|were|it|this|that|would|should|because|since|"
    r"however|therefore|based|on|a|an|in|of|for|to|and|or|not|but|"
    r"i|will|can|please|note|query|search|return|result|user|"
    r"request|generate|provide|here|below|above|following)\b",
    re.IGNORECASE,
)
# Detects Mistral pattern: operator sits INSIDE outer quotes
# "cat:cs.LG OR cat:cs.AI" ← bad   |   all:"machine learning" ← fine
_OUTER_QUOTE_PATTERN = re.compile(
    r'(?:^|\s)"[^"]*(?:cat:|au:|all:|ti:)[^"]*"'
)
_CONTENT_OPS = re.compile(r"\b(all|ti|abs|co|jr|rn|id):", re.IGNORECASE)
_CAT_FORMAT  = re.compile(r"^cat:[a-z][\w.-]+$", re.IGNORECASE)
_AU_FORMAT   = re.compile(r"^au:[A-Za-z][\w-]*$")


def _is_prose_contaminated(text: str) -> bool:
    if "\n" in text:
        return True
    return len(_PROSE_PATTERN.findall(text)) > 6

def _has_outer_quotes(text: str) -> bool:
    return bool(_OUTER_QUOTE_PATTERN.search(text))

def _is_valid_category_code(code: str) -> bool:
    return code in _VALID_CATEGORIES or code in _VALID_TOP_LEVEL


class ComponentResult(TypedDict):
    raw: str
    is_none: bool
    structural_ok: bool
    contaminated: bool
    quoted_operators: bool
    invalid_codes: list
    valid: bool


def validate_content_component(raw: str) -> ComponentResult:
    text = raw.strip()
    is_none      = text.upper() == "NONE"
    contaminated = not is_none and _is_prose_contaminated(text)
    quoted_ops   = not is_none and _has_outer_quotes(text)
    struct_ok    = is_none or bool(_CONTENT_OPS.search(text))
    valid        = struct_ok and not contaminated and not quoted_ops
    return ComponentResult(raw=raw, is_none=is_none, structural_ok=struct_ok,
                           contaminated=contaminated, quoted_operators=quoted_ops,
                           invalid_codes=[], valid=valid)


def validate_author_component(raw: str) -> ComponentResult:
    """
    Known limitation: au:topic_word (e.g. au:machine_learning) passes
    structural checks — detecting non-person strings requires semantics.
    """
    text = raw.strip()
    is_none      = text.upper() == "NONE"
    contaminated = not is_none and _is_prose_contaminated(text)
    quoted_ops   = not is_none and _has_outer_quotes(text)
    if is_none:
        struct_ok = True
    else:
        tokens    = re.split(r"\s+(?:AND|OR|ANDNOT)\s+", text, flags=re.IGNORECASE)
        struct_ok = bool(tokens) and all(_AU_FORMAT.match(t.strip()) for t in tokens if t.strip())
    valid = struct_ok and not contaminated and not quoted_ops
    return ComponentResult(raw=raw, is_none=is_none, structural_ok=struct_ok,
                           contaminated=contaminated, quoted_operators=quoted_ops,
                           invalid_codes=[], valid=valid)


def validate_category_component(raw: str) -> ComponentResult:
    text = raw.strip()
    is_none      = text.upper() == "NONE"
    contaminated = not is_none and _is_prose_contaminated(text)
    quoted_ops   = not is_none and _has_outer_quotes(text)
    invalid_codes: list = []
    if is_none:
        struct_ok = True
    else:
        inner  = text.strip('"').strip("'")
        tokens = [t.strip() for t in re.split(r"\s+(?:OR|AND|ANDNOT)\s+", inner, flags=re.IGNORECASE) if t.strip()]
        format_ok = bool(tokens) and all(_CAT_FORMAT.match(t) for t in tokens)
        if format_ok:
            invalid_codes = [t for t in tokens if not _is_valid_category_code(t.split("cat:")[-1])]
        struct_ok = format_ok and not invalid_codes
    valid = struct_ok and not contaminated and not quoted_ops
    return ComponentResult(raw=raw, is_none=is_none, structural_ok=struct_ok,
                           contaminated=contaminated, quoted_operators=quoted_ops,
                           invalid_codes=invalid_codes, valid=valid)


class ValidationResult(TypedDict):
    content:  ComponentResult
    author:   ComponentResult
    category: ComponentResult
    all_components_valid: bool
    any_contaminated:     bool
    any_quoted_operators: bool
    any_invalid_codes:    bool
    retrieval_success:    bool


def validate_query_components(
    raw_content:  Optional[str],
    raw_author:   Optional[str],
    raw_category: Optional[str],
    num_papers:   int,
) -> ValidationResult:
    c = validate_content_component(raw_content  or "NONE")
    a = validate_author_component(raw_author    or "NONE")
    k = validate_category_component(raw_category or "NONE")
    return ValidationResult(
        content=c, author=a, category=k,
        all_components_valid   = c["valid"] and a["valid"] and k["valid"],
        any_contaminated       = c["contaminated"] or a["contaminated"] or k["contaminated"],
        any_quoted_operators   = c["quoted_operators"] or a["quoted_operators"] or k["quoted_operators"],
        any_invalid_codes      = bool(k["invalid_codes"]),
        retrieval_success      = num_papers > 0,
    )
