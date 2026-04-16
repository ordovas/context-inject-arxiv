# ArXiv Paper Retrieval Agent

A LangGraph-based intelligent agent for retrieving arXiv papers using natural language queries. Built with hexagonal architecture for clean separation of concerns.

## Overview

This project implements an intelligent agent that:
1. Takes natural language queries from users
2. Uses an LLM to convert queries into proper arXiv API search strings (three parallel calls: content, author, category)
3. Optionally injects external domain context (glossaries, team rosters, project timelines) into LLM prompts
4. Cleans raw LLM outputs to remove contamination (prose prefixes, outer quotes, markdown fences, category syntax issues)
5. Validates each query component structurally before sending to the API
6. Retrieves papers from arXiv and returns results as a pandas DataFrame

## Installation

```bash
pip install langgraph pandas requests lmstudio anthropic ollama python-dotenv pyyaml
```

## Usage

### Python API (Primary)

```python
from src.application.usecases.arxiv import run_arxiv_agent

# Basic usage (two-step category mode, default)
result = await run_arxiv_agent(
    "papers about machine learning",
    llm_provider="ollama",
    model_id="mistral"
)
papers_df = result['papers_df']

# With context injection (glossary, teams, project phases)
result = await run_arxiv_agent(
    "Papers by the NTE team about LINER classification",
    llm_provider="ollama",
    model_id="gemma2:27b",
    context_yaml_path="src/domain/context/context.yaml"
)

# With Claude API
result = await run_arxiv_agent(
    "papers about machine learning",
    llm_provider="claude",
    model_id="claude-3-5-sonnet-20241022"
)

# With single-step category mode (original behaviour)
result = await run_arxiv_agent(
    "papers about quantum computing",
    llm_provider="ollama",
    model_id="qwen3:8b",
    category_mode="single_step"
)
```

### REST API

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Search papers
curl -X POST http://localhost:5000/api/v1/papers/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_prompt": "papers about machine learning",
    "llm_provider": "lmstudio",
    "model_id": "qwen/qwen3-32b"
  }'
```

## Agent Workflow

The agent uses a LangGraph workflow with these steps:

1. **Query Generation** (`query_generation_node`): Converts natural language to arXiv query
   - Three LLM calls run in parallel: content (`all:`/`ti:` etc.), author (`au:`), category (`cat:`)
   - When context injection is enabled, domain-specific knowledge is inserted into each prompt before the user request
   - Each raw output is cleaned by `output_cleaner.clean_component()` before combination
   - Cleaned components are stored in state (`query_content`, `query_author`, `query_category`)
   - Combined query sent to arXiv API

2. **Paper Retrieval** (`paper_retrieval_node`): Fetches papers from arXiv API
   - Calls arXiv API with generated query
   - Parses XML response
   - Returns pandas DataFrame

### Category Classification Modes

| Mode | Description |
|---|---|
| `single_step` | One LLM call with the full 150+ code list. Works well for large models (27B+). |
| `two_step` (default) | Two-stage: domain classification → domain-specific code selection. Reduces the code list from 150+ to 5–40, improving accuracy for small models. |

## Context Injection

The agent supports injecting external domain-specific knowledge into LLM prompts via a YAML configuration file. This lets the LLM resolve terms it wouldn't otherwise understand.

### Supported context types

| Type | Purpose | Example |
|---|---|---|
| **Glossary** | Acronyms, disambiguation | LINER ≠ AGN; RAG = Retrieval-Augmented Generation |
| **Teams** | Named groups → member lists | NTE team → 4 specific researchers |
| **Project phases** | Named periods → date ranges | "Phase 1 of NX Project" → `submittedDate:[20210101 TO 20220630]` |
| **Instruments** | Telescope/facility names → categories | eROSITA → `astro-ph.HE`, `astro-ph.CO` |
| **Datasets** | Named surveys → categories | SDSS → `astro-ph.GA`, `astro-ph.CO` |

### How it works

The `ContextResolver` scans the user query for matching terms, then `ContextQueryBuilder` injects prompt-type-specific context blocks. Each prompt type gets tailored information: author prompts include team member lists with OR-join instructions, content prompts include date arithmetic for project phases, and category prompts include instrument/dataset category hints.

```python
# Enable context injection by passing the YAML path
result = await run_arxiv_agent(
    "Papers about RAG methods before Phase 1 of the NX Project",
    llm_provider="ollama",
    model_id="gemma2:27b",
    context_yaml_path="src/domain/context/context.yaml"
)
```

When no context file is provided, or when no terms match the query, behaviour is identical to the base pipeline.

## Output Cleaning (Strategy D)

`src/domain/output_cleaner.py` normalises raw LLM outputs before they are combined or validated.

| Pattern | Example (raw) | After cleaning |
|---------|--------------|----------------|
| Outer wrapper quotes (Mistral) | `"cat:cs.LG OR cat:cs.AI"` | `cat:cs.LG OR cat:cs.AI` |
| Prose + blank line + query (Llama) | `Based on the request...\n\ncat:cs.LG` | `cat:cs.LG` |
| Explanation prefix | `The query is: all:"machine learning"` | `all:"machine learning"` |
| Markdown code fences | ` ```\ncat:cs.LG\n``` ` | `cat:cs.LG` |
| Lowercase boolean (category) | `cat:cs.LG or cat:cs.AI` | `cat:cs.LG OR cat:cs.AI` |
| Comma-separated categories | `cat:cs.LG, cat:cs.AI` | `cat:cs.LG OR cat:cs.AI` |
| Trailing "or none" | `cat:cs.LG or none` | `cat:cs.LG` |

The cleaner is conservative — valid arXiv syntax (quoted phrases like `all:"machine learning"`, date ranges, boolean logic) is never modified.

## Query Validation

`src/query_validator.py` checks each component structurally, independently of whether papers were retrieved.

```python
from src.query_validator import validate_query_components

vr = validate_query_components(cleaned_content, cleaned_author, cleaned_category, num_papers)
vr["all_components_valid"]   # True only if all three pass
vr["any_contaminated"]       # Prose detected in any component
vr["any_quoted_operators"]   # arXiv operators wrapped in outer quotes
vr["any_invalid_codes"]      # Hallucinated/typo category codes
vr["content"]["valid"]       # Per-component breakdown
vr["author"]["valid"]
vr["category"]["valid"]
```

Checks performed per component:

- **Content**: Must contain a valid field operator (`all:`, `ti:`, `abs:`, etc.); no prose; no quoted operators
- **Author**: Each token must match `au:Lastname` or `au:Compound-Surname`; no prose
- **Category**: Each token must match `cat:code` and exist in the whitelist (150+ valid arXiv codes); no prose; no outer quotes

## DataFrame Schema

The returned DataFrame contains:

| Column | Description |
|--------|-------------|
| `arxiv_id` | arXiv paper identifier |
| `title` | Paper title |
| `summary` | Abstract |
| `published` | Publication date |
| `updated` | Last update date |
| `authors` | List of author names |
| `categories` | List of subject categories |
| `primary_category` | Primary subject category |
| `doi` | Digital Object Identifier (if available) |
| `journal_ref` | Journal reference (if available) |
| `comment` | Additional comments (if available) |

## Supported LLM Providers

| Provider | Type | Notes |
|----------|------|-------|
| **LMStudio** | Local | Free, no API key; model must be loaded in LMStudio UI |
| **Ollama** | Local | Free; supports `auto_pull` and `auto_unload` |
| **Claude API** | Cloud | Paid; requires `ANTHROPIC_API_KEY` env var |

## License

MIT License