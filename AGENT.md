# ArXiv Paper Retrieval Agent - Project Context

## Project Overview

This is a **LangGraph-based intelligent agent** designed to retrieve arXiv research papers using natural language queries. The agent translates conversational user input into properly formatted arXiv API search queries, retrieves papers, and returns results as a pandas DataFrame.

### Key Problem Solved
Users don't need to understand arXiv's complex query syntax. They can ask for papers in plain English (e.g., "Papers about machine learning in biology from 2023"), and the agent automatically converts this to a valid arXiv API query. The agent also supports **external context injection** — domain glossaries, team rosters, project timelines, and instrument/dataset registries — so that queries containing terms the LLM wouldn't otherwise understand (acronyms, team names, project phases) are resolved correctly.

---

## Architecture Overview

The project follows a hexagonal architecture:

```
src/
├── ports/                      # PORTS LAYER - Abstract interfaces
│   ├── llm_provider.py         # LLM provider interface
│   ├── paper_source.py         # Paper source interface
│   ├── query_builder.py        # Query builder interface
│   └── __init__.py
├── adapters/                   # ADAPTERS LAYER - Concrete implementations
│   ├── inbound/
│   │   └── rest_api/           # REST API HTTP adapter
│   │       ├── app.py          # Flask application
│   │       ├── routes.py       # API endpoints
│   │       ├── models.py       # Request/response schemas
│   │       └── __init__.py
│   └── outbound/               # Outbound adapters (external systems)
│       ├── llm/                # LLM implementations
│       │   ├── lmstudio_adapter.py
│       │   ├── claude_adapter.py
│       │   ├── ollama_adapter.py
│       │   └── __init__.py
│       ├── arxiv/              # ArXiv implementation
│       │   ├── arxiv_adapter.py
│       │   ├── arxiv_client.py
│       │   └── __init__.py
│       └── __init__.py
├── domain/                     # DOMAIN LAYER - Business logic
│   ├── query_builder.py        # Parallel LLM calls + cleanup; single_step & two_step modes
│   ├── context_query_builder.py # Extends QueryBuilder with external context injection
│   ├── output_cleaner.py       # LLM output normalisation (Strategy D + category syntax fixes)
│   ├── context/                # External domain knowledge
│   │   ├── context.yaml        # Glossary, teams, project phases, instruments, datasets
│   │   ├── context_resolver.py # Term matching + prompt-type-specific formatting
│   │   └── __init__.py
│   ├── prompts/
│   │   ├── prompts.yaml        # Prompt templates (YAML configuration)
│   │   └── prompts.py          # YAML loader — exports prompt variables
│   ├── graphs/
│   │   └── arxiv/
│   │       └── nodes.py        # LangGraph workflow nodes
│   └── repositories/
│       └── states/
│           └── arxiv.py        # Agent state schema
├── application/                # APPLICATION LAYER - Orchestration
│   ├── agents/
│   │   └── arxiv_agent.py      # Main agent orchestrator (supports optional context)
│   ├── factories.py            # Dependency injection factories
│   └── usecases/
│       └── arxiv.py            # Use case entry point (supports optional context)
├── query_validator.py          # Per-component structural validation
├── config.py                   # Configuration
└── __init__.py
```

---

## Hexagonal Architecture (Ports & Adapters)

### 1. **Domain Layer** (Core Business Logic)
The innermost layer containing pure business logic.
- `domain/query_builder.py` — Parallel LLM query generation with output cleanup; supports `single_step` (full 150+ code list) and `two_step` (domain → codes) category classification
- `domain/context_query_builder.py` — Extends `QueryBuilder` with optional external context injection; injects domain-specific knowledge into each LLM prompt before the user request
- `domain/output_cleaner.py` — Removes contamination patterns from raw LLM outputs (prose prefixes, outer quotes, markdown fences, category syntax normalisation)
- `domain/context/context.yaml` — External domain knowledge: glossary (acronyms, disambiguation), teams (named groups → member lists), project phases (named periods → date ranges), instruments, datasets/surveys
- `domain/context/context_resolver.py` — Scans user queries for matching context entries; produces prompt-type-specific formatted blocks (author prompts get team OR-join instructions, content prompts get date arithmetic, category prompts get instrument/dataset hints)
- `domain/graphs/arxiv/nodes.py` — LangGraph workflow nodes; stores cleaned per-component outputs in state
- `domain/prompts/prompts.yaml` — YAML configuration with all prompt templates (content, author, single-step category, two-step domain classifier, 8 domain-specific category prompts)
- `domain/prompts/prompts.py` — YAML loader that exports prompts as module-level variables
- `domain/repositories/states/arxiv.py` — State schema including per-component fields

### 2. **Ports Layer** (Abstractions/Interfaces)
- `ports/llm_provider.py` — Abstract interface for LLM providers
- `ports/paper_source.py` — Abstract interface for paper retrieval systems
- `ports/query_builder.py` — Abstract interface for query builders

### 3. **Adapters Layer** (Concrete Implementations)

**Outbound Adapters:**
- `adapters/outbound/llm/lmstudio_adapter.py` — Connects to LMStudio
- `adapters/outbound/llm/claude_adapter.py` — Connects to Claude API
- `adapters/outbound/llm/ollama_adapter.py` — Connects to Ollama server
- `adapters/outbound/arxiv/arxiv_adapter.py` — Connects to ArXiv API

**Inbound Adapters:**
- `adapters/inbound/rest_api/` — REST API HTTP adapter

### 4. **Application Layer** (Orchestration)
- `application/agents/arxiv_agent.py` — Orchestrates workflow via LangGraph; accepts optional `context_yaml_path` to enable context injection
- `application/factories.py` — Creates adapter instances
- `application/usecases/arxiv.py` — High-level use case orchestration; passes `context_yaml_path` through to the agent

### 5. **Cross-cutting (src/)**
- `src/query_validator.py` — Per-component structural validation; used by the benchmark notebooks to assess query quality independently of retrieval success

### Dependency Flow (Always Inward)

```
External Systems (LMStudio, Claude, ArXiv API)
    ↓ (implements)
Adapters Layer
    ↓ (implement)
Ports Layer (abstractions)
    ↓ (depend on)
Domain Layer (business logic)
```

---

## Query Generation Pipeline

Each user query triggers three parallel LLM calls, producing independent components that are then cleaned, validated, and combined. When context injection is enabled, matched domain knowledge is inserted into each prompt before the user request.

```
User Input
    │
    ├── [ContextResolver] ─── resolve(user_input)
    │        │
    │        ▼
    │   ResolvedContext (glossary, teams, phases, instruments, datasets)
    │        │
    │        ├── format_for_prompt("content")  → content context block
    │        ├── format_for_prompt("author")   → author context block
    │        └── format_for_prompt("category") → category context block
    │
    ├─── [LLM call 1] PROMPT_QUERY_ARXIV + content context   → raw_content
    ├─── [LLM call 2] AUTHOR_QUERY_ARXIV + author context    → raw_author
    └─── [LLM call 3] Category (single_step or two_step)     → raw_category
                                    │
                        clean_component() × 3     ← output_cleaner.py
                                    │
                    (content, author, category)   ← stored in ArxivState
                                    │
                        Combined arXiv query
                                    │
                            arXiv HTTP API
                                    │
                           Pandas DataFrame
```

### Two-Step Category Classification

Instead of one prompt with 150+ codes (which causes attention dilution in small models), the two-step mode splits classification into:

1. **Domain classification** — Classify the user request into a broad domain (CS, PHYSICS, MATH, STATS, BIOLOGY, ECON, ENGINEERING, NONLINEAR)
2. **Domain-specific code selection** — Map to specific arXiv codes using only that domain's short list (5–40 codes instead of 150+)

This runs in parallel with content and author generation (phase 1), then the domain-specific call runs sequentially (phase 2).

### Context Injection

When `context_yaml_path` is provided, the `ContextQueryBuilder` resolves matching entries for the user query and injects prompt-type-specific context blocks:

| Prompt type | What gets injected |
|---|---|
| **Content** | Glossary expansions, project phase date ranges with arXiv `submittedDate` format and before/after semantics |
| **Author** | Glossary terms, team member lists with OR-join instructions |
| **Category** | Glossary terms, instrument/dataset category hints |

The context block is inserted right before the `USER REQUEST:` line in each prompt template. When no context entries match a query, the prompts are sent unmodified.

### Output Cleaner (`src/domain/output_cleaner.py`)

Normalises raw LLM outputs before they propagate. Applied inside `QueryBuilder._combine()` immediately after each LLM response is extracted.

Cleanup steps (applied in order):
1. Strip markdown code fences
2. Strip outer wrapper quotes — handles Mistral's `"cat:cs.LG OR cat:cs.AI"` pattern
3. Single-line: strip known explanation prefixes (`The query is:`, `ARXIV QUERY:`, etc.) / Multi-line: extract the first line containing an arXiv operator
4. Category syntax normalisation: lowercase `or` → `OR`, commas → `OR`, trailing `or none` removal, bare codes → `cat:` prefix

The cleaner is **conservative**: if no pattern matches, the input is returned unchanged so the downstream validator can report the failure with full context. It is also **idempotent**: `clean_component(clean_component(x)) == clean_component(x)`.

### Query Validator (`src/query_validator.py`)

Per-component structural checks that replace the old "did we get papers back?" heuristic.

```python
vr = validate_query_components(cleaned_content, cleaned_author, cleaned_category, num_papers)
```

**Content component checks:**
- Must contain a valid field operator (`all:`, `ti:`, `abs:`, `co:`, `jr:`, `rn:`, `id:`)
- Must not contain prose (>6 common English words or any newline)
- Must not have arXiv operators wrapped in outer quotes

**Author component checks:**
- Each token must match `au:Lastname` or `au:Compound-Surname` (hyphens allowed, no spaces)
- Must not contain prose or outer-quoted operators

**Category component checks:**
- Each token must match `cat:code` format
- The code part must exist in the whitelist of 150+ valid arXiv codes (synced from `prompts.yaml`)
- Must not contain prose or outer-quoted operators

**Known limitation:** `au:topic_word` (e.g. `au:machine_learning`) passes structural checks because underscore is valid in arXiv author syntax (`au:garcia_i`). Distinguishing topic words from surnames requires semantic knowledge.

---

## Agent State (`src/domain/repositories/states/arxiv.py`)

```python
class ArxivState(TypedDict):
    user_input:    str                          # Original user query
    arxiv_query:   str                          # Combined query sent to arXiv API
    query_content:  Optional[str]               # Cleaned content component (all:/ti: etc.)
    query_author:   Optional[str]               # Cleaned author component (au:)
    query_category: Optional[str]               # Cleaned category component (cat:)
    papers_df:     Optional[pd.DataFrame]       # Retrieved papers
    error:         Optional[str]                # Error message if any
    messages:      Annotated[List[str], add]    # Message history (accumulates)
```

The three `query_*` fields expose per-component outputs to the benchmark notebooks without requiring any changes to the agent or adapter layers.

---

## Workflow Nodes (`src/domain/graphs/arxiv/nodes.py`)

### `query_generation_node(state, query_builder)`
- Calls `query_builder.build_query_with_components_async(user_input)`
- Receives 4-tuple: `(combined, cleaned_content, cleaned_author, cleaned_category)`
- Stores all four in state
- Sets `error` if query builder raises

### `paper_retrieval_node(state, paper_source)`
- Reads `arxiv_query` from state
- Calls `paper_source.search(arxiv_query)`
- Stores resulting DataFrame in `papers_df`
- Sets `error` on failure

---

## LLM Prompts (`src/domain/prompts/`)

**Architecture:**
- `prompts.yaml` — YAML configuration file with all prompt templates
- `prompts.py` — Python loader that reads YAML and exports prompts as module-level variables

**Prompts:**
- `PROMPT_QUERY_ARXIV` — Generates content-based search query; instructs model to return `NONE` if only author queries are given
- `AUTHOR_QUERY_ARXIV` — Extracts and formats author names; covers compound surnames, accent removal, hyphens vs underscores
- `PROMPT_ARXIV_CATEGORY` — (single-step) Maps user input to arXiv category codes; includes the full list of 150+ valid codes
- `CATEGORY_DOMAIN_PROMPT` — (two-step, step 1) Classifies user request into a broad academic domain
- `CATEGORY_CS_PROMPT`, `CATEGORY_PHYSICS_PROMPT`, etc. — (two-step, step 2) Domain-specific prompts with only that domain's arXiv codes

---

## External Context (`src/domain/context/`)

### Context YAML (`context.yaml`)

Stores domain-specific knowledge in five sections:

| Section | Purpose | Example |
|---|---|---|
| `glossary` | Acronyms, abbreviations, disambiguation | LINER ≠ AGN; RAG = Retrieval-Augmented Generation |
| `teams` | Named researcher groups → member lists | NTE team = 4 specific members |
| `project_phases` | Named time periods → date ranges | NX Project Phase 1 = 2021-01-01 to 2022-06-30 |
| `instruments` | Telescopes, detectors → research areas | eROSITA → astro-ph.HE, astro-ph.CO |
| `datasets` | Named surveys → research areas | SDSS → astro-ph.GA, astro-ph.CO |

### Context Resolver (`context_resolver.py`)

Scans user queries for matching terms using word-boundary regex. Each entry and its aliases are compiled into patterns at init time. Project phase matching uses a two-tier system: strong matches (explicit "Phase 1 of NX Project") and weak matches ("Phase 1" only triggers if the project name also appears in the query).

The resolver produces a `ResolvedContext` dataclass with `format_for_prompt(prompt_type)` that generates different text blocks depending on the target prompt (author prompts get team OR-join instructions, content prompts get date arithmetic, etc.).

### Context Query Builder (`context_query_builder.py`)

Extends `QueryBuilder` to inject context blocks into each LLM prompt before the `USER REQUEST:` line. Falls back to base `QueryBuilder` behaviour when no context resolver is provided or when no terms match the current query.

---

## REST API Adapter

Simple HTTP wrapper around `run_arxiv_agent()`.

### Starting the Server
```bash
python -m src.adapters.inbound.rest_api.app
```

### Endpoints

#### Health Check
```
GET /api/v1/health
→ {"status": "healthy", "service": "ArXiv Paper Retrieval API"}
```

#### Search Papers
```
POST /api/v1/papers/search
{
    "user_prompt": "papers about machine learning in biology",
    "llm_provider": "lmstudio",
    "model_id": "qwen/qwen3-32b"
}
```

---

## Benchmarking Notebooks

### LLM Benchmark (`notebooks/llm_benchmark_test.ipynb`)

Tests and compares LLM models on arXiv query generation using the full validation pipeline across both `single_step` and `two_step` category modes.

- Configurable `N_RUNS` per model×query×mode for variance analysis
- `Useful` metric: Valid AND papers > 0
- Accuracy excluding exceptions reported alongside raw accuracy
- Latency recorded per test
- Queries tagged by category for grouped analysis
- ETA displayed during long runs

### Context Injection Benchmark (`notebooks/context_benchmark_test.ipynb`)

Tests the impact of external context injection on query quality. Runs each test query in two modes (baseline vs. context-enriched) across all selected models.

- 10 context-dependent queries spanning all 5 context types
- 2 control queries (no context terms) to verify no regression
- Analysis sections: per-model comparison, model-size impact, context-type breakdown, side-by-side per-query view

---

## Supported LLM Providers

### LMStudio (Local)
```python
from src.application.factories import LLMProviderFactory
adapter = LLMProviderFactory.create("lmstudio", "qwen/qwen3-32b")
```
Requires LMStudio desktop app running with a model loaded.

### Claude API (Cloud)
```python
adapter = LLMProviderFactory.create("claude", "claude-sonnet-4-6")
```
Requires `ANTHROPIC_API_KEY` environment variable.

### Ollama (Local)
```python
adapter = LLMProviderFactory.create("ollama", "mistral", auto_pull=True, auto_unload=True)
```
Requires `ollama serve` running. `auto_pull=True` downloads missing models automatically.

---

## Data Flow Diagram

```
┌─────────────────────────────┐
│   User Natural Language      │
│   "Papers on ML + biology"   │
└──────────────┬──────────────┘
               │
        ┌──────▼──────┐
        │ Context     │  (optional)
        │ Resolver    │  Scans for glossary, teams, phases,
        │             │  instruments, datasets
        └──────┬──────┘
               │
               ▼
    ┌──────────────────────┐
    │ Query Builder        │
    │ (3 parallel LLM calls│
    │  + context injection)│
    └──────┬───────────────┘
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
raw_content raw_author raw_category
           │
  output_cleaner.clean_component() × 3
           │
  ┌────────┼────────┐
  ▼        ▼        ▼
content  author  category  ──→ stored in ArxivState
           │
  Combined arXiv Query
           │
    ┌──────▼───────┐
    │ ArXiv API    │
    │ XML Response │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Parse XML    │
    │ → DataFrame  │
    └──────────────┘
```

---

## Error Handling Strategy

1. **Query Generation Errors** — If all three LLM components return NONE after cleaning, `QueryBuilder` raises `ValueError`; the conditional edge prevents paper retrieval
2. **Paper Retrieval Errors** — HTTP errors and timeouts caught; returns empty DataFrame with error context
3. **XML Parsing Errors** — `ElementTree` exceptions caught; returns empty DataFrame
4. **State Validation** — Each node validates required state fields before proceeding
5. **Context Resolution Errors** — If context YAML fails to load, `ContextResolver` raises at init; if no terms match, prompts are sent unmodified (graceful degradation)

---

## Configuration (`src/config.py`)

```python
LLM_MODEL = "qwen/qwen3-32b"   # Default model for LMStudio
MAX_ARXIV_RESULTS = 200         # Default max papers per query
```

---

## Adding New Features

### Adding a New LLM Provider

1. Create `adapters/outbound/llm/your_adapter.py` implementing the `LLMProvider` port
2. Register it in `application/factories.py` under `LLMProviderFactory.create()`
3. Domain layer remains unchanged ✓

### Adding Workflow Nodes

1. Define the new node function in `domain/graphs/arxiv/nodes.py`
2. Add it to the graph in `application/agents/arxiv_agent.py`
3. Extend `ArxivState` if the node needs new fields

### Adding Context Entries

1. Edit `src/domain/context/context.yaml` — add entries to the appropriate section (`glossary`, `teams`, `project_phases`, `instruments`, `datasets`)
2. The `ContextResolver` picks them up automatically at next initialisation
3. For entirely new section types, extend `ContextResolver._build_patterns()` and `ResolvedContext.format_for_prompt()`

---
