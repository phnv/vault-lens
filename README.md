# vaultlens

Local-first search workbench for markdown vaults (Obsidian-style folders of `.md` files).
Elasticsearch does the searching; an optional LLM answers questions from the matching notes.

- **Search**: keyword search with filters, facets, highlights, "more like this", link counts. No LLM involved.
- **Ask**: the same retrieval, then an LLM (local Ollama or Gemini, your choice) answers with citations.
- **Workbench**: see and tune the exact Elasticsearch query, explain/profile it, compare analyzers.
- **Evaluate**: judge a set of queries once, then measure each ranking change (nDCG@10).
- Vaults are mounted **read-only**; the app never modifies your notes.

> Status: scaffold. Core logic is unit-tested offline; Elasticsearch, Streamlit and LLM integration are
> **not yet verified** (see `docs/ROADMAP.md` M0-M1). UI wireframes: the "Vault Search - Streamlit wireframes" artifact.

## Quick start

```bash
cp .env.example .env          # set VAULT_PATH to your vault notes folder
make up                       # starts Elasticsearch + Streamlit app
make sync                     # incremental index of the vault
open http://localhost:8501
```

Ask mode (optional):

| Provider | Setup |
|----------|-------|
| `ollama` (Windows host) | Run Ollama natively on Windows. The container connects via `http://host.docker.internal:11434`. Pull model on host: `ollama pull qwen2.5:1.5b-instruct`. |
| `gemini` (API) | Set `GEMINI_API_KEY` in `.env`, pick `gemini` in sidebar. Note excerpts leave your machine. |
| `off` (default) | Plain Elasticsearch search only (zero LLM overhead). |

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `VAULT_PATH` | `/mnt/c/Users/Home/Documents/notes` | Host path to Markdown notes, mounted read-only (`:ro`) |
| `AIRFLOW_UID` | `1000` | Host WSL User ID to avoid Airflow volume permission issues |
| `ES_VERSION`, `ES_HEAP` | `8.17.0`, `512m` | Elasticsearch Docker tag and JVM heap cap |
| `LLM_PROVIDER` | `ollama` | Active provider (`off`, `ollama`, or `gemini`) |
| `OLLAMA_BASE_URL` | `http://host.docker.internal:11434` | Endpoint to host Windows Ollama instance |
| `OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | Compact LLM model suited for WSL memory limits |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | empty, `gemini-2.5-flash` | Gemini API credentials |

More vaults: add a read-only mount in `docker-compose.yml`, then an entry in `config/vaultlens.yaml`.

## Suite of Commands

### Stack Lifecycle

| Command | Description |
|---------|-------------|
| `make up` | Start core stack in background: Elasticsearch (`:9200`) + Streamlit app (`:8501`). |
| `make up-airflow` | Start full stack including optional Airflow (`:8080`), Scheduler, Postgres, and Redis. |
| `make down` | Stop all running containers (volumes preserved). |
| `make down-airflow` | Stop Airflow and metadata databases only, keeping core search active. |
| `make logs` | Follow logs for the core stack (`app` and `elasticsearch`). |
| `make logs-airflow` | Follow logs for Airflow webserver and scheduler. |
| `make build` | Rebuild the Streamlit app container image. |
| `make shell` | Open an interactive bash shell inside the running `app` container. |
| `make clean` | **DANGER**: Stop and remove all containers, networks, and named volumes (`esdata`, `appdata`, `postgres-db-volume`). |

### Vault & Index Operations

| Command | Description |
|---------|-------------|
| `make status` | Display Elasticsearch cluster health and index document counts. |
| `make sync` | Fast incremental index: scans notes, computes hash diff, and indexes new/modified files. |
| `make reindex` | Full zero-downtime reindex: builds `notes_v{N+1}`, indexes all vaults, swaps alias, deletes old index. |

### Headless CLI (`vaultlens` inside container or venv)

Run via `docker compose exec app vaultlens <cmd>`:

```bash
docker compose exec app vaultlens status                  # Check cluster & alias status
docker compose exec app vaultlens sync                    # Incremental scan & indexing
docker compose exec app vaultlens reindex                 # Zero-downtime reindex
docker compose exec app vaultlens search "my search"      # Keyword search with facets & scores
docker compose exec app vaultlens ask "What is X?"        # Grounded Q&A with citations
docker compose exec app vaultlens eval                    # Run ranking evaluations (nDCG@10)
```

### Quality & Tests

| Command | Description |
|---------|-------------|
| `make test` | Run pytest unit tests (offline; no Elasticsearch, Streamlit, or LLM required). |
| `make lint` | Run `ruff check src tests` code linting and formatting verification. |


## Elasticsearch & Multi-Language Support

### Why Elasticsearch?

Markdown vaults (such as Obsidian vaults) for personal knowledge management are commonly **bilingual or multilingual** — notes may be authored in Portuguese, English, or contain mixed-language terminology, technical documentation, and code.

Elasticsearch was selected as the retrieval engine over embedded alternatives (such as SQLite FTS5) or pure vector search for several key reasons:

1. **Mature Multilingual Lucene Analyzers**:
   - **Algorithmic Stemming**: Language-specific stemmers (such as `light_portuguese` and `light_english`) collapse grammatical inflections and suffixes (e.g., plurals, verb tenses) to common stems without sacrificing precision.
   - **Stopword Filtering**: Built-in language stopword dictionaries (`_portuguese_`, `_english_`, etc.) filter low-information words tailored to each language.
   - **Accent and Diacritic Folding (`asciifolding`)**: Essential for Romance languages like Portuguese. A search for `reuniao` seamlessly matches notes containing `reuniões`.

2. **Multi-Field Text Indexing Without Duplication**:
   - Rather than forcing documents to be tagged with a single language or duplicating stored text, Elasticsearch allows a single field (`body` and `chunks.text`) to be indexed across multiple analyzers simultaneously:
     - `body`: tokenized with standard lowercase ASCII folding (`std_folded`).
     - `body.pt`: analyzed with Portuguese stemming, stopwords, and accent folding (`pt_folded`).
     - `body.en`: analyzed with English stemming, stopwords, and accent folding (`en_folded`).
   - Raw source text is stored only once, but searchable across any configured linguistic lens.

3. **Unified Cross-Language Queries**:
   - Queries execute using `multi_match` across all language subfields (`body.pt`, `body.en`, `title`, etc.) at query time. Notes written in either language match their corresponding linguistic stems automatically without requiring manual language switching.

4. **Transparent Inspection and Tuning**:
   - Elasticsearch exposes diagnostic APIs (`_analyze`, `_explain`, `_profile`). Vaultlens surfaces this directly in the **Workbench** tab, allowing users to inspect token streams side-by-side (e.g. comparing `std_folded`, `pt_folded`, and `en_folded`) and see exact BM25 scoring breakdowns.


### How to Edit Supported Languages (Backend Files)

All language definitions, tokenizers, and query targets reside in the backend search modules under `src/vaultlens/`. To add a new language (for example, Spanish `es`), modify an existing one, or remove one, update the following files:

#### 1. Define Analyzers and Multi-Fields in the Mapping
**File:** `src/vaultlens/search/mappings/notes_v1.json` *(or create `notes_v<N+1>.json` if versioning an existing index)*

- **Add language token filters** under `settings.analysis.filter`:
  ```json
  "es_stop": {"type": "stop", "stopwords": "_spanish_"},
  "es_stemmer": {"type": "stemmer", "language": "light_spanish"}
  ```
- **Define the custom analyzer** under `settings.analysis.analyzer`:
  ```json
  "es_folded": {
    "type": "custom",
    "tokenizer": "standard",
    "filter": ["lowercase", "es_stop", "es_stemmer", "asciifolding"]
  }
  ```
- **Add multi-field mappings** under `mappings.properties`:
  - Under `body.fields`:
    ```json
    "es": {"type": "text", "analyzer": "es_folded"}
    ```
  - Under `chunks.properties.text.fields`:
    ```json
    "es": {"type": "text", "analyzer": "es_folded"}
    ```

#### 2. Update Query Builder Whitelists & Clauses
**File:** `src/vaultlens/search/query_builder.py`

- **Whitelist query fields** in `ALLOWED_QUERY_FIELDS`:
  ```python
  ALLOWED_QUERY_FIELDS = {"title", "title.edge", "aliases", "body", "body.pt", "body.en", "body.es"}
  ```
- **Include in chunk search** in `_chunk_clause()`:
  ```python
  "fields": ["chunks.text.pt", "chunks.text.en", "chunks.text.es", "chunks.heading^2"]
  ```
- **(Optional) Include in More-Like-This & Strategy Evaluations**:
  - Update `more_like_this_body()`: add `"body.es"` to `fields`.
  - Update `STRATEGIES`: update or add strategy entries (e.g., in Strategy `C`).

#### 3. Update Default Search Fields
**File:** `src/vaultlens/models.py`

If the new language should be searched by default, update `SearchSpec.fields`:
```python
fields: dict[str, float] = field(
    default_factory=lambda: {"title": 3.0, "aliases": 2.0, "body.pt": 1.0, "body.en": 1.0, "body.es": 1.0}
)
```

#### 4. Expose in the Workbench Analyzer Tab
**File:** `src/vaultlens/ui/views/workbench.py`

Add the new analyzer to the `tab_analyzer` loop to allow interactive token inspection in the Streamlit UI:
```python
for name in ("standard", "std_folded", "pt_folded", "en_folded", "es_folded"):
```

#### 5. Verify and Reindex

1. **Run offline tests**:
   ```bash
   make test
   ```
   *(Validates that all analyzers referenced in `notes_v*.json` are defined and match `ALLOWED_QUERY_FIELDS` in `tests/test_mapping.py`)*.

2. **Reindex the vault**:
   - If you introduced a new mapping file (e.g., `notes_v2.json`), bump `MAPPING_VERSION` in `src/vaultlens/search/mapping.py` (see `docs/project-updates-guide.md` checklist C2).
   - Run the zero-downtime reindex:
     ```bash
     make reindex
     # or headless inside the container:
     docker compose exec app vaultlens reindex
     ```


## Development

Read `AGENTS.md` first (rules for humans and coding agents), then `docs/ARCHITECTURE.md`, `docs/SPEC.md`,
`docs/ROADMAP.md`. When you change something, follow `docs/project-updates-guide.md`.

