# BookRAG

BookRAG is a hierarchical, structure-aware RAG system for complex documents. The current repository contains both:

- an **offline indexing / evaluation pipeline** for document-level experiments, and
- a **multi-tenant FastAPI service** for authenticated document upload, indexing, chat, and entity management.

This README reflects the **current implementation status in this repository**, not only the original paper/project description.

## Current implementation status

### Implemented today

- **Document indexing pipeline** in `main.py` and `Core/construct_index.py`
  - document tree construction
  - knowledge graph construction
  - vector database construction
  - staged indexing via `index --stage ...`
  - split-based dataset processing for parallel workers
- **RAG / inference pipeline** in `main.py` and `Core/inference.py`
  - dataset-driven batch inference
  - result persistence per query and per document
  - token/cost tracking
- **Multiple RAG strategy implementations** under `Core/rag/`
  - `gbc`
  - `graph`
  - `mm`
  - `vanilla`
- **FastAPI application** in `api/main.py`
  - auth router
  - tenants router
  - documents router
  - chat router
  - entities router
  - request ID middleware
  - structured JSON logging
  - health endpoint
  - startup recovery for stale indexing jobs
- **Background indexing service** in `api/services/indexing.py`
- **Entity editing operations** in the API
  - list entities
  - rename
  - merge
  - split
  - suggest merge candidates
- **Conversation/session support** in chat APIs
  - session creation
  - session listing
  - session message history
  - history-aware query rewriting
- **Parser selection** in the tree builder
  - `docling`
  - `mineru`

### Implemented, but disabled by default in `config/gbc.yaml`

- **Ontology-backed canonical entity alignment**
  - `ontology.enabled: false` by default
  - see `docs/ontology-usage-guide.md`
- **Tenant/global cross-document entity resolution**
  - `entity_resolution.enabled: false` by default
  - global VDB sync is implemented but opt-in

### Present as docs / design material, not part of the default runtime

- `MCP.md` documents a proposed MCP integration path.
- This repository does **not** currently include a checked-in `mcp_server.py` runtime file.
- Treat `MCP.md` as design/implementation guidance, not as a feature already wired into the default app.

## Main entrypoints

- `main.py` — CLI for indexing and dataset-based RAG runs
- `api/main.py` — FastAPI application entrypoint
- `Eval/evaluation.py` — dataset evaluation entrypoint
- `Scripts/example-index.sh` — example indexing script
- `Scripts/example-rag.sh` — example batch RAG script
- `Scripts/example-eval.sh` — example evaluation script

## Repository layout

- `Core/` — indexing, graph, retrieval, and model/provider logic
- `api/` — FastAPI app, routers, DB helpers, and services
- `config/` — system configs for different retrieval/index strategies
- `Scripts/` — example run scripts and dataset preprocessing notebooks
- `Eval/` — evaluation utilities
- `tests/` — targeted tests for ontology and language-aware document processing
- `docs/` — architecture and feature-specific docs

## Configuration and setup notes

### Config files

- `config/gbc.yaml` is the most feature-complete example config in the repo.
- `config/gbc.yaml` currently defaults to `parser: docling`.
- `config/docling.yaml` documents Docling-specific usage more explicitly.
- `config/default.yaml` is a template with many `TODO` placeholders.

### Environment / dependencies

This repository currently does **not** include a root dependency manifest such as `pyproject.toml` or `requirements.txt`, so this README intentionally avoids inventing package installation commands.

Practical implication:

- use the checked-in config files as the source of truth for required model/backend settings,
- use the parser you intend to run (`docling` or `mineru`),
- and treat supporting infrastructure such as MongoDB / FalkorDB / vector DB backends as runtime requirements of the API and advanced graph workflows.

### API environment variables

The FastAPI service reads configuration from environment variables. Important ones include:

- `BOOKRAG_SECRET_KEY` — required; the API will fail fast if it is not set
- `BOOKRAG_CONFIG_PATH` — defaults to `config/gbc.yaml`
- `BOOKRAG_MONGO_URI` — defaults to `mongodb://localhost:27017`
- `BOOKRAG_FALKORDB_HOST` / `BOOKRAG_FALKORDB_PORT`
- `BOOKRAG_UPLOAD_DIR` — defaults to `./uploads`
- `BOOKRAG_INDEX_DIR` — defaults to `./indices`

## CLI workflow

The best-supported CLI workflow is **dataset-driven batch processing**.

### 1. Prepare configs

You will typically need:

- a **system config**, for example `config/gbc.yaml` or `config/docling.yaml`
- a **dataset config**, for example `Scripts/cfg/example-Qasper.yaml`

Dataset configs specify:

- `dataset_path`
- `working_dir`
- `dataset_name`

### 2. Build indexes

`main.py` supports staged indexing:

- `tree`
- `graph`
- `vdb`
- `all`
- `mm_reranker`
- `rebuild_graph_vdb`

Example:

```bash
python main.py -c config/gbc.yaml -d Scripts/cfg/example-Qasper.yaml index --stage all
```

Or use the provided shell example:

```bash
bash Scripts/example-index.sh
```

### 3. Run RAG

Example batch run:

```bash
python main.py -c config/gbc.yaml -d Scripts/cfg/example-Qasper.yaml rag
```

Or use the provided shell example:

```bash
bash Scripts/example-rag.sh
```

### CLI behavior worth knowing

- documents are grouped by `doc_uuid` and `doc_path`
- splits can be distributed with `--nsplit` and `--num`
- each document gets its own output directory under the dataset `working_dir`
- config snapshots are written into the output directories for reproducibility
- per-run logs are also written into document output folders

## API service

The API entrypoint is `api/main.py`.

Implemented router surface:

- `POST /auth/register`
- `POST /auth/login`
- `POST /auth/refresh`
- `POST /auth/logout`
- `POST /tenants`
- `GET /tenants/{tenant_id}`
- `POST /tenants/{tenant_id}/permissions`
- `POST /documents`
- `GET /documents`
- `GET /documents/{doc_id}`
- `DELETE /documents/{doc_id}`
- `GET /documents/{doc_id}/raw`
- `POST /chat/query`
- session management under `/chat/sessions`
- entity management under `/entities/{doc_id}`

Notable current API capabilities:

- JWT-based auth with refresh-token rotation
- tenant isolation
- background indexing after upload
- cross-document chat mode
- document access filtering
- editable entity graph operations
- `/health` endpoint with MongoDB and optional FalkorDB checks

## Ontology and cross-document resolution

Ontology and entity-resolution support are real parts of the current codebase, but they are **not enabled by default**.

- Ontology config model: `Core/configs/ontology_config.py`
- Ontology matching utilities: `Core/utils/ontology_utils.py`
- Tenant/global resolution service: `api/services/entity_resolution.py`
- Usage guide: `docs/ontology-usage-guide.md`

Recommended documentation stance:

- enable ontology support deliberately,
- validate mappings on your own data,
- enable tenant/global entity resolution only when you want shared canonical entities across documents.

## Evaluation

Dataset evaluation is implemented in `Eval/evaluation.py` and currently includes handlers for:

- `MMLongBench`
- `m3docrag`
- `qasper`

Example:

```bash
bash Scripts/example-eval.sh
```

## Tests currently present

The checked-in test suite is focused and confirms several recent implementation details, including:

- ontology integration
- language-aware PDF paragraph refinement
- legal heading detection / language detection

Current test files:

- `tests/test_ontology_integration.py`
- `tests/test_pdf_refiner_lang.py`
- `tests/test_legal_heading_detector.py`

## Dataset format

The repo uses a unified JSON dataset format for batch indexing and evaluation workflows.

Supported/used datasets in the repo docs and scripts include:

- MMLongBench-Doc: [MMLONGBENCH-DOC](https://github.com/mayubo2333/MMLongBench-Doc)
- m3docvqa / M3DocRAG: [M3DocRAG](https://github.com/bloomberg/m3docrag)
- Qasper: [Qasper](https://huggingface.co/datasets/allenai/qasper)

Example shape:

```json
[
  {
    "question": "THE FIRST QUESTION",
    "answer": "THE ANSWER OF FIRST QUESTION",
    "doc_uuid": "UUID OF THE DOCUMENT PDF",
    "doc_path": "PATH TO THE DOCUMENT PDF",
    "xxx": "other attributes"
  }
]
```

See preprocessing examples under `Scripts/preprocess/`.

## Additional docs

- `docs/bookrag-architecture-review.md` — architecture review and implementation analysis
- `docs/ontology-usage-guide.md` — ontology and global resolution notes
- `docs/research-behavior-detection.md` — research-oriented notes

If you are trying to understand what is implemented **right now**, prefer the code entrypoints plus this README over older paper-era setup text.
