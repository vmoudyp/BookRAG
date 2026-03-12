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
- **Role-aware GBC answer augmentation** in `Core/rag/gbc_rag.py`
  - detects role-oriented queries
  - injects structured role evidence into answer generation when role assignments exist
  - excludes rejected role assignments from role-context augmentation
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
- **Domain role assignment / curation support** across `Core/Index/Graph.py`, `Core/utils/role_normalizer.py`, and `api/routers/entities.py`
  - `RoleAssignment` / `RoleEvidence` models attached to entities
  - extracted vs manual role provenance
  - review state tracking (`suggested`, `confirmed`, `disputed`, `rejected`)
  - evidence, source-node IDs, normalization status/confidence, and audit metadata
  - role listing, review, bulk-review, update, delete, and manual add endpoints
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
- **Visual sidecar query/fusion**
  - `visual_sidecar_query_enabled: false` by default
  - `visual_sidecar_fusion_enabled: false` by default
  - request-level overrides are supported on `POST /chat/query`

### Implemented, but opt-in unless added/enabled in your active YAML config

- **Automatic domain role extraction during ingestion**
  - controlled by `graph.role_extraction_enabled`
  - implemented in `Core/pipelines/kg_extractor.py`
- **FalkorDB role graph materialization**
  - controlled by `graph.role_graph_materialization`
  - role-aware RAG can also fall back to in-memory `role_assignments`

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
- `config/` — system configs for different retrieval/index strategies plus `role_vocab.yaml`
- `Scripts/` — example run scripts and dataset preprocessing notebooks
- `Eval/` — evaluation utilities
- `tests/` — focused tests for API docs, chat visual sidecar behavior, HTML/tree handling, role logic, ontology, and language-aware document processing
- `docs/` — architecture and feature-specific docs

## Configuration and setup notes

### Config files

- `config/gbc.yaml` is the most feature-complete example config in the repo.
- `config/gbc.yaml` currently defaults to `parser: docling`.
- `config/gbc.yaml` includes current defaults for ontology/entity resolution and visual-sidecar behavior.
- `config/docling.yaml` documents Docling-specific usage more explicitly.
- `config/role_vocab.yaml` is the current YAML vocabulary used by role normalization utilities.
- role extraction flags live on `GraphConfig` (`graph.role_extraction_enabled`, `graph.role_graph_materialization`) and can be added to your active YAML config when needed.
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
- `BOOKRAG_MONGO_PREFIX` / `BOOKRAG_MONGO_SYSTEM_DB`
- `BOOKRAG_FALKORDB_HOST` / `BOOKRAG_FALKORDB_PORT` / `BOOKRAG_FALKORDB_USERNAME` / `BOOKRAG_FALKORDB_PASSWORD`
- `BOOKRAG_UPLOAD_DIR` — defaults to `./uploads`
- `BOOKRAG_INDEX_DIR` — defaults to `./indices`
- `BOOKRAG_ROLE_VOCAB` — optional override path for `config/role_vocab.yaml`
- `BOOKRAG_CORS_ORIGINS` — comma-separated allowed origins for the API
- `BOOKRAG_LOG_LEVEL` — defaults to `INFO`
- `BOOKRAG_MAX_UPLOAD_MB` — upload size limit for `POST /documents`
- `BOOKRAG_TOKEN_EXPIRE` / `BOOKRAG_REFRESH_TOKEN_DAYS`
- `BOOKRAG_THREAD_POOL_SIZE`, `BOOKRAG_GBC_CACHE_TTL`, and `BOOKRAG_GBC_CACHE_MAX` for service/runtime tuning

`api/main.py` also calls `load_dotenv()`, so a local `.env` file is read before startup config is validated.

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
- `POST /chat/sessions`
- `GET /chat/sessions`
- `GET /chat/sessions/{session_id}/messages`
- `DELETE /chat/sessions/{session_id}`
- `GET /entities/{doc_id}`
- `PATCH /entities/{doc_id}/rename`
- `POST /entities/{doc_id}/merge`
- `POST /entities/{doc_id}/split`
- `GET /entities/{doc_id}/suggestions`
- `POST /entities/{doc_id}/roles`
- `GET /entities/{doc_id}/roles`
- `PATCH /entities/{doc_id}/roles/{assignment_id}`
- `POST /entities/{doc_id}/roles/{assignment_id}/review`
- `DELETE /entities/{doc_id}/roles/{assignment_id}`
- `POST /entities/{doc_id}/roles/bulk-review`
- `GET /health`

Notable current API capabilities:

- JWT-based auth with refresh-token rotation
- tenant isolation
- background indexing after upload
- cross-document chat mode
- document access filtering
- editable entity graph operations
- role curation export and batch review workflows
- `/health` endpoint with MongoDB and optional FalkorDB checks

### Document upload metadata

`POST /documents` accepts multipart upload of one or more **PDF** files and starts background indexing for each accepted file.

Useful form fields:

- `files`: one or more PDF files
- `document_date`: optional ISO-8601 timestamp applied to **all files in the request**
- `document_lang`: optional ISO 639-1 code like `en` or `id`, or `auto`, also applied to **all files in the request**

Current behavior:

- non-PDF files are rejected per-file in the `failed` list
- accepted files return `202 Accepted`
- the response includes both `uploaded` and `failed` entries so partial success is visible to callers

### Auth and token lifecycle

The auth endpoints are tenant-scoped: the same username may exist in different tenants, and login/registration always include `tenant_id`.

Typical flow:

1. `POST /auth/register` to create a tenant user
2. `POST /auth/login` to obtain an access token and refresh token
3. call protected endpoints with `Authorization: Bearer <access_token>`
4. `POST /auth/refresh` when the access token expires
5. `POST /auth/logout` to revoke the refresh token

Example register request:

```json
{
  "username": "alice",
  "password": "StrongPass1",
  "tenant_id": "tenant-a"
}
```

Example register response (`201 Created`):

```json
{
  "message": "User registered successfully"
}
```

Example login request:

```json
{
  "username": "alice",
  "password": "StrongPass1",
  "tenant_id": "tenant-a"
}
```

Example login response (`200 OK`):

```json
{
  "access_token": "access.jwt.token",
  "refresh_token": "refresh.jwt.token",
  "token_type": "bearer"
}
```

Example refresh request:

```json
{
  "refresh_token": "refresh.jwt.token"
}
```

Example refresh response (`200 OK`):

```json
{
  "access_token": "new.access.jwt.token",
  "refresh_token": "new.refresh.jwt.token",
  "token_type": "bearer"
}
```

Logout behavior:

- `POST /auth/logout`
- request body: `{"refresh_token": "..."}`
- response: `204 No Content`
- the provided refresh token is revoked and should not be reused

### Chat query visual sidecar overrides

`POST /chat/query` accepts optional per-request visual-sidecar overrides. When a field is omitted or set to `null`, the server keeps the loaded `GBCRAGConfig` value from its config file.

Current conservative fusion defaults are:

- `visual_sidecar_fusion_score_mode: "max_norm"`
- `visual_sidecar_fusion_min_score: 0.2`
- visual querying and fusion still remain disabled unless enabled in config or per request

Useful optional request fields:

- `visual_sidecar_query_enabled`
- `visual_sidecar_query_topk`
- `visual_sidecar_fusion_enabled`
- `visual_sidecar_fusion_weight`
- `visual_sidecar_fusion_score_mode`
- `visual_sidecar_fusion_min_score`

Example request body:

```json
{
  "query": "What does the revenue chart show?",
  "doc_ids": ["doc-123"],
  "cross_doc": false,
  "visual_sidecar_query_enabled": true,
  "visual_sidecar_query_topk": 5,
  "visual_sidecar_fusion_enabled": true,
  "visual_sidecar_fusion_weight": 1.25,
  "visual_sidecar_fusion_score_mode": "max_norm",
  "visual_sidecar_fusion_min_score": 0.2
}
```

### Chat sessions and history

The chat router also exposes session/history endpoints under `/chat/sessions`.

- `POST /chat/sessions`
  - creates a new session for the current user
  - optional `doc_ids` are filtered to accessible documents before they are stored on the session
- `GET /chat/sessions`
  - lists the current user's sessions
  - supports `limit` and `offset`
- `GET /chat/sessions/{session_id}/messages`
  - returns paginated message history for the session
  - supports `limit` and `offset`
- `DELETE /chat/sessions/{session_id}`
  - deletes a session and its messages
  - allowed for the session owner or an admin

### Role assignments and curation workflow

Entities may carry structured domain role assignments such as political offices, organizational positions, or executive titles.

Current role-assignment shape includes fields such as:

- `assignment_id`
- `role_name` and optional stable `role_id`
- scope metadata (`scope_entity_name`, `scope_entity_type`, `scope_entity_id`)
- temporal metadata (`start_date`, `end_date`, `tenure_status`)
- workflow metadata (`origin`, `review_status`, `normalization_status`, `normalization_confidence`)
- evidence/source trace (`source_ids`, `evidence`)
- audit metadata (`created_by`, `updated_by`, `created_at`, `updated_at`)

Current workflow support:

- LLM extraction can create `origin="extracted"`, `review_status="suggested"` assignments during ingestion when role extraction is enabled
- reviewers can confirm / dispute / reject assignments and optionally override canonical `role_name` / `role_id`
- bulk review supports partial success reporting per item
- role export supports filtering by `review_status` and `entity_type`
- role-name normalization utilities use the YAML vocabulary in `config/role_vocab.yaml`

Operational note:

- `TODONEXT.md` currently tracks follow-up work for role re-normalization, router functional tests, role stats, vocab cache invalidation, tenure-aware filtering, and vocabulary management APIs

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

- role assignment merge/CRUD/review logic and extraction helpers
- chat visual-sidecar config propagation and router request handling
- documents router metadata and OpenAPI behavior
- remaining router OpenAPI documentation coverage
- HTML/JSON document-tree and visual-sidecar processing behavior
- ontology integration
- language-aware PDF paragraph refinement
- legal heading detection / language detection

Current test files:

- `tests/test_chat_router_visual_sidecar.py`
- `tests/test_chat_service_visual_sidecar.py`
- `tests/test_doc_tree_builder_html.py`
- `tests/test_documents_router_openapi.py`
- `tests/test_entity_roles.py`
- `tests/test_ontology_integration.py`
- `tests/test_pdf_refiner_lang.py`
- `tests/test_legal_heading_detector.py`
- `tests/test_remaining_routers_openapi.py`

Current testing gap worth knowing:

- the newer `GET /entities/{doc_id}/roles` and `POST /entities/{doc_id}/roles/bulk-review` router paths have OpenAPI/schema coverage and service-level logic coverage, but not dedicated functional router tests yet

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
- `docs/gemini-leaf-embedding-design.md` — notes on Gemini leaf embedding design
- `docs/ontology-usage-guide.md` — ontology and global resolution notes
- `docs/research-behavior-detection.md` — research-oriented notes
- `TODONEXT.md` — short near-term backlog for the next high-value improvements

If you are trying to understand what is implemented **right now**, prefer the code entrypoints plus this README over older paper-era setup text.
