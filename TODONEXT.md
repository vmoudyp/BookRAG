# TODONEXT

## 1. Retroactive re-normalization endpoint (high value, low effort)
The vocabulary grew from 19 to 129 entries. Role assignments ingested before the expansion that landed as `normalization_status: unresolved` are still unresolved. A single endpoint or CLI command that scans all `suggested` assignments in a document, re-runs `apply_normalization`, and writes back those that now match would immediately increase coverage without any new ingestion.

## 2. Functional tests for the new curation endpoints (important gap)
`GET /{doc_id}/roles` and `POST /{doc_id}/roles/bulk-review` are tested for OpenAPI shape only. Functional tests that exercise list filtering (by `review_status`, by `entity_type`) and the bulk-review batch semantics (including the per-item error path) would give real confidence in the curation workflow.

## 3. Role statistics / coverage endpoint (useful for curation dashboards)
A lightweight `GET /{doc_id}/roles/stats` response like:
```json
{ "total": 142, "by_status": { "suggested": 98, "confirmed": 31, "rejected": 13 }, "unresolved": 44, "coverage_pct": 69 }
```
Gives curators a single glance at how much work is left and how effective extraction has been, without loading the full role list.

## 4. Vocabulary cache invalidation (important correctness issue)
`_load_vocab` in `Core/utils/role_normalizer.py` uses `@lru_cache(maxsize=1)`. If the vocab YAML is updated at runtime the running process keeps the stale cache. Add a `reload_vocab()` helper and wire it into any future vocab-mutation path, or expose `POST /vocab/reload` as an admin endpoint.

## 5. Role tenure-aware RAG filtering (meaningful UX lift)
The role-aware Cypher path in `gbc_rag.py` (`_query_role_context`) fetches all matching `HAS_ROLE` edges but does not filter on `tenure_status`. A query like *"who is the current Minister of Finance?"* should return only assignments with `tenure_status: current`, not historical holders.

## 6. Vocabulary management API (medium effort, admin-only)
Curators who discover a new alias (`"Plt. Menkes"`, `"Pj. Gubernur"`) must currently edit the YAML directly. An admin-only API (`POST /vocab/roles`, `PATCH /vocab/roles/{role_id}`, `GET /vocab/roles`) backed by the YAML (or optionally MongoDB for multi-instance deployments) would let the curation loop drive vocabulary growth.

