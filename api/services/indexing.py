"""Background indexing service: PDF → GBC Index."""
import asyncio
import logging
import os
import shutil

from api.db import mongodb as db
from api.dependencies import MONGO_URI, MONGO_DB_PREFIX, INDEX_SAVE_DIR, THREAD_POOL

log = logging.getLogger(__name__)
_executor = THREAD_POOL


def _build_index_sync(
    pdf_path: str, save_path: str, tenant_id: str, doc_id: str,
    config_path: str, document_date=None, document_lang=None,
):
    """Synchronous index build for the current parser-backed flow."""
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.construct_index import construct_gbc_index

    cfg = load_system_config(config_path)
    cfg.pdf_path = pdf_path
    cfg.save_path = save_path
    cfg.source_type = "pdf"
    cfg.source_path = pdf_path
    cfg.tenant_id = tenant_id
    cfg.doc_id = doc_id
    # Propagate document_date into the config for temporal awareness
    if document_date is not None:
        cfg.document_date = document_date
    # Propagate document_lang into the config for language-aware processing
    if document_lang is not None:
        cfg.document_lang = document_lang
    # FalkorDB will be used if BOOKRAG_FALKORDB_HOST is set
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        from api.dependencies import FALKORDB_HOST, FALKORDB_PORT, FALKORDB_USERNAME, FALKORDB_PASSWORD
        cfg.falkordb = FalkorDBConfig(
            host=FALKORDB_HOST, port=FALKORDB_PORT,
            username=FALKORDB_USERNAME, password=FALKORDB_PASSWORD,
        )
    construct_gbc_index(cfg)


def _clear_stale_index_caches(save_path: str) -> None:
    """Delete per-document caches that must be regenerated on every indexing run.

    We intentionally KEEP the Docling / MinerU PDF-parse cache
    (``docling/`` and ``<method>/`` subdirectories) because reparsing a PDF is
    expensive (~30-150 s) and is not affected by extractor changes.

    Cleared artefacts
    -----------------
    * ``tree.pkl`` / ``tree.json``  — document tree (built from the parse cache)
    * ``kg_extractor_res/``         — per-node KG extraction results
    * ``graph_data_basic.json``     — compiled graph (rebuilt from KG results)
    """
    # Tree cache
    for fname in ("tree.pkl", "tree.json"):
        fpath = os.path.join(save_path, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            log.info(f"[cache-clear] Removed stale {fname} from {save_path}")

    # KG extraction cache (per-node JSON files)
    kg_res_dir = os.path.join(save_path, "kg_extractor_res")
    if os.path.isdir(kg_res_dir):
        shutil.rmtree(kg_res_dir)
        log.info(f"[cache-clear] Removed stale kg_extractor_res/ from {save_path}")

    # Compiled graph data (rebuilt during KG refinement)
    graph_data = os.path.join(save_path, "graph_data_basic.json")
    if os.path.exists(graph_data):
        os.remove(graph_data)
        log.info(f"[cache-clear] Removed stale graph_data_basic.json from {save_path}")


async def run_indexing(
    tenant_id: str,
    doc_id: str,
    pdf_path: str,
    config_path: str,
    document_date=None,
    document_lang=None,
):
    """Async wrapper: update status in MongoDB before/after indexing."""
    save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
    os.makedirs(save_path, exist_ok=True)

    # Clear stale per-document caches so every indexing run starts fresh.
    # This prevents broken tree/KG caches from a previous (possibly failed or
    # differently-configured) run from silently masking the current results.
    _clear_stale_index_caches(save_path)

    await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "indexing")
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            _build_index_sync,
            pdf_path, save_path, tenant_id, doc_id, config_path, document_date,
            document_lang,
        )
        await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "ready")
        log.info(f"Indexing complete for doc '{doc_id}' in tenant '{tenant_id}'")
        # Phase 3: Run entity resolution to merge into global graph
        try:
            from api.services.entity_resolution import run_entity_resolution
            await run_entity_resolution(tenant_id, doc_id, config_path)
        except Exception as er_err:
            log.warning(f"Entity resolution skipped (non-fatal): {er_err}")
    except Exception as e:
        log.error(f"Indexing failed for doc '{doc_id}': {e}", exc_info=True)
        await db.update_document_status(MONGO_URI, MONGO_DB_PREFIX, tenant_id, doc_id, "error", str(e))

