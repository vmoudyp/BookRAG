"""Entity editor service: rename, merge, split, suggest-merges on NER entities.

All mutating operations work on the in-memory NetworkX graph (loaded from
graph_data.json), persist changes to graph_data.json *and* FalkorDB (when
configured), then best-effort rebuild the entity VDB so search stays fresh.

Entities are NOT stored in MongoDB — their source of truth is FalkorDB +
graph_data.json.  A lightweight audit entry is written to MongoDB's
``entity_edits`` collection for every mutating operation.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from api.dependencies import (
    FALKORDB_HOST, FALKORDB_PORT, FALKORDB_USERNAME, FALKORDB_PASSWORD,
    INDEX_SAVE_DIR, MONGO_URI, MONGO_DB_PREFIX,
    THREAD_POOL,
)
from api.db import mongodb as db

log = logging.getLogger(__name__)
_executor = THREAD_POOL

# Per-document asyncio lock — keyed by "{tenant_id}:{doc_id}"
_doc_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Dedup tracker: prevents redundant VDB rebuilds when multiple edits fire
# in quick succession on the same document.
_rebuild_pending: set[str] = set()
_rebuild_pending_lock = asyncio.Lock()

# Global lock for shared vocabulary-file mutations.
_role_vocab_lock = asyncio.Lock()


def _get_lock(tenant_id: str, doc_id: str) -> asyncio.Lock:
    return _doc_locks[f"{tenant_id}:{doc_id}"]


# ── Graph loader ─────────────────────────────────────────────────────────────

def _load_graph_sync(tenant_id: str, doc_id: str, config_path: str):
    """Load a Graph from JSON (never from FalkorDB) for in-memory editing.

    Returns ``(graph, save_path, falkordb_cfg | None)``.
    """
    from Core.configs.system_config import load_system_config
    from Core.configs.falkordb_config import FalkorDBConfig
    from Core.Index.Graph import Graph

    cfg = load_system_config(config_path)
    save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
    variant = "basic" if cfg.graph.refine_type == "basic" else None

    # Always load from JSON so we get the full in-memory graph
    graph = Graph.load_from_dir(
        load_dir=save_path,
        variant=variant,
        tenant_id=tenant_id,
        doc_id=doc_id,
        falkordb_cfg=None,  # load from JSON only
    )

    # Attach FalkorDB cfg for saving if the host env var is set
    falkordb_cfg = None
    fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
    if fdb_host:
        falkordb_cfg = FalkorDBConfig(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            username=FALKORDB_USERNAME,
            password=FALKORDB_PASSWORD,
        )
        graph.falkordb_cfg = falkordb_cfg
        graph.tenant_id = tenant_id
        graph.doc_id = doc_id
        graph.use_falkordb = True
        graph._fdb_graph_name = falkordb_cfg.graph_name_for_doc(tenant_id, doc_id)

    return graph, save_path, falkordb_cfg


def _rebuild_vdb_sync(tenant_id: str, doc_id: str, config_path: str) -> None:
    """Best-effort VDB rebuild after any graph mutation."""
    try:
        from Core.configs.system_config import load_system_config
        from Core.configs.falkordb_config import FalkorDBConfig
        from Core.Index.GBCIndex import GBC

        cfg = load_system_config(config_path)
        cfg.tenant_id = tenant_id
        cfg.doc_id = doc_id
        cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)

        fdb_host = os.getenv("BOOKRAG_FALKORDB_HOST", "")
        if fdb_host:
            cfg.falkordb = FalkorDBConfig(
                host=FALKORDB_HOST, port=FALKORDB_PORT,
                username=FALKORDB_USERNAME, password=FALKORDB_PASSWORD,
            )

        gbc = GBC.load_gbc_index(cfg)
        gbc.rebuild_vdb()
        log.info(f"VDB rebuilt for {tenant_id}/{doc_id}")
    except Exception as exc:
        log.warning(f"VDB rebuild failed for {tenant_id}/{doc_id}: {exc}")


async def _schedule_vdb_rebuild(tenant_id: str, doc_id: str, config_path: str) -> None:
    """Await a VDB rebuild, deduplicating concurrent requests for the same doc.

    If a rebuild is already in-flight for this ``tenant_id:doc_id``, the call
    is skipped (the already-running rebuild will pick up the latest graph JSON).
    """
    key = f"{tenant_id}:{doc_id}"
    async with _rebuild_pending_lock:
        if key in _rebuild_pending:
            log.debug(f"VDB rebuild already pending for {key} — skipping")
            return
        _rebuild_pending.add(key)
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(_executor, _rebuild_vdb_sync, tenant_id, doc_id, config_path)
    finally:
        async with _rebuild_pending_lock:
            _rebuild_pending.discard(key)


# ── List entities ─────────────────────────────────────────────────────────────

def _list_entities_sync(tenant_id: str, doc_id: str, config_path: str) -> List[dict]:
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    result = []
    for node_name in graph.get_all_nodes():
        entity = graph.get_entity_by_node_name(node_name)
        result.append({
            "entity_name": entity.entity_name,
            "entity_type": entity.entity_type,
            "description": entity.description,
            "source_ids": sorted(entity.source_ids),
            "node_name": node_name,
            "role_assignments": [ra.model_dump() for ra in entity.role_assignments],
        })
    return sorted(result, key=lambda e: e["entity_name"].lower())


async def list_entities(tenant_id: str, doc_id: str, config_path: str) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _list_entities_sync, tenant_id, doc_id, config_path
    )


# ── Rename entity ─────────────────────────────────────────────────────────────

def _rename_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entity_name: str, new_entity_type: str, new_description: Optional[str],
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    old_entity = graph.get_entity(entity_name, entity_type)

    effective_type = new_entity_type if new_entity_type else old_entity.entity_type
    effective_desc = new_description if new_description is not None else old_entity.description

    new_entity = Entity(
        entity_name=new_entity_name,
        entity_type=effective_type,
        description=effective_desc,
        source_ids=old_entity.source_ids,
        role_assignments=old_entity.role_assignments,  # preserve role assignments across renames
    )
    graph.update_entity(entity_name, entity_type, new_entity)
    graph.save_graph()

    new_node = graph.get_node_name_from_str(new_entity_name, effective_type)
    return [{
        "entity_name": new_entity_name,
        "entity_type": effective_type,
        "description": effective_desc,
        "source_ids": sorted(new_entity.source_ids),
        "node_name": new_node,
        "role_assignments": [ra.model_dump() for ra in new_entity.role_assignments],
    }]


async def rename_entity(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entity_name: str, new_entity_type: str, new_description: Optional[str],
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _rename_sync,
            tenant_id, doc_id, config_path,
            entity_name, entity_type, new_entity_name, new_entity_type, new_description,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "rename", "doc_id": doc_id, "user_id": user_id,
        "before": {"entity_name": entity_name, "entity_type": entity_type},
        "after": {"entity_name": new_entity_name, "entity_type": new_entity_type or entity_type},
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result


# ── Split entity ──────────────────────────────────────────────────────────────

def _split_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entities: List[dict],
    edge_mode: str,
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    old_node = graph.get_node_name_from_str(entity_name, entity_type)
    if old_node not in graph.kg:
        raise KeyError(f"Entity '{old_node}' not found in graph.")

    old_entity = graph.get_entity_by_node_name(old_node)
    old_neighbors = list(graph.kg.neighbors(old_node))
    old_edge_data = {n: graph.kg.get_edge_data(old_node, n) for n in old_neighbors}

    created: List[dict] = []
    for spec in new_entities:
        spec_name = spec["entity_name"]
        spec_type = spec["entity_type"]
        spec_desc = spec.get("description") or old_entity.description
        spec_sids = set(spec.get("source_ids") or old_entity.source_ids)

        # Conservative role assignment split: only carry roles whose evidence
        # source_ids intersect with this child entity's source_ids.
        child_roles = [
            ra for ra in old_entity.role_assignments
            if set(ra.source_ids) & spec_sids
        ]

        new_node = graph.get_node_name_from_str(spec_name, spec_type)
        graph.add_kg_node(Entity(
            entity_name=spec_name, entity_type=spec_type,
            description=spec_desc, source_ids=spec_sids,
            role_assignments=child_roles,
        ))

        if edge_mode == "duplicate":
            for neighbor, edata in old_edge_data.items():
                if neighbor != old_node and not graph.kg.has_edge(new_node, neighbor):
                    graph.kg.add_edge(new_node, neighbor, **edata)

        for tree_id in spec_sids:
            graph.tree2kg[tree_id].add(new_node)

        created.append({
            "entity_name": spec_name, "entity_type": spec_type,
            "description": spec_desc, "source_ids": sorted(spec_sids), "node_name": new_node,
            "role_assignments": [ra.model_dump() for ra in child_roles],
        })

    for _, nodes in graph.tree2kg.items():
        nodes.discard(old_node)
    graph.kg.remove_node(old_node)
    graph.save_graph()
    return created


async def split_entity(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    new_entities: List[dict], edge_mode: str,
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _split_sync,
            tenant_id, doc_id, config_path,
            entity_name, entity_type, new_entities, edge_mode,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "split", "doc_id": doc_id, "user_id": user_id,
        "before": {"entity_name": entity_name, "entity_type": entity_type},
        "after": [{"entity_name": e["entity_name"], "entity_type": e["entity_type"]} for e in result],
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result


# ── Suggest merge candidates ──────────────────────────────────────────────────

def _suggest_merges_sync(
    tenant_id: str, doc_id: str, config_path: str,
    min_score: float, top_k: int, use_embeddings: bool,
) -> List[dict]:
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    nodes = list(graph.get_all_nodes())
    entities = []
    for node in nodes:
        ent = graph.get_entity_by_node_name(node)
        entities.append({"entity_name": ent.entity_name, "entity_type": ent.entity_type, "node": node})

    suggestions: List[dict] = []

    # Pre-group entities by type so we only compare within same-type groups
    # This reduces O(n²) to O(Σ nᵢ²) where nᵢ is entity count per type
    by_type: Dict[str, List[dict]] = defaultdict(list)
    for ent in entities:
        by_type[ent["entity_type"]].append(ent)

    # String similarity — only within same-type groups
    for group in by_type.values():
        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = group[i], group[j]
                score = SequenceMatcher(None, a["entity_name"].lower(), b["entity_name"].lower()).ratio()
                if score >= min_score:
                    suggestions.append({
                        "entity_a": {"entity_name": a["entity_name"], "entity_type": a["entity_type"]},
                        "entity_b": {"entity_name": b["entity_name"], "entity_type": b["entity_type"]},
                        "score": round(score, 4),
                        "method": "string_similarity",
                    })

    # Embedding similarity (optional)
    if use_embeddings:
        try:
            from Core.configs.system_config import load_system_config
            from Core.Index.GBCIndex import GBC

            cfg = load_system_config(config_path)
            cfg.tenant_id = tenant_id
            cfg.doc_id = doc_id
            cfg.save_path = os.path.join(INDEX_SAVE_DIR, tenant_id, doc_id)
            gbc = GBC.load_gbc_index(cfg)

            seen_pairs: set = set()
            for ent in entities:
                for hit in gbc.entity_vdb.search(ent["node"], top_k=5):
                    sim = 1.0 - hit["distance"]
                    if sim < min_score:
                        continue
                    meta = hit.get("metadata", {})
                    b_name = meta.get("entity_name", "")
                    b_type = meta.get("entity_type", "")
                    if (b_name == ent["entity_name"] and b_type == ent["entity_type"]) or b_type != ent["entity_type"]:
                        continue
                    pair = tuple(sorted([(ent["entity_name"], ent["entity_type"]), (b_name, b_type)]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    suggestions.append({
                        "entity_a": {"entity_name": ent["entity_name"], "entity_type": ent["entity_type"]},
                        "entity_b": {"entity_name": b_name, "entity_type": b_type},
                        "score": round(sim, 4),
                        "method": "embedding_similarity",
                    })
        except Exception as exc:
            log.warning(f"Embedding suggestions failed: {exc}")

    suggestions.sort(key=lambda s: s["score"], reverse=True)
    return suggestions[:top_k]


async def suggest_merges(
    tenant_id: str, doc_id: str, config_path: str,
    min_score: float = 0.80,
    top_k: int = 50,
    use_embeddings: bool = False,
) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _suggest_merges_sync,
        tenant_id, doc_id, config_path, min_score, top_k, use_embeddings,
    )
# ── Merge entities ────────────────────────────────────────────────────────────

def _merge_role_assignments(all_lists) -> list:
    """Union and deduplicate role assignments from multiple entity sources.

    Deduplication key: (role_id or normalised role_name, normalised scope, tenure_status, start_date, end_date).
    Conflict resolution priority: confirmed > suggested, manual > extracted,
    higher normalization_confidence wins.  source_ids and evidence are always unioned.
    """
    from Core.Index.Graph import RoleAssignment

    seen: dict = {}  # key -> winning RoleAssignment

    def _dedup_key(ra: RoleAssignment) -> tuple:
        role_key = (ra.role_id or ra.role_name.strip().lower())
        scope_key = (ra.scope_entity_name or "").strip().lower()
        return (role_key, scope_key, ra.tenure_status, ra.start_date or "", ra.end_date or "")

    _review_priority = {"confirmed": 0, "disputed": 1, "suggested": 2, "rejected": 3}
    _origin_priority = {"manual": 0, "extracted": 1}

    for role_list in all_lists:
        for ra in role_list:
            key = _dedup_key(ra)
            if key not in seen:
                seen[key] = ra
            else:
                existing = seen[key]
                # prefer stronger review_status
                replace_existing = (
                    _review_priority.get(ra.review_status, 99) < _review_priority.get(existing.review_status, 99)
                    or _origin_priority.get(ra.origin, 99) < _origin_priority.get(existing.origin, 99)
                )
                if replace_existing:
                    seen[key] = ra
                    existing = ra
                # union source_ids and evidence
                merged_sids = list(set(existing.source_ids) | set(ra.source_ids))
                existing_ev_ids = {(e.source_id, e.observed_role_text) for e in existing.evidence}
                merged_ev = list(existing.evidence) + [
                    e for e in ra.evidence if (e.source_id, e.observed_role_text) not in existing_ev_ids
                ]
                seen[key] = RoleAssignment(
                    **{**existing.model_dump(), "source_ids": merged_sids, "evidence": [e.model_dump() for e in merged_ev]}
                )

    return list(seen.values())


def _merge_sync(
    tenant_id: str, doc_id: str, config_path: str,
    source_entities: List[dict],
    canonical_name: str, canonical_type: str, canonical_desc: str,
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)

    # Collect all source_ids and role_assignments from entities being merged
    merged_source_ids: set = set()
    all_role_lists = []
    for src in source_entities:
        try:
            ent = graph.get_entity(src["entity_name"], src["entity_type"])
            merged_source_ids.update(ent.source_ids)
            all_role_lists.append(ent.role_assignments)
        except KeyError:
            log.warning(f"Merge: source entity not found: {src}")

    canonical_node = graph.get_node_name_from_str(canonical_name, canonical_type)

    # Ensure canonical node exists (may be one of the sources or brand new)
    if canonical_node not in graph.kg:
        merged_roles = _merge_role_assignments(all_role_lists)
        canonical_entity = Entity(
            entity_name=canonical_name,
            entity_type=canonical_type,
            description=canonical_desc,
            source_ids=merged_source_ids,
            role_assignments=merged_roles,
        )
        graph.add_kg_node(canonical_entity)
    else:
        # Update description, source_ids, and role_assignments on the existing node
        existing = graph.get_entity_by_node_name(canonical_node)
        merged_source_ids.update(existing.source_ids)
        all_role_lists.append(existing.role_assignments)
        merged_roles = _merge_role_assignments(all_role_lists)
        updated = Entity(
            entity_name=canonical_name,
            entity_type=canonical_type,
            description=canonical_desc or existing.description,
            source_ids=merged_source_ids,
            role_assignments=merged_roles,
        )
        graph.kg.nodes[canonical_node].update(updated.model_dump())

    # Transfer edges from each source to canonical, then remove source
    for src in source_entities:
        src_node = graph.get_node_name_from_str(src["entity_name"], src["entity_type"])
        if src_node == canonical_node or src_node not in graph.kg:
            continue
        for neighbor in graph.kg.neighbors(src_node):
            if neighbor == canonical_node:
                continue
            edge_data = graph.kg.get_edge_data(src_node, neighbor)
            if not graph.kg.has_edge(canonical_node, neighbor):
                graph.kg.add_edge(canonical_node, neighbor, **edge_data)
        # Update tree2kg
        for _, nodes in graph.tree2kg.items():
            if src_node in nodes:
                nodes.discard(src_node)
                nodes.add(canonical_node)
        graph.kg.remove_node(src_node)

    # Persist source_ids on canonical node
    graph.kg.nodes[canonical_node]["source_ids"] = list(merged_source_ids)
    graph.save_graph()

    canonical_ent = graph.get_entity_by_node_name(canonical_node)
    return [{
        "entity_name": canonical_ent.entity_name,
        "entity_type": canonical_ent.entity_type,
        "description": canonical_ent.description,
        "source_ids": sorted(canonical_ent.source_ids),
        "node_name": canonical_node,
        "role_assignments": [ra.model_dump() for ra in canonical_ent.role_assignments],
    }]


async def merge_entities(
    tenant_id: str, doc_id: str, config_path: str,
    source_entities: List[dict],
    canonical_name: str, canonical_type: str, canonical_desc: str,
    user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _merge_sync,
            tenant_id, doc_id, config_path,
            source_entities, canonical_name, canonical_type, canonical_desc,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "merge", "doc_id": doc_id, "user_id": user_id,
        "before": source_entities,
        "after": {"entity_name": canonical_name, "entity_type": canonical_type},
    })
    await _schedule_vdb_rebuild(tenant_id, doc_id, config_path)
    return result


# ── Domain role assignment CRUD ───────────────────────────────────────────────

def _add_role_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    role_input: dict, user_id: str,
) -> List[dict]:
    import uuid
    from datetime import datetime, timezone
    from Core.Index.Graph import Entity, RoleAssignment, RoleEvidence

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    entity = graph.get_entity(entity_name, entity_type)

    evidence = [RoleEvidence(**e) for e in role_input.pop("evidence", [])]
    now = datetime.now(timezone.utc).isoformat()
    assignment = RoleAssignment(
        assignment_id=str(uuid.uuid4()),
        origin="manual",
        review_status="confirmed",
        normalization_status="manual_override",
        created_by=user_id,
        created_at=now,
        updated_by=user_id,
        updated_at=now,
        evidence=evidence,
        **role_input,
    )
    # Apply normalization unless the caller has already supplied a role_id
    if not assignment.role_id:
        try:
            from Core.utils.role_normalizer import apply_normalization
            assignment = apply_normalization(assignment)
        except Exception as norm_exc:
            log.warning("Role normalization skipped: %s", norm_exc)
    updated_entity = Entity(
        **{**entity.model_dump(), "role_assignments": entity.role_assignments + [assignment]}
    )
    graph.update_entity(entity_name, entity_type, updated_entity)
    graph.save_graph()

    refreshed = graph.get_entity(updated_entity.entity_name, updated_entity.entity_type)
    return [{
        "entity_name": refreshed.entity_name,
        "entity_type": refreshed.entity_type,
        "description": refreshed.description,
        "source_ids": sorted(refreshed.source_ids),
        "node_name": graph.get_node_name_from_entity(refreshed),
        "role_assignments": [ra.model_dump() for ra in refreshed.role_assignments],
    }]


async def add_role_assignment(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    role_input: dict, user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _add_role_sync,
            tenant_id, doc_id, config_path, entity_name, entity_type, role_input, user_id,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "add_role", "doc_id": doc_id, "user_id": user_id,
        "entity": {"entity_name": entity_name, "entity_type": entity_type},
    })
    return result


def _update_role_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str, update_fields: dict, user_id: str,
) -> List[dict]:
    from datetime import datetime, timezone
    from Core.Index.Graph import Entity, RoleAssignment

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    entity = graph.get_entity(entity_name, entity_type)

    new_roles = []
    found = False
    for ra in entity.role_assignments:
        if ra.assignment_id == assignment_id:
            found = True
            merged = {**ra.model_dump(), **{k: v for k, v in update_fields.items() if v is not None}}
            merged["updated_by"] = user_id
            merged["updated_at"] = datetime.now(timezone.utc).isoformat()
            new_roles.append(RoleAssignment(**merged))
        else:
            new_roles.append(ra)

    if not found:
        raise KeyError(f"Role assignment '{assignment_id}' not found on entity '{entity_name}'.")

    updated_entity = Entity(**{**entity.model_dump(), "role_assignments": new_roles})
    graph.update_entity(entity_name, entity_type, updated_entity)
    graph.save_graph()

    refreshed = graph.get_entity(updated_entity.entity_name, updated_entity.entity_type)
    return [{
        "entity_name": refreshed.entity_name,
        "entity_type": refreshed.entity_type,
        "description": refreshed.description,
        "source_ids": sorted(refreshed.source_ids),
        "node_name": graph.get_node_name_from_entity(refreshed),
        "role_assignments": [ra.model_dump() for ra in refreshed.role_assignments],
    }]


async def update_role_assignment(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str, update_fields: dict, user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _update_role_sync,
            tenant_id, doc_id, config_path, entity_name, entity_type,
            assignment_id, update_fields, user_id,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "update_role", "doc_id": doc_id, "user_id": user_id,
        "entity": {"entity_name": entity_name, "entity_type": entity_type},
        "assignment_id": assignment_id,
    })
    return result


def _review_role_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str, review_status: str,
    role_name: Optional[str], role_id: Optional[str], user_id: str,
) -> List[dict]:
    from datetime import datetime, timezone
    from Core.Index.Graph import Entity, RoleAssignment

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    entity = graph.get_entity(entity_name, entity_type)

    new_roles = []
    found = False
    for ra in entity.role_assignments:
        if ra.assignment_id == assignment_id:
            found = True
            overrides = {"review_status": review_status, "updated_by": user_id,
                         "updated_at": datetime.now(timezone.utc).isoformat()}
            if role_name is not None:
                overrides["role_name"] = role_name
                overrides["normalization_status"] = "manual_override"
            if role_id is not None:
                overrides["role_id"] = role_id
            new_roles.append(RoleAssignment(**{**ra.model_dump(), **overrides}))
        else:
            new_roles.append(ra)

    if not found:
        raise KeyError(f"Role assignment '{assignment_id}' not found on entity '{entity_name}'.")

    updated_entity = Entity(**{**entity.model_dump(), "role_assignments": new_roles})
    graph.update_entity(entity_name, entity_type, updated_entity)
    graph.save_graph()

    refreshed = graph.get_entity(updated_entity.entity_name, updated_entity.entity_type)
    return [{
        "entity_name": refreshed.entity_name,
        "entity_type": refreshed.entity_type,
        "description": refreshed.description,
        "source_ids": sorted(refreshed.source_ids),
        "node_name": graph.get_node_name_from_entity(refreshed),
        "role_assignments": [ra.model_dump() for ra in refreshed.role_assignments],
    }]


async def review_role_assignment(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str, review_status: str,
    role_name: Optional[str], role_id: Optional[str], user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _review_role_sync,
            tenant_id, doc_id, config_path, entity_name, entity_type,
            assignment_id, review_status, role_name, role_id, user_id,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "review_role", "doc_id": doc_id, "user_id": user_id,
        "entity": {"entity_name": entity_name, "entity_type": entity_type},
        "assignment_id": assignment_id, "review_status": review_status,
    })
    return result


def _delete_role_sync(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str,
) -> List[dict]:
    from Core.Index.Graph import Entity

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    entity = graph.get_entity(entity_name, entity_type)

    new_roles = [ra for ra in entity.role_assignments if ra.assignment_id != assignment_id]
    if len(new_roles) == len(entity.role_assignments):
        raise KeyError(f"Role assignment '{assignment_id}' not found on entity '{entity_name}'.")

    updated_entity = Entity(**{**entity.model_dump(), "role_assignments": new_roles})
    graph.update_entity(entity_name, entity_type, updated_entity)
    graph.save_graph()

    refreshed = graph.get_entity(updated_entity.entity_name, updated_entity.entity_type)
    return [{
        "entity_name": refreshed.entity_name,
        "entity_type": refreshed.entity_type,
        "description": refreshed.description,
        "source_ids": sorted(refreshed.source_ids),
        "node_name": graph.get_node_name_from_entity(refreshed),
        "role_assignments": [ra.model_dump() for ra in refreshed.role_assignments],
    }]


async def delete_role_assignment(
    tenant_id: str, doc_id: str, config_path: str,
    entity_name: str, entity_type: str,
    assignment_id: str, user_id: str,
) -> List[dict]:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _delete_role_sync,
            tenant_id, doc_id, config_path, entity_name, entity_type, assignment_id,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "delete_role", "doc_id": doc_id, "user_id": user_id,
        "entity": {"entity_name": entity_name, "entity_type": entity_type},
        "assignment_id": assignment_id,
    })
    return result


# ── Role curation export ──────────────────────────────────────────────────────

def _list_roles_sync(
    tenant_id: str, doc_id: str, config_path: str,
    review_status: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> List[dict]:
    """Collect all role assignments across every entity in the graph.

    Optionally filter by ``review_status`` and/or ``entity_type``.
    Returns a flat list of dicts suitable for the ``RoleListResponse.roles`` field.
    """
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    rows: List[dict] = []
    for _, node_data in graph.kg.nodes(data=True):
        ent_name = node_data.get("entity_name", "")
        ent_type = node_data.get("entity_type", "")
        if entity_type and ent_type.upper() != entity_type.upper():
            continue
        role_assignments = node_data.get("role_assignments", [])
        for ra in role_assignments:
            # ra may be a RoleAssignment object or a plain dict (depending on graph state)
            if hasattr(ra, "model_dump"):
                ra_dict = ra.model_dump()
            else:
                ra_dict = dict(ra)
            if review_status and ra_dict.get("review_status") != review_status:
                continue
            rows.append({
                "entity_name": ent_name,
                "entity_type": ent_type,
                **ra_dict,
            })
    return rows


def _bulk_review_roles_sync(
    tenant_id: str, doc_id: str, config_path: str,
    reviews: List[dict], user_id: str,
) -> dict:
    """Apply multiple role review actions in a single graph load/save cycle.

    ``reviews`` is a list of dicts with keys:
        entity_name, entity_type, assignment_id, review_status,
        role_name (optional), role_id (optional)

    Returns ``{"processed": int, "errors": list[dict]}``.
    """
    from datetime import datetime, timezone
    from Core.Index.Graph import Entity, RoleAssignment

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    processed = 0
    errors: List[dict] = []
    now = datetime.now(timezone.utc).isoformat()

    for item in reviews:
        ent_name = item["entity_name"]
        ent_type = item["entity_type"]
        assignment_id = item["assignment_id"]
        new_status = item["review_status"]
        override_name = item.get("role_name")
        override_id = item.get("role_id")

        try:
            entity = graph.get_entity(ent_name, ent_type)
            found = False
            new_roles = []
            for ra in entity.role_assignments:
                if ra.assignment_id != assignment_id:
                    new_roles.append(ra)
                    continue

                found = True
                overrides = {
                    "review_status": new_status,
                    "updated_by": user_id,
                    "updated_at": now,
                }
                if override_name is not None:
                    overrides["role_name"] = override_name
                    overrides["normalization_status"] = "manual_override"
                if override_id is not None:
                    overrides["role_id"] = override_id
                    overrides["normalization_status"] = "manual_override"
                new_roles.append(RoleAssignment(**{**ra.model_dump(), **overrides}))

            if not found:
                errors.append({
                    "assignment_id": assignment_id,
                    "entity_name": ent_name,
                    "error": "assignment_id not found",
                })
                continue
            updated_entity = Entity(**{**entity.model_dump(), "role_assignments": new_roles})
            graph.update_entity(ent_name, ent_type, updated_entity)
            processed += 1
        except KeyError:
            errors.append({
                "assignment_id": assignment_id,
                "entity_name": ent_name,
                "error": f"entity '{ent_name}' ({ent_type}) not found",
            })
        except Exception as exc:
            errors.append({
                "assignment_id": assignment_id,
                "entity_name": ent_name,
                "error": str(exc),
            })

    if processed:
        graph.save_graph()

    return {"processed": processed, "errors": errors}


async def list_roles(
    tenant_id: str, doc_id: str, config_path: str,
    review_status: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _list_roles_sync,
        tenant_id, doc_id, config_path, review_status, entity_type,
    )


async def bulk_review_roles(
    tenant_id: str, doc_id: str, config_path: str,
    reviews: List[dict], user_id: str,
) -> dict:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, _bulk_review_roles_sync,
            tenant_id, doc_id, config_path, reviews, user_id,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "bulk_review_roles", "doc_id": doc_id, "user_id": user_id,
        "processed": result["processed"],
    })
    return result


def _re_normalize_roles_sync(
    tenant_id: str,
    doc_id: str,
    config_path: str,
    user_id: str,
    review_status: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> dict:
    from datetime import datetime, timezone
    from Core.Index.Graph import Entity
    from Core.utils.role_normalizer import apply_normalization

    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    processed = 0
    updated = 0
    skipped = 0
    unresolved = 0
    graph_changed = False
    now = datetime.now(timezone.utc).isoformat()

    for node_name in graph.get_all_nodes():
        entity = graph.get_entity_by_node_name(node_name)
        if entity_type and entity.entity_type.upper() != entity_type.upper():
            continue

        entity_changed = False
        new_roles = []
        for assignment in entity.role_assignments:
            if review_status and assignment.review_status != review_status:
                new_roles.append(assignment)
                continue

            if assignment.normalization_status == "manual_override" or (
                assignment.review_status == "confirmed" and assignment.role_id
            ):
                skipped += 1
                new_roles.append(assignment)
                continue

            processed += 1
            normalized = apply_normalization(assignment)
            if normalized.model_dump() != assignment.model_dump():
                normalized = normalized.model_copy(update={"updated_by": user_id, "updated_at": now})
                updated += 1
                entity_changed = True
                graph_changed = True
            if normalized.normalization_status == "unresolved":
                unresolved += 1
            new_roles.append(normalized)

        if entity_changed:
            graph.update_entity(
                entity.entity_name,
                entity.entity_type,
                Entity(**{**entity.model_dump(), "role_assignments": new_roles}),
            )

    if graph_changed:
        graph.save_graph()

    return {
        "doc_id": doc_id,
        "processed": processed,
        "updated": updated,
        "skipped": skipped,
        "unresolved": unresolved,
    }


async def re_normalize_roles(
    tenant_id: str,
    doc_id: str,
    config_path: str,
    user_id: str,
    review_status: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> dict:
    async with _get_lock(tenant_id, doc_id):
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _re_normalize_roles_sync,
            tenant_id,
            doc_id,
            config_path,
            user_id,
            review_status,
            entity_type,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "re_normalize_roles",
        "doc_id": doc_id,
        "user_id": user_id,
        "processed": result["processed"],
        "updated": result["updated"],
        "review_status": review_status,
        "entity_type": entity_type,
    })
    return result


def _role_stats_sync(
    tenant_id: str,
    doc_id: str,
    config_path: str,
) -> dict:
    graph, _, _ = _load_graph_sync(tenant_id, doc_id, config_path)
    roles = _list_roles_sync(tenant_id, doc_id, config_path)
    total_entities = len(list(graph.get_all_nodes()))
    entities_with_roles = len({(row["entity_name"], row["entity_type"]) for row in roles})

    def _count(field: str, fallback: str = "unknown") -> dict[str, int]:
        return dict(Counter((row.get(field) or fallback) for row in roles))

    return {
        "doc_id": doc_id,
        "total_entities": total_entities,
        "entities_with_roles": entities_with_roles,
        "coverage_ratio": (entities_with_roles / total_entities) if total_entities else 0.0,
        "total_roles": len(roles),
        "unresolved_roles": sum(1 for row in roles if row.get("normalization_status") == "unresolved"),
        "review_status_counts": _count("review_status"),
        "normalization_status_counts": _count("normalization_status"),
        "origin_counts": _count("origin"),
        "tenure_status_counts": _count("tenure_status"),
    }


async def role_stats(
    tenant_id: str,
    doc_id: str,
    config_path: str,
) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _role_stats_sync,
        tenant_id,
        doc_id,
        config_path,
    )


def _list_role_vocab_sync() -> List[dict]:
    from Core.utils.role_normalizer import list_vocab_entries

    return list_vocab_entries()


async def list_role_vocab() -> List[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _list_role_vocab_sync)


def _create_role_vocab_entry_sync(role_input: dict) -> dict:
    from Core.utils.role_normalizer import add_vocab_entry

    return add_vocab_entry(
        role_id=role_input["role_id"],
        canonical=role_input["canonical"],
        aliases=role_input.get("aliases") or [],
    )


async def create_role_vocab_entry(tenant_id: str, role_input: dict, user_id: str) -> dict:
    async with _role_vocab_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _create_role_vocab_entry_sync, role_input)
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "create_role_vocab_entry",
        "doc_id": None,
        "user_id": user_id,
        "role_id": result["role_id"],
    })
    return result


def _update_role_vocab_entry_sync(role_id: str, update_fields: dict) -> dict:
    from Core.utils.role_normalizer import update_vocab_entry

    return update_vocab_entry(
        role_id,
        canonical=update_fields.get("canonical"),
        aliases=update_fields.get("aliases"),
    )


async def update_role_vocab_entry(
    tenant_id: str,
    role_id: str,
    update_fields: dict,
    user_id: str,
) -> dict:
    async with _role_vocab_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _update_role_vocab_entry_sync,
            role_id,
            update_fields,
        )
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "update_role_vocab_entry",
        "doc_id": None,
        "user_id": user_id,
        "role_id": role_id,
    })
    return result


def _reload_role_vocab_sync() -> List[dict]:
    from Core.utils.role_normalizer import reload_vocab

    return reload_vocab()


async def reload_role_vocab(tenant_id: str, user_id: str) -> List[dict]:
    async with _role_vocab_lock:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, _reload_role_vocab_sync)
    await db.log_entity_edit(MONGO_URI, MONGO_DB_PREFIX, tenant_id, {
        "operation": "reload_role_vocab",
        "doc_id": None,
        "user_id": user_id,
        "total": len(result),
    })
    return result

