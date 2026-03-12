"""Entity management router: list, rename, merge, split, suggest-merges."""
import logging
import os

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.models.requests import (
    EntityListResponse, EntityInfo,
    EntityOperationResponse,
    RenameEntityRequest,
    MergeEntitiesRequest,
    SplitEntityRequest,
    SuggestMergesResponse, MergeSuggestion, EntityRef,
    AddRoleRequest, UpdateRoleRequest, ReviewRoleRequest,
    RoleListResponse, BulkReviewRolesRequest, BulkReviewRolesResponse,
)
from api.dependencies import get_current_user, check_doc_access
import api.services.entity_editor as svc

log = logging.getLogger(__name__)
router = APIRouter(prefix="/entities", tags=["entities"])

CONFIG_PATH = os.getenv("BOOKRAG_CONFIG_PATH", "config/gbc.yaml")


async def _require_access(tenant_id: str, user_id: str, doc_id: str):
    if not await check_doc_access(user_id, tenant_id, doc_id):
        raise HTTPException(status_code=403, detail="Access denied to this document")


# ── List ──────────────────────────────────────────────────────────────────────

@router.get(
    "/{doc_id}",
    response_model=EntityListResponse,
    summary="List document entities",
    description="Return all extracted entities for a document the caller can access.",
    responses={
        403: {"description": "Access denied to this document."},
        500: {"description": "Entity listing failed."},
    },
)
async def list_entities(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Return all NER entities for the given document."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        entities = await svc.list_entities(tenant_id, doc_id, CONFIG_PATH)
    except Exception as exc:
        log.exception(f"list_entities failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityListResponse(
        entities=[EntityInfo(**e) for e in entities],
        total=len(entities),
    )


# ── Rename ────────────────────────────────────────────────────────────────────

@router.patch(
    "/{doc_id}/rename",
    response_model=EntityOperationResponse,
    summary="Rename an entity",
    description="Rename an entity node and optionally update its type and description.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity not found."},
        500: {"description": "Entity rename failed."},
    },
)
async def rename_entity(
    doc_id: str,
    req: RenameEntityRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename an entity node (name and/or type)."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        updated = await svc.rename_entity(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            new_entity_name=req.new_entity_name,
            new_entity_type=req.new_entity_type,
            new_description=req.new_description,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"rename_entity failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Renamed '{req.entity_name}' → '{req.new_entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


# ── Merge ─────────────────────────────────────────────────────────────────────

@router.post(
    "/{doc_id}/merge",
    response_model=EntityOperationResponse,
    summary="Merge entities",
    description="Merge two or more entities into a single canonical entity.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "One or more entities were not found."},
        422: {"description": "At least two source entities are required."},
        500: {"description": "Entity merge failed."},
    },
)
async def merge_entities(
    doc_id: str,
    req: MergeEntitiesRequest,
    current_user: dict = Depends(get_current_user),
):
    """Merge two or more entities into a single canonical entity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if len(req.source_entities) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 source_entities to merge")

    try:
        updated = await svc.merge_entities(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            source_entities=[e.model_dump() for e in req.source_entities],
            canonical_name=req.canonical_entity_name,
            canonical_type=req.canonical_entity_type,
            canonical_desc=req.canonical_description,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"merge_entities failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Merged {len(req.source_entities)} entities → '{req.canonical_entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


# ── Split ─────────────────────────────────────────────────────────────────────

@router.post(
    "/{doc_id}/split",
    response_model=EntityOperationResponse,
    summary="Split an entity",
    description="Split one existing entity into two or more new entities.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity not found."},
        422: {"description": "At least two new entities are required and `edge_mode` must be valid."},
        500: {"description": "Entity split failed."},
    },
)
async def split_entity(
    doc_id: str,
    req: SplitEntityRequest,
    current_user: dict = Depends(get_current_user),
):
    """Split one entity into two or more new entities."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if len(req.new_entities) < 2:
        raise HTTPException(status_code=422, detail="Provide at least 2 new_entities for a split")
    if req.edge_mode not in ("duplicate", "none"):
        raise HTTPException(status_code=422, detail="edge_mode must be 'duplicate' or 'none'")

    try:
        created = await svc.split_entity(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            new_entities=[e.model_dump() for e in req.new_entities],
            edge_mode=req.edge_mode,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"split_entity failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Split '{req.entity_name}' into {len(created)} entities",
        entities=[EntityInfo(**e) for e in created],
    )


# ── Domain role assignments ───────────────────────────────────────────────────

@router.post(
    "/{doc_id}/roles",
    response_model=EntityOperationResponse,
    summary="Add a domain role assignment",
    description="Manually add a domain-level role assignment (e.g. President, CEO) to an entity.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity not found."},
        500: {"description": "Role assignment creation failed."},
    },
)
async def add_role_assignment(
    doc_id: str,
    req: AddRoleRequest,
    current_user: dict = Depends(get_current_user),
):
    """Add a manual domain role assignment to an entity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        updated = await svc.add_role_assignment(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            role_input=req.role_assignment.model_dump(),
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"add_role_assignment failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Added role '{req.role_assignment.role_name}' to '{req.entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


@router.patch(
    "/{doc_id}/roles/{assignment_id}",
    response_model=EntityOperationResponse,
    summary="Update a domain role assignment",
    description="Update fields on an existing role assignment by its assignment_id.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity or role assignment not found."},
        500: {"description": "Role assignment update failed."},
    },
)
async def update_role_assignment(
    doc_id: str,
    assignment_id: str,
    req: UpdateRoleRequest,
    current_user: dict = Depends(get_current_user),
):
    """Update an existing domain role assignment."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    update_fields = req.model_dump(exclude={"entity_name", "entity_type"}, exclude_none=True)

    try:
        updated = await svc.update_role_assignment(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            assignment_id=assignment_id, update_fields=update_fields,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"update_role_assignment failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Updated role assignment '{assignment_id}' on '{req.entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


@router.post(
    "/{doc_id}/roles/{assignment_id}/review",
    response_model=EntityOperationResponse,
    summary="Review a domain role assignment",
    description="Set review status (confirmed/disputed/rejected) on a role assignment, optionally overriding its canonical role name.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity or role assignment not found."},
        422: {"description": "Invalid review_status value."},
        500: {"description": "Role assignment review failed."},
    },
)
async def review_role_assignment(
    doc_id: str,
    assignment_id: str,
    req: ReviewRoleRequest,
    current_user: dict = Depends(get_current_user),
):
    """Set review state on a domain role assignment."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    valid_statuses = {"confirmed", "disputed", "rejected"}
    if req.review_status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"review_status must be one of: {', '.join(sorted(valid_statuses))}",
        )

    try:
        updated = await svc.review_role_assignment(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=req.entity_name, entity_type=req.entity_type,
            assignment_id=assignment_id, review_status=req.review_status,
            role_name=req.role_name, role_id=req.role_id,
            user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"review_role_assignment failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Role assignment '{assignment_id}' marked '{req.review_status}'",
        entities=[EntityInfo(**e) for e in updated],
    )


@router.delete(
    "/{doc_id}/roles/{assignment_id}",
    response_model=EntityOperationResponse,
    summary="Delete a domain role assignment",
    description="Permanently remove a role assignment from an entity by its assignment_id.",
    responses={
        403: {"description": "Access denied to this document."},
        404: {"description": "Entity or role assignment not found."},
        500: {"description": "Role assignment deletion failed."},
    },
)
async def delete_role_assignment(
    doc_id: str,
    assignment_id: str,
    entity_name: str = Query(..., description="Name of the entity owning the assignment."),
    entity_type: str = Query(..., description="Type of the entity owning the assignment."),
    current_user: dict = Depends(get_current_user),
):
    """Delete a domain role assignment from an entity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        updated = await svc.delete_role_assignment(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            entity_name=entity_name, entity_type=entity_type,
            assignment_id=assignment_id, user_id=user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        log.exception(f"delete_role_assignment failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return EntityOperationResponse(
        success=True,
        message=f"Deleted role assignment '{assignment_id}' from '{entity_name}'",
        entities=[EntityInfo(**e) for e in updated],
    )


# ── Role curation export ──────────────────────────────────────────────────────

@router.get(
    "/{doc_id}/roles",
    response_model=RoleListResponse,
    summary="List role assignments",
    description=(
        "Return all domain role assignments for a document, optionally filtered by "
        "review_status (e.g. 'suggested', 'confirmed', 'disputed', 'rejected') and/or entity_type."
    ),
    responses={
        403: {"description": "Access denied to this document."},
        500: {"description": "Internal server error."},
    },
)
async def list_role_assignments(
    doc_id: str,
    review_status: Optional[str] = Query(
        None,
        description="Filter by review status: suggested | confirmed | disputed | rejected.",
    ),
    entity_type: Optional[str] = Query(
        None,
        description="Filter by entity type, e.g. PERSON or ORGANIZATION.",
    ),
    current_user: dict = Depends(get_current_user),
):
    """Return all role assignments for a document, with optional filters."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        roles = await svc.list_roles(
            tenant_id=tenant_id,
            doc_id=doc_id,
            config_path=CONFIG_PATH,
            review_status=review_status,
            entity_type=entity_type,
        )
    except Exception as exc:
        log.exception(f"list_role_assignments failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return RoleListResponse(doc_id=doc_id, total=len(roles), roles=roles)


@router.post(
    "/{doc_id}/roles/bulk-review",
    response_model=BulkReviewRolesResponse,
    summary="Bulk-review role assignments",
    description=(
        "Set review status (confirmed / disputed / rejected) on multiple role assignments "
        "in a single request. Partial failures are reported per-item without aborting the "
        "rest of the batch."
    ),
    responses={
        403: {"description": "Access denied to this document."},
        500: {"description": "Internal server error."},
    },
)
async def bulk_review_role_assignments(
    doc_id: str,
    req: BulkReviewRolesRequest,
    current_user: dict = Depends(get_current_user),
):
    """Bulk-set review status on a list of role assignments."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    try:
        result = await svc.bulk_review_roles(
            tenant_id=tenant_id,
            doc_id=doc_id,
            config_path=CONFIG_PATH,
            reviews=[item.model_dump() for item in req.reviews],
            user_id=user_id,
        )
    except Exception as exc:
        log.exception(f"bulk_review_role_assignments failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    ok = result["processed"]
    errs = result["errors"]
    return BulkReviewRolesResponse(
        success=ok > 0 or not errs,
        message=f"Processed {ok} assignment(s); {len(errs)} error(s).",
        processed=ok,
        errors=errs,
    )


# ── Suggest merges ────────────────────────────────────────────────────────────

@router.get(
    "/{doc_id}/suggestions",
    response_model=SuggestMergesResponse,
    summary="Suggest entity merges",
    description="Return ranked merge candidates for a document using string similarity or optional embedding similarity.",
    responses={
        403: {"description": "Access denied to this document."},
        422: {"description": "Invalid suggestion query parameters."},
        500: {"description": "Suggestion generation failed."},
    },
)
async def suggest_merges(
    doc_id: str,
    min_score: float = Query(default=0.80, ge=0.0, le=1.0, description="Minimum similarity score required to return a suggestion."),
    top_k: int = Query(default=50, ge=1, description="Maximum number of merge suggestions to return."),
    use_embeddings: bool = Query(default=False, description="Also use embedding similarity when available."),
    current_user: dict = Depends(get_current_user),
):
    """Return ranked merge-candidate pairs based on string/embedding similarity."""
    tenant_id = current_user["tenant_id"]
    user_id = current_user["user_id"]
    await _require_access(tenant_id, user_id, doc_id)

    if not (0.0 <= min_score <= 1.0):
        raise HTTPException(status_code=422, detail="min_score must be between 0.0 and 1.0")

    try:
        raw = await svc.suggest_merges(
            tenant_id=tenant_id, doc_id=doc_id, config_path=CONFIG_PATH,
            min_score=min_score, top_k=top_k, use_embeddings=use_embeddings,
        )
    except Exception as exc:
        log.exception(f"suggest_merges failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return SuggestMergesResponse(suggestions=[
        MergeSuggestion(
            entity_a=EntityRef(**s["entity_a"]),
            entity_b=EntityRef(**s["entity_b"]),
            score=s["score"],
            method=s["method"],
        )
        for s in raw
    ])

