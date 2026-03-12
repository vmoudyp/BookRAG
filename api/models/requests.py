"""Pydantic request and response models for the BookRAG API."""
from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Reusable length constraints ──────────────────────────────────────────────
_SHORT_STR = 128        # usernames, tenant_ids, role names
_PASSWORD_MIN = 8
_PASSWORD_MAX = 128
_QUERY_MAX = 10_000     # max characters for a chat query


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"username": "alice", "password": "<redacted>", "tenant_id": "tenant-a"}  # NOSONAR
            ]
        }
    )

    username: str = Field(..., min_length=3, max_length=_SHORT_STR, description="Username within the tenant.")
    password: str = Field(
        ...,
        min_length=_PASSWORD_MIN,
        max_length=_PASSWORD_MAX,
        description="Password with at least one uppercase letter, one lowercase letter, and one digit.",
    )
    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Tenant identifier.")

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v: str) -> str:
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class LoginRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"username": "alice", "password": "<redacted>", "tenant_id": "tenant-a"}  # NOSONAR
            ]
        }
    )

    username: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Username within the tenant.")
    password: str = Field(..., min_length=1, max_length=_PASSWORD_MAX, description="User password.")
    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Tenant identifier.")


class TokenResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "access_token": "access.jwt.token",
                    "refresh_token": "refresh.jwt.token",
                    "token_type": "bearer",
                }
            ]
        }
    )

    access_token: str = Field(..., description="Short-lived JWT access token (default 60 min)")
    refresh_token: str = Field(..., description="Long-lived refresh token for rotation (default 7 days)")
    token_type: str = Field(default="bearer", description="OAuth2 token type")


class RefreshRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"refresh_token": "refresh.jwt.token"}
            ]
        }
    )

    refresh_token: str = Field(..., description="Previously issued refresh token.")


# ── Tenant ────────────────────────────────────────────────────────────────────

class TenantCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tenant_id": "tenant-a",
                    "name": "Tenant A",
                    "description": "Internal business unit workspace.",
                }
            ]
        }
    )

    tenant_id: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Unique tenant identifier.")
    name: str = Field(..., min_length=1, max_length=256, description="Human-readable tenant name.")
    description: Optional[str] = Field(default="", max_length=1000, description="Optional tenant description.")


class TenantResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "tenant_id": "tenant-a",
                    "name": "Tenant A",
                    "description": "Internal business unit workspace.",
                }
            ]
        }
    )

    tenant_id: str = Field(..., description="Tenant identifier.")
    name: str = Field(..., description="Human-readable tenant name.")
    description: Optional[str] = Field(default="", description="Optional tenant description.")


class SimpleMessageResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "Operation completed successfully"}
            ]
        }
    )

    message: str = Field(..., description="Human-readable operation status message.")


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "doc_id": "doc-123",
                    "filename": "report.pdf",
                    "status": "ready",
                    "error": None,
                    "created_at": "2026-03-11T12:00:00Z",
                    "document_date": "2025-06-15T00:00:00Z",
                    "document_lang": "en",
                }
            ]
        }
    )

    doc_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Indexing status: pending | indexing | ready | error")
    error: Optional[str] = Field(default=None, description="Error message if status is 'error'")
    created_at: Optional[datetime] = Field(default=None, description="Upload timestamp (UTC)")
    document_date: Optional[datetime] = Field(
        default=None,
        description="User-provided original authoring/publishing date of the document. "
                    "Used for temporal awareness in cross-document RAG.",
    )
    document_lang: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'en', 'id') or 'auto' for auto-detection. "
                    "Used for legal heading detection and language-aware text processing.",
    )


class BatchUploadResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "uploaded": [
                        {
                            "doc_id": "doc-123",
                            "filename": "report.pdf",
                            "status": "pending",
                            "document_date": "2025-06-15T00:00:00Z",
                            "document_lang": "en",
                        }
                    ],
                    "failed": [
                        {
                            "filename": "notes.txt",
                            "error": "Only PDF files are supported",
                        }
                    ],
                }
            ]
        }
    )

    uploaded: List["DocumentResponse"] = Field(..., description="Successfully uploaded documents")
    failed: List[dict] = Field(default_factory=list, description="Files that failed: [{filename, error}]")


class PermissionGrantRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"user_id": "alice", "doc_id": "doc-123", "role": "reader"}
            ]
        }
    )

    user_id: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Target user identifier.")
    doc_id: str = Field(..., min_length=1, max_length=_SHORT_STR, description="Target document identifier.")
    role: str = Field(default="reader", max_length=32, description="Permission role to grant, for example `reader` or `owner`.")


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatQueryRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What does the revenue chart show?",
                    "doc_ids": ["doc-123"],
                    "cross_doc": False,
                    "visual_sidecar_query_enabled": True,
                    "visual_sidecar_query_topk": 5,
                    "visual_sidecar_fusion_enabled": True,
                    "visual_sidecar_fusion_weight": 1.25,
                    "visual_sidecar_fusion_score_mode": "max_norm",
                    "visual_sidecar_fusion_min_score": 0.2,
                }
            ]
        }
    )

    query: str = Field(..., min_length=1, max_length=_QUERY_MAX, description="User question.")
    session_id: Optional[str] = Field(
        default=None,
        max_length=_SHORT_STR,
        description="Existing session ID for history-aware queries.",
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to specific documents. `null` means all documents the caller can access.",
    )
    cross_doc: bool = Field(default=False, description="Enable cross-document retrieval mode.")
    visual_sidecar_query_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Optional per-request override for query-time visual sidecar retrieval. "
            "Omit or set to `null` to keep the loaded config value."
        ),
    )
    visual_sidecar_query_topk: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Optional per-request override for how many visual sidecar hits to retrieve "
            "before fusion and augmentation. Omit or set to `null` to keep the loaded config value."
        ),
    )
    visual_sidecar_fusion_enabled: Optional[bool] = Field(
        default=None,
        description=(
            "Optional per-request override for allowing visual sidecar scores to participate "
            "in skyline fusion. Omit or set to `null` to keep the loaded config value."
        ),
    )
    visual_sidecar_fusion_weight: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional per-request override for the visual sidecar fusion weight. "
            "Only relevant when fusion is enabled. Omit or set to `null` to keep the loaded config value."
        ),
    )
    visual_sidecar_fusion_score_mode: Optional[Literal["raw", "max_norm", "rank"]] = Field(
        default=None,
        description=(
            "Optional per-request override for visual score calibration before skyline fusion. "
            "Supported values: `raw`, `max_norm`, `rank`. Omit or set to `null` to keep the loaded config value."
        ),
    )
    visual_sidecar_fusion_min_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Optional per-request override for the minimum calibrated visual score used in fusion. "
            "Omit or set to `null` to keep the loaded config value."
        ),
    )


class ChatQueryResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "answer": "The revenue chart shows quarter-over-quarter growth.",
                    "session_id": "session-123",
                    "doc_ids_used": ["doc-123"],
                    "rewritten_query": None,
                }
            ]
        }
    )

    answer: str = Field(..., description="LLM-generated answer")
    session_id: str = Field(..., description="Session ID (created if not provided)")
    doc_ids_used: List[str] = Field(default_factory=list, description="Document IDs used for retrieval")
    rewritten_query: Optional[str] = Field(default=None, description="Rewritten query when history was used")


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"doc_ids": ["doc-123", "doc-456"]}
            ]
        }
    )

    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional document IDs to attach to the session. Requested IDs are filtered to documents the caller can access.",
    )


class SessionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"session_id": "session-123"}
            ]
        }
    )

    session_id: str = Field(..., description="Created session identifier.")


class SessionListItem(BaseModel):
    session_id: str = Field(..., description="Session identifier.")
    created_at: Optional[datetime] = Field(default=None, description="Session creation timestamp when available.")
    message_count: int = Field(default=0, description="Number of messages currently stored in the session.")
    doc_ids: List[str] = Field(default_factory=list, description="Document IDs associated with the session.")


class SessionListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sessions": [
                        {
                            "session_id": "session-123",
                            "created_at": "2026-03-11T12:00:00Z",
                            "message_count": 4,
                            "doc_ids": ["doc-123"],
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    )

    sessions: List[SessionListItem] = Field(..., description="Paginated session records for the current user.")
    total: int = Field(..., description="Total number of sessions before pagination.")


class MessageResponse(BaseModel):
    role: str = Field(..., description="Message author role, typically `user` or `assistant`.")
    content: str = Field(..., description="Message content.")
    ts: Optional[str] = Field(default=None, description="Serialized message timestamp when available.")


class SessionMessagesResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "session_id": "session-123",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Summarize the report",
                            "ts": "2026-03-11T12:00:00Z",
                        },
                        {
                            "role": "assistant",
                            "content": "The report focuses on revenue growth and cost changes.",
                            "ts": "2026-03-11T12:00:03Z",
                        },
                    ],
                    "total": 2,
                }
            ]
        }
    )

    session_id: str = Field(..., description="Session identifier.")
    messages: List[MessageResponse] = Field(..., description="Paginated messages for the session.")
    total: int = Field(0, description="Total messages in session (before pagination)")


# ── Domain Role Assignments ───────────────────────────────────────────────────

class RoleEvidenceInfo(BaseModel):
    """Response model: a single evidence item supporting a role assignment."""

    source_id: int = Field(..., description="Tree/source node ID containing this evidence.")
    observed_role_text: str = Field(..., description="Literal wording from the source document.")
    evidence_text: Optional[str] = Field(None, description="Short excerpt supporting the role.")
    language: Optional[str] = Field(None, description="Language code, e.g. 'id', 'en'.")
    normalization_method: str = Field(..., description="How observed text was mapped to canonical role.")
    confidence: Optional[float] = Field(None, description="Per-evidence confidence score.")


class RoleAssignmentInfo(BaseModel):
    """Response model: a domain-level role assignment on an entity."""

    assignment_id: str = Field(..., description="Stable UUID for this assignment.")
    role_name: str = Field(..., description="Canonical role label, e.g. 'President'.")
    role_id: Optional[str] = Field(None, description="Optional stable vocabulary role identifier.")
    scope_entity_name: Optional[str] = Field(None, description="Context entity, e.g. 'Indonesia'.")
    scope_entity_type: Optional[str] = Field(None, description="Type of scope entity, e.g. 'COUNTRY'.")
    scope_entity_id: Optional[str] = Field(None, description="Optional canonical ID for the scope entity.")
    start_date: Optional[str] = Field(None, description="Start date in partial ISO format: YYYY, YYYY-MM, or YYYY-MM-DD.")
    end_date: Optional[str] = Field(None, description="End date in the same format as start_date.")
    tenure_status: str = Field(..., description="Real-world tenure state: current/former/acting/interim/candidate/unknown.")
    origin: str = Field(..., description="How the assignment was created: extracted or manual.")
    review_status: str = Field(..., description="Curation state: suggested/confirmed/disputed/rejected.")
    confidence: Optional[float] = Field(None, description="Overall assignment confidence score.")
    normalization_status: str = Field(..., description="Normalization outcome: matched/ambiguous/unresolved/manual_override.")
    normalization_confidence: Optional[float] = Field(None, description="Confidence of role-name normalization.")
    source_ids: List[int] = Field(..., description="Supporting tree/source node IDs.")
    evidence: List[RoleEvidenceInfo] = Field(default_factory=list, description="Evidence items supporting this assignment.")


class RoleEvidenceInput(BaseModel):
    """Input model: evidence item for a new or updated role assignment."""

    source_id: int = Field(..., description="Tree/source node ID containing this evidence.")
    observed_role_text: str = Field(..., description="Literal wording as it appears in the document.")
    evidence_text: Optional[str] = Field(None, description="Short excerpt supporting the role.")
    language: Optional[str] = Field(None, description="Language code, e.g. 'id', 'en'.")
    normalization_method: str = Field("manual", description="Normalization method used.")
    confidence: Optional[float] = Field(None, description="Per-evidence confidence score.")


class RoleAssignmentInput(BaseModel):
    """Input model: payload for adding a new role assignment to an entity."""

    role_name: str = Field(..., description="Canonical role label, e.g. 'Koordinator Sektor'.")
    role_id: Optional[str] = Field(None, description="Optional stable vocabulary role identifier.")
    scope_entity_name: Optional[str] = Field(None, description="Context entity, e.g. 'Indonesia'.")
    scope_entity_type: Optional[str] = Field(None, description="Type of scope entity, e.g. 'COUNTRY'.")
    scope_entity_id: Optional[str] = Field(None, description="Optional canonical ID for the scope entity.")
    start_date: Optional[str] = Field(None, description="Start date in partial ISO format.")
    end_date: Optional[str] = Field(None, description="End date in partial ISO format.")
    tenure_status: str = Field("unknown", description="Tenure state: current/former/acting/interim/candidate/unknown.")
    source_ids: List[int] = Field(default_factory=list, description="Supporting tree/source node IDs.")
    evidence: List[RoleEvidenceInput] = Field(default_factory=list, description="Evidence items.")


class AddRoleRequest(BaseModel):
    """Request body for POST /{doc_id}/roles."""

    entity_name: str = Field(..., description="Name of the entity to add the role to.")
    entity_type: str = Field(..., description="Type of the entity.")
    role_assignment: RoleAssignmentInput = Field(..., description="Role assignment to add.")


class UpdateRoleRequest(BaseModel):
    """Request body for PATCH /{doc_id}/roles/{assignment_id}."""

    entity_name: str = Field(..., description="Name of the entity owning the assignment.")
    entity_type: str = Field(..., description="Type of the entity.")
    role_name: Optional[str] = Field(None, description="Updated canonical role label.")
    role_id: Optional[str] = Field(None, description="Updated vocabulary role identifier.")
    scope_entity_name: Optional[str] = Field(None, description="Updated scope entity name.")
    scope_entity_type: Optional[str] = Field(None, description="Updated scope entity type.")
    scope_entity_id: Optional[str] = Field(None, description="Updated scope entity canonical ID.")
    start_date: Optional[str] = Field(None, description="Updated start date.")
    end_date: Optional[str] = Field(None, description="Updated end date.")
    tenure_status: Optional[str] = Field(None, description="Updated tenure state.")


class ReviewRoleRequest(BaseModel):
    """Request body for POST /{doc_id}/roles/{assignment_id}/review."""

    entity_name: str = Field(..., description="Name of the entity owning the assignment.")
    entity_type: str = Field(..., description="Type of the entity.")
    review_status: str = Field(..., description="New review state: confirmed/disputed/rejected.")
    role_name: Optional[str] = Field(None, description="Override canonical role label during review.")
    role_id: Optional[str] = Field(None, description="Override vocabulary role identifier during review.")
    note: Optional[str] = Field(None, description="Optional internal reviewer note.")


# ── Entity Management ─────────────────────────────────────────────────────────

class EntityRef(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"entity_name": "Acme Corp", "entity_type": "organization"}
            ]
        }
    )

    entity_name: str = Field(..., description="Entity display name.")
    entity_type: str = Field(..., description="Entity type label.")


class EntityInfo(BaseModel):
    entity_name: str = Field(..., description="Entity display name.")
    entity_type: str = Field(..., description="Entity type label.")
    description: str = Field(..., description="Entity description or canonical summary.")
    source_ids: List[int] = Field(..., description="Source node identifiers supporting this entity.")
    node_name: str = Field(..., description="Underlying graph node name.")
    role_assignments: List[RoleAssignmentInfo] = Field(
        default_factory=list,
        description="Domain-level role assertions for this entity (e.g. President, CEO).",
    )


class EntityListResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "entities": [
                        {
                            "entity_name": "Acme Corp",
                            "entity_type": "organization",
                            "description": "Manufacturer mentioned in the report.",
                            "source_ids": [12, 19],
                            "node_name": "organization::Acme Corp",
                        }
                    ],
                    "total": 1,
                }
            ]
        }
    )

    entities: List[EntityInfo] = Field(..., description="Entities extracted for the document.")
    total: int = Field(..., description="Total number of returned entities.")


class RenameEntityRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "entity_name": "ACME",
                    "entity_type": "organization",
                    "new_entity_name": "Acme Corp",
                    "new_entity_type": "organization",
                    "new_description": "Canonical organization entity.",
                }
            ]
        }
    )

    entity_name: str = Field(..., description="Current entity name.")
    entity_type: str = Field(..., description="Current entity type.")
    new_entity_name: str = Field(..., description="New canonical entity name.")
    new_entity_type: str = Field(default="", description="New entity type. Empty string keeps the existing type.")
    new_description: Optional[str] = Field(default=None, description="Optional replacement description. `null` keeps the existing description.")


class MergeEntitiesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "source_entities": [
                        {"entity_name": "ACME", "entity_type": "organization"},
                        {"entity_name": "Acme Corp", "entity_type": "organization"},
                    ],
                    "canonical_entity_name": "Acme Corp",
                    "canonical_entity_type": "organization",
                    "canonical_description": "Canonical organization entity.",
                }
            ]
        }
    )

    source_entities: List[EntityRef] = Field(..., description="Entities to merge. Provide at least two.")
    canonical_entity_name: str = Field(..., description="Name of the merged canonical entity.")
    canonical_entity_type: str = Field(..., description="Type of the merged canonical entity.")
    canonical_description: str = Field(default="", description="Optional description for the merged canonical entity.")


class NewEntitySpec(BaseModel):
    entity_name: str = Field(..., description="Name of the new entity.")
    entity_type: str = Field(..., description="Type of the new entity.")
    description: str = Field(default="", description="Optional description for the new entity.")
    source_ids: List[int] = Field(default_factory=list, description="Supporting source node IDs for the new entity.")


class SplitEntityRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "entity_name": "Acme",
                    "entity_type": "organization",
                    "new_entities": [
                        {
                            "entity_name": "Acme Manufacturing",
                            "entity_type": "organization",
                            "description": "Manufacturing division.",
                            "source_ids": [12],
                        },
                        {
                            "entity_name": "Acme Logistics",
                            "entity_type": "organization",
                            "description": "Logistics division.",
                            "source_ids": [19],
                        },
                    ],
                    "edge_mode": "duplicate",
                }
            ]
        }
    )

    entity_name: str = Field(..., description="Existing entity name to split.")
    entity_type: str = Field(..., description="Existing entity type to split.")
    new_entities: List[NewEntitySpec] = Field(..., description="Replacement entities. Provide at least two.")
    edge_mode: str = Field(default="duplicate", description="How to handle edges for the new entities: `duplicate` or `none`.")


class MergeSuggestion(BaseModel):
    entity_a: EntityRef = Field(..., description="First candidate entity.")
    entity_b: EntityRef = Field(..., description="Second candidate entity.")
    score: float = Field(..., description="Similarity score between 0.0 and 1.0.")
    method: str = Field(..., description="Scoring method, for example `string_similarity` or `embedding_similarity`.")


class SuggestMergesResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "suggestions": [
                        {
                            "entity_a": {"entity_name": "ACME", "entity_type": "organization"},
                            "entity_b": {"entity_name": "Acme Corp", "entity_type": "organization"},
                            "score": 0.97,
                            "method": "string_similarity",
                        }
                    ]
                }
            ]
        }
    )

    suggestions: List[MergeSuggestion] = Field(..., description="Ranked merge suggestions for the document.")


class EntityOperationResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "success": True,
                    "message": "Renamed 'ACME' → 'Acme Corp'",
                    "entities": [
                        {
                            "entity_name": "Acme Corp",
                            "entity_type": "organization",
                            "description": "Canonical organization entity.",
                            "source_ids": [12, 19],
                            "node_name": "organization::Acme Corp",
                        }
                    ],
                }
            ]
        }
    )

    success: bool = Field(..., description="Whether the requested entity operation succeeded.")
    message: str = Field(..., description="Human-readable operation summary.")
    entities: List[EntityInfo] = Field(default_factory=list, description="Updated or created entities returned by the operation.")


class RoleListResponse(BaseModel):
    """Response for GET /{doc_id}/roles — filtered list of role assignments."""

    doc_id: str
    total: int = Field(..., description="Total number of role assignments returned.")
    roles: List[dict] = Field(
        default_factory=list,
        description=(
            "Each item contains entity_name, entity_type, and the role assignment "
            "fields (assignment_id, role_name, role_id, scope_entity_name, tenure_status, "
            "review_status, start_date, end_date, confidence, origin, source_ids)."
        ),
    )


class BulkReviewItem(BaseModel):
    """One item in a bulk-review request."""

    entity_name: str = Field(..., description="Name of the entity that owns the assignment.")
    entity_type: str = Field(..., description="Type of the entity.")
    assignment_id: str = Field(..., description="UUID of the role assignment to review.")
    review_status: str = Field(
        ...,
        description="New review status: confirmed | disputed | rejected.",
        pattern="^(confirmed|disputed|rejected)$",
    )
    role_name: Optional[str] = Field(
        None,
        description="Override canonical role name when confirming (optional).",
    )
    role_id: Optional[str] = Field(
        None,
        description="Override canonical role_id when confirming (optional).",
    )


class BulkReviewRolesRequest(BaseModel):
    """Request body for POST /{doc_id}/roles/bulk-review."""

    reviews: List[BulkReviewItem] = Field(
        ...,
        min_length=1,
        description="List of role assignments to review in one shot.",
    )


class BulkReviewRolesResponse(BaseModel):
    """Response for POST /{doc_id}/roles/bulk-review."""

    success: bool
    message: str
    processed: int = Field(..., description="Number of assignments successfully processed.")
    errors: List[dict] = Field(
        default_factory=list,
        description="Any per-item errors; items with errors are skipped but the rest succeed.",
    )


class ReNormalizeRolesRequest(BaseModel):
    """Request body for POST /{doc_id}/roles/re-normalize."""

    review_status: Optional[str] = Field(
        None,
        description="Optionally restrict re-normalization to one review state.",
    )
    entity_type: Optional[str] = Field(
        None,
        description="Optionally restrict re-normalization to one entity type.",
    )


class ReNormalizeRolesResponse(BaseModel):
    """Response for POST /{doc_id}/roles/re-normalize."""

    success: bool
    message: str
    doc_id: str
    processed: int = Field(..., description="Assignments considered for re-normalization.")
    updated: int = Field(..., description="Assignments whose normalized fields changed.")
    skipped: int = Field(..., description="Assignments skipped because they were manually overridden or curator-locked.")
    unresolved: int = Field(..., description="Assignments that remain unresolved after re-normalization.")


class RoleStatsResponse(BaseModel):
    """Response for GET /{doc_id}/roles/stats."""

    doc_id: str
    total_entities: int = Field(..., description="Total entities present in the document graph.")
    entities_with_roles: int = Field(..., description="Entities that currently have at least one role assignment.")
    coverage_ratio: float = Field(..., description="entities_with_roles / total_entities.")
    total_roles: int = Field(..., description="Total role assignments across the document.")
    unresolved_roles: int = Field(..., description="Role assignments still marked unresolved.")
    review_status_counts: dict[str, int] = Field(default_factory=dict, description="Counts grouped by review_status.")
    normalization_status_counts: dict[str, int] = Field(default_factory=dict, description="Counts grouped by normalization_status.")
    origin_counts: dict[str, int] = Field(default_factory=dict, description="Counts grouped by origin.")
    tenure_status_counts: dict[str, int] = Field(default_factory=dict, description="Counts grouped by tenure_status.")


class RoleVocabularyEntryInfo(BaseModel):
    """One entry in the curated role vocabulary."""

    role_id: str = Field(..., description="Stable vocabulary identifier.")
    canonical: str = Field(..., description="Preferred canonical role label.")
    aliases: List[str] = Field(default_factory=list, description="Known aliases or surface forms.")


class RoleVocabularyListResponse(BaseModel):
    """Response for GET /role-vocab."""

    total: int = Field(..., description="Total vocabulary entries returned.")
    roles: List[RoleVocabularyEntryInfo] = Field(default_factory=list, description="Curated role vocabulary entries.")


class CreateRoleVocabularyEntryRequest(BaseModel):
    """Request body for POST /role-vocab."""

    role_id: str = Field(..., description="Stable vocabulary identifier to create.")
    canonical: str = Field(..., description="Preferred canonical role label.")
    aliases: List[str] = Field(default_factory=list, description="Known aliases or alternate spellings.")


class UpdateRoleVocabularyEntryRequest(BaseModel):
    """Request body for PATCH /role-vocab/{role_id}."""

    canonical: Optional[str] = Field(None, description="Updated canonical role label.")
    aliases: Optional[List[str]] = Field(None, description="Replacement alias list.")


class RoleVocabularyMutationResponse(BaseModel):
    """Response for creating or updating one vocabulary entry."""

    success: bool
    message: str
    role: RoleVocabularyEntryInfo


class RoleVocabularyReloadResponse(BaseModel):
    """Response for POST /role-vocab/reload."""

    success: bool
    message: str
    total: int = Field(..., description="Vocabulary entry count after reload.")

