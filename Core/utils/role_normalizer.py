"""Role-name normalization for BookRAG domain role assignments.

Resolves observed surface forms (e.g. "Korsek", "Presiden RI") to canonical
role names and stable ``role_id`` values using a curated YAML vocabulary.

Normalization flow (tried in order):
    1. Exact match (case-insensitive) against canonical name or any alias.
    2. Prefix/substring match against canonical name or alias.
    3. Returns ``None`` (unresolved) if no match found.

The normalizer is intentionally conservative — it never changes a role that
has already been marked ``confirmed`` or ``manual_override``.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default vocabulary path — can be overridden via env var BOOKRAG_ROLE_VOCAB
_DEFAULT_VOCAB_PATH = Path(__file__).resolve().parents[2] / "config" / "role_vocab.yaml"


@lru_cache(maxsize=1)
def _load_vocab(path: str) -> list[dict]:
    """Load and cache the role vocabulary from YAML."""
    import yaml  # optional dependency; only needed at normalizer call time

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    roles = data.get("roles", [])
    # Pre-compute lower-cased lookup sets for fast matching
    for entry in roles:
        entry["_canonical_lower"] = entry["canonical"].lower()
        entry["_aliases_lower"] = [a.lower() for a in entry.get("aliases", [])]
    log.info("Loaded %d role vocabulary entries from %s", len(roles), path)
    return roles


def _vocab_path() -> str:
    return os.environ.get("BOOKRAG_ROLE_VOCAB", str(_DEFAULT_VOCAB_PATH))


def normalize_role(
    observed_text: str,
    *,
    vocab_path: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], str]:
    """Map an observed role surface form to a canonical entry.

    Args:
        observed_text: Raw text from document, e.g. ``"Korsek"``.
        vocab_path:    Override path to role_vocab.yaml (mainly for testing).

    Returns:
        ``(role_id, canonical_name, normalization_method)`` where:
        - ``role_id``           is the stable vocab identifier, or ``None``
        - ``canonical_name``    is the preferred display name, or ``None``
        - ``normalization_method`` is one of
          ``"alias"``, ``"exact"``, ``"prefix"``, or ``"unresolved"``
    """
    path = vocab_path or _vocab_path()
    try:
        vocab = _load_vocab(path)
    except Exception as exc:
        log.warning("Could not load role vocab: %s", exc)
        return None, None, "unresolved"

    needle = observed_text.strip().lower()
    if not needle:
        return None, None, "unresolved"

    # Pass 1: exact match on canonical name
    for entry in vocab:
        if needle == entry["_canonical_lower"]:
            return entry["role_id"], entry["canonical"], "exact"

    # Pass 2: exact match on any alias
    for entry in vocab:
        if needle in entry["_aliases_lower"]:
            return entry["role_id"], entry["canonical"], "alias"

    # Pass 3: prefix/substring match on canonical name or alias
    for entry in vocab:
        if needle in entry["_canonical_lower"] or entry["_canonical_lower"] in needle:
            return entry["role_id"], entry["canonical"], "prefix"
        for alias_lower in entry["_aliases_lower"]:
            if needle in alias_lower or alias_lower in needle:
                return entry["role_id"], entry["canonical"], "prefix"

    return None, None, "unresolved"


def apply_normalization(assignment: "RoleAssignment") -> "RoleAssignment":  # noqa: F821
    """Normalise a ``RoleAssignment`` in-place (returns new model instance).

    Skips assignments already marked ``confirmed`` or ``manual_override``.
    Populates ``role_id``, ``role_name``, ``normalization_status``, and
    updates each evidence item's ``normalization_method``.
    """
    # Don't re-normalise human-reviewed or manually set roles
    if assignment.normalization_status == "manual_override":
        return assignment
    if assignment.review_status == "confirmed" and assignment.role_id:
        return assignment

    # Try to normalise using the observed text from any evidence item
    observed_texts = [e.observed_role_text for e in assignment.evidence] or [assignment.role_name]
    best_role_id: Optional[str] = None
    best_canonical: Optional[str] = None
    best_method: str = "unresolved"

    for text in observed_texts:
        role_id, canonical, method = normalize_role(text)
        if role_id is not None:
            best_role_id, best_canonical, best_method = role_id, canonical, method
            break

    if best_role_id is None:
        # Fall back to normalising the role_name itself
        best_role_id, best_canonical, best_method = normalize_role(assignment.role_name)

    norm_status = "matched" if best_role_id else "unresolved"

    updates = {
        "normalization_status": norm_status,
        "normalization_confidence": 1.0 if best_method == "exact" else 0.8 if best_method == "alias" else 0.5 if best_method == "prefix" else None,
    }
    if best_role_id:
        updates["role_id"] = best_role_id
        updates["role_name"] = best_canonical  # promote to canonical name

    # Update evidence items' normalization_method
    updated_evidence = [
        e.model_copy(update={"normalization_method": best_method}) for e in assignment.evidence
    ]
    updates["evidence"] = updated_evidence

    return assignment.model_copy(update=updates)

