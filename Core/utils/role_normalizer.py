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


def _prepare_role_entry(entry: dict) -> dict:
    """Normalize a raw YAML role entry into a validated runtime form."""
    role_id = str(entry.get("role_id", "")).strip()
    canonical = str(entry.get("canonical", "")).strip()
    if not role_id:
        raise ValueError("role_id is required")
    if not canonical:
        raise ValueError("canonical is required")

    aliases = entry.get("aliases") or []
    cleaned_aliases: list[str] = []
    seen_aliases: set[str] = set()
    for alias in aliases:
        alias_text = str(alias).strip()
        if not alias_text:
            continue
        alias_lower = alias_text.lower()
        if alias_lower == canonical.lower() or alias_lower in seen_aliases:
            continue
        seen_aliases.add(alias_lower)
        cleaned_aliases.append(alias_text)

    prepared = {
        "role_id": role_id,
        "canonical": canonical,
        "aliases": cleaned_aliases,
    }
    prepared["_canonical_lower"] = canonical.lower()
    prepared["_aliases_lower"] = [alias.lower() for alias in cleaned_aliases]
    return prepared


def _serialize_role_entry(entry: dict) -> dict:
    """Strip runtime-only fields before persisting to YAML."""
    return {
        "role_id": entry["role_id"],
        "canonical": entry["canonical"],
        "aliases": list(entry.get("aliases") or []),
    }


def _read_vocab_file(path: str) -> list[dict]:
    import yaml  # optional dependency; only needed at normalizer call time

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    roles = data.get("roles", []) or []
    return [_prepare_role_entry(entry) for entry in roles]


def _write_vocab_file(path: str, roles: list[dict]) -> None:
    import yaml  # optional dependency; only needed at normalizer call time

    payload = {"roles": [_serialize_role_entry(entry) for entry in roles]}
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=True)


@lru_cache(maxsize=1)
def _load_vocab(path: str) -> list[dict]:
    """Load and cache the role vocabulary from YAML."""
    roles = _read_vocab_file(path)
    log.info("Loaded %d role vocabulary entries from %s", len(roles), path)
    return roles


def _vocab_path() -> str:
    return os.environ.get("BOOKRAG_ROLE_VOCAB", str(_DEFAULT_VOCAB_PATH))


def list_vocab_entries(*, vocab_path: Optional[str] = None) -> list[dict]:
    """Return vocabulary entries without runtime-only cache fields."""
    path = vocab_path or _vocab_path()
    return [_serialize_role_entry(entry) for entry in _load_vocab(path)]


def reload_vocab(*, vocab_path: Optional[str] = None) -> list[dict]:
    """Clear the cached vocabulary and return the freshly loaded entries."""
    path = vocab_path or _vocab_path()
    _load_vocab.cache_clear()
    return list_vocab_entries(vocab_path=path)


def add_vocab_entry(
    role_id: str,
    canonical: str,
    aliases: Optional[list[str]] = None,
    *,
    vocab_path: Optional[str] = None,
) -> dict:
    """Add a new vocabulary entry and reload the cache."""
    path = vocab_path or _vocab_path()
    roles = _read_vocab_file(path)
    if any(entry["role_id"] == role_id for entry in roles):
        raise ValueError(f"Role vocabulary entry '{role_id}' already exists.")

    roles.append(_prepare_role_entry({
        "role_id": role_id,
        "canonical": canonical,
        "aliases": aliases or [],
    }))
    roles.sort(key=lambda item: item["canonical"].lower())
    _write_vocab_file(path, roles)
    reload_vocab(vocab_path=path)
    return {"role_id": role_id, "canonical": canonical.strip(), "aliases": list(aliases or [])}


def update_vocab_entry(
    role_id: str,
    *,
    canonical: Optional[str] = None,
    aliases: Optional[list[str]] = None,
    vocab_path: Optional[str] = None,
) -> dict:
    """Update an existing vocabulary entry and reload the cache."""
    path = vocab_path or _vocab_path()
    roles = _read_vocab_file(path)
    for idx, entry in enumerate(roles):
        if entry["role_id"] != role_id:
            continue
        updated = _prepare_role_entry({
            "role_id": role_id,
            "canonical": canonical if canonical is not None else entry["canonical"],
            "aliases": aliases if aliases is not None else entry.get("aliases", []),
        })
        roles[idx] = updated
        roles.sort(key=lambda item: item["canonical"].lower())
        _write_vocab_file(path, roles)
        reload_vocab(vocab_path=path)
        return _serialize_role_entry(updated)
    raise KeyError(f"Role vocabulary entry '{role_id}' not found.")


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

