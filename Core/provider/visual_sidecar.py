import gc
import importlib
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence, Tuple

from Core.configs.visual_sidecar_config import VisualSidecarConfig

log = logging.getLogger(__name__)

VISUAL_SIDECAR_INDEX_NAME = "visual_leaf_sidecar"


class VisualSidecarDependencyError(RuntimeError):
    """Raised when an optional visual sidecar runtime dependency is unavailable."""


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str) -> Dict[str, Any] | List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _prepare_backend_dir(
    sidecar_dir: str,
    backend_type: str,
    retriever_family: str,
    retriever_version: str,
    force_rebuild: bool,
) -> Tuple[str, str]:
    backend_key = f"{backend_type}__{retriever_family}__{retriever_version}"
    backend_dir = os.path.join(sidecar_dir, "backends", backend_key)
    if force_rebuild and os.path.exists(backend_dir):
        shutil.rmtree(backend_dir)
    os.makedirs(backend_dir, exist_ok=True)
    return backend_key, backend_dir


def _build_prepared_documents(
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prepared_documents: List[Dict[str, Any]] = []
    skipped_documents: List[Dict[str, Any]] = []

    for candidate in candidates:
        image_path = candidate.get("img_path")
        if not image_path or not os.path.exists(image_path):
            skipped_documents.append(
                {
                    "node_id": candidate.get("node_id"),
                    "reason": "missing_local_asset",
                    "img_path": image_path,
                }
            )
            continue

        prepared_documents.append(
            {
                "document_id": str(candidate["node_id"]),
                "image_path": image_path,
                "text_context": {
                    "text_surrogate": candidate.get("text_surrogate"),
                    "caption": candidate.get("caption"),
                    "footnote": candidate.get("footnote"),
                    "table_body": candidate.get("table_body"),
                    "content": candidate.get("content"),
                },
                "metadata": candidate,
            }
        )

    return prepared_documents, skipped_documents


def _candidate_to_prepared_document(candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "document_id": str(candidate["node_id"]),
        "image_path": candidate.get("img_path"),
        "text_context": {
            "text_surrogate": candidate.get("text_surrogate"),
            "caption": candidate.get("caption"),
            "footnote": candidate.get("footnote"),
            "table_body": candidate.get("table_body"),
            "content": candidate.get("content"),
        },
        "metadata": candidate,
    }


def _tokenize_visual_query_text(text: str | None) -> List[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_visual_search_text(document: Dict[str, Any]) -> str:
    metadata = document.get("metadata") or {}
    text_context = document.get("text_context") or {}
    text_parts = [
        text_context.get("text_surrogate"),
        text_context.get("caption"),
        text_context.get("footnote"),
        text_context.get("table_body"),
        text_context.get("content"),
        metadata.get("caption"),
        metadata.get("footnote"),
        metadata.get("table_body"),
        metadata.get("content"),
        metadata.get("source_role"),
        metadata.get("node_type"),
    ]
    return " ".join(str(part) for part in text_parts if part)


def _score_visual_query_match(query_tokens: Sequence[str], search_text: str) -> float:
    if not query_tokens or not search_text:
        return 0.0

    document_tokens = _tokenize_visual_query_text(search_text)
    if not document_tokens:
        return 0.0

    query_counter = Counter(query_tokens)
    document_counter = Counter(document_tokens)
    overlap_count = sum(
        min(count, document_counter.get(token, 0))
        for token, count in query_counter.items()
    )
    if overlap_count <= 0:
        return 0.0

    unique_overlap = len(set(query_counter) & set(document_counter))
    coverage = overlap_count / max(len(query_tokens), 1)
    unique_coverage = unique_overlap / max(len(query_counter), 1)
    return (coverage * 0.7) + (unique_coverage * 0.3)


def _resolve_query_backend_build(
    sidecar_manifest: Dict[str, Any],
    sidecar_cfg: VisualSidecarConfig,
) -> Dict[str, Any] | None:
    backend_builds = sidecar_manifest.get("backend_builds") or []
    if not isinstance(backend_builds, list):
        return None

    for build in backend_builds:
        if (
            build.get("backend_type") == sidecar_cfg.backend_type
            and build.get("retriever_family") == sidecar_cfg.retriever_family
            and build.get("retriever_version") == sidecar_cfg.retriever_version
        ):
            return build

    for build in backend_builds:
        if (
            build.get("retriever_family") == sidecar_cfg.retriever_family
            and build.get("retriever_version") == sidecar_cfg.retriever_version
        ):
            return build

    return None


def _load_query_documents(
    sidecar_dir: str,
    sidecar_manifest: Dict[str, Any],
    sidecar_cfg: VisualSidecarConfig,
) -> Tuple[List[Dict[str, Any]], str]:
    backend_build = _resolve_query_backend_build(sidecar_manifest, sidecar_cfg)
    if backend_build:
        backend_rel_path = backend_build.get("path")
        if backend_rel_path:
            documents_path = os.path.join(sidecar_dir, backend_rel_path, "documents.jsonl")
            if os.path.exists(documents_path):
                return _read_jsonl(documents_path), backend_build.get("backend_type", "backend")

    candidates_path = os.path.join(sidecar_dir, "candidates.json")
    if os.path.exists(candidates_path):
        candidates = _read_json(candidates_path)
        if isinstance(candidates, list):
            return [
                _candidate_to_prepared_document(candidate)
                for candidate in candidates
                if isinstance(candidate, dict)
            ], "stub_candidates"

    return [], "unavailable"


def _torch_load_payload(torch_module: Any, path: str) -> Any:
    load_fn = getattr(torch_module, "load", None)
    if not callable(load_fn):
        raise ValueError("The visual sidecar torch runtime does not provide torch.load().")

    try:
        return load_fn(path, map_location="cpu")
    except TypeError:
        return load_fn(path)


def _move_embeddings_to_device(embeddings: Sequence[Any], device: str) -> List[Any]:
    return [embedding.to(device) if hasattr(embedding, "to") else embedding for embedding in embeddings]


def _to_python_value(value: Any) -> Any:
    value = _to_cpu_embedding(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _normalize_score_values(scores: Any) -> List[float]:
    python_scores = _to_python_value(scores)
    if isinstance(python_scores, tuple):
        python_scores = list(python_scores)
    if not isinstance(python_scores, list):
        return [float(python_scores)]
    if python_scores and isinstance(python_scores[0], tuple):
        python_scores = list(python_scores[0])
    elif python_scores and isinstance(python_scores[0], list):
        python_scores = python_scores[0]
    return [float(_to_python_value(score)) for score in python_scores]


def _query_visual_sidecar_textual(
    documents: Sequence[Dict[str, Any]],
    retrieval_source: str,
    query_tokens: Sequence[str],
    *,
    allowed_id_set: set[int] | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for document in documents:
        metadata = document.get("metadata") or {}
        node_id = metadata.get("node_id", document.get("document_id"))
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            continue

        if allowed_id_set is not None and node_id not in allowed_id_set:
            continue

        score = _score_visual_query_match(
            query_tokens,
            _build_visual_search_text(document),
        )
        if score <= 0:
            continue

        hits.append(
            {
                "node_id": node_id,
                "score": score,
                "document_id": document.get("document_id"),
                "image_path": document.get("image_path"),
                "node_type": metadata.get("node_type"),
                "metadata": metadata,
                "retrieval_source": retrieval_source,
            }
        )

    hits.sort(key=lambda item: (-item["score"], item["node_id"]))
    return hits[:top_k]


def _query_colqwen2_visual_sidecar(
    sidecar_dir: str,
    backend_build: Dict[str, Any],
    sidecar_cfg: VisualSidecarConfig,
    query_text: str,
    *,
    allowed_id_set: set[int] | None,
    top_k: int,
) -> List[Dict[str, Any]]:
    backend_rel_path = backend_build.get("path")
    if not backend_rel_path:
        raise FileNotFoundError("The visual sidecar backend build does not include a relative path.")

    backend_dir = os.path.join(sidecar_dir, backend_rel_path)
    documents_path = os.path.join(backend_dir, "documents.jsonl")
    embeddings_path = os.path.join(backend_dir, "multivector_embeddings.pt")
    if not os.path.exists(documents_path) or not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"Model-backed visual sidecar query artifacts are missing under {backend_dir}."
        )

    runtime = _load_colqwen2_runtime()
    torch_module = runtime["torch"]
    documents = _read_jsonl(documents_path)
    encoded_payload = _torch_load_payload(torch_module, embeddings_path)
    encoded_documents = encoded_payload.get("documents") if isinstance(encoded_payload, dict) else None
    if not isinstance(encoded_documents, list):
        raise ValueError("The visual sidecar ColQwen2 embedding payload is malformed.")

    embedding_by_document_id = {
        str(item.get("document_id")): item.get("embedding")
        for item in encoded_documents
        if isinstance(item, dict)
        and item.get("document_id") is not None
        and item.get("embedding") is not None
    }

    candidate_documents: List[Tuple[Dict[str, Any], int, Dict[str, Any], Any]] = []
    for document in documents:
        metadata = document.get("metadata") or {}
        node_id = metadata.get("node_id", document.get("document_id"))
        try:
            node_id = int(node_id)
        except (TypeError, ValueError):
            continue

        if allowed_id_set is not None and node_id not in allowed_id_set:
            continue

        document_id = str(document.get("document_id"))
        embedding = embedding_by_document_id.get(document_id)
        if embedding is None:
            continue

        candidate_documents.append((document, node_id, metadata, embedding))

    if not candidate_documents:
        return []

    resolved_device = _resolve_runtime_device(torch_module, sidecar_cfg.device)
    resolved_dtype = _resolve_torch_dtype(torch_module, sidecar_cfg.torch_dtype)
    resolved_attn = _resolve_attn_implementation(
        sidecar_cfg.attn_implementation,
        runtime["is_flash_attn_2_available"],
    )
    retriever_model = (
        backend_build.get("retriever_model")
        or sidecar_cfg.retriever_model
        or sidecar_cfg.retriever_family
    )

    model = None
    try:
        model = runtime["ColQwen2"].from_pretrained(
            retriever_model,
            torch_dtype=resolved_dtype,
            device_map=resolved_device,
            attn_implementation=resolved_attn,
        ).eval()
        processor = runtime["ColQwen2Processor"].from_pretrained(retriever_model)
        model_device = getattr(model, "device", resolved_device)
        batch_queries = processor.process_queries([query_text])
        batch_queries = _move_batch_to_device(batch_queries, model_device)
        with torch_module.no_grad():
            query_embeddings = model(**batch_queries)

        image_embeddings = _move_embeddings_to_device(
            [embedding for _, _, _, embedding in candidate_documents],
            model_device,
        )
        scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    finally:
        _cleanup_torch_model(torch_module, model)

    score_values = _normalize_score_values(scores)
    if len(score_values) != len(candidate_documents):
        raise ValueError(
            "ColQwen2 query-time score count does not match the number of visual sidecar documents."
        )

    hits: List[Dict[str, Any]] = []
    retrieval_source = backend_build.get("backend_type", "colqwen2_local")
    for (document, node_id, metadata, _), score in zip(candidate_documents, score_values):
        if score <= 0:
            continue
        hits.append(
            {
                "node_id": node_id,
                "score": float(score),
                "document_id": document.get("document_id"),
                "image_path": document.get("image_path"),
                "node_type": metadata.get("node_type"),
                "metadata": metadata,
                "retrieval_source": retrieval_source,
            }
        )

    hits.sort(key=lambda item: (-item["score"], item["node_id"]))
    return hits[:top_k]


def query_visual_sidecar(
    save_path: str,
    sidecar_cfg: VisualSidecarConfig,
    query_text: str,
    *,
    allowed_node_ids: Sequence[int] | None = None,
    top_k: int = 3,
    index_name: str = VISUAL_SIDECAR_INDEX_NAME,
) -> List[Dict[str, Any]]:
    """Query visual leaf sidecar artifacts with optional model-backed retrieval.

    When a compatible `colqwen2_local` backend build is available, this function
    will try to embed the query and score it against persisted multivector image
    embeddings. If that optional runtime path is unavailable, it safely falls back
    to the prepared textual surrogate/context ranking path.
    """
    if not save_path or not query_text or top_k <= 0:
        return []

    sidecar_dir = os.path.join(save_path, index_name)
    manifest_path = os.path.join(sidecar_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return []

    try:
        sidecar_manifest = _read_json(manifest_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Failed to load visual sidecar query artifacts from %s: %s", sidecar_dir, exc)
        return []

    allowed_id_set = {int(node_id) for node_id in allowed_node_ids} if allowed_node_ids else None
    query_tokens = _tokenize_visual_query_text(query_text)
    if not query_tokens:
        return []

    backend_build = _resolve_query_backend_build(sidecar_manifest, sidecar_cfg)
    if backend_build and backend_build.get("backend_type") == "colqwen2_local":
        try:
            return _query_colqwen2_visual_sidecar(
                sidecar_dir,
                backend_build,
                sidecar_cfg,
                query_text,
                allowed_id_set=allowed_id_set,
                top_k=top_k,
            )
        except Exception as exc:
            log.warning(
                "Model-backed visual sidecar query failed for %s; falling back to textual surrogate search: %s",
                sidecar_dir,
                exc,
            )

    try:
        documents, retrieval_source = _load_query_documents(
            sidecar_dir,
            sidecar_manifest,
            sidecar_cfg,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        log.warning("Failed to load visual sidecar query artifacts from %s: %s", sidecar_dir, exc)
        return []

    return _query_visual_sidecar_textual(
        documents,
        retrieval_source,
        query_tokens,
        allowed_id_set=allowed_id_set,
        top_k=top_k,
    )


def _build_manifest_base(
    *,
    backend_type: str,
    retriever_family: str,
    retriever_model: str,
    retriever_version: str,
    source_candidate_count: int,
    prepared_documents: Sequence[Dict[str, Any]],
    skipped_documents: Sequence[Dict[str, Any]],
    sidecar_cfg: VisualSidecarConfig,
) -> Dict[str, Any]:
    return {
        "backend_type": backend_type,
        "retriever_family": retriever_family,
        "retriever_model": retriever_model,
        "retriever_version": retriever_version,
        "source_candidate_count": source_candidate_count,
        "document_count": len(prepared_documents),
        "skipped_count": len(skipped_documents),
        "created_at": _utc_now_isoformat(),
        "config": asdict(sidecar_cfg),
    }


def _write_prepared_document_artifacts(
    backend_dir: str,
    prepared_documents: List[Dict[str, Any]],
    skipped_documents: List[Dict[str, Any]],
) -> Tuple[str, str, str | None, List[Dict[str, Any]]]:
    documents_filename = "documents.jsonl"
    manifest_filename = "backend_manifest.json"
    skipped_filename = "skipped.json"

    documents_path = os.path.join(backend_dir, documents_filename)
    manifest_path = os.path.join(backend_dir, manifest_filename)
    skipped_path = os.path.join(backend_dir, skipped_filename)

    _write_jsonl(documents_path, prepared_documents)

    artifacts = [
        {
            "artifact_type": "prepared_documents",
            "path": documents_filename,
            "count": len(prepared_documents),
        }
    ]

    skipped_path_result: str | None = None
    if skipped_documents:
        _write_json(skipped_path, {"items": skipped_documents})
        skipped_path_result = skipped_path
        artifacts.append(
            {
                "artifact_type": "skipped_candidates",
                "path": skipped_filename,
                "count": len(skipped_documents),
            }
        )

    return documents_path, manifest_path, skipped_path_result, artifacts


def _resolve_runtime_device(torch_module: Any, requested_device: str) -> str:
    if requested_device and requested_device != "auto":
        return requested_device

    cuda = getattr(torch_module, "cuda", None)
    if cuda and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        return "cuda:0"

    backends = getattr(torch_module, "backends", None)
    mps = getattr(backends, "mps", None)
    if mps and callable(getattr(mps, "is_available", None)) and mps.is_available():
        return "mps"

    return "cpu"


def _resolve_torch_dtype(torch_module: Any, requested_dtype: str | None) -> Any:
    if not requested_dtype or requested_dtype == "auto":
        return "auto"
    if hasattr(torch_module, requested_dtype):
        return getattr(torch_module, requested_dtype)
    raise ValueError(f"Unsupported visual sidecar torch_dtype: {requested_dtype}")


def _resolve_attn_implementation(
    requested_attn: str | None,
    is_flash_attn_2_available: Any,
) -> str | None:
    if requested_attn in (None, "", "none"):
        return None
    if requested_attn != "auto":
        return requested_attn
    if callable(is_flash_attn_2_available) and is_flash_attn_2_available():
        return "flash_attention_2"
    return None


def _move_batch_to_device(batch_inputs: Any, device: str) -> Any:
    if hasattr(batch_inputs, "to"):
        return batch_inputs.to(device)
    if isinstance(batch_inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch_inputs.items()
        }
    return batch_inputs


def _to_cpu_embedding(embedding: Any) -> Any:
    if hasattr(embedding, "detach"):
        embedding = embedding.detach()
    if hasattr(embedding, "to"):
        try:
            return embedding.to("cpu")
        except TypeError:
            pass
    if hasattr(embedding, "cpu"):
        return embedding.cpu()
    return embedding


def _split_batch_embeddings(batch_embeddings: Any, torch_module: Any) -> List[Any]:
    cpu_embeddings = _to_cpu_embedding(batch_embeddings)
    if isinstance(cpu_embeddings, (list, tuple)):
        return list(cpu_embeddings)
    if hasattr(torch_module, "unbind"):
        return list(torch_module.unbind(cpu_embeddings, dim=0))
    if hasattr(cpu_embeddings, "unbind"):
        return list(cpu_embeddings.unbind(0))
    return list(cpu_embeddings)


def _cleanup_torch_model(torch_module: Any, model: Any) -> None:
    if model is not None:
        del model
    cuda = getattr(torch_module, "cuda", None)
    if cuda and callable(getattr(cuda, "is_available", None)) and cuda.is_available():
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
    gc.collect()


def _load_colqwen2_runtime() -> Dict[str, Any]:
    missing_packages: List[str] = []
    imported_modules: Dict[str, Any] = {}

    required_modules = {
        "torch": "torch",
        "PIL.Image": "Pillow",
        "colpali_engine.models": "colpali-engine",
    }

    for module_name, package_name in required_modules.items():
        try:
            imported_modules[module_name] = importlib.import_module(module_name)
        except ImportError:
            missing_packages.append(package_name)

    if missing_packages:
        package_list = ", ".join(sorted(set(missing_packages)))
        raise VisualSidecarDependencyError(
            "The 'colqwen2_local' visual sidecar backend requires optional runtime "
            f"dependencies that are not available in this environment: {package_list}. "
            "Install the appropriate packages in your environment before running this backend."
        )

    try:
        transformers_import_utils = importlib.import_module("transformers.utils.import_utils")
        is_flash_attn_2_available = getattr(
            transformers_import_utils,
            "is_flash_attn_2_available",
            lambda: False,
        )
    except ImportError:
        is_flash_attn_2_available = lambda: False

    model_module = imported_modules["colpali_engine.models"]
    return {
        "torch": imported_modules["torch"],
        "image_module": imported_modules["PIL.Image"],
        "ColQwen2": getattr(model_module, "ColQwen2"),
        "ColQwen2Processor": getattr(model_module, "ColQwen2Processor"),
        "is_flash_attn_2_available": is_flash_attn_2_available,
    }


class BaseVisualSidecarBackend(ABC):
    backend_type: str

    @abstractmethod
    def build(
        self,
        *,
        sidecar_dir: str,
        candidates: List[Dict[str, Any]],
        sidecar_cfg: VisualSidecarConfig,
        retriever_family: str,
        retriever_model: str,
        retriever_version: str,
        force_rebuild: bool,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class PreparedCorpusVisualSidecarBackend(BaseVisualSidecarBackend):
    backend_type = "prepared_corpus"

    def build(
        self,
        *,
        sidecar_dir: str,
        candidates: List[Dict[str, Any]],
        sidecar_cfg: VisualSidecarConfig,
        retriever_family: str,
        retriever_model: str,
        retriever_version: str,
        force_rebuild: bool,
    ) -> Dict[str, Any]:
        backend_key, backend_dir = _prepare_backend_dir(
            sidecar_dir,
            self.backend_type,
            retriever_family,
            retriever_version,
            force_rebuild,
        )
        prepared_documents, skipped_documents = _build_prepared_documents(candidates)
        _, manifest_path, _, artifacts = _write_prepared_document_artifacts(
            backend_dir,
            prepared_documents,
            skipped_documents,
        )

        manifest = _build_manifest_base(
            backend_type=self.backend_type,
            retriever_family=retriever_family,
            retriever_model=retriever_model,
            retriever_version=retriever_version,
            source_candidate_count=len(candidates),
            prepared_documents=prepared_documents,
            skipped_documents=skipped_documents,
            sidecar_cfg=sidecar_cfg,
        )
        manifest["artifacts"] = artifacts

        _write_json(manifest_path, manifest)
        log.info(
            "Visual sidecar backend '%s' prepared %s document(s) in %s.",
            backend_key,
            len(prepared_documents),
            backend_dir,
        )
        return {
            **manifest,
            "backend_dir": backend_dir,
            "manifest_path": manifest_path,
        }


class ColQwen2LocalVisualSidecarBackend(BaseVisualSidecarBackend):
    backend_type = "colqwen2_local"

    def build(
        self,
        *,
        sidecar_dir: str,
        candidates: List[Dict[str, Any]],
        sidecar_cfg: VisualSidecarConfig,
        retriever_family: str,
        retriever_model: str,
        retriever_version: str,
        force_rebuild: bool,
    ) -> Dict[str, Any]:
        if retriever_family != "colqwen2":
            raise ValueError(
                "The 'colqwen2_local' backend currently supports retriever_family='colqwen2' only."
            )

        backend_key, backend_dir = _prepare_backend_dir(
            sidecar_dir,
            self.backend_type,
            retriever_family,
            retriever_version,
            force_rebuild,
        )
        prepared_documents, skipped_documents = _build_prepared_documents(candidates)
        _, manifest_path, _, artifacts = _write_prepared_document_artifacts(
            backend_dir,
            prepared_documents,
            skipped_documents,
        )

        manifest = _build_manifest_base(
            backend_type=self.backend_type,
            retriever_family=retriever_family,
            retriever_model=retriever_model,
            retriever_version=retriever_version,
            source_candidate_count=len(candidates),
            prepared_documents=prepared_documents,
            skipped_documents=skipped_documents,
            sidecar_cfg=sidecar_cfg,
        )
        manifest["artifacts"] = artifacts

        if not prepared_documents:
            manifest["encoded_count"] = 0
            manifest["encoding_status"] = "skipped_no_documents"
            _write_json(manifest_path, manifest)
            return {
                **manifest,
                "backend_dir": backend_dir,
                "manifest_path": manifest_path,
            }

        runtime = _load_colqwen2_runtime()
        torch_module = runtime["torch"]
        resolved_device = _resolve_runtime_device(torch_module, sidecar_cfg.device)
        resolved_dtype = _resolve_torch_dtype(torch_module, sidecar_cfg.torch_dtype)
        resolved_attn = _resolve_attn_implementation(
            sidecar_cfg.attn_implementation,
            runtime["is_flash_attn_2_available"],
        )

        model = None
        try:
            model = runtime["ColQwen2"].from_pretrained(
                retriever_model,
                torch_dtype=resolved_dtype,
                device_map=resolved_device,
                attn_implementation=resolved_attn,
            ).eval()
            processor = runtime["ColQwen2Processor"].from_pretrained(retriever_model)
            encoded_documents = self._encode_documents(
                runtime=runtime,
                model=model,
                processor=processor,
                prepared_documents=prepared_documents,
                batch_size=max(1, sidecar_cfg.image_batch_size),
            )
        finally:
            _cleanup_torch_model(torch_module, model)

        embeddings_filename = "multivector_embeddings.pt"
        embeddings_path = os.path.join(backend_dir, embeddings_filename)
        torch_module.save(
            {
                "backend_type": self.backend_type,
                "retriever_family": retriever_family,
                "retriever_model": retriever_model,
                "retriever_version": retriever_version,
                "documents": encoded_documents,
            },
            embeddings_path,
        )

        manifest["encoded_count"] = len(encoded_documents)
        manifest["encoding_status"] = "encoded"
        manifest["runtime"] = {
            "device": resolved_device,
            "torch_dtype": sidecar_cfg.torch_dtype,
            "attn_implementation": resolved_attn,
            "image_batch_size": max(1, sidecar_cfg.image_batch_size),
        }
        manifest["artifacts"].append(
            {
                "artifact_type": "colqwen2_multivector_embeddings",
                "path": embeddings_filename,
                "count": len(encoded_documents),
            }
        )

        _write_json(manifest_path, manifest)
        log.info(
            "Visual sidecar backend '%s' encoded %s document(s) in %s.",
            backend_key,
            len(encoded_documents),
            backend_dir,
        )
        return {
            **manifest,
            "backend_dir": backend_dir,
            "manifest_path": manifest_path,
        }

    def _encode_documents(
        self,
        *,
        runtime: Dict[str, Any],
        model: Any,
        processor: Any,
        prepared_documents: Sequence[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        torch_module = runtime["torch"]
        image_module = runtime["image_module"]
        model_device = getattr(model, "device", "cpu")
        encoded_documents: List[Dict[str, Any]] = []

        for start in range(0, len(prepared_documents), batch_size):
            batch_documents = prepared_documents[start : start + batch_size]
            batch_images = []
            try:
                for document in batch_documents:
                    with image_module.open(document["image_path"]) as image:
                        batch_images.append(image.convert("RGB"))

                batch_inputs = processor.process_images(batch_images)
                batch_inputs = _move_batch_to_device(batch_inputs, model_device)

                with torch_module.no_grad():
                    batch_embeddings = model(**batch_inputs)

                split_embeddings = _split_batch_embeddings(batch_embeddings, torch_module)
                if len(split_embeddings) != len(batch_documents):
                    raise ValueError(
                        "ColQwen2 image embedding batch size does not match the number of prepared documents."
                    )

                encoded_documents.extend(
                    {
                        "document_id": document["document_id"],
                        "image_path": document["image_path"],
                        "embedding": _to_cpu_embedding(embedding),
                    }
                    for document, embedding in zip(batch_documents, split_embeddings)
                )
            finally:
                for image in batch_images:
                    close = getattr(image, "close", None)
                    if callable(close):
                        close()

        return encoded_documents


def get_visual_sidecar_backend(backend_type: str) -> BaseVisualSidecarBackend:
    registry = {
        PreparedCorpusVisualSidecarBackend.backend_type: PreparedCorpusVisualSidecarBackend(),
        ColQwen2LocalVisualSidecarBackend.backend_type: ColQwen2LocalVisualSidecarBackend(),
    }
    if backend_type not in registry:
        raise ValueError(f"Unsupported visual sidecar backend_type: {backend_type}")
    return registry[backend_type]


def build_visual_sidecar_backend(
    save_path: str,
    sidecar_cfg: VisualSidecarConfig,
    *,
    index_name: str = VISUAL_SIDECAR_INDEX_NAME,
) -> Dict[str, Any]:
    sidecar_dir = os.path.join(save_path, index_name)
    manifest_path = os.path.join(sidecar_dir, "manifest.json")
    candidates_path = os.path.join(sidecar_dir, "candidates.json")
    if not os.path.exists(manifest_path) or not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"Visual sidecar stub artifacts not found under {sidecar_dir}. Build the stub first."
        )

    sidecar_manifest = _read_json(manifest_path)
    candidates = _read_json(candidates_path)
    if not isinstance(candidates, list):
        raise ValueError("Visual sidecar candidates.json must contain a list of candidate records.")

    retriever_model = sidecar_cfg.retriever_model or sidecar_cfg.retriever_family
    backend = get_visual_sidecar_backend(sidecar_cfg.backend_type)
    backend_manifest = backend.build(
        sidecar_dir=sidecar_dir,
        candidates=candidates,
        sidecar_cfg=sidecar_cfg,
        retriever_family=sidecar_cfg.retriever_family,
        retriever_model=retriever_model,
        retriever_version=sidecar_cfg.retriever_version,
        force_rebuild=sidecar_cfg.force_rebuild,
    )

    backend_build_record = {
        "backend_type": sidecar_cfg.backend_type,
        "retriever_family": sidecar_cfg.retriever_family,
        "retriever_model": retriever_model,
        "retriever_version": sidecar_cfg.retriever_version,
        "path": os.path.relpath(backend_manifest["backend_dir"], sidecar_dir),
        "manifest_path": os.path.relpath(backend_manifest["manifest_path"], sidecar_dir),
        "document_count": backend_manifest["document_count"],
        "encoded_count": backend_manifest.get("encoded_count"),
        "created_at": backend_manifest["created_at"],
    }

    existing_builds = sidecar_manifest.get("backend_builds", [])
    filtered_builds = [
        build
        for build in existing_builds
        if not (
            build.get("backend_type") == backend_build_record["backend_type"]
            and build.get("retriever_family") == backend_build_record["retriever_family"]
            and build.get("retriever_version") == backend_build_record["retriever_version"]
        )
    ]
    filtered_builds.append(backend_build_record)

    sidecar_manifest["status"] = "backend_materialized"
    sidecar_manifest["backend_builds"] = filtered_builds
    _write_json(manifest_path, sidecar_manifest)

    return backend_manifest