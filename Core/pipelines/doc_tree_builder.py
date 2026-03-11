import json
import os
import logging
from pathlib import Path
from typing import Any

from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines.tree_node_builder import create_node_by_type
# MinerU imports are deferred to avoid top-level dependency on doclayout_yolo
# when using the Docling parser.  See the ``else`` branch below.
from Core.configs.system_config import SystemConfig

log = logging.getLogger(__name__)

HTML_SOURCE_TYPES = {"html", "html_json", "preprocessed_html_json"}


def _get_source_type(cfg: SystemConfig) -> str:
    source_type = (getattr(cfg, "source_type", None) or "").strip().lower()
    if getattr(cfg, "html_json_path", None) and source_type in {"", "pdf"}:
        return "html_json"
    if source_type:
        return source_type
    if getattr(cfg, "html_json_path", None):
        return "html_json"
    return "pdf"


def _normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_normalize_text_value(item) for item in value]
        return " ".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "value", "body"):
            if key in value:
                return _normalize_text_value(value.get(key))
        return ""
    return str(value).strip()


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _extract_provenance(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}

    provenance = {}
    if isinstance(value.get("provenance"), dict):
        provenance.update(value["provenance"])

    for key in (
        "id",
        "block_id",
        "source_id",
        "xpath",
        "selector",
        "url",
        "page_idx",
        "page_path",
    ):
        if value.get(key) not in (None, ""):
            provenance[key] = value[key]
    return provenance


def _resolve_document_file_info(payload: Any, fallback_path: str) -> tuple[str, str]:
    if not isinstance(payload, dict):
        return os.path.basename(fallback_path), fallback_path

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    file_path = (
        payload.get("file_path")
        or payload.get("source_path")
        or metadata.get("file_path")
        or metadata.get("source_path")
        or fallback_path
    )
    file_name = payload.get("file_name") or metadata.get("file_name") or os.path.basename(file_path)
    return file_name, file_path


def _create_tree_node(node_type: NodeType, meta_dict: dict[str, Any]) -> TreeNode:
    node = TreeNode(meta_dict)
    node.type = node_type
    node.outline_node = node_type == NodeType.TITLE
    return node


def _append_html_text_blocks(
    tree_index: DocumentTree,
    parent_node: TreeNode,
    content_blocks: Any,
    next_pdf_id: int,
    *,
    source_type: str,
    source_path: str,
    file_name: str,
    file_path: str,
    inherited_provenance: dict,
) -> int:
    for block in _ensure_list(content_blocks):
        if isinstance(block, dict):
            block_role = (block.get("source_role") or block.get("role") or "body_text").strip()
            if block_role in {"image", "table"} or block.get("type") in {"image", "table"}:
                continue
            text_content = _normalize_text_value(
                block.get("text") or block.get("content") or block.get("body")
            )
            page_idx = block.get("page_idx")
            page_path = block.get("page_path")
            provenance = {**inherited_provenance, **_extract_provenance(block)}
        else:
            block_role = "body_text"
            text_content = _normalize_text_value(block)
            page_idx = None
            page_path = None
            provenance = dict(inherited_provenance)

        if not text_content:
            continue

        node = _create_tree_node(
            NodeType.TEXT,
            {
                "content": text_content,
                "pdf_id": next_pdf_id,
                "page_idx": page_idx,
                "page_path": page_path,
                "file_name": file_name,
                "file_path": file_path,
                "source_type": source_type,
                "source_role": block_role,
                "source_path": source_path,
                "provenance": provenance,
            },
        )
        tree_index.add_node(node)
        parent_node.add_child(node)
        next_pdf_id += 1

    return next_pdf_id


def _append_html_visual_blocks(
    tree_index: DocumentTree,
    parent_node: TreeNode,
    items: Any,
    next_pdf_id: int,
    *,
    node_type: NodeType,
    default_role: str,
    source_type: str,
    source_path: str,
    file_name: str,
    file_path: str,
    inherited_provenance: dict,
) -> int:
    for item in _ensure_list(items):
        if isinstance(item, dict):
            img_path = item.get("img_path") or item.get("path") or item.get("local_path")
            caption = _normalize_text_value(item.get("caption") or item.get("title"))
            footnote = _normalize_text_value(item.get("footnote") or item.get("ocr_text"))
            table_body = _normalize_text_value(item.get("table_body") or item.get("content"))
            page_idx = item.get("page_idx")
            page_path = item.get("page_path")
            source_role = item.get("source_role") or item.get("role") or default_role
            provenance = {**inherited_provenance, **_extract_provenance(item)}
        else:
            img_path = str(item)
            caption = ""
            footnote = ""
            table_body = ""
            page_idx = None
            page_path = None
            source_role = default_role
            provenance = dict(inherited_provenance)

        content = " ".join(part for part in [caption, footnote] if part).strip()
        if node_type == NodeType.TABLE and table_body:
            content = " ".join(part for part in [content, table_body] if part).strip()

        node = _create_tree_node(
            node_type,
            {
                "content": content,
                "img_path": img_path,
                "caption": caption,
                "footnote": footnote,
                "table_body": table_body if node_type == NodeType.TABLE else None,
                "pdf_id": next_pdf_id,
                "page_idx": page_idx,
                "page_path": page_path,
                "file_name": file_name,
                "file_path": file_path,
                "source_type": source_type,
                "source_role": source_role,
                "source_path": source_path,
                "provenance": provenance,
            },
        )
        tree_index.add_node(node)
        parent_node.add_child(node)
        next_pdf_id += 1

    return next_pdf_id


def _append_html_section(
    tree_index: DocumentTree,
    parent_node: TreeNode,
    section: Any,
    next_pdf_id: int,
    *,
    title_level: int,
    source_type: str,
    source_path: str,
    file_name: str,
    file_path: str,
    inherited_provenance: dict,
) -> int:
    if isinstance(section, str):
        return _append_html_text_blocks(
            tree_index,
            parent_node,
            section,
            next_pdf_id,
            source_type=source_type,
            source_path=source_path,
            file_name=file_name,
            file_path=file_path,
            inherited_provenance=inherited_provenance,
        )

    if not isinstance(section, dict):
        return next_pdf_id

    section_provenance = {**inherited_provenance, **_extract_provenance(section)}
    section_title = _normalize_text_value(section.get("title"))
    anchor_node = parent_node

    if section_title:
        title_node = _create_tree_node(
            NodeType.TITLE,
            {
                "content": section_title,
                "pdf_id": next_pdf_id,
                "title_level": title_level,
                "page_idx": section.get("page_idx"),
                "page_path": section.get("page_path"),
                "file_name": file_name,
                "file_path": file_path,
                "source_type": source_type,
                "source_role": section.get("source_role") or section.get("role") or "title",
                "source_path": source_path,
                "provenance": section_provenance,
            },
        )
        tree_index.add_node(title_node)
        parent_node.add_child(title_node)
        anchor_node = title_node
        next_pdf_id += 1

    next_pdf_id = _append_html_text_blocks(
        tree_index,
        anchor_node,
        section.get("content") or section.get("body"),
        next_pdf_id,
        source_type=source_type,
        source_path=source_path,
        file_name=file_name,
        file_path=file_path,
        inherited_provenance=section_provenance,
    )
    next_pdf_id = _append_html_visual_blocks(
        tree_index,
        anchor_node,
        section.get("images"),
        next_pdf_id,
        node_type=NodeType.IMAGE,
        default_role="image",
        source_type=source_type,
        source_path=source_path,
        file_name=file_name,
        file_path=file_path,
        inherited_provenance=section_provenance,
    )
    next_pdf_id = _append_html_visual_blocks(
        tree_index,
        anchor_node,
        section.get("tables"),
        next_pdf_id,
        node_type=NodeType.TABLE,
        default_role="table",
        source_type=source_type,
        source_path=source_path,
        file_name=file_name,
        file_path=file_path,
        inherited_provenance=section_provenance,
    )

    child_sections = (
        section.get("sections")
        or section.get("children")
        or section.get("subsections")
        or []
    )
    next_title_level = title_level + 1 if section_title else title_level
    for child_section in _ensure_list(child_sections):
        next_pdf_id = _append_html_section(
            tree_index,
            anchor_node,
            child_section,
            next_pdf_id,
            title_level=next_title_level,
            source_type=source_type,
            source_path=source_path,
            file_name=file_name,
            file_path=file_path,
            inherited_provenance=section_provenance,
        )

    return next_pdf_id


def build_tree_from_source(cfg: SystemConfig, reforce: bool = False) -> DocumentTree:
    source_type = _get_source_type(cfg)
    if source_type in HTML_SOURCE_TYPES:
        return build_tree_from_preprocessed_html_json(cfg, reforce=reforce)
    return build_tree_from_pdf(cfg, reforce=reforce)


def build_tree_from_preprocessed_html_json(
    cfg: SystemConfig, reforce: bool = False
) -> DocumentTree:
    tree_index_path = DocumentTree.get_save_path(cfg.save_path)
    if os.path.exists(tree_index_path) and not reforce:
        log.info(f"Loading existing tree index from {tree_index_path}...")
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info("Tree index loaded successfully.")
        return tree_index

    html_json_path = getattr(cfg, "html_json_path", None) or getattr(cfg, "source_path", None)
    if not html_json_path:
        raise ValueError("html_json_path or source_path must be set for html_json indexing.")

    with open(html_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    source_type = _get_source_type(cfg)
    file_name, file_path = _resolve_document_file_info(payload, html_json_path)
    meta_dict = {
        "file_name": file_name,
        "file_path": file_path,
        "source_type": source_type,
        "source_path": html_json_path,
        "source_role": "document",
        "provenance": _extract_provenance(payload),
    }

    os.makedirs(cfg.save_path, exist_ok=True)
    tree_index = DocumentTree(meta_dict=meta_dict, cfg=cfg)

    next_pdf_id = 1
    sections = payload if isinstance(payload, list) else [payload]
    for section in sections:
        next_pdf_id = _append_html_section(
            tree_index,
            tree_index.root_node,
            section,
            next_pdf_id,
            title_level=0,
            source_type=source_type,
            source_path=html_json_path,
            file_name=file_name,
            file_path=file_path,
            inherited_provenance=_extract_provenance(payload),
        )

    if cfg.tree.node_summary:
        from Core.pipelines.tree_node_summary import generate_tree_node_summary
        from Core.provider.llm import LLM
        from Core.provider.vlm import VLM
        from Core.provider.TokenTracker import TokenTracker

        llm = LLM(cfg.llm)
        vlm = VLM(cfg.vlm) if cfg.tree.use_vlm else None
        tree_index = generate_tree_node_summary(
            tree_index=tree_index,
            llm=llm,
            use_VLM=cfg.tree.use_vlm,
            vlm=vlm,
        )
        token_tracker = TokenTracker.get_instance()
        summary_cost = token_tracker.record_stage("tree_node_summary")
        log.info(f"Tree node summary generation cost: {summary_cost}")

    tree_index.save_to_file()
    log.info(
        "HTML JSON normalized into DocumentTree with %s nodes.",
        len(tree_index.nodes),
    )
    return tree_index


def construct_tree_index(
    tree_index: DocumentTree, pdf_list: list[dict], title_outline: list[dict]
) -> DocumentTree:
    """Constructs the tree index from the provided PDF content and title outline.
    :param tree_index: DocumentTree instance to construct the index.
    :param pdf_list: List of dictionaries containing PDF content.
    :param title_outline: List of dictionaries containing title outline information.
    :return: The updated DocumentTree instance with the constructed index.
    """

    for content in title_outline:
        node = create_node_by_type(pdf_content=content, isTitle=True)
        tree_index.add_node(node)

        # Add parent node by parent_id
        text_level = content.get("text_level", -1)
        if text_level == 0:
            # If text_level is 0, it is a root node
            tree_index.root_node.add_child(node)
        else:
            parent_id = content.get("parent_id", None)
            if parent_id is not None:
                parent_node = tree_index.get_node_by_pdf_id(parent_id)
                if parent_node:
                    parent_node.add_child(node)
            else:
                # If no parent_id, add to root
                tree_index.root_node.add_child(node)

        # Add child nodes
        end_idx = content["end_id"]
        for i in range(content["pdf_id"], end_idx):
            if i == len(pdf_list):
                break  # Avoid index out of range
            child_i = pdf_list[i]
            content_id = child_i.get("pdf_id", -1)
            if content_id > content["pdf_id"] and content_id < end_idx:
                child_node = create_node_by_type(pdf_content=child_i, isTitle=False)
                tree_index.add_node(child_node)
                node.add_child(child_node)

    log.info(f"Total {len(tree_index.nodes)} nodes added to the tree index.")
    return tree_index


def build_tree_from_pdf(cfg: SystemConfig, reforce: bool = False) -> DocumentTree:

    from Core.pipelines.outline_extractor import extract_pdf_outline_in_chunks
    from Core.pipelines.pdf_refiner import pdf_info_refiner
    from Core.pipelines.legal_heading_detector import (
        detect_document_language,
        detect_legal_headings,
    )
    from Core.provider.llm import LLM
    from Core.provider.vlm import VLM
    from Core.provider.TokenTracker import TokenTracker

    tree_index_path = DocumentTree.get_save_path(cfg.save_path)
    if os.path.exists(tree_index_path) and not reforce:
        # Load existing tree index
        log.info(f"Loading existing tree index from {tree_index_path}...")
        tree_index = DocumentTree.load_from_file(tree_index_path)
        log.info("Tree index loaded successfully.")
        return tree_index
    else:
        # Create a new tree index
        log.info("Creating a new tree index...")

    meta_dict = {
        "file_name": os.path.basename(cfg.pdf_path),
        "file_path": cfg.pdf_path,
    }

    os.makedirs(cfg.save_path, exist_ok=True)

    tree_index = DocumentTree(meta_dict=meta_dict, cfg=cfg)

    import json

    parser = getattr(cfg, "parser", "mineru") or "mineru"
    base_file_name = Path(cfg.pdf_path).stem

    # Each parser writes its cached pdf_list to its own sub-directory so the
    # two caches never collide even when the same save_path is reused.
    if parser == "docling":
        tmp_save_path = os.path.join(
            cfg.save_path, "docling", f"{base_file_name}_merged_content.json"
        )
    else:
        method = cfg.mineru.method
        tmp_save_path = os.path.join(
            cfg.save_path, method, f"{base_file_name}_merged_content.json"
        )

    if os.path.exists(tmp_save_path) and not reforce:
        # Load cached pdf_list (parser-agnostic from this point on)
        with open(tmp_save_path, "rb") as f:
            pdf_list = json.load(f)
        log.info(f"Loaded cached content from {tmp_save_path}")
    else:
        log.info(f"Extracting content from '{cfg.pdf_path}' using parser='{parser}' …")

        if parser == "docling":
            from Core.provider.extract_pdf_info_docling import parse_doc_with_docling

            pdf_list = parse_doc_with_docling(
                pdf_path=cfg.pdf_path,
                output_dir=cfg.save_path,
                cfg=cfg.docling,
            )
            # Persist for subsequent fast loads (mirrors what merge_middle_content
            # does for the MinerU path).
            docling_cache_dir = os.path.join(cfg.save_path, "docling")
            os.makedirs(docling_cache_dir, exist_ok=True)
            with open(tmp_save_path, "w", encoding="utf-8") as f:
                json.dump(pdf_list, f, ensure_ascii=False, indent=4)
            log.info(f"[Docling] Extracted content cached to '{tmp_save_path}'")
        else:
            # ── MinerU (default) ──────────────────────────────────────────
            from Core.provider.extract_pdf_info import parse_doc, merge_middle_content

            backend = cfg.mineru.backend
            server_url = cfg.mineru.server_url
            method = cfg.mineru.method

            middle_json, content_list = parse_doc(
                cfg.pdf_path,
                output_dir=cfg.save_path,
                backend=backend,
                method=method,
                server_url=server_url,
                lang=cfg.mineru.lang,
            )

            file_name = str(Path(cfg.pdf_path).stem)
            save_dir = os.path.join(cfg.save_path, method)
            pdf_list = merge_middle_content(
                middle_json,
                content_list,
                parse_dir=os.path.join(cfg.save_path, method),
                save_dir=save_dir,
                file_name=file_name,
            )
            log.info(f"[MinerU] Extracted content saved to '{tmp_save_path}'")

    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.tree.use_vlm else None

    lang = getattr(cfg, "document_lang", "auto") or "auto"
    if lang == "auto":
        lang = detect_document_language(pdf_list, fallback="en")
    pdf_list = pdf_info_refiner(pdf_list, llm, lang=lang)
    pdf_list = detect_legal_headings(pdf_list, lang=lang)
    title_outline = extract_pdf_outline_in_chunks(pdf_list, llm, lang=lang)
    tree_index = construct_tree_index(
        tree_index=tree_index, pdf_list=pdf_list, title_outline=title_outline
    )
    token_tracker = TokenTracker.get_instance()
    tree_index_cost = token_tracker.record_stage("tree_index_construction")
    log.info(f"Tree index construction cost: {tree_index_cost}")

    if cfg.tree.node_summary:
        from Core.pipelines.tree_node_summary import generate_tree_node_summary

        # Generate summaries for each node
        tree_index = generate_tree_node_summary(
            tree_index=tree_index,
            llm=llm,
            use_VLM=cfg.tree.use_vlm,
            vlm=vlm,
        )
        token_tracker = TokenTracker.get_instance()
        summary_cost = token_tracker.record_stage("tree_node_summary")
        log.info(f"Tree node summary generation cost: {summary_cost}")

    # save
    tree_index.save_to_file()
    return tree_index
