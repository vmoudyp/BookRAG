"""Tests for HTML JSON normalization into DocumentTree."""
import json
import yaml

from Core.Index.Tree import NodeType
from Core.configs.system_config import load_system_config
from Core.pipelines.doc_tree_builder import build_tree_from_source
from Core.pipelines.vdb_index import (
    build_visual_sidecar_stub,
    process_tree_nodes,
    select_visual_leaf_candidates,
)


def _build_test_cfg(tmp_path, html_json_path):
    config_path = tmp_path / "test-config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "mineru": {"backend": "pipeline", "method": "auto", "lang": "en"},
                "rag": {"strategy": "gbc"},
                "tree": {"node_summary": False, "use_vlm": False},
            }
        ),
        encoding="utf-8",
    )
    cfg = load_system_config(str(config_path))
    cfg.save_path = str(tmp_path / "index")
    cfg.html_json_path = str(html_json_path)
    return cfg


def test_build_tree_from_html_json_normalizes_sections_and_metadata(tmp_path):
    image_path = tmp_path / "figure-1.png"
    image_path.write_bytes(b"fake-image")

    payload = {
        "title": "Handbook",
        "content": [
            "Intro paragraph.",
            {"text": "Second paragraph.", "role": "body_text", "id": "p2"},
        ],
        "images": [
            {"img_path": str(image_path), "caption": "Figure 1", "footnote": "Overview", "id": "img-1"}
        ],
        "sections": [
            {
                "title": "Section One",
                "content": "Nested body copy.",
                "provenance": {"section_id": "sec-1"},
            }
        ],
        "metadata": {"file_name": "example.html", "file_path": "/virtual/example.html"},
        "provenance": {"doc_id": "doc-1"},
    }
    html_json_path = tmp_path / "doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    tree = build_tree_from_source(cfg, reforce=True)

    titles = [node for node in tree.get_nodes() if node.type == NodeType.TITLE]
    texts = [node for node in tree.get_nodes() if node.type == NodeType.TEXT]
    images = [node for node in tree.get_nodes() if node.type == NodeType.IMAGE]

    assert [node.meta_info.content for node in titles] == ["Handbook", "Section One"]
    assert [node.meta_info.content for node in texts] == [
        "Intro paragraph.",
        "Second paragraph.",
        "Nested body copy.",
    ]
    assert len(images) == 1

    image_node = images[0]
    assert image_node.meta_info.caption == "Figure 1"
    assert image_node.meta_info.file_path == "/virtual/example.html"
    assert image_node.meta_info.source_type == "html_json"
    assert image_node.meta_info.source_role == "image"
    assert image_node.meta_info.provenance["doc_id"] == "doc-1"
    assert image_node.meta_info.provenance["id"] == "img-1"


def test_visual_leaf_candidate_selection_uses_local_assets_and_text_surrogates(tmp_path):
    shared_image_path = tmp_path / "visual.png"
    shared_image_path.write_bytes(b"visual")

    payload = {
        "title": "Visual Doc",
        "images": [
            {"img_path": str(shared_image_path), "caption": "Figure A", "footnote": "Alpha"},
            {"img_path": str(tmp_path / "missing.png"), "caption": "Figure B"},
        ],
        "sections": [
            {
                "title": "Data",
                "tables": [
                    {"img_path": str(shared_image_path), "caption": "Table 1", "table_body": "A | B"}
                ],
            }
        ],
    }
    html_json_path = tmp_path / "visual-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    tree = build_tree_from_source(cfg, reforce=True)

    text_dict, image_dict = process_tree_nodes(tree)
    candidates = select_visual_leaf_candidates(tree)

    assert any("Figure A Alpha" in text for text in text_dict["text"])
    assert any("Table 1" in text and "A | B" in text for text in text_dict["text"])
    assert len(image_dict["image"]) == 2
    assert len(candidates) == 2
    assert {candidate["node_type"] for candidate in candidates} == {"image", "table"}
    assert all(candidate["img_path"] == str(shared_image_path) for candidate in candidates)


def test_visual_sidecar_stub_writes_manifest_and_candidate_metadata(tmp_path):
    image_path = tmp_path / "sidecar.png"
    image_path.write_bytes(b"visual-sidecar")

    payload = {
        "title": "Sidecar Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [
                    {"img_path": str(image_path), "caption": "Figure S", "footnote": "Support"}
                ],
                "tables": [
                    {"img_path": str(image_path), "caption": "Table S", "table_body": "L | R"}
                ],
            }
        ],
    }
    html_json_path = tmp_path / "sidecar-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.tenant_id = "tenant-a"
    cfg.doc_id = "doc-a"
    tree = build_tree_from_source(cfg, reforce=True)

    manifest = build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
    )

    sidecar_dir = tmp_path / "index" / "visual_leaf_sidecar"
    manifest_path = sidecar_dir / "manifest.json"
    candidates_path = sidecar_dir / "candidates.json"

    assert manifest_path.exists()
    assert candidates_path.exists()
    assert manifest["status"] == "stub_only"
    assert manifest["candidate_count"] == 2

    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    assert {candidate["node_type"] for candidate in candidates} == {"image", "table"}
    assert all(candidate["tenant_id"] == "tenant-a" for candidate in candidates)
    assert all(candidate["doc_id"] == "doc-a" for candidate in candidates)
    assert all(candidate["path_from_root_ids"][-1] == candidate["node_id"] for candidate in candidates)
    assert all(candidate["nearest_title_ancestor_id"] is not None for candidate in candidates)
    assert all(candidate["retriever_family"] == "colqwen2" for candidate in candidates)