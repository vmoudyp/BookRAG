import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

from Core.configs.graph_config import GraphConfig
import Core.pipelines.doc_tree_builder as doc_tree_builder
import Core.pipelines.kg_builder as kg_builder
from Core.provider.extract_pdf_info_docling import is_known_docling_meta_tensor_error
from Core.Index.Tree import DocumentTree, NodeType, TreeNode
from Core.pipelines.kg_extractor import HybridExtractor


def _install_doc_tree_runtime_stubs(monkeypatch):
    def fake_extract_pdf_outline_in_chunks(_pdf_list, _llm, lang=None):
        _ = (_pdf_list, _llm, lang)
        return [{"text": "Section", "pdf_id": 1}]

    def fake_pdf_info_refiner(pdf_list, _llm, lang=None):
        _ = (_llm, lang)
        return pdf_list

    def fake_detect_document_language(_pdf_list, fallback="en"):
        _ = (_pdf_list, fallback)
        return "en"

    def fake_detect_legal_headings(pdf_list, lang=None):
        _ = lang
        return pdf_list

    outline = ModuleType("Core.pipelines.outline_extractor")
    outline.extract_pdf_outline_in_chunks = fake_extract_pdf_outline_in_chunks
    refiner = ModuleType("Core.pipelines.pdf_refiner")
    refiner.pdf_info_refiner = fake_pdf_info_refiner
    headings = ModuleType("Core.pipelines.legal_heading_detector")
    headings.detect_document_language = fake_detect_document_language
    headings.detect_legal_headings = fake_detect_legal_headings
    llm_mod = ModuleType("Core.provider.llm")
    llm_mod.LLM = lambda cfg: SimpleNamespace(config=cfg)
    vlm_mod = ModuleType("Core.provider.vlm")
    vlm_mod.VLM = lambda cfg: SimpleNamespace(config=cfg)
    tracker_mod = ModuleType("Core.provider.TokenTracker")

    class _Tracker:
        @staticmethod
        def get_instance():
            def record_stage(stage):
                _ = stage
                return 0

            return SimpleNamespace(record_stage=record_stage)

    tracker_mod.TokenTracker = _Tracker
    for name, module in {
        "Core.pipelines.outline_extractor": outline,
        "Core.pipelines.pdf_refiner": refiner,
        "Core.pipelines.legal_heading_detector": headings,
        "Core.provider.llm": llm_mod,
        "Core.provider.vlm": vlm_mod,
        "Core.provider.TokenTracker": tracker_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_build_tree_from_pdf_recovers_from_root_only_tree(tmp_path, monkeypatch):
    _install_doc_tree_runtime_stubs(monkeypatch)
    save_path = tmp_path / "index"
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")
    cached_pdf_list = [
        {"type": "text", "text": "Body paragraph.", "pdf_id": 1, "page_idx": 0},
        {"type": "table", "table_caption": ["Revenue Table"], "pdf_id": 2, "page_idx": 0},
    ]
    cache_path = save_path / "auto" / "sample_merged_content.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cached_pdf_list), encoding="utf-8")
    cfg = SimpleNamespace(
        save_path=str(save_path),
        pdf_path=str(pdf_path),
        parser="mineru",
        mineru=SimpleNamespace(method="auto", backend="pipeline", server_url="", lang="en"),
        tree=SimpleNamespace(use_vlm=False, node_summary=False),
        llm=SimpleNamespace(),
        vlm=SimpleNamespace(),
        document_lang="en",
    )
    real_construct = doc_tree_builder.construct_tree_index

    def fake_construct(tree_index, pdf_list, title_outline):
        return tree_index if title_outline else real_construct(tree_index, pdf_list, title_outline)

    monkeypatch.setattr(doc_tree_builder, "construct_tree_index", fake_construct)
    tree = doc_tree_builder.build_tree_from_pdf(cast(Any, cfg))
    assert len(tree.nodes) == 3
    assert [node.type for node in tree.get_nodes()] == [NodeType.TEXT, NodeType.TABLE]
    assert not tree.root_node.is_leaf()
    assert Path(DocumentTree.get_save_path(str(save_path))).exists()


def test_build_tree_from_pdf_falls_back_to_mineru_when_docling_hits_meta_tensor(
    tmp_path, monkeypatch
):
    _install_doc_tree_runtime_stubs(monkeypatch)
    save_path = tmp_path / "index"
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")
    fallback_pdf_list = [
        {"type": "text", "text": "Fallback paragraph.", "pdf_id": 0, "page_idx": 0}
    ]
    calls = {"docling": 0, "mineru": 0}

    docling_mod = ModuleType("Core.provider.extract_pdf_info_docling")

    def fake_parse_doc_with_docling(pdf_path, output_dir, cfg):
        _ = (pdf_path, output_dir, cfg)
        calls["docling"] += 1
        raise NotImplementedError("Cannot copy out of meta tensor; no data!")

    docling_mod.parse_doc_with_docling = fake_parse_doc_with_docling
    docling_mod.is_known_docling_meta_tensor_error = is_known_docling_meta_tensor_error

    mineru_mod = ModuleType("Core.provider.extract_pdf_info")

    def fake_parse_doc(
        pdf_path, output_dir, lang="en", backend="pipeline", method="auto", server_url=None
    ):
        _ = (pdf_path, output_dir, lang, backend, method, server_url)
        calls["mineru"] += 1
        return {"pdf_info": []}, fallback_pdf_list

    mineru_mod.parse_doc = fake_parse_doc

    def fake_merge_middle_content(middle_json, content_list, parse_dir, save_dir=None, file_name=None):
        _ = (middle_json, parse_dir, save_dir, file_name)
        return content_list

    mineru_mod.merge_middle_content = fake_merge_middle_content

    monkeypatch.setitem(sys.modules, "Core.provider.extract_pdf_info_docling", docling_mod)
    monkeypatch.setitem(sys.modules, "Core.provider.extract_pdf_info", mineru_mod)

    cfg = SimpleNamespace(
        save_path=str(save_path),
        pdf_path=str(pdf_path),
        parser="docling",
        docling=SimpleNamespace(images_scale=2.0, ocr_engine="easyocr", force_full_page_ocr=False, lang="en"),
        mineru=SimpleNamespace(method="auto", backend="pipeline", server_url="", lang="en"),
        tree=SimpleNamespace(use_vlm=False, node_summary=False),
        llm=SimpleNamespace(),
        vlm=SimpleNamespace(),
        document_lang="en",
    )
    real_construct = doc_tree_builder.construct_tree_index

    def fake_construct(tree_index, pdf_list, title_outline):
        return tree_index if title_outline else real_construct(tree_index, pdf_list, title_outline)

    monkeypatch.setattr(doc_tree_builder, "construct_tree_index", fake_construct)
    tree = doc_tree_builder.build_tree_from_pdf(cast(Any, cfg))
    cache_path = save_path / "docling" / "sample_merged_content.json"

    assert calls == {"docling": 1, "mineru": 1}
    assert cache_path.exists()
    assert json.loads(cache_path.read_text(encoding="utf-8")) == fallback_pdf_list
    assert len(tree.nodes) == 2
    assert tree.get_nodes()[0].meta_info.content == "Fallback paragraph."


def test_is_known_docling_meta_tensor_error_walks_exception_chain():
    try:
        try:
            raise NotImplementedError("Cannot copy out of meta tensor; no data!")
        except NotImplementedError as exc:
            raise RuntimeError("Docling convert failed") from exc
    except RuntimeError as exc:
        assert is_known_docling_meta_tensor_error(exc)


def test_hybrid_extractor_uses_bert_for_title_and_text_nodes(monkeypatch):
    extractor = HybridExtractor(
        graph_config=SimpleNamespace(
            max_gleaning=0,
            image_description_force=False,
            hybrid_ner_model="stub-model",
            ner_confidence_threshold=0.5,
        ),
        llm=SimpleNamespace(),
        vlm=None,
    )
    monkeypatch.setattr(
        extractor,
        "_extract_kg_from_text",
        lambda node: ([{"entity_name": node.meta_info.content.lower(), "entity_type": "LAW"}], []),
    )
    monkeypatch.setattr(extractor._llm_extractor, "extract", lambda node: {"entities": [{"entity_name": "visual"}], "relations": [], "node_idx": node.index_id})
    title_node = TreeNode({"content": "Addendum", "pdf_id": 1})
    title_node.type = NodeType.TITLE
    title_node.index_id = 7
    text_node = TreeNode({"content": "Pasal 12", "pdf_id": 2})
    text_node.type = NodeType.TEXT
    text_node.index_id = 8
    table_node = TreeNode({"content": "Table", "pdf_id": 3})
    table_node.type = NodeType.TABLE
    table_node.index_id = 9
    assert extractor.extract_title(title_node, [], [])["entities"][0]["entity_name"] == "addendum"
    assert extractor.extract(text_node)["entities"][0]["entity_name"] == "pasal 12"
    assert extractor.extract(table_node)["entities"][0]["entity_name"] == "visual"


def test_hybrid_extractor_reconstructs_wordpiece_entities_from_offsets(monkeypatch):
    extractor = HybridExtractor(
        graph_config=SimpleNamespace(
            max_gleaning=0,
            image_description_force=False,
            hybrid_ner_model="stub-model",
            ner_confidence_threshold=0.5,
        ),
        llm=SimpleNamespace(),
        vlm=None,
    )
    text = (
        "PT Avrist Assurance dan PT Reasuransi Indonesia Utama (Persero) berlaku "
        "01 September 2018 di Jakarta."
    )
    raw_tokens = [
        {"entity": "I-ORG", "score": 0.60268456, "index": 1, "word": "Av", "start": 3, "end": 5},
        {"entity": "I-LAW", "score": 0.56280196, "index": 2, "word": "##rist", "start": 5, "end": 9},
        {"entity": "I-ORG", "score": 0.599163, "index": 3, "word": "Ass", "start": 10, "end": 13},
        {"entity": "I-ORG", "score": 0.49348155, "index": 4, "word": "##urance", "start": 13, "end": 19},
        {"entity": "B-ORG", "score": 0.97788584, "index": 6, "word": "PT", "start": 24, "end": 26},
        {"entity": "I-ORG", "score": 0.98870057, "index": 7, "word": "Re", "start": 27, "end": 29},
        {"entity": "I-ORG", "score": 0.9876466, "index": 8, "word": "##asuransi", "start": 29, "end": 37},
        {"entity": "I-ORG", "score": 0.99516064, "index": 9, "word": "Indonesia", "start": 38, "end": 47},
        {"entity": "I-ORG", "score": 0.9932288, "index": 10, "word": "Utama", "start": 48, "end": 53},
        {"entity": "I-ORG", "score": 0.9627691, "index": 11, "word": "(", "start": 54, "end": 55},
        {"entity": "I-ORG", "score": 0.9652233, "index": 12, "word": "Persero", "start": 55, "end": 62},
        {"entity": "I-LAW", "score": 0.32361704, "index": 13, "word": ")", "start": 62, "end": 63},
        {"entity": "B-DAT", "score": 0.99870694, "index": 15, "word": "01", "start": 72, "end": 74},
        {"entity": "I-DAT", "score": 0.9989754, "index": 16, "word": "September", "start": 75, "end": 84},
        {"entity": "I-DAT", "score": 0.9988427, "index": 17, "word": "2018", "start": 85, "end": 89},
        {"entity": "B-GPE", "score": 0.9938984, "index": 19, "word": "Jakarta", "start": 93, "end": 100},
    ]

    def fake_ner_pipeline(chunk):
        _ = chunk
        return raw_tokens

    monkeypatch.setattr(extractor, "_get_ner_pipeline", lambda: fake_ner_pipeline)

    entities = extractor._bert_extract_entities(text, node_id=12)
    names = {entity.entity_name: entity.entity_type for entity in entities}

    assert "Avrist Assurance" in names
    assert names["Avrist Assurance"] == "ORGANIZATION"
    assert "PT Reasuransi Indonesia Utama (Persero)" in names
    assert names["PT Reasuransi Indonesia Utama (Persero)"] == "ORGANIZATION"
    assert "01 September 2018" in names
    assert "Jakarta" in names
    assert all("##" not in entity.entity_name for entity in entities)


def test_hybrid_extractor_moves_trailing_pt_prefix_to_following_org(monkeypatch):
    extractor = HybridExtractor(
        graph_config=SimpleNamespace(
            max_gleaning=0,
            image_description_force=False,
            hybrid_ner_model="stub-model",
            ner_confidence_threshold=0.5,
        ),
        llm=SimpleNamespace(),
        vlm=None,
    )
    text = "Addendum 1 Perjanjian PT Avrist Assurance"
    raw_tokens = [
        {"entity": "B-LAW", "score": 0.96, "index": 1, "word": "Addendum", "start": 0, "end": 8},
        {"entity": "I-LAW", "score": 0.95, "index": 2, "word": "1", "start": 9, "end": 10},
        {"entity": "I-LAW", "score": 0.94, "index": 3, "word": "Perjanjian", "start": 11, "end": 21},
        {"entity": "I-LAW", "score": 0.93, "index": 4, "word": "PT", "start": 22, "end": 24},
        {"entity": "I-ORG", "score": 0.92, "index": 5, "word": "Av", "start": 25, "end": 27},
        {"entity": "I-LAW", "score": 0.91, "index": 6, "word": "##rist", "start": 27, "end": 31},
        {"entity": "I-ORG", "score": 0.9, "index": 7, "word": "Ass", "start": 32, "end": 35},
        {"entity": "I-ORG", "score": 0.89, "index": 8, "word": "##urance", "start": 35, "end": 41},
    ]

    def fake_ner_pipeline(chunk):
        _ = chunk
        return raw_tokens

    monkeypatch.setattr(extractor, "_get_ner_pipeline", lambda: fake_ner_pipeline)

    entities = extractor._bert_extract_entities(text, node_id=13)
    names = {entity.entity_name: entity.entity_type for entity in entities}

    assert names["Addendum 1 Perjanjian"] == "LAW"
    assert names["PT Avrist Assurance"] == "ORGANIZATION"
    assert "Addendum 1 Perjanjian PT" not in names


def test_graph_config_defaults_to_cahya_hybrid_ner_model():
    assert GraphConfig().hybrid_ner_model == "cahya/bert-base-indonesian-NER"


def test_hybrid_extractor_maps_prd_label_to_product(monkeypatch):
    extractor = HybridExtractor(
        graph_config=SimpleNamespace(
            max_gleaning=0,
            image_description_force=False,
            hybrid_ner_model="stub-model",
            ner_confidence_threshold=0.5,
        ),
        llm=SimpleNamespace(),
        vlm=None,
    )
    text = "BookRAG Premium"
    raw_tokens = [
        {"entity": "B-PRD", "score": 0.99, "index": 1, "word": "Book", "start": 0, "end": 4},
        {"entity": "I-PRD", "score": 0.99, "index": 2, "word": "##RAG", "start": 4, "end": 7},
        {"entity": "I-PRD", "score": 0.99, "index": 3, "word": "Premium", "start": 8, "end": 15},
    ]

    def fake_ner_pipeline(chunk):
        _ = chunk
        return raw_tokens

    monkeypatch.setattr(extractor, "_get_ner_pipeline", lambda: fake_ner_pipeline)

    entities = extractor._bert_extract_entities(text, node_id=21)

    assert [(entity.entity_name, entity.entity_type) for entity in entities] == [
        ("BookRAG Premium", "PRODUCT")
    ]


def test_hybrid_extractor_resolves_existing_local_model_dir_before_loading(monkeypatch, tmp_path):
    local_model_dir = tmp_path / "local-ner-model"
    local_model_dir.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path))

    fake_transformers = ModuleType("transformers")
    captured = {}

    def fake_pipeline(task, model, device):
        captured.update({"task": task, "model": model, "device": device})
        return "stub-pipeline"

    fake_transformers.pipeline = fake_pipeline
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    extractor = HybridExtractor(
        graph_config=SimpleNamespace(
            max_gleaning=0,
            image_description_force=False,
            hybrid_ner_model="~/local-ner-model",
            ner_confidence_threshold=0.5,
        ),
        llm=SimpleNamespace(),
        vlm=None,
    )

    pipeline = extractor._get_ner_pipeline()

    assert pipeline == "stub-pipeline"
    assert captured == {
        "task": "ner",
        "model": str(local_model_dir.resolve()),
        "device": -1,
    }


def test_build_knowledge_graph_batches_nodes_by_tree_role(tmp_path, monkeypatch):
    cfg = SimpleNamespace(
        save_path=str(tmp_path),
        llm=SimpleNamespace(),
        vlm=SimpleNamespace(),
        tenant_id=None,
        doc_id=None,
        falkordb=None,
        ontology=SimpleNamespace(),
        graph=SimpleNamespace(
            image_description_force=False,
            refine_type="basic",
            role_graph_materialization=False,
            text_extraction_scope="body_text_leaves",
        ),
    )
    tree = DocumentTree(meta_dict={"file_name": "doc.pdf", "file_path": "doc.pdf"}, cfg=cfg)
    title, leaf, internal, child, table, caption, blank = [
        TreeNode({"content": name, "pdf_id": i})
        for i, name in enumerate(["Title", "Leaf", "Internal", "Child", "Table", "Caption", "   "], start=1)
    ]
    title.type, leaf.type, internal.type, child.type, table.type, caption.type, blank.type = (
        NodeType.TITLE,
        NodeType.TEXT,
        NodeType.TEXT,
        NodeType.TEXT,
        NodeType.TABLE,
        NodeType.TEXT,
        NodeType.TEXT,
    )
    leaf.meta_info.source_role = "body_text"
    internal.meta_info.source_role = "body_text"
    child.meta_info.source_role = "body_text"
    caption.meta_info.source_role = "caption"
    blank.meta_info.source_role = "body_text"
    for node in [title, leaf, internal, child, table, caption, blank]:
        tree.add_node(node)
    tree.root_node.add_child(title)
    tree.root_node.add_child(leaf)
    tree.root_node.add_child(internal)
    internal.add_child(child)
    tree.root_node.add_child(table)
    tree.root_node.add_child(caption)
    tree.root_node.add_child(blank)
    calls = {"titles": [], "batches": []}

    class FakeKGExtractor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def batch_extract_titles(self, nodes, title_paths, sibling_nodes_list):
            _ = (title_paths, sibling_nodes_list)
            calls["titles"] = [node.index_id for node in nodes]
            return [{"node_idx": node.index_id, "entities": [], "relations": []} for node in nodes]

        def batch_extract_kg(self, nodes, _max_workers=4):
            _ = _max_workers
            calls["batches"].append([node.index_id for node in nodes])
            return [{"node_idx": node.index_id, "entities": [], "relations": []} for node in nodes]

    class FakeKGRefiner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def basic_kg_refiner(self, entities, relationships, source_id):
            _ = (entities, relationships, source_id)
            return None

        def refine_entities(self):
            return None

        def refine_relation(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(kg_builder, "LLM", lambda cfg: SimpleNamespace(config=cfg))
    monkeypatch.setattr(kg_builder, "VLM", lambda cfg: SimpleNamespace(config=cfg))
    monkeypatch.setattr(kg_builder, "Graph", lambda **kwargs: SimpleNamespace(kwargs=kwargs))
    monkeypatch.setattr(kg_builder, "KGExtractor", FakeKGExtractor)
    monkeypatch.setattr(kg_builder, "KGRefiner", FakeKGRefiner)
    monkeypatch.setattr(
        kg_builder,
        "align_entities_to_ontology",
        lambda entities, relationships, ontology_cfg: (ontology_cfg, entities, relationships)[1:],
    )
    monkeypatch.setattr(
        kg_builder,
        "TokenTracker",
        SimpleNamespace(
            get_instance=lambda: SimpleNamespace(
                record_stage=lambda stage: (stage, 0)[1]
            )
        ),
    )
    kg_builder.build_knowledge_graph(tree, cast(Any, cfg))
    assert leaf.is_leaf() and child.is_leaf() and not internal.is_leaf()
    assert calls["titles"] == [title.index_id]
    assert calls["batches"] == [[leaf.index_id, child.index_id]]