"""Tests for HTML JSON normalization into DocumentTree."""
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import yaml

from Core.Index.Tree import NodeType
from Core.configs.system_config import load_system_config
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.pipelines.doc_tree_builder import build_tree_from_source
import Core.provider.visual_sidecar as visual_sidecar_module
from Core.provider.visual_sidecar import build_visual_sidecar_backend, query_visual_sidecar
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


def _patch_fake_colqwen2_runtime(monkeypatch):
    class FakeImage:
        def __init__(self, path):
            self.path = str(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def convert(self, mode):
            return self

        def close(self):
            return None

    class FakeImageModule:
        @staticmethod
        def open(path):
            return FakeImage(path)

    class FakeBatch(dict):
        def to(self, device):
            self["device"] = device
            return self

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def to(self, device):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeQueryEmbedding:
        def __init__(self, query_text):
            self.query_text = query_text

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        bfloat16 = "bfloat16"

        class cuda:
            @staticmethod
            def is_available():
                return False

        class backends:
            class mps:
                @staticmethod
                def is_available():
                    return False

        @staticmethod
        def no_grad():
            return FakeNoGrad()

        @staticmethod
        def unbind(batch_embeddings, dim=0):
            return list(batch_embeddings)

        @staticmethod
        def save(payload, path):
            serializable_payload = {
                "backend_type": payload["backend_type"],
                "retriever_family": payload["retriever_family"],
                "retriever_model": payload["retriever_model"],
                "retriever_version": payload["retriever_version"],
                "documents": [
                    {
                        "document_id": doc["document_id"],
                        "image_path": doc["image_path"],
                        "embedding": doc["embedding"].tolist(),
                    }
                    for doc in payload["documents"]
                ],
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serializable_payload, f)

        @staticmethod
        def load(path, map_location=None):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            payload["documents"] = [
                {
                    **doc,
                    "embedding": FakeTensor(doc["embedding"]),
                }
                for doc in payload["documents"]
            ]
            return payload

    class FakeColQwen2:
        last_from_pretrained_kwargs = None

        def __init__(self):
            self.device = "cpu"

        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            cls.last_from_pretrained_kwargs = {"model_name": model_name, **kwargs}
            return cls()

        def eval(self):
            return self

        def __call__(self, **batch_inputs):
            if "images" in batch_inputs:
                return [
                    FakeTensor([float(index + 1), float(len(image.path))])
                    for index, image in enumerate(batch_inputs["images"])
                ]
            return FakeQueryEmbedding(batch_inputs["queries"][0])

    class FakeColQwen2Processor:
        @classmethod
        def from_pretrained(cls, model_name):
            instance = cls()
            instance.model_name = model_name
            return instance

        def process_images(self, images):
            return FakeBatch({"images": images})

        def process_queries(self, queries):
            return FakeBatch({"queries": queries})

        def score_multi_vector(self, query_embeddings, image_embeddings):
            query_text = query_embeddings.query_text.lower()
            if "finance" in query_text:
                return [[float(embedding.values[0]) for embedding in image_embeddings]]
            if "diagram" in query_text:
                return [[10.0 - float(embedding.values[0]) for embedding in image_embeddings]]
            return [[0.0 for _ in image_embeddings]]

    monkeypatch.setattr(
        visual_sidecar_module,
        "_load_colqwen2_runtime",
        lambda: {
            "torch": FakeTorch,
            "image_module": FakeImageModule,
            "ColQwen2": FakeColQwen2,
            "ColQwen2Processor": FakeColQwen2Processor,
            "is_flash_attn_2_available": lambda: False,
        },
    )

    return SimpleNamespace(
        FakeTorch=FakeTorch,
        FakeColQwen2=FakeColQwen2,
        FakeColQwen2Processor=FakeColQwen2Processor,
    )


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


def test_visual_sidecar_backend_materializes_prepared_corpus_artifacts(tmp_path):
    image_path = tmp_path / "backend.png"
    image_path.write_bytes(b"visual-backend")

    payload = {
        "title": "Backend Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [
                    {"img_path": str(image_path), "caption": "Figure B", "footnote": "Beta"}
                ],
                "tables": [
                    {"img_path": str(image_path), "caption": "Table B", "table_body": "X | Y"}
                ],
            }
        ],
    }
    html_json_path = tmp_path / "backend-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.visual_sidecar.retriever_model = "vidore/colqwen2-v1.0"
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    backend_manifest = build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    backend_dir = (
        tmp_path
        / "index"
        / "visual_leaf_sidecar"
        / "backends"
        / "prepared_corpus__colqwen2__v1"
    )
    backend_manifest_path = backend_dir / "backend_manifest.json"
    documents_path = backend_dir / "documents.jsonl"
    root_manifest_path = tmp_path / "index" / "visual_leaf_sidecar" / "manifest.json"

    assert backend_manifest_path.exists()
    assert documents_path.exists()
    assert backend_manifest["backend_type"] == "prepared_corpus"
    assert backend_manifest["document_count"] == 2
    assert backend_manifest["retriever_model"] == "vidore/colqwen2-v1.0"

    document_rows = [
        json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(document_rows) == 2
    assert all(row["image_path"] == str(image_path) for row in document_rows)
    assert all("metadata" in row for row in document_rows)

    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    assert root_manifest["status"] == "backend_materialized"
    assert root_manifest["backend_builds"][0]["backend_type"] == "prepared_corpus"


def test_query_visual_sidecar_returns_empty_when_artifacts_are_missing(tmp_path):
    html_json_path = tmp_path / "missing-sidecar-doc.json"
    html_json_path.write_text(json.dumps({"title": "Missing Sidecar"}), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)

    assert query_visual_sidecar(cfg.save_path, cfg.visual_sidecar, "figure evidence") == []


def test_query_visual_sidecar_returns_ranked_visual_hits_from_prepared_corpus(tmp_path):
    image_path = tmp_path / "query-backend.png"
    image_path.write_bytes(b"visual-backend")

    payload = {
        "title": "Query Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [{"img_path": str(image_path), "caption": "Architecture Figure", "footnote": "Overview"}],
                "tables": [{"img_path": str(image_path), "caption": "Revenue Table", "table_body": "North | 10"}],
            }
        ],
    }
    html_json_path = tmp_path / "query-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    table_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.TABLE and node.meta_info.caption == "Revenue Table"
    )

    hits = query_visual_sidecar(
        cfg.save_path,
        cfg.visual_sidecar,
        "revenue north",
        top_k=2,
    )

    assert hits
    assert hits[0]["node_id"] == table_node.index_id
    assert hits[0]["node_type"] == "table"
    assert hits[0]["retrieval_source"] == "prepared_corpus"


def test_gbc_get_info_augments_tree_nodes_with_visual_sidecar_hits(tmp_path, monkeypatch):
    image_path = tmp_path / "query-gbc.png"
    image_path.write_bytes(b"visual-backend")

    payload = {
        "title": "Query Doc",
        "sections": [
            {
                "title": "Evidence",
                "content": "Body paragraph.",
                "images": [{"img_path": str(image_path), "caption": "Support Figure", "footnote": "Diagram"}],
            }
        ],
    }
    html_json_path = tmp_path / "query-gbc-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    section_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.TITLE and node.meta_info.content == "Evidence"
    )
    text_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.TEXT and node.meta_info.content == "Body paragraph."
    )
    image_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.IMAGE and node.meta_info.caption == "Support Figure"
    )

    class FakeRetriever:
        def skyline_filter(self, sub_query, subtree_nodes, subgraph, start_ent_map):
            return [text_node.index_id], []

    class FakeGraphIndex:
        @staticmethod
        def get_kg_subgraph(subtree_ids):
            return object()

        @staticmethod
        def get_subgraph_data(res_entities):
            return {"nodes": []}

    fake_networkx = ModuleType("networkx")

    class FakeNetworkXGraph:
        pass

    fake_networkx.Graph = FakeNetworkXGraph
    fake_networkx.pagerank = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "networkx", fake_networkx)

    fake_rag_package = ModuleType("Core.rag")
    fake_rag_package.__path__ = []
    monkeypatch.setitem(sys.modules, "Core.rag", fake_rag_package)

    fake_base_rag = ModuleType("Core.rag.base_rag")

    class FakeBaseRAG:
        pass

    fake_base_rag.BaseRAG = FakeBaseRAG
    monkeypatch.setitem(sys.modules, "Core.rag.base_rag", fake_base_rag)

    fake_llm = ModuleType("Core.provider.llm")

    class FakeLLM:
        pass

    fake_llm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "Core.provider.llm", fake_llm)

    fake_vlm = ModuleType("Core.provider.vlm")

    class FakeVLM:
        pass

    fake_vlm.VLM = FakeVLM
    monkeypatch.setitem(sys.modules, "Core.provider.vlm", fake_vlm)

    fake_rerank = ModuleType("Core.provider.rerank")

    class FakeTextRerankerProvider:
        def __init__(self, *args, **kwargs):
            pass

    fake_rerank.TextRerankerProvider = FakeTextRerankerProvider
    monkeypatch.setitem(sys.modules, "Core.provider.rerank", fake_rerank)

    fake_gbc_index = ModuleType("Core.Index.GBCIndex")

    class FakeGBC:
        pass

    fake_gbc_index.GBC = FakeGBC
    monkeypatch.setitem(sys.modules, "Core.Index.GBCIndex", fake_gbc_index)

    fake_prompt = ModuleType("Core.prompts.gbc_prompt")
    fake_prompt.LLM_EXPANSION_SELECT_PROMPT = ""

    class FakeQuestionEntity:
        pass

    class FakeQuestionEntityExtraction:
        pass

    class FakeSecEXPSelection:
        pass

    fake_prompt.QuestionEntity = FakeQuestionEntity
    fake_prompt.QuestionEntityExtraction = FakeQuestionEntityExtraction
    fake_prompt.QUESTION_ENT_PROMPT = ""
    fake_prompt.QUESTION_ENTITY_TYPES = []
    fake_prompt.SecEXPSelection = FakeSecEXPSelection
    monkeypatch.setitem(sys.modules, "Core.prompts.gbc_prompt", fake_prompt)

    fake_graph = ModuleType("Core.Index.Graph")

    class FakeEntity:
        pass

    fake_graph.Entity = FakeEntity
    monkeypatch.setitem(sys.modules, "Core.Index.Graph", fake_graph)

    fake_answer = ModuleType("Core.rag.gbc_answer")

    class FakeAnswerAgent:
        pass

    fake_answer.AnswerAgent = FakeAnswerAgent
    monkeypatch.setitem(sys.modules, "Core.rag.gbc_answer", fake_answer)

    fake_plan = ModuleType("Core.rag.gbc_plan")

    class FakeTaskPlanner:
        pass

    class FakePlanResult:
        pass

    fake_plan.TaskPlanner = FakeTaskPlanner
    fake_plan.PlanResult = FakePlanResult
    monkeypatch.setitem(sys.modules, "Core.rag.gbc_plan", fake_plan)

    fake_retrieval = ModuleType("Core.rag.gbc_retrieval")

    class FakeRetrieverType:
        pass

    fake_retrieval.Retriever = FakeRetrieverType
    monkeypatch.setitem(sys.modules, "Core.rag.gbc_retrieval", fake_retrieval)

    fake_utils = ModuleType("Core.rag.gbc_utils")

    class FakeGBCRAGContext:
        pass

    class FakeSubStep:
        pass

    fake_utils.GBCRAGContext = FakeGBCRAGContext
    fake_utils.SubStep = FakeSubStep
    fake_utils.filter_tree_nodes = lambda *args, **kwargs: []
    monkeypatch.setitem(sys.modules, "Core.rag.gbc_utils", fake_utils)

    fake_ontology = ModuleType("Core.utils.ontology_utils")
    fake_ontology.find_best_graph_ontology_node = lambda *args, **kwargs: None
    fake_ontology.normalize_entity_name = lambda value: value
    fake_ontology.normalize_entity_type = lambda value: value
    monkeypatch.setitem(sys.modules, "Core.utils.ontology_utils", fake_ontology)

    gbc_rag_path = Path(__file__).resolve().parents[1] / "Core" / "rag" / "gbc_rag.py"
    spec = importlib.util.spec_from_file_location("_test_gbc_rag_module", gbc_rag_path)
    gbc_rag_module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(gbc_rag_module)
    GBCRAG = gbc_rag_module.GBCRAG

    rag = GBCRAG.__new__(GBCRAG)
    rag.cfg = GBCRAGConfig(
        visual_sidecar_query_enabled=True,
        visual_sidecar_query_topk=2,
    )
    rag.varient = "standard"
    rag.gbc_index = SimpleNamespace(
        TreeIndex=tree,
        GraphIndex=FakeGraphIndex(),
        config=cfg,
        save_dir=cfg.save_path,
    )
    rag.retriever = FakeRetriever()

    iter_context = SimpleNamespace(
        sub_query="support figure",
        retrieval_sec_ids=[section_node.index_id],
        gbc_entity_map={},
        supplementary_ids=[],
    )

    rag.get_GBC_info(iter_context)

    retrieved_ids = [node["index_id"] for node in iter_context.retrieval_nodes]
    assert text_node.index_id in retrieved_ids
    assert image_node.index_id in retrieved_ids
    assert iter_context.supplementary_ids == [image_node.index_id]


def test_visual_sidecar_backend_colqwen2_local_reports_missing_runtime_dependencies(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "colqwen2-missing.png"
    image_path.write_bytes(b"visual-backend")

    payload = {
        "title": "Backend Doc",
        "images": [{"img_path": str(image_path), "caption": "Figure M"}],
    }
    html_json_path = tmp_path / "backend-missing-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.visual_sidecar.backend_type = "colqwen2_local"
    cfg.visual_sidecar.retriever_model = "vidore/colqwen2-v1.0"
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )

    original_import_module = visual_sidecar_module.importlib.import_module

    def fake_import_module(name):
        if name == "colpali_engine.models":
            raise ImportError("colpali-engine unavailable for test")
        return original_import_module(name)

    monkeypatch.setattr(visual_sidecar_module.importlib, "import_module", fake_import_module)

    with pytest.raises(
        visual_sidecar_module.VisualSidecarDependencyError,
        match="colpali-engine",
    ):
        build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)


def test_visual_sidecar_backend_materializes_colqwen2_local_artifacts_with_mocks(
    tmp_path, monkeypatch
):
    image_path = tmp_path / "colqwen2-local.png"
    image_path.write_bytes(b"visual-backend")

    payload = {
        "title": "Backend Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [{"img_path": str(image_path), "caption": "Figure C"}],
                "tables": [{"img_path": str(image_path), "caption": "Table C", "table_body": "1 | 2"}],
            }
        ],
    }
    html_json_path = tmp_path / "colqwen2-local-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.visual_sidecar.backend_type = "colqwen2_local"
    cfg.visual_sidecar.retriever_model = "vidore/colqwen2-v1.0"
    cfg.visual_sidecar.image_batch_size = 2
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    fake_runtime = _patch_fake_colqwen2_runtime(monkeypatch)

    backend_manifest = build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    backend_dir = (
        tmp_path
        / "index"
        / "visual_leaf_sidecar"
        / "backends"
        / "colqwen2_local__colqwen2__v1"
    )
    backend_manifest_path = backend_dir / "backend_manifest.json"
    embeddings_path = backend_dir / "multivector_embeddings.pt"
    root_manifest_path = tmp_path / "index" / "visual_leaf_sidecar" / "manifest.json"

    assert backend_manifest_path.exists()
    assert embeddings_path.exists()
    assert backend_manifest["backend_type"] == "colqwen2_local"
    assert backend_manifest["document_count"] == 2
    assert backend_manifest["encoded_count"] == 2
    assert backend_manifest["runtime"]["device"] == "cpu"

    saved_embeddings = json.loads(embeddings_path.read_text(encoding="utf-8"))
    assert saved_embeddings["retriever_model"] == "vidore/colqwen2-v1.0"
    assert len(saved_embeddings["documents"]) == 2

    assert fake_runtime.FakeColQwen2.last_from_pretrained_kwargs["device_map"] == "cpu"
    assert fake_runtime.FakeColQwen2.last_from_pretrained_kwargs["torch_dtype"] == fake_runtime.FakeTorch.bfloat16

    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    assert root_manifest["status"] == "backend_materialized"
    assert root_manifest["backend_builds"][0]["backend_type"] == "colqwen2_local"
    assert root_manifest["backend_builds"][0]["encoded_count"] == 2


def test_query_visual_sidecar_uses_model_backed_colqwen2_scores_with_mocks(
    tmp_path, monkeypatch
):
    image_a_path = tmp_path / "colqwen2-query-a.png"
    image_b_path = tmp_path / "colqwen2-query-b.png"
    image_a_path.write_bytes(b"visual-a")
    image_b_path.write_bytes(b"visual-b")

    payload = {
        "title": "Backend Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [{"img_path": str(image_a_path), "caption": "Alpha Panel"}],
                "tables": [{"img_path": str(image_b_path), "caption": "Beta Panel", "table_body": "1 | 2"}],
            }
        ],
    }
    html_json_path = tmp_path / "colqwen2-query-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.visual_sidecar.backend_type = "colqwen2_local"
    cfg.visual_sidecar.retriever_model = "vidore/colqwen2-v1.0"
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    _patch_fake_colqwen2_runtime(monkeypatch)
    build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    table_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.TABLE and node.meta_info.caption == "Beta Panel"
    )

    hits = query_visual_sidecar(
        cfg.save_path,
        cfg.visual_sidecar,
        "finance intent",
        top_k=2,
    )

    assert hits
    assert hits[0]["node_id"] == table_node.index_id
    assert hits[0]["retrieval_source"] == "colqwen2_local"


def test_query_visual_sidecar_falls_back_to_textual_matching_when_model_query_fails(
    tmp_path, monkeypatch
):
    image_a_path = tmp_path / "colqwen2-fallback-a.png"
    image_b_path = tmp_path / "colqwen2-fallback-b.png"
    image_a_path.write_bytes(b"visual-a")
    image_b_path.write_bytes(b"visual-b")

    payload = {
        "title": "Backend Doc",
        "sections": [
            {
                "title": "Evidence",
                "images": [{"img_path": str(image_a_path), "caption": "Alpha Panel"}],
                "tables": [{"img_path": str(image_b_path), "caption": "Beta Panel", "table_body": "1 | 2"}],
            }
        ],
    }
    html_json_path = tmp_path / "colqwen2-fallback-doc.json"
    html_json_path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = _build_test_cfg(tmp_path, html_json_path)
    cfg.visual_sidecar.backend_type = "colqwen2_local"
    cfg.visual_sidecar.retriever_model = "vidore/colqwen2-v1.0"
    tree = build_tree_from_source(cfg, reforce=True)

    build_visual_sidecar_stub(
        tree,
        save_path=cfg.save_path,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        retriever_family=cfg.visual_sidecar.retriever_family,
        retriever_model=cfg.visual_sidecar.retriever_model,
        retriever_version=cfg.visual_sidecar.retriever_version,
    )
    _patch_fake_colqwen2_runtime(monkeypatch)
    build_visual_sidecar_backend(cfg.save_path, cfg.visual_sidecar)

    image_node = next(
        node
        for node in tree.get_nodes()
        if node.type == NodeType.IMAGE and node.meta_info.caption == "Alpha Panel"
    )

    monkeypatch.setattr(
        visual_sidecar_module,
        "_load_colqwen2_runtime",
        lambda: (_ for _ in ()).throw(
            visual_sidecar_module.VisualSidecarDependencyError("query runtime unavailable")
        ),
    )

    hits = query_visual_sidecar(
        cfg.save_path,
        cfg.visual_sidecar,
        "alpha panel",
        top_k=2,
    )

    assert hits
    assert hits[0]["node_id"] == image_node.index_id
    assert hits[0]["retrieval_source"] == "colqwen2_local"