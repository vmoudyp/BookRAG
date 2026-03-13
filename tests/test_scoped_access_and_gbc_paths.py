import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from api.db.mongodb import _document_scope_filter, _normalize_sub_tenant


def test_document_scope_filter_includes_shared_docs_and_matching_sub_tenant():
    shared_only = {
        "$or": [
            {"sub_tenant": {"$exists": False}},
            {"sub_tenant": None},
            {"sub_tenant": ""},
        ]
    }

    assert _normalize_sub_tenant("  ") is None
    assert _normalize_sub_tenant(" finance ") == "finance"
    assert _document_scope_filter(None) == shared_only
    assert _document_scope_filter(" finance ") == {
        "$or": [
            {"sub_tenant": {"$exists": False}},
            {"sub_tenant": None},
            {"sub_tenant": ""},
            {"sub_tenant": "finance"},
        ]
    }


def test_gbc_entity_vdb_path_uses_document_scoped_save_dir(monkeypatch):
    fake_tree = ModuleType("Core.Index.Tree")
    fake_tree.DocumentTree = type("FakeDocumentTree", (), {})

    fake_graph = ModuleType("Core.Index.Graph")
    fake_graph.Graph = type("FakeGraph", (), {})

    fake_system_config = ModuleType("Core.configs.system_config")
    fake_system_config.SystemConfig = object

    fake_llm = ModuleType("Core.provider.llm")
    fake_llm.LLM = lambda cfg: SimpleNamespace(config=cfg)

    fake_embedding = ModuleType("Core.provider.embedding")

    class FakeTextEmbeddingProvider:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_embedding.TextEmbeddingProvider = FakeTextEmbeddingProvider

    fake_vdb = ModuleType("Core.provider.vdb")

    class FakeVectorStore:
        def __init__(self, db_path, embedding_model, collection_name):
            self.db_path = db_path
            self.embedding_model = embedding_model
            self.collection_name = collection_name

    fake_vdb.VectorStore = FakeVectorStore

    monkeypatch.setitem(sys.modules, "Core.Index.Tree", fake_tree)
    monkeypatch.setitem(sys.modules, "Core.Index.Graph", fake_graph)
    monkeypatch.setitem(sys.modules, "Core.configs.system_config", fake_system_config)
    monkeypatch.setitem(sys.modules, "Core.provider.llm", fake_llm)
    monkeypatch.setitem(sys.modules, "Core.provider.embedding", fake_embedding)
    monkeypatch.setitem(sys.modules, "Core.provider.vdb", fake_vdb)

    gbc_path = Path(__file__).resolve().parents[1] / "Core" / "Index" / "GBCIndex.py"
    spec = importlib.util.spec_from_file_location("_test_gbcindex_module", gbc_path)
    gbc_module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(gbc_module)

    cfg = SimpleNamespace(
        save_path="/tmp/bookrag-indices/tenant-a/doc-1",
        llm=SimpleNamespace(),
        graph=SimpleNamespace(
            refine_type="basic",
            embedding_config=SimpleNamespace(
                model_name="test-model",
                backend="local",
                max_length=512,
                device="cpu",
                api_base=None,
                api_key=None,
            ),
        ),
    )

    gbc = gbc_module.GBC(config=cfg)

    assert gbc.entity_vdb_path == "/tmp/bookrag-indices/tenant-a/doc-1/kg_vdb_basic"
    assert gbc.entity_vdb.db_path == gbc.entity_vdb_path