import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from Core.configs.rag.gbc_config import GBCRAGConfig


def _load_chat_service_module(monkeypatch):
    fake_api = ModuleType("api")
    fake_api.__path__ = []
    fake_api_db = ModuleType("api.db")
    fake_api_db.__path__ = []
    fake_db_module = ModuleType("api.db.mongodb")
    fake_api_db.mongodb = fake_db_module

    fake_dependencies = ModuleType("api.dependencies")
    fake_dependencies.MONGO_URI = "mongodb://test"
    fake_dependencies.MONGO_DB_PREFIX = "bookrag_test"
    fake_dependencies.INDEX_SAVE_DIR = "./indices"
    fake_dependencies.FALKORDB_HOST = "localhost"
    fake_dependencies.FALKORDB_PORT = 6379
    fake_dependencies.FALKORDB_USERNAME = ""
    fake_dependencies.FALKORDB_PASSWORD = ""
    fake_dependencies.THREAD_POOL = None

    monkeypatch.setitem(sys.modules, "api", fake_api)
    monkeypatch.setitem(sys.modules, "api.db", fake_api_db)
    monkeypatch.setitem(sys.modules, "api.db.mongodb", fake_db_module)
    monkeypatch.setitem(sys.modules, "api.dependencies", fake_dependencies)

    chat_path = Path(__file__).resolve().parents[1] / "api" / "services" / "chat.py"
    spec = importlib.util.spec_from_file_location("_test_chat_service_module", chat_path)
    chat_module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(chat_module)
    return chat_module


def test_gbc_rag_visual_sidecar_fusion_defaults_are_conservative():
    cfg = GBCRAGConfig()

    assert cfg.visual_sidecar_query_enabled is False
    assert cfg.visual_sidecar_fusion_enabled is False
    assert cfg.visual_sidecar_fusion_score_mode == "max_norm"
    assert cfg.visual_sidecar_fusion_min_score == pytest.approx(0.2)


def test_build_chat_gbc_rag_config_inherits_base_strategy_and_applies_overrides(monkeypatch):
    chat_module = _load_chat_service_module(monkeypatch)
    base_cfg = GBCRAGConfig(
        visual_sidecar_query_enabled=False,
        visual_sidecar_query_topk=3,
        visual_sidecar_fusion_enabled=False,
        visual_sidecar_fusion_weight=1.0,
    )
    monkeypatch.setattr(
        chat_module,
        "_get_system_config",
        lambda config_path: SimpleNamespace(rag=SimpleNamespace(strategy_config=base_cfg)),
    )

    rag_cfg = chat_module._build_chat_gbc_rag_config(
        "config/gbc.yaml",
        visual_sidecar_query_enabled=True,
        visual_sidecar_query_topk=5,
        visual_sidecar_fusion_enabled=True,
        visual_sidecar_fusion_weight=1.5,
        visual_sidecar_fusion_score_mode="rank",
        visual_sidecar_fusion_min_score=0.25,
    )

    assert rag_cfg.visual_sidecar_query_enabled is True
    assert rag_cfg.visual_sidecar_query_topk == 5
    assert rag_cfg.visual_sidecar_fusion_enabled is True
    assert rag_cfg.visual_sidecar_fusion_weight == pytest.approx(1.5)
    assert rag_cfg.visual_sidecar_fusion_score_mode == "rank"
    assert rag_cfg.visual_sidecar_fusion_min_score == pytest.approx(0.25)
    assert base_cfg.visual_sidecar_query_enabled is False
    assert base_cfg.visual_sidecar_fusion_enabled is False


def test_query_single_doc_sync_uses_chat_rag_config_and_answer_query(monkeypatch):
    chat_module = _load_chat_service_module(monkeypatch)
    monkeypatch.setattr(chat_module, "_get_gbc_index", lambda *args: "gbc-index")
    monkeypatch.setattr(chat_module, "_get_llm", lambda *args: "llm")
    monkeypatch.setattr(chat_module, "_get_vlm", lambda *args: "vlm")
    monkeypatch.setattr(
        chat_module,
        "_build_chat_gbc_rag_config",
        lambda *args, **kwargs: SimpleNamespace(query_topk=kwargs.get("visual_sidecar_query_topk")),
    )

    fake_rag_package = ModuleType("Core.rag")
    fake_rag_package.__path__ = []
    fake_gbc_rag = ModuleType("Core.rag.gbc_rag")
    captured = {}

    class FakeGBCRAG:
        def __init__(self, llm, vlm, config, gbc_index, lang="en"):
            captured["init"] = {
                "llm": llm,
                "vlm": vlm,
                "config": config,
                "gbc_index": gbc_index,
                "lang": lang,
            }

        def answer_query(self, query):
            captured["query"] = query
            return "single-doc-answer"

    fake_gbc_rag.GBCRAG = FakeGBCRAG
    monkeypatch.setitem(sys.modules, "Core.rag", fake_rag_package)
    monkeypatch.setitem(sys.modules, "Core.rag.gbc_rag", fake_gbc_rag)

    answer = chat_module._query_single_doc_sync(
        "find the figure",
        "tenant-a",
        "doc-1",
        "config/gbc.yaml",
        "id",
        visual_sidecar_query_topk=7,
    )

    assert answer == "single-doc-answer"
    assert captured["query"] == "find the figure"
    assert captured["init"]["lang"] == "id"
    assert captured["init"]["config"].query_topk == 7


def test_handle_query_forwards_visual_sidecar_overrides(monkeypatch):
    chat_module = _load_chat_service_module(monkeypatch)
    persisted_messages = []
    captured = {}

    async def fake_create_session(*_args, **_kwargs):
        await asyncio.sleep(0)
        return "created"

    async def fake_append_message(*args, **_kwargs):
        await asyncio.sleep(0)
        persisted_messages.append(args[4]["role"])

    async def fake_get_document(*_args, **_kwargs):
        await asyncio.sleep(0)
        return None

    monkeypatch.setattr(chat_module.db, "create_session", fake_create_session, raising=False)
    monkeypatch.setattr(chat_module.db, "append_message", fake_append_message, raising=False)
    monkeypatch.setattr(chat_module.db, "get_document", fake_get_document, raising=False)

    def fake_query_single_doc_sync(query, tenant_id, doc_id, config_path, lang="en", **kwargs):
        captured.update(
            {
                "query": query,
                "tenant_id": tenant_id,
                "doc_id": doc_id,
                "config_path": config_path,
                "lang": lang,
                "kwargs": kwargs,
            }
        )
        return "document answer"

    monkeypatch.setattr(chat_module, "_query_single_doc_sync", fake_query_single_doc_sync)

    result = asyncio.run(
        chat_module.handle_query(
            query="find the chart",
            tenant_id="tenant-a",
            user_id="user-1",
            doc_ids=["doc-1"],
            session_id=None,
            config_path="config/gbc.yaml",
            cross_doc=False,
            visual_sidecar_query_enabled=True,
            visual_sidecar_query_topk=4,
            visual_sidecar_fusion_enabled=True,
            visual_sidecar_fusion_weight=1.75,
            visual_sidecar_fusion_score_mode="rank",
            visual_sidecar_fusion_min_score=0.2,
        )
    )

    assert result["answer"] == "document answer"
    assert result["rewritten_query"] is None
    assert persisted_messages == ["user", "assistant"]
    assert captured["query"] == "find the chart"
    assert captured["kwargs"] == {
        "visual_sidecar_query_enabled": True,
        "visual_sidecar_query_topk": 4,
        "visual_sidecar_fusion_enabled": True,
        "visual_sidecar_fusion_weight": 1.75,
        "visual_sidecar_fusion_score_mode": "rank",
        "visual_sidecar_fusion_min_score": 0.2,
    }