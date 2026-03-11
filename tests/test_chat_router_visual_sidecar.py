import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_chat_router_module(monkeypatch):
    fake_api = ModuleType("api")
    fake_api.__path__ = []
    fake_api_db = ModuleType("api.db")
    fake_api_db.__path__ = []
    fake_db_module = ModuleType("api.db.mongodb")
    fake_api_db.mongodb = fake_db_module

    fake_models = ModuleType("api.models")
    fake_models.__path__ = []
    requests_path = Path(__file__).resolve().parents[1] / "api" / "models" / "requests.py"
    requests_spec = importlib.util.spec_from_file_location("api.models.requests", requests_path)
    requests_module = importlib.util.module_from_spec(requests_spec)
    assert requests_spec is not None and requests_spec.loader is not None
    requests_spec.loader.exec_module(requests_module)
    fake_models.requests = requests_module

    fake_services = ModuleType("api.services")
    fake_services.__path__ = []
    fake_chat_service = ModuleType("api.services.chat")

    async def fake_handle_query(**kwargs):
        return {
            "answer": "unused",
            "session_id": "session-1",
            "doc_ids_used": kwargs.get("doc_ids", []),
            "rewritten_query": None,
        }

    fake_chat_service.handle_query = fake_handle_query
    fake_services.chat = fake_chat_service

    fake_dependencies = ModuleType("api.dependencies")
    fake_dependencies.MONGO_URI = "mongodb://test"
    fake_dependencies.MONGO_DB_PREFIX = "bookrag_test"

    async def fake_get_current_user():
        return {"user_id": "user-1", "tenant_id": "tenant-a", "role": "user"}

    async def fake_filter_accessible_docs(user_id, tenant_id, requested_doc_ids):
        return requested_doc_ids or ["doc-1"]

    async def fake_rate_limit_query():
        return {"user_id": "user-1", "tenant_id": "tenant-a", "role": "user"}

    fake_dependencies.get_current_user = fake_get_current_user
    fake_dependencies.filter_accessible_docs = fake_filter_accessible_docs
    fake_dependencies.rate_limit_query = fake_rate_limit_query

    fake_routers = ModuleType("api.routers")
    fake_routers.__path__ = []

    monkeypatch.setitem(sys.modules, "api", fake_api)
    monkeypatch.setitem(sys.modules, "api.db", fake_api_db)
    monkeypatch.setitem(sys.modules, "api.db.mongodb", fake_db_module)
    monkeypatch.setitem(sys.modules, "api.models", fake_models)
    monkeypatch.setitem(sys.modules, "api.models.requests", requests_module)
    monkeypatch.setitem(sys.modules, "api.services", fake_services)
    monkeypatch.setitem(sys.modules, "api.services.chat", fake_chat_service)
    monkeypatch.setitem(sys.modules, "api.dependencies", fake_dependencies)
    monkeypatch.setitem(sys.modules, "api.routers", fake_routers)

    router_path = Path(__file__).resolve().parents[1] / "api" / "routers" / "chat.py"
    router_spec = importlib.util.spec_from_file_location("api.routers.chat", router_path)
    chat_module = importlib.util.module_from_spec(router_spec)
    assert router_spec is not None and router_spec.loader is not None
    router_spec.loader.exec_module(chat_module)
    return chat_module


def _build_test_client(chat_module):
    app = FastAPI()
    app.include_router(chat_module.router)
    app.dependency_overrides[chat_module.rate_limit_query] = lambda: {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "role": "user",
    }
    return TestClient(app)


def test_chat_query_router_forwards_visual_sidecar_overrides(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)
    captured = {}

    async def fake_filter_accessible_docs(user_id, tenant_id, requested_doc_ids):
        captured["filter"] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "requested_doc_ids": requested_doc_ids,
        }
        return ["doc-allowed"]

    async def fake_handle_query(**kwargs):
        captured["handle_query"] = kwargs
        return {
            "answer": "router-answer",
            "session_id": "session-42",
            "doc_ids_used": kwargs["doc_ids"],
            "rewritten_query": None,
        }

    monkeypatch.setattr(chat_module, "filter_accessible_docs", fake_filter_accessible_docs)
    monkeypatch.setattr(chat_module, "handle_query", fake_handle_query)

    with _build_test_client(chat_module) as client:
        response = client.post(
            "/chat/query",
            json={
                "query": "find the chart",
                "session_id": "session-existing",
                "doc_ids": ["doc-allowed", "doc-denied"],
                "cross_doc": True,
                "visual_sidecar_query_enabled": True,
                "visual_sidecar_query_topk": 4,
                "visual_sidecar_fusion_enabled": True,
                "visual_sidecar_fusion_weight": 1.75,
                "visual_sidecar_fusion_score_mode": "rank",
                "visual_sidecar_fusion_min_score": 0.25,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "router-answer",
        "session_id": "session-42",
        "doc_ids_used": ["doc-allowed"],
        "rewritten_query": None,
    }
    assert captured["filter"] == {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "requested_doc_ids": ["doc-allowed", "doc-denied"],
    }
    assert captured["handle_query"] == {
        "query": "find the chart",
        "tenant_id": "tenant-a",
        "user_id": "user-1",
        "doc_ids": ["doc-allowed"],
        "session_id": "session-existing",
        "config_path": "config/gbc.yaml",
        "cross_doc": True,
        "visual_sidecar_query_enabled": True,
        "visual_sidecar_query_topk": 4,
        "visual_sidecar_fusion_enabled": True,
        "visual_sidecar_fusion_weight": 1.75,
        "visual_sidecar_fusion_score_mode": "rank",
        "visual_sidecar_fusion_min_score": 0.25,
    }


def test_chat_query_router_returns_403_when_no_accessible_docs(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)

    async def fake_filter_accessible_docs(user_id, tenant_id, requested_doc_ids):
        return []

    async def fail_handle_query(**kwargs):
        raise AssertionError("handle_query should not be called when access filtering returns no docs")

    monkeypatch.setattr(chat_module, "filter_accessible_docs", fake_filter_accessible_docs)
    monkeypatch.setattr(chat_module, "handle_query", fail_handle_query)

    with _build_test_client(chat_module) as client:
        response = client.post(
            "/chat/query",
            json={
                "query": "find the chart",
                "doc_ids": ["doc-missing"],
                "visual_sidecar_query_enabled": True,
            },
        )

    assert response.status_code == 403
    assert response.json() == {"detail": "No accessible documents for this query"}