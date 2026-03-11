import importlib.util
import sys
import uuid
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
    app.dependency_overrides[chat_module.get_current_user] = lambda: {
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


def test_chat_query_router_openapi_documents_visual_sidecar_overrides(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)

    with _build_test_client(chat_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    query_op = schema["paths"]["/chat/query"]["post"]
    assert query_op["summary"] == "Submit a chat query"
    assert "optional per-request overrides" in query_op["description"]

    request_ref = query_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_schema_name = request_ref.rsplit("/", 1)[-1]
    request_schema = schema["components"]["schemas"][request_schema_name]
    properties = request_schema["properties"]

    assert "visual_sidecar_query_enabled" in properties
    assert "keep the loaded config value" in properties["visual_sidecar_query_enabled"]["description"]
    assert "visual_sidecar_fusion_score_mode" in properties
    assert "raw" in properties["visual_sidecar_fusion_score_mode"]["description"]
    assert request_schema["examples"][0]["visual_sidecar_fusion_score_mode"] == "max_norm"

    response_ref = query_op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    response_schema_name = response_ref.rsplit("/", 1)[-1]
    response_schema = schema["components"]["schemas"][response_schema_name]
    assert response_schema["examples"][0]["session_id"] == "session-123"


def test_chat_session_router_openapi_documents_session_endpoints(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)

    with _build_test_client(chat_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    create_op = schema["paths"]["/chat/sessions"]["post"]
    assert create_op["summary"] == "Create a chat session"
    assert "filtered to documents the caller can access" in create_op["description"]
    create_ref = create_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    create_schema_name = create_ref.rsplit("/", 1)[-1]
    create_schema = schema["components"]["schemas"][create_schema_name]
    assert create_schema["examples"][0]["doc_ids"] == ["doc-123", "doc-456"]

    list_op = schema["paths"]["/chat/sessions"]["get"]
    assert list_op["summary"] == "List chat sessions"
    list_params = {param["name"]: param for param in list_op["parameters"]}
    assert list_params["limit"]["description"] == "Max sessions to return"

    delete_op = schema["paths"]["/chat/sessions/{session_id}"]["delete"]
    assert delete_op["summary"] == "Delete a chat session"
    assert delete_op["responses"]["204"]["description"] == "Session deleted successfully."

    messages_op = schema["paths"]["/chat/sessions/{session_id}/messages"]["get"]
    assert messages_op["summary"] == "Get session messages"
    assert messages_op["responses"]["404"]["description"] == "Session not found."

    messages_schema = schema["components"]["schemas"]["SessionMessagesResponse"]
    assert messages_schema["examples"][0]["messages"][0]["role"] == "user"

    message_schema = schema["components"]["schemas"]["MessageResponse"]
    assert "user" in message_schema["properties"]["role"]["description"]


def test_chat_session_router_creates_session_with_accessible_docs(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)
    captured = {}

    async def fake_filter_accessible_docs(user_id, tenant_id, requested_doc_ids):
        captured["filter"] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "requested_doc_ids": requested_doc_ids,
        }
        return ["doc-allowed"]

    async def fake_create_session(uri, db_prefix, tenant_id, session_data):
        captured["create_session"] = {
            "uri": uri,
            "db_prefix": db_prefix,
            "tenant_id": tenant_id,
            "session_data": session_data,
        }
        return "mongo-id-1"

    monkeypatch.setattr(chat_module, "filter_accessible_docs", fake_filter_accessible_docs)
    monkeypatch.setattr(chat_module.db, "create_session", fake_create_session, raising=False)
    monkeypatch.setattr(uuid, "uuid4", lambda: "session-created")

    with _build_test_client(chat_module) as client:
        response = client.post("/chat/sessions", json={"doc_ids": ["doc-allowed", "doc-denied"]})

    assert response.status_code == 201
    assert response.json() == {"session_id": "session-created"}
    assert captured["filter"] == {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "requested_doc_ids": ["doc-allowed", "doc-denied"],
    }
    assert captured["create_session"] == {
        "uri": "mongodb://test",
        "db_prefix": "bookrag_test",
        "tenant_id": "tenant-a",
        "session_data": {
            "session_id": "session-created",
            "user_id": "user-1",
            "doc_ids": ["doc-allowed"],
            "messages": [],
        },
    }


def test_chat_session_router_lists_sessions(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)
    captured = {}

    async def fake_list_sessions(uri, db_prefix, tenant_id, user_id, limit=50, offset=0):
        captured["list_sessions"] = {
            "uri": uri,
            "db_prefix": db_prefix,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "limit": limit,
            "offset": offset,
        }
        return ([{
            "session_id": "session-1",
            "created_at": "2026-03-11T12:00:00Z",
            "message_count": 3,
            "doc_ids": ["doc-1"],
        }], 1)

    monkeypatch.setattr(chat_module.db, "list_sessions", fake_list_sessions, raising=False)

    with _build_test_client(chat_module) as client:
        response = client.get("/chat/sessions?limit=10&offset=2")

    assert response.status_code == 200
    assert response.json() == {
        "sessions": [{
            "session_id": "session-1",
            "created_at": "2026-03-11T12:00:00Z",
            "message_count": 3,
            "doc_ids": ["doc-1"],
        }],
        "total": 1,
    }
    assert captured["list_sessions"] == {
        "uri": "mongodb://test",
        "db_prefix": "bookrag_test",
        "tenant_id": "tenant-a",
        "user_id": "user-1",
        "limit": 10,
        "offset": 2,
    }


def test_chat_session_router_deletes_owned_session(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)
    captured = {}

    async def fake_get_session(uri, db_prefix, tenant_id, session_id):
        captured["get_session"] = {
            "uri": uri,
            "db_prefix": db_prefix,
            "tenant_id": tenant_id,
            "session_id": session_id,
        }
        return {"session_id": session_id, "user_id": "user-1"}

    async def fake_delete_session(uri, db_prefix, tenant_id, session_id):
        captured["delete_session"] = {
            "uri": uri,
            "db_prefix": db_prefix,
            "tenant_id": tenant_id,
            "session_id": session_id,
        }

    monkeypatch.setattr(chat_module.db, "get_session", fake_get_session, raising=False)
    monkeypatch.setattr(chat_module.db, "delete_session", fake_delete_session, raising=False)

    with _build_test_client(chat_module) as client:
        response = client.delete("/chat/sessions/session-1")

    assert response.status_code == 204
    assert response.text == ""
    assert captured["get_session"]["session_id"] == "session-1"
    assert captured["delete_session"] == {
        "uri": "mongodb://test",
        "db_prefix": "bookrag_test",
        "tenant_id": "tenant-a",
        "session_id": "session-1",
    }


def test_chat_session_router_returns_paginated_message_history(monkeypatch):
    chat_module = _load_chat_router_module(monkeypatch)

    async def fake_get_session(uri, db_prefix, tenant_id, session_id):
        return {
            "session_id": session_id,
            "user_id": "user-1",
            "messages": [
                {"role": "user", "content": "first", "ts": "2026-03-11T12:00:00Z"},
                {"role": "assistant", "content": "second", "ts": "2026-03-11T12:00:01Z"},
                {"role": "user", "content": "third", "ts": "2026-03-11T12:00:02Z"},
            ],
        }

    monkeypatch.setattr(chat_module.db, "get_session", fake_get_session, raising=False)

    with _build_test_client(chat_module) as client:
        response = client.get("/chat/sessions/session-1/messages?limit=2&offset=1")

    assert response.status_code == 200
    assert response.json() == {
        "session_id": "session-1",
        "messages": [
            {"role": "assistant", "content": "second", "ts": "2026-03-11T12:00:01Z"},
            {"role": "user", "content": "third", "ts": "2026-03-11T12:00:02Z"},
        ],
        "total": 3,
    }