import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


async def _async_return(value):
    await asyncio.sleep(0)
    return value


def _load_requests_module():
    requests_path = Path(__file__).resolve().parents[1] / "api" / "models" / "requests.py"
    spec = importlib.util.spec_from_file_location("api.models.requests", requests_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_entities_router_module(monkeypatch):
    fake_api = ModuleType("api")
    fake_api.__path__ = []

    fake_models = ModuleType("api.models")
    fake_models.__path__ = []
    requests_module = _load_requests_module()
    fake_models.requests = requests_module

    fake_dependencies = ModuleType("api.dependencies")

    async def fake_get_current_user():
        return await _async_return({"user_id": "user-1", "tenant_id": "tenant-a", "role": "admin"})

    async def fake_require_admin():
        return await _async_return({"user_id": "user-1", "tenant_id": "tenant-a", "role": "admin"})

    async def fake_check_doc_access(*args, **kwargs):
        return await _async_return(True)

    fake_dependencies.get_current_user = fake_get_current_user
    fake_dependencies.require_admin = fake_require_admin
    fake_dependencies.check_doc_access = fake_check_doc_access

    fake_services = ModuleType("api.services")
    fake_services.__path__ = []
    fake_entity_editor = ModuleType("api.services.entity_editor")

    async def fake_noop(*args, **kwargs):
        return await _async_return(None)

    for name in [
        "list_roles",
        "bulk_review_roles",
        "re_normalize_roles",
        "role_stats",
        "list_role_vocab",
        "create_role_vocab_entry",
        "update_role_vocab_entry",
        "reload_role_vocab",
    ]:
        setattr(fake_entity_editor, name, fake_noop)

    fake_services.entity_editor = fake_entity_editor
    fake_routers = ModuleType("api.routers")
    fake_routers.__path__ = []

    monkeypatch.setitem(sys.modules, "api", fake_api)
    monkeypatch.setitem(sys.modules, "api.models", fake_models)
    monkeypatch.setitem(sys.modules, "api.models.requests", requests_module)
    monkeypatch.setitem(sys.modules, "api.dependencies", fake_dependencies)
    monkeypatch.setitem(sys.modules, "api.services", fake_services)
    monkeypatch.setitem(sys.modules, "api.services.entity_editor", fake_entity_editor)
    monkeypatch.setitem(sys.modules, "api.routers", fake_routers)

    router_path = Path(__file__).resolve().parents[1] / "api" / "routers" / "entities.py"
    spec = importlib.util.spec_from_file_location("api.routers.entities", router_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _build_test_client(entities_module):
    app = FastAPI()
    app.include_router(entities_module.router)
    app.dependency_overrides[entities_module.get_current_user] = lambda: {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "role": "admin",
    }
    app.dependency_overrides[entities_module.require_admin] = lambda: {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "role": "admin",
    }
    return TestClient(app)


def test_entities_role_list_and_bulk_review_routes(monkeypatch):
    entities_module = _load_entities_router_module(monkeypatch)
    captured = {}

    async def fake_list_roles(tenant_id, doc_id, config_path, review_status=None, entity_type=None):
        captured["list_roles"] = {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "config_path": config_path,
            "review_status": review_status,
            "entity_type": entity_type,
        }
        return await _async_return([
            {"entity_name": "Alice", "entity_type": "PERSON", "assignment_id": "ra-1", "role_name": "President"}
        ])

    async def fake_bulk_review_roles(tenant_id, doc_id, config_path, reviews, user_id):
        captured["bulk_review_roles"] = {
            "tenant_id": tenant_id,
            "doc_id": doc_id,
            "config_path": config_path,
            "reviews": reviews,
            "user_id": user_id,
        }
        return await _async_return({"processed": 1, "errors": [{"assignment_id": "ra-2", "error": "missing"}]})

    monkeypatch.setattr(entities_module.svc, "list_roles", fake_list_roles)
    monkeypatch.setattr(entities_module.svc, "bulk_review_roles", fake_bulk_review_roles)

    with _build_test_client(entities_module) as client:
        list_response = client.get("/entities/doc-1/roles?review_status=confirmed&entity_type=PERSON")
        bulk_response = client.post(
            "/entities/doc-1/roles/bulk-review",
            json={
                "reviews": [
                    {
                        "entity_name": "Alice",
                        "entity_type": "PERSON",
                        "assignment_id": "ra-1",
                        "review_status": "confirmed",
                    },
                    {
                        "entity_name": "Bob",
                        "entity_type": "PERSON",
                        "assignment_id": "ra-2",
                        "review_status": "rejected",
                    },
                ]
            },
        )

    assert list_response.status_code == 200
    assert list_response.json() == {
        "doc_id": "doc-1",
        "total": 1,
        "roles": [{"entity_name": "Alice", "entity_type": "PERSON", "assignment_id": "ra-1", "role_name": "President"}],
    }
    assert captured["list_roles"] == {
        "tenant_id": "tenant-a",
        "doc_id": "doc-1",
        "config_path": "config/gbc.yaml",
        "review_status": "confirmed",
        "entity_type": "PERSON",
    }

    assert bulk_response.status_code == 200
    assert bulk_response.json() == {
        "success": True,
        "message": "Processed 1 assignment(s); 1 error(s).",
        "processed": 1,
        "errors": [{"assignment_id": "ra-2", "error": "missing"}],
    }
    assert captured["bulk_review_roles"]["tenant_id"] == "tenant-a"
    assert captured["bulk_review_roles"]["user_id"] == "user-1"
    assert len(captured["bulk_review_roles"]["reviews"]) == 2


def test_entities_re_normalize_and_stats_routes(monkeypatch):
    entities_module = _load_entities_router_module(monkeypatch)
    captured = {}

    async def fake_re_normalize_roles(**kwargs):
        captured["re_normalize_roles"] = kwargs
        return await _async_return({"doc_id": "doc-1", "processed": 3, "updated": 2, "skipped": 1, "unresolved": 1})

    async def fake_role_stats(**kwargs):
        captured["role_stats"] = kwargs
        return await _async_return({
            "doc_id": "doc-1",
            "total_entities": 4,
            "entities_with_roles": 2,
            "coverage_ratio": 0.5,
            "total_roles": 3,
            "unresolved_roles": 1,
            "review_status_counts": {"confirmed": 2, "suggested": 1},
            "normalization_status_counts": {"matched": 2, "unresolved": 1},
            "origin_counts": {"manual": 1, "extracted": 2},
            "tenure_status_counts": {"current": 2, "former": 1},
        })

    monkeypatch.setattr(entities_module.svc, "re_normalize_roles", fake_re_normalize_roles)
    monkeypatch.setattr(entities_module.svc, "role_stats", fake_role_stats)

    with _build_test_client(entities_module) as client:
        renorm_response = client.post(
            "/entities/doc-1/roles/re-normalize",
            json={"review_status": "suggested", "entity_type": "PERSON"},
        )
        stats_response = client.get("/entities/doc-1/roles/stats")

    assert renorm_response.status_code == 200
    assert renorm_response.json()["updated"] == 2
    assert captured["re_normalize_roles"]["review_status"] == "suggested"
    assert captured["re_normalize_roles"]["entity_type"] == "PERSON"
    assert captured["re_normalize_roles"]["user_id"] == "user-1"

    assert stats_response.status_code == 200
    assert stats_response.json()["coverage_ratio"] == pytest.approx(0.5)
    assert captured["role_stats"] == {
        "tenant_id": "tenant-a",
        "doc_id": "doc-1",
        "config_path": "config/gbc.yaml",
    }


def test_entities_role_vocab_admin_routes(monkeypatch):
    entities_module = _load_entities_router_module(monkeypatch)
    captured = {}

    async def fake_list_role_vocab():
        captured["list_role_vocab"] = True
        return await _async_return([
            {"role_id": "role:president", "canonical": "President", "aliases": ["Head of State"]}
        ])

    async def fake_create_role_vocab_entry(tenant_id, role_input, user_id):
        captured["create_role_vocab_entry"] = {
            "tenant_id": tenant_id,
            "role_input": role_input,
            "user_id": user_id,
        }
        return await _async_return({
            "role_id": role_input["role_id"],
            "canonical": role_input["canonical"],
            "aliases": role_input["aliases"],
        })

    async def fake_update_role_vocab_entry(tenant_id, role_id, update_fields, user_id):
        captured["update_role_vocab_entry"] = {
            "tenant_id": tenant_id,
            "role_id": role_id,
            "update_fields": update_fields,
            "user_id": user_id,
        }
        return await _async_return({"role_id": role_id, "canonical": "Prime Minister", "aliases": ["Premier"]})

    async def fake_reload_role_vocab(tenant_id, user_id):
        captured["reload_role_vocab"] = {"tenant_id": tenant_id, "user_id": user_id}
        return await _async_return([
            {"role_id": "role:president", "canonical": "President", "aliases": []}
        ])

    monkeypatch.setattr(entities_module.svc, "list_role_vocab", fake_list_role_vocab)
    monkeypatch.setattr(entities_module.svc, "create_role_vocab_entry", fake_create_role_vocab_entry)
    monkeypatch.setattr(entities_module.svc, "update_role_vocab_entry", fake_update_role_vocab_entry)
    monkeypatch.setattr(entities_module.svc, "reload_role_vocab", fake_reload_role_vocab)

    with _build_test_client(entities_module) as client:
        list_response = client.get("/entities/role-vocab")
        create_response = client.post(
            "/entities/role-vocab",
            json={"role_id": "role:pm", "canonical": "Prime Minister", "aliases": ["Premier"]},
        )
        update_response = client.patch(
            "/entities/role-vocab/role:pm",
            json={"canonical": "Prime Minister", "aliases": ["Premier"]},
        )
        reload_response = client.post("/entities/role-vocab/reload")

    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1
    assert captured["list_role_vocab"] is True

    assert create_response.status_code == 200
    assert create_response.json()["role"]["role_id"] == "role:pm"
    assert captured["create_role_vocab_entry"]["tenant_id"] == "tenant-a"

    assert update_response.status_code == 200
    assert update_response.json()["role"]["canonical"] == "Prime Minister"
    assert captured["update_role_vocab_entry"]["role_id"] == "role:pm"

    assert reload_response.status_code == 200
    assert reload_response.json() == {
        "success": True,
        "message": "Reloaded role vocabulary.",
        "total": 1,
    }
    assert captured["reload_role_vocab"] == {"tenant_id": "tenant-a", "user_id": "user-1"}