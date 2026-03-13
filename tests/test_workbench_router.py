import asyncio
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import networkx as nx
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_workbench_router_module(monkeypatch):
    async def _yield_once():
        await asyncio.sleep(0)

    fake_api = ModuleType("api")
    fake_api.__path__ = []

    fake_api_db = ModuleType("api.db")
    fake_api_db.__path__ = []
    fake_db_module = ModuleType("api.db.mongodb")

    async def fake_get_document(_mongo_uri, _mongo_db_prefix, tenant_id, doc_id):
        await _yield_once()
        _ = (_mongo_uri, _mongo_db_prefix)
        return {
            "doc_id": doc_id,
            "filename": "report.pdf",
            "status": "ready",
            "document_lang": "en",
            "tenant_id": tenant_id,
        }

    fake_db_module.get_document = fake_get_document
    fake_api_db.mongodb = fake_db_module

    fake_models = ModuleType("api.models")
    fake_models.__path__ = []
    requests_path = Path(__file__).resolve().parents[1] / "api" / "models" / "requests.py"
    requests_spec = importlib.util.spec_from_file_location("api.models.requests", requests_path)
    requests_module = importlib.util.module_from_spec(requests_spec)
    assert requests_spec is not None and requests_spec.loader is not None
    requests_spec.loader.exec_module(requests_module)
    fake_models.requests = requests_module

    fake_dependencies = ModuleType("api.dependencies")
    fake_dependencies.MONGO_URI = "mongodb://test"
    fake_dependencies.MONGO_DB_PREFIX = "bookrag_test"
    fake_dependencies.THREAD_POOL = ThreadPoolExecutor(max_workers=1)

    async def fake_get_current_user():
        await _yield_once()
        return {"user_id": "user-1", "tenant_id": "tenant-a", "role": "user"}

    async def fake_check_doc_access(_user_id, _tenant_id, _doc_id, sub_tenant=None):
        await _yield_once()
        _ = (_user_id, _tenant_id, _doc_id, sub_tenant)
        return True

    fake_dependencies.get_current_user = fake_get_current_user
    fake_dependencies.check_doc_access = fake_check_doc_access

    fake_services = ModuleType("api.services")
    fake_services.__path__ = []
    fake_entity_editor = ModuleType("api.services.entity_editor")

    async def fake_list_entities(tenant_id, doc_id, config_path):
        await _yield_once()
        _ = (tenant_id, doc_id, config_path)
        return [{"entity_name": "Alice", "entity_type": "PERSON", "description": "Leader", "source_ids": [1], "node_name": "person::Alice", "role_assignments": []}]

    async def fake_list_roles(tenant_id, doc_id, config_path):
        await _yield_once()
        _ = (tenant_id, doc_id, config_path)
        return [{"entity_name": "Alice", "entity_type": "PERSON", "role_name": "President", "review_status": "suggested", "tenure_status": "current"}]

    async def fake_role_stats(tenant_id, doc_id, config_path):
        await _yield_once()
        _ = (tenant_id, doc_id, config_path)
        return {"doc_id": "doc-1", "total_entities": 1, "entities_with_roles": 1, "coverage_ratio": 1.0, "total_roles": 1, "unresolved_roles": 0, "review_status_counts": {"suggested": 1}, "normalization_status_counts": {"normalized": 1}, "origin_counts": {"model": 1}, "tenure_status_counts": {"current": 1}}

    def fake_load_graph_sync(tenant_id, doc_id, config_path):
        _ = config_path
        graph = type("FakeGraph", (), {})()
        graph.kg = nx.DiGraph()
        graph.kg.add_node("person::Alice", entity_name="Alice", entity_type="PERSON", description="Leader", source_ids={1}, role_assignments=[{"role_name": "President"}])
        graph.kg.add_node("organization::Acme", entity_name="Acme", entity_type="ORGANIZATION", description="Company", source_ids={2}, role_assignments=[])
        graph.kg.add_edge("person::Alice", "organization::Acme", relation_name="leads", weight=0.9, description="Leadership edge", source_ids={1, 2})
        return graph, f"/tmp/bookrag-indices/{tenant_id}/{doc_id}", None

    fake_entity_editor.list_entities = fake_list_entities
    fake_entity_editor.list_roles = fake_list_roles
    fake_entity_editor.role_stats = fake_role_stats
    fake_entity_editor._load_graph_sync = fake_load_graph_sync
    fake_services.entity_editor = fake_entity_editor

    fake_routers = ModuleType("api.routers")
    fake_routers.__path__ = []

    monkeypatch.setitem(sys.modules, "api", fake_api)
    monkeypatch.setitem(sys.modules, "api.db", fake_api_db)
    monkeypatch.setitem(sys.modules, "api.db.mongodb", fake_db_module)
    monkeypatch.setitem(sys.modules, "api.models", fake_models)
    monkeypatch.setitem(sys.modules, "api.models.requests", requests_module)
    monkeypatch.setitem(sys.modules, "api.dependencies", fake_dependencies)
    monkeypatch.setitem(sys.modules, "api.services", fake_services)
    monkeypatch.setitem(sys.modules, "api.services.entity_editor", fake_entity_editor)
    monkeypatch.setitem(sys.modules, "api.routers", fake_routers)

    router_path = Path(__file__).resolve().parents[1] / "api" / "routers" / "workbench.py"
    router_spec = importlib.util.spec_from_file_location("api.routers.workbench", router_path)
    workbench_module = importlib.util.module_from_spec(router_spec)
    assert router_spec is not None and router_spec.loader is not None
    router_spec.loader.exec_module(workbench_module)
    return workbench_module


def _build_test_client(workbench_module):
    app = FastAPI()
    app.include_router(workbench_module.router)
    app.dependency_overrides[workbench_module.get_current_user] = lambda: {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "role": "user",
    }
    return TestClient(app)


def test_workbench_home_serves_internal_ui(monkeypatch):
    workbench_module = _load_workbench_router_module(monkeypatch)

    with _build_test_client(workbench_module) as client:
        response = client.get("/workbench")

    assert response.status_code == 200
    assert "BookRAG Workbench" in response.text
    assert "/auth/login" in response.text
    assert "cytoscape" in response.text


def test_workbench_bundle_aggregates_document_results(monkeypatch):
    workbench_module = _load_workbench_router_module(monkeypatch)

    async def fake_check_doc_access(_user_id, _tenant_id, _doc_id, sub_tenant=None):
        await asyncio.sleep(0)
        _ = (_user_id, _tenant_id, _doc_id, sub_tenant)
        return True

    monkeypatch.setattr(workbench_module, "check_doc_access", fake_check_doc_access)

    with _build_test_client(workbench_module) as client:
        response = client.get("/workbench/api/documents/doc-1/bundle")

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc"]["doc_id"] == "doc-1"
    assert payload["doc"]["status"] == "ready"
    assert payload["entities"][0]["entity_name"] == "Alice"
    assert payload["roles"][0]["role_name"] == "President"
    assert payload["stats"]["total_roles"] == 1


def test_workbench_graph_serializes_nodes_and_edges(monkeypatch):
    workbench_module = _load_workbench_router_module(monkeypatch)

    async def fake_check_doc_access(_user_id, _tenant_id, _doc_id, sub_tenant=None):
        await asyncio.sleep(0)
        _ = (_user_id, _tenant_id, _doc_id, sub_tenant)
        return True

    monkeypatch.setattr(workbench_module, "check_doc_access", fake_check_doc_access)

    with _build_test_client(workbench_module) as client:
        response = client.get("/workbench/api/documents/doc-1/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"] == {"node_count": 2, "edge_count": 1}
    assert payload["nodes"][0]["label"] == "Acme"
    assert payload["nodes"][1]["role_count"] == 1
    assert payload["edges"][0]["relation_name"] == "leads"


def test_workbench_graph_returns_403_when_access_denied(monkeypatch):
    workbench_module = _load_workbench_router_module(monkeypatch)

    async def fake_check_doc_access(_user_id, _tenant_id, _doc_id, sub_tenant=None):
        await asyncio.sleep(0)
        _ = (_user_id, _tenant_id, _doc_id, sub_tenant)
        return False

    monkeypatch.setattr(workbench_module, "check_doc_access", fake_check_doc_access)

    with _build_test_client(workbench_module) as client:
        response = client.get("/workbench/api/documents/doc-1/graph")

    assert response.status_code == 403
    assert response.json() == {"detail": "Access denied to this document"}