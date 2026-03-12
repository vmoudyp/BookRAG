import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_requests_module():
    requests_path = Path(__file__).resolve().parents[1] / "api" / "models" / "requests.py"
    requests_spec = importlib.util.spec_from_file_location("api.models.requests", requests_path)
    requests_module = importlib.util.module_from_spec(requests_spec)
    assert requests_spec is not None and requests_spec.loader is not None
    requests_spec.loader.exec_module(requests_module)
    return requests_module


def _install_base_modules(monkeypatch):
    fake_api = ModuleType("api")
    fake_api.__path__ = []
    fake_api_db = ModuleType("api.db")
    fake_api_db.__path__ = []
    fake_db_module = ModuleType("api.db.mongodb")

    async def fake_noop(*args, **kwargs):
        return None

    async def fake_false(*args, **kwargs):
        return False

    fake_db_module.get_tenant = fake_noop
    fake_db_module.get_user_by_username = fake_noop
    fake_db_module.create_user = fake_noop
    fake_db_module.store_refresh_token = fake_noop
    fake_db_module.is_refresh_token_valid = fake_false
    fake_db_module.revoke_refresh_token = fake_noop
    fake_db_module.create_tenant = fake_noop
    fake_db_module.get_permission = fake_noop
    fake_db_module.grant_permission = fake_noop
    fake_api_db.mongodb = fake_db_module

    fake_models = ModuleType("api.models")
    fake_models.__path__ = []
    requests_module = _load_requests_module()
    fake_models.requests = requests_module

    fake_dependencies = ModuleType("api.dependencies")
    fake_dependencies.MONGO_URI = "mongodb://test"
    fake_dependencies.MONGO_DB_PREFIX = "bookrag_test"
    fake_dependencies.MONGO_SYSTEM_DB = "bookrag_system"
    fake_dependencies.REFRESH_TOKEN_EXPIRE_DAYS = 7
    fake_dependencies.hash_password = lambda value: f"hashed:{value}"
    fake_dependencies.verify_password = lambda plain, hashed: plain == hashed
    fake_dependencies.create_access_token = lambda claims: "access.jwt.token"
    fake_dependencies.create_refresh_token = lambda claims: "refresh.jwt.token"
    fake_dependencies.decode_refresh_token = lambda token: {"tenant_id": "tenant-a", "sub": "alice", "role": "user"}

    async def fake_rate_limit_login():
        return None

    async def fake_get_current_user():
        return {"user_id": "alice", "tenant_id": "tenant-a", "role": "admin"}

    async def fake_require_admin():
        return {"user_id": "alice", "tenant_id": "tenant-a", "role": "admin"}

    async def fake_check_doc_access(*args, **kwargs):
        return True

    fake_dependencies.rate_limit_login = fake_rate_limit_login
    fake_dependencies.get_current_user = fake_get_current_user
    fake_dependencies.require_admin = fake_require_admin
    fake_dependencies.check_doc_access = fake_check_doc_access

    fake_services = ModuleType("api.services")
    fake_services.__path__ = []
    fake_entity_editor = ModuleType("api.services.entity_editor")
    fake_entity_editor.list_entities = fake_noop
    fake_entity_editor.rename_entity = fake_noop
    fake_entity_editor.merge_entities = fake_noop
    fake_entity_editor.split_entity = fake_noop
    fake_entity_editor.suggest_merges = fake_noop
    fake_entity_editor.list_roles = fake_noop
    fake_entity_editor.bulk_review_roles = fake_noop
    fake_entity_editor.re_normalize_roles = fake_noop
    fake_entity_editor.role_stats = fake_noop
    fake_entity_editor.list_role_vocab = fake_noop
    fake_entity_editor.create_role_vocab_entry = fake_noop
    fake_entity_editor.update_role_vocab_entry = fake_noop
    fake_entity_editor.reload_role_vocab = fake_noop
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


def _load_router_module(monkeypatch, router_name):
    _install_base_modules(monkeypatch)
    router_path = Path(__file__).resolve().parents[1] / "api" / "routers" / f"{router_name}.py"
    router_spec = importlib.util.spec_from_file_location(f"api.routers.{router_name}", router_path)
    router_module = importlib.util.module_from_spec(router_spec)
    assert router_spec is not None and router_spec.loader is not None
    router_spec.loader.exec_module(router_module)
    return router_module


def _build_test_client(router_module):
    app = FastAPI()
    app.include_router(router_module.router)
    return TestClient(app)


def test_auth_router_openapi_includes_summaries_and_examples(monkeypatch):
    auth_module = _load_router_module(monkeypatch, "auth")

    with _build_test_client(auth_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    register_op = schema["paths"]["/auth/register"]["post"]
    assert register_op["summary"] == "Register a tenant user"
    assert register_op["responses"]["201"]["description"] == "User registered successfully."

    login_op = schema["paths"]["/auth/login"]["post"]
    assert login_op["summary"] == "Authenticate and obtain tokens"
    assert login_op["responses"]["200"]["description"] == "Access and refresh tokens issued successfully."
    login_body_ref = login_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    login_schema_name = login_body_ref.rsplit("/", 1)[-1]
    login_schema = schema["components"]["schemas"][login_schema_name]
    assert login_schema["examples"][0]["tenant_id"] == "tenant-a"

    refresh_op = schema["paths"]["/auth/refresh"]["post"]
    assert refresh_op["responses"]["200"]["description"] == "New access and refresh tokens issued successfully."

    token_schema = schema["components"]["schemas"]["TokenResponse"]
    assert token_schema["examples"][0]["token_type"] == "bearer"


def test_tenants_router_openapi_includes_permission_docs(monkeypatch):
    tenants_module = _load_router_module(monkeypatch, "tenants")

    with _build_test_client(tenants_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    create_op = schema["paths"]["/tenants"]["post"]
    assert create_op["summary"] == "Create a tenant"

    perm_op = schema["paths"]["/tenants/{tenant_id}/permissions"]["post"]
    assert perm_op["summary"] == "Grant document permission"
    assert "document owners" in perm_op["description"]

    permission_ref = perm_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    permission_schema_name = permission_ref.rsplit("/", 1)[-1]
    permission_schema = schema["components"]["schemas"][permission_schema_name]
    assert permission_schema["examples"][0]["role"] == "reader"


def test_entities_router_openapi_includes_operation_docs_and_examples(monkeypatch):
    entities_module = _load_router_module(monkeypatch, "entities")

    with _build_test_client(entities_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    list_op = schema["paths"]["/entities/{doc_id}"]["get"]
    assert list_op["summary"] == "List document entities"

    merge_op = schema["paths"]["/entities/{doc_id}/merge"]["post"]
    assert merge_op["summary"] == "Merge entities"
    merge_ref = merge_op["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    merge_schema_name = merge_ref.rsplit("/", 1)[-1]
    merge_schema = schema["components"]["schemas"][merge_schema_name]
    assert len(merge_schema["examples"][0]["source_entities"]) == 2

    suggestions_op = schema["paths"]["/entities/{doc_id}/suggestions"]["get"]
    assert suggestions_op["summary"] == "Suggest entity merges"
    params = {param["name"]: param for param in suggestions_op["parameters"]}
    assert "Minimum similarity score" in params["min_score"]["description"]

    role_list_op = schema["paths"]["/entities/{doc_id}/roles"]["get"]
    assert role_list_op["summary"] == "List role assignments"

    bulk_review_op = schema["paths"]["/entities/{doc_id}/roles/bulk-review"]["post"]
    assert bulk_review_op["summary"] == "Bulk-review role assignments"

    renorm_op = schema["paths"]["/entities/{doc_id}/roles/re-normalize"]["post"]
    assert renorm_op["summary"] == "Re-normalize extracted role assignments"

    stats_op = schema["paths"]["/entities/{doc_id}/roles/stats"]["get"]
    assert stats_op["summary"] == "Get role curation statistics"

    vocab_list_op = schema["paths"]["/entities/role-vocab"]["get"]
    assert vocab_list_op["summary"] == "List curated role vocabulary"

    vocab_reload_op = schema["paths"]["/entities/role-vocab/reload"]["post"]
    assert vocab_reload_op["summary"] == "Reload the curated role vocabulary"

    renorm_schema = schema["components"]["schemas"]["ReNormalizeRolesRequest"]
    assert "review_status" in renorm_schema["properties"]

    vocab_schema = schema["components"]["schemas"]["RoleVocabularyMutationResponse"]
    assert vocab_schema["properties"]["role"]["$ref"].endswith("/RoleVocabularyEntryInfo")

    operation_schema = schema["components"]["schemas"]["EntityOperationResponse"]
    assert operation_schema["examples"][0]["success"] is True