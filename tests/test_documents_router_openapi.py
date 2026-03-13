import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_documents_router_module(monkeypatch):
    fake_api = ModuleType("api")
    fake_api.__path__ = []

    fake_api_db = ModuleType("api.db")
    fake_api_db.__path__ = []
    fake_db_module = ModuleType("api.db.mongodb")

    async def fake_create_document(*args, **kwargs):
        return None

    async def fake_grant_permission(*args, **kwargs):
        return None

    async def fake_list_documents(*args, **kwargs):
        return [], 0

    async def fake_get_document(*args, **kwargs):
        return None

    async def fake_get_document_raw_path(*args, **kwargs):
        return None

    async def fake_get_permission(*args, **kwargs):
        return None

    async def fake_delete_document(*args, **kwargs):
        return None

    fake_db_module.create_document = fake_create_document
    fake_db_module.grant_permission = fake_grant_permission
    fake_db_module.list_documents = fake_list_documents
    fake_db_module.get_document = fake_get_document
    fake_db_module.get_document_raw_path = fake_get_document_raw_path
    fake_db_module.get_permission = fake_get_permission
    fake_db_module.delete_document = fake_delete_document
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
    fake_dependencies.UPLOAD_DIR = "/tmp/bookrag-uploads"
    fake_dependencies.INDEX_SAVE_DIR = "/tmp/bookrag-indices"

    async def fake_get_current_user():
        return {"user_id": "user-1", "tenant_id": "tenant-a", "role": "user"}

    async def fake_check_doc_access(*args, **kwargs):
        return True

    fake_dependencies.get_current_user = fake_get_current_user
    fake_dependencies.check_doc_access = fake_check_doc_access

    fake_services = ModuleType("api.services")
    fake_services.__path__ = []
    fake_indexing_service = ModuleType("api.services.indexing")
    fake_indexing_service.run_indexing = lambda *args, **kwargs: None
    fake_services.indexing = fake_indexing_service

    fake_routers = ModuleType("api.routers")
    fake_routers.__path__ = []

    fake_aiofiles = ModuleType("aiofiles")
    fake_aiofiles.open = object()

    monkeypatch.setitem(sys.modules, "api", fake_api)
    monkeypatch.setitem(sys.modules, "api.db", fake_api_db)
    monkeypatch.setitem(sys.modules, "api.db.mongodb", fake_db_module)
    monkeypatch.setitem(sys.modules, "api.models", fake_models)
    monkeypatch.setitem(sys.modules, "api.models.requests", requests_module)
    monkeypatch.setitem(sys.modules, "api.dependencies", fake_dependencies)
    monkeypatch.setitem(sys.modules, "api.services", fake_services)
    monkeypatch.setitem(sys.modules, "api.services.indexing", fake_indexing_service)
    monkeypatch.setitem(sys.modules, "api.routers", fake_routers)
    monkeypatch.setitem(sys.modules, "aiofiles", fake_aiofiles)

    router_path = Path(__file__).resolve().parents[1] / "api" / "routers" / "documents.py"
    router_spec = importlib.util.spec_from_file_location("api.routers.documents", router_path)
    documents_module = importlib.util.module_from_spec(router_spec)
    assert router_spec is not None and router_spec.loader is not None
    router_spec.loader.exec_module(documents_module)
    return documents_module


def _build_test_client(documents_module):
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[documents_module.get_current_user] = lambda: {
        "user_id": "user-1",
        "tenant_id": "tenant-a",
        "role": "user",
    }
    return TestClient(app)


def test_documents_router_openapi_describes_upload_metadata_and_examples(monkeypatch):
    documents_module = _load_documents_router_module(monkeypatch)

    with _build_test_client(documents_module) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()

    upload_op = schema["paths"]["/documents"]["post"]
    assert upload_op["summary"] == "Upload PDF documents"
    assert "Optional `document_date`, `document_lang`, and `sub_tenant`" in upload_op["description"]

    upload_body_ref = upload_op["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    upload_schema_name = upload_body_ref.rsplit("/", 1)[-1]
    upload_schema = schema["components"]["schemas"][upload_schema_name]
    assert "document_date" in upload_schema["properties"]
    assert "ISO-8601" in upload_schema["properties"]["document_date"]["description"]
    assert "document_lang" in upload_schema["properties"]
    assert "ISO 639-1" in upload_schema["properties"]["document_lang"]["description"]
    assert "sub_tenant" in upload_schema["properties"]

    status_op = schema["paths"]["/documents/{doc_id}"]["get"]
    raw_op = schema["paths"]["/documents/{doc_id}/raw"]["get"]
    assert status_op["summary"] == "Get document status"
    assert raw_op["summary"] == "Download the original PDF"
    status_params = {param["name"]: param for param in status_op["parameters"]}
    raw_params = {param["name"]: param for param in raw_op["parameters"]}
    assert "sub_tenant" in status_params
    assert "sub_tenant" in raw_params

    batch_schema = schema["components"]["schemas"]["BatchUploadResponse"]
    assert batch_schema["examples"][0]["uploaded"][0]["sub_tenant"] == "finance"
    assert batch_schema["examples"][0]["failed"][0]["error"] == "Only PDF files are supported"


def test_documents_router_list_and_status_include_sub_tenant_and_forward_scope(monkeypatch):
    documents_module = _load_documents_router_module(monkeypatch)
    captured = {}

    async def fake_list_documents(*args, **kwargs):
        captured["list_documents"] = kwargs
        return [
            {
                "doc_id": "doc-1",
                "filename": "report.pdf",
                "status": "ready",
                "sub_tenant": "finance",
                "document_lang": "en",
            }
        ], 1

    async def fake_get_document(*args, **kwargs):
        return {
            "doc_id": "doc-1",
            "filename": "report.pdf",
            "status": "ready",
            "sub_tenant": "finance",
            "document_lang": "en",
        }

    async def fake_check_doc_access(*args, **kwargs):
        captured["check_doc_access"] = kwargs
        return True

    monkeypatch.setattr(documents_module.db, "list_documents", fake_list_documents)
    monkeypatch.setattr(documents_module.db, "get_document", fake_get_document)
    monkeypatch.setattr(documents_module, "check_doc_access", fake_check_doc_access)

    with _build_test_client(documents_module) as client:
        list_response = client.get("/documents?sub_tenant=finance")
        status_response = client.get("/documents/doc-1?sub_tenant=finance")

    assert list_response.status_code == 200
    assert list_response.json()[0]["sub_tenant"] == "finance"
    assert list_response.json()[0]["document_lang"] == "en"
    assert captured["list_documents"]["sub_tenant"] == "finance"

    assert status_response.status_code == 200
    assert status_response.json()["sub_tenant"] == "finance"
    assert status_response.json()["document_lang"] == "en"
    assert captured["check_doc_access"]["sub_tenant"] == "finance"