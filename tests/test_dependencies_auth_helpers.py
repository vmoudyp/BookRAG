import asyncio
import importlib
import sys


def _load_dependencies(monkeypatch):
    monkeypatch.setenv("BOOKRAG_SECRET_KEY", "test-secret")
    module_name = "api.dependencies"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


def test_password_hash_round_trip(monkeypatch):
    deps = _load_dependencies(monkeypatch)

    hashed = deps.hash_password("StrongPass1")

    assert hashed != "StrongPass1"
    assert hashed.startswith("$2")
    assert deps.verify_password("StrongPass1", hashed) is True
    assert deps.verify_password("WrongPass1", hashed) is False


def test_verify_password_rejects_invalid_hash(monkeypatch):
    deps = _load_dependencies(monkeypatch)

    assert deps.verify_password("StrongPass1", "not-a-valid-hash") is False


def test_check_doc_access_forwards_sub_tenant(monkeypatch):
    deps = _load_dependencies(monkeypatch)
    captured = {}

    async def fake_get_accessible_doc_ids(uri, db_prefix, tenant_id, user_id, sub_tenant=None, include_permissions=True):
        captured.update(
            {
                "uri": uri,
                "db_prefix": db_prefix,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "sub_tenant": sub_tenant,
                "include_permissions": include_permissions,
            }
        )
        return ["doc-1"]

    monkeypatch.setattr(deps.db, "get_accessible_doc_ids", fake_get_accessible_doc_ids)

    assert asyncio.run(deps.check_doc_access("user-1", "tenant-a", "doc-1", sub_tenant="finance")) is True
    assert captured["tenant_id"] == "tenant-a"
    assert captured["user_id"] == "user-1"
    assert captured["sub_tenant"] == "finance"
    assert captured["include_permissions"] is True


def test_filter_accessible_docs_uses_visibility_scope_when_doc_ids_omitted(monkeypatch):
    deps = _load_dependencies(monkeypatch)
    captured = {}

    async def fake_get_accessible_doc_ids(uri, db_prefix, tenant_id, user_id, sub_tenant=None, include_permissions=True):
        captured.update(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "sub_tenant": sub_tenant,
                "include_permissions": include_permissions,
            }
        )
        return ["shared-doc", "finance-doc"]

    monkeypatch.setattr(deps.db, "get_accessible_doc_ids", fake_get_accessible_doc_ids)

    result = asyncio.run(
        deps.filter_accessible_docs("user-1", "tenant-a", None, sub_tenant="finance")
    )

    assert result == ["shared-doc", "finance-doc"]
    assert captured == {
        "tenant_id": "tenant-a",
        "user_id": "user-1",
        "sub_tenant": "finance",
        "include_permissions": False,
    }


def test_filter_accessible_docs_intersects_requested_ids_with_permission_fallback(monkeypatch):
    deps = _load_dependencies(monkeypatch)
    captured = {}

    async def fake_get_accessible_doc_ids(uri, db_prefix, tenant_id, user_id, sub_tenant=None, include_permissions=True):
        captured.update(
            {
                "tenant_id": tenant_id,
                "user_id": user_id,
                "sub_tenant": sub_tenant,
                "include_permissions": include_permissions,
            }
        )
        return ["doc-2", "doc-1"]

    monkeypatch.setattr(deps.db, "get_accessible_doc_ids", fake_get_accessible_doc_ids)

    result = asyncio.run(
        deps.filter_accessible_docs(
            "user-1",
            "tenant-a",
            ["doc-1", "doc-missing", "doc-2"],
            sub_tenant="finance",
        )
    )

    assert result == ["doc-1", "doc-2"]
    assert captured == {
        "tenant_id": "tenant-a",
        "user_id": "user-1",
        "sub_tenant": "finance",
        "include_permissions": True,
    }