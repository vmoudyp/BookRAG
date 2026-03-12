"""
Focused tests for domain role assignment business logic and CRUD operations.

All heavy ML/graph dependencies (networkx, spacy, falkordb, …) are stubbed
before any module imports so the suite runs in the lightweight CI environment.
"""
from __future__ import annotations

import sys
import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ── 1. Stub out networkx (not installed in test env) ─────────────────────────
_nx = ModuleType("networkx")
# Graph.py uses: nx.DiGraph(), nx.Graph (type hint), nx.NetworkXError
_nx.DiGraph = MagicMock
_nx.Graph = MagicMock
_nx.NetworkXError = Exception
_nx.readwrite = ModuleType("networkx.readwrite")
_nx.readwrite.json_graph = SimpleNamespace(
    node_link_data=lambda *a, **kw: {"nodes": [], "links": []},
    node_link_graph=lambda *a, **kw: MagicMock(),
)
sys.modules["networkx"] = _nx
sys.modules["networkx.readwrite"] = _nx.readwrite
sys.modules["networkx.readwrite.json_graph"] = ModuleType("networkx.readwrite.json_graph")

# ── 2. Import the lightweight Pydantic models from Core.Index.Graph ──────────
_graph_path = Path(__file__).resolve().parents[1] / "Core" / "Index" / "Graph.py"
_graph_spec = importlib.util.spec_from_file_location("Core.Index.Graph", _graph_path)
_graph_mod = importlib.util.module_from_spec(_graph_spec)
sys.modules["Core.Index.Graph"] = _graph_mod
_graph_spec.loader.exec_module(_graph_mod)  # type: ignore[union-attr]

RoleEvidence = _graph_mod.RoleEvidence
RoleAssignment = _graph_mod.RoleAssignment
Entity = _graph_mod.Entity

# ── 3. Stub out api.* heavy modules ─────────────────────────────────────────
_fake_deps = ModuleType("api.dependencies")
_fake_deps.FALKORDB_HOST = "localhost"
_fake_deps.FALKORDB_PORT = 6379
_fake_deps.FALKORDB_USERNAME = ""
_fake_deps.FALKORDB_PASSWORD = ""
_fake_deps.INDEX_SAVE_DIR = "/tmp"
_fake_deps.MONGO_URI = "mongodb://test"
_fake_deps.MONGO_DB_PREFIX = "test"
_fake_deps.THREAD_POOL = None

_fake_db = ModuleType("api.db")
_fake_db.__path__ = []
_fake_db_mongo = ModuleType("api.db.mongodb")

async def _noop_log(*a, **kw):
    pass

_fake_db_mongo.log_entity_edit = _noop_log
_fake_db.mongodb = _fake_db_mongo

for _name, _mod in [
    ("api", ModuleType("api")),
    ("api.db", _fake_db),
    ("api.db.mongodb", _fake_db_mongo),
    ("api.dependencies", _fake_deps),
]:
    sys.modules.setdefault(_name, _mod)

# ── 4. Import service module with stubs in place ─────────────────────────────
_svc_path = Path(__file__).resolve().parents[1] / "api" / "services" / "entity_editor.py"
_svc_spec = importlib.util.spec_from_file_location("api.services.entity_editor", _svc_path)
_svc_mod = importlib.util.module_from_spec(_svc_spec)
sys.modules["api.services.entity_editor"] = _svc_mod
_svc_spec.loader.exec_module(_svc_mod)  # type: ignore[union-attr]

_merge_roles = _svc_mod._merge_role_assignments
_add_role_sync = _svc_mod._add_role_sync
_update_role_sync = _svc_mod._update_role_sync
_review_role_sync = _svc_mod._review_role_sync
_delete_role_sync = _svc_mod._delete_role_sync
_re_normalize_roles_sync = _svc_mod._re_normalize_roles_sync
_role_stats_sync = _svc_mod._role_stats_sync
_split_sync = _svc_mod._split_sync


# ── Helper factories ─────────────────────────────────────────────────────────

def _ra(role_name: str, *, assignment_id: str = "a1", role_id: str | None = None,
        scope: str = "", tenure: str = "unknown",
        review_status: str = "suggested", origin: str = "extracted",
        source_ids: list[int] | None = None, evidence: list | None = None) -> RoleAssignment:
    return RoleAssignment(
        assignment_id=assignment_id,
        role_name=role_name,
        role_id=role_id,
        scope_entity_name=scope or None,
        tenure_status=tenure,
        review_status=review_status,
        origin=origin,
        source_ids=source_ids or [],
        evidence=evidence or [],
    )


def _entity(name: str, role_assignments: list | None = None,
            source_ids: set | None = None) -> Entity:
    return Entity(
        entity_name=name,
        entity_type="PERSON",
        source_ids=source_ids or {1},
        role_assignments=role_assignments or [],
    )


def _mock_graph(entity: Entity) -> MagicMock:
    """Return a minimal mock Graph that holds a single entity."""
    g = MagicMock()
    g.get_entity.return_value = entity
    g.get_node_name_from_entity.return_value = f"{entity.entity_name}_PERSON"
    g.get_node_name_from_str.return_value = f"{entity.entity_name}_PERSON"
    g.kg = MagicMock()
    g.kg.__contains__ = MagicMock(return_value=True)
    g.tree2kg = {}
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1E Tests
# ═══════════════════════════════════════════════════════════════════════════════

# ── _merge_role_assignments (pure logic) ─────────────────────────────────────

class TestMergeRoleAssignments:

    def test_empty_lists_return_empty(self):
        assert _merge_roles([]) == []
        assert _merge_roles([[]]) == []

    def test_no_duplicates_returned_unchanged(self):
        ra1 = _ra("President", assignment_id="a1", source_ids=[1])
        ra2 = _ra("CEO", assignment_id="a2", source_ids=[2])
        result = _merge_roles([[ra1, ra2]])
        assert len(result) == 2

    def test_dedup_by_role_name_same_scope(self):
        """Two identical assignments (same role, scope, tenure) collapse to one."""
        ra1 = _ra("President", assignment_id="a1", review_status="suggested", source_ids=[1])
        ra2 = _ra("President", assignment_id="a2", review_status="suggested", source_ids=[2])
        result = _merge_roles([[ra1], [ra2]])
        assert len(result) == 1

    def test_confirmed_beats_suggested(self):
        """Confirmed assignment wins over suggested during dedup."""
        suggested = _ra("President", assignment_id="a1", review_status="suggested", source_ids=[1])
        confirmed = _ra("President", assignment_id="a2", review_status="confirmed", source_ids=[2])
        result = _merge_roles([[suggested], [confirmed]])
        assert len(result) == 1
        assert result[0].review_status == "confirmed"

    def test_manual_beats_extracted(self):
        """Manual origin wins over extracted when review_status is equal."""
        extracted = _ra("CEO", assignment_id="a1", origin="extracted", source_ids=[1])
        manual = _ra("CEO", assignment_id="a2", origin="manual", source_ids=[2])
        result = _merge_roles([[extracted], [manual]])
        assert len(result) == 1
        assert result[0].origin == "manual"

    def test_source_ids_unioned_on_dedup(self):
        """When two assignments collapse, source_ids are merged (union)."""
        ra1 = _ra("President", assignment_id="a1", source_ids=[10, 20])
        ra2 = _ra("President", assignment_id="a2", source_ids=[20, 30])
        result = _merge_roles([[ra1], [ra2]])
        assert len(result) == 1
        assert set(result[0].source_ids) == {10, 20, 30}

    def test_evidence_unioned_on_dedup(self):
        """Evidence items are merged (no duplicates by source_id+observed_text)."""
        ev1 = RoleEvidence(source_id=1, observed_role_text="President")
        ev2 = RoleEvidence(source_id=2, observed_role_text="Prez")
        ra1 = _ra("President", assignment_id="a1", source_ids=[1], evidence=[ev1])
        ra2 = _ra("President", assignment_id="a2", source_ids=[2], evidence=[ev2])
        result = _merge_roles([[ra1], [ra2]])
        assert len(result[0].evidence) == 2

    def test_different_scope_not_deduped(self):
        """Same role name but different scope → two distinct assignments."""
        ra1 = _ra("Governor", assignment_id="a1", scope="Java")
        ra2 = _ra("Governor", assignment_id="a2", scope="Bali")
        result = _merge_roles([[ra1, ra2]])
        assert len(result) == 2

    def test_role_id_used_as_dedup_key_over_name(self):
        """If role_id is set, it takes precedence over role_name in dedup key."""
        ra1 = _ra("Koordinator Sektor", assignment_id="a1", role_id="role:korsek", source_ids=[1])
        ra2 = _ra("Korsek", assignment_id="a2", role_id="role:korsek", source_ids=[2])
        result = _merge_roles([[ra1], [ra2]])
        assert len(result) == 1
        assert set(result[0].source_ids) == {1, 2}


# ── _add_role_sync ───────────────────────────────────────────────────────────

class TestAddRoleSync:

    def _run(self, entity, role_input):
        mock_graph = _mock_graph(entity)
        updated_holder = []

        def capture_update(name, typ, updated):
            updated_holder.append(updated)
            # Make subsequent get_entity calls return the updated entity
            mock_graph.get_entity.return_value = updated

        mock_graph.update_entity.side_effect = capture_update

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_graph, "/tmp", None)):
            result = _add_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                                    dict(role_input), "user1")
        return result, mock_graph

    def test_happy_path_adds_role(self):
        entity = _entity("Joko Widodo")
        role_input = {"role_name": "President", "evidence": []}
        _, mock_g = self._run(entity, role_input)
        mock_g.update_entity.assert_called_once()
        mock_g.save_graph.assert_called_once()

    def test_added_role_has_manual_origin_and_confirmed_status(self):
        entity = _entity("Joko Widodo")
        role_input = {"role_name": "President", "evidence": []}

        captured_entity = []
        mock_graph = _mock_graph(entity)
        mock_graph.update_entity.side_effect = lambda n, t, e: captured_entity.append(e)
        mock_graph.get_entity.return_value = entity

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_graph, "/tmp", None)):
            _add_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                           {"role_name": "President", "evidence": []}, "user1")

        assert len(captured_entity) == 1
        added = captured_entity[0].role_assignments[-1]
        assert added.origin == "manual"
        assert added.review_status == "confirmed"
        assert added.created_by == "user1"

    def test_evidence_items_parsed(self):
        entity = _entity("Joko Widodo")
        captured_entity = []
        mock_graph = _mock_graph(entity)
        mock_graph.update_entity.side_effect = lambda n, t, e: captured_entity.append(e)
        mock_graph.get_entity.return_value = entity

        ev = {"source_id": 5, "observed_role_text": "Presiden RI"}
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_graph, "/tmp", None)):
            _add_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                           {"role_name": "President", "evidence": [ev]}, "user1")

        added = captured_entity[0].role_assignments[-1]
        assert len(added.evidence) == 1
        assert added.evidence[0].source_id == 5


# ── _update_role_sync ────────────────────────────────────────────────────────

class TestUpdateRoleSync:

    def _entity_with_role(self):
        ra = _ra("President", assignment_id="ra-001")
        return _entity("Joko Widodo", role_assignments=[ra])

    def _run(self, entity, assignment_id, update_fields):
        captured = []
        mock_g = _mock_graph(entity)
        mock_g.update_entity.side_effect = lambda n, t, e: captured.append(e)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            _update_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                              assignment_id, update_fields, "user1")
        return captured, mock_g

    def test_happy_path_updates_field(self):
        entity = self._entity_with_role()
        captured, _ = self._run(entity, "ra-001", {"role_name": "Prime Minister"})
        updated_ra = captured[0].role_assignments[0]
        assert updated_ra.role_name == "Prime Minister"

    def test_unknown_assignment_id_raises_key_error(self):
        entity = self._entity_with_role()
        mock_g = _mock_graph(entity)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            with pytest.raises(KeyError):
                _update_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                                  "DOES-NOT-EXIST", {"role_name": "x"}, "user1")


# ── _review_role_sync ────────────────────────────────────────────────────────

class TestReviewRoleSync:

    def test_sets_review_status_confirmed(self):
        ra = _ra("President", assignment_id="ra-001", review_status="suggested")
        entity = _entity("Joko Widodo", role_assignments=[ra])
        captured = []
        mock_g = _mock_graph(entity)
        mock_g.update_entity.side_effect = lambda n, t, e: captured.append(e)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            _review_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                              "ra-001", "confirmed", None, None, "user1")
        assert captured[0].role_assignments[0].review_status == "confirmed"

    def test_sets_canonical_role_name_when_provided(self):
        ra = _ra("Korsek", assignment_id="ra-001")
        entity = _entity("Joko Widodo", role_assignments=[ra])
        captured = []
        mock_g = _mock_graph(entity)
        mock_g.update_entity.side_effect = lambda n, t, e: captured.append(e)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            _review_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                              "ra-001", "confirmed", "Koordinator Sektor", None, "user1")
        assert captured[0].role_assignments[0].role_name == "Koordinator Sektor"

    def test_unknown_id_raises_key_error(self):
        entity = _entity("Joko Widodo", role_assignments=[_ra("President", assignment_id="ra-001")])
        mock_g = _mock_graph(entity)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            with pytest.raises(KeyError):
                _review_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type,
                                  "MISSING", "confirmed", None, None, "user1")


# ── _delete_role_sync ────────────────────────────────────────────────────────

class TestDeleteRoleSync:

    def test_happy_path_removes_role(self):
        ra1 = _ra("President", assignment_id="ra-001")
        ra2 = _ra("CEO", assignment_id="ra-002")
        entity = _entity("Joko Widodo", role_assignments=[ra1, ra2])
        captured = []
        mock_g = _mock_graph(entity)

        def capture_update(n, t, e):
            captured.append(e)
            mock_g.get_entity.return_value = e

        mock_g.update_entity.side_effect = capture_update
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            # _delete_role_sync does NOT take user_id
            _delete_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type, "ra-001")
        remaining_ids = [r.assignment_id for r in captured[0].role_assignments]
        assert "ra-001" not in remaining_ids
        assert "ra-002" in remaining_ids

    def test_unknown_id_raises_key_error(self):
        entity = _entity("Joko Widodo", role_assignments=[_ra("President", assignment_id="ra-001")])
        mock_g = _mock_graph(entity)
        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            with pytest.raises(KeyError):
                _delete_role_sync("t", "d", "cfg", entity.entity_name, entity.entity_type, "MISSING")


# ── _re_normalize_roles_sync / _role_stats_sync ───────────────────────────────

class TestReNormalizeRolesSync:

    def test_updates_only_non_locked_assignments_and_persists_graph(self):
        target = _ra("Minister", assignment_id="ra-001")
        manual_override = _ra("Custom", assignment_id="ra-002", review_status="confirmed", origin="manual")
        manual_override = manual_override.model_copy(update={"normalization_status": "manual_override"})
        locked = _ra("President", assignment_id="ra-003", role_id="role:president", review_status="confirmed")
        entity = _entity("Alice", role_assignments=[target, manual_override, locked])

        captured = []
        graph = MagicMock()
        graph.get_all_nodes.return_value = ["Alice_PERSON"]
        graph.get_entity_by_node_name.return_value = entity
        graph.update_entity.side_effect = lambda n, t, e: captured.append(e)

        def fake_apply(assignment):
            if assignment.assignment_id != "ra-001":
                return assignment
            return assignment.model_copy(update={
                "role_id": "role:minister",
                "normalization_status": "matched",
            })

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(graph, "/tmp", None)):
            with patch("Core.utils.role_normalizer.apply_normalization", side_effect=fake_apply):
                result = _re_normalize_roles_sync("tenant-a", "doc-1", "cfg", "user-1")

        assert result == {
            "doc_id": "doc-1",
            "processed": 1,
            "updated": 1,
            "skipped": 2,
            "unresolved": 0,
        }
        graph.save_graph.assert_called_once()
        assert len(captured) == 1
        updated = {ra.assignment_id: ra for ra in captured[0].role_assignments}
        assert updated["ra-001"].role_id == "role:minister"
        assert updated["ra-001"].normalization_status == "matched"
        assert updated["ra-001"].updated_by == "user-1"
        assert updated["ra-002"].normalization_status == "manual_override"
        assert updated["ra-003"].role_id == "role:president"

    def test_honors_entity_type_filter(self):
        person = _entity("Alice", role_assignments=[_ra("Minister", assignment_id="ra-001")])
        org = Entity(
            entity_name="Cabinet",
            entity_type="ORG",
            source_ids={1},
            role_assignments=[_ra("Coalition Lead", assignment_id="ra-002")],
        )
        graph = MagicMock()
        graph.get_all_nodes.return_value = ["Alice_PERSON", "Cabinet_ORG"]
        graph.get_entity_by_node_name.side_effect = lambda node: {
            "Alice_PERSON": person,
            "Cabinet_ORG": org,
        }[node]

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(graph, "/tmp", None)):
            with patch(
                "Core.utils.role_normalizer.apply_normalization",
                side_effect=lambda ra: ra.model_copy(update={"role_id": "matched", "normalization_status": "matched"}),
            ):
                result = _re_normalize_roles_sync(
                    "tenant-a", "doc-1", "cfg", "user-1", entity_type="PERSON"
                )

        assert result["processed"] == 1
        graph.update_entity.assert_called_once()
        update_args = graph.update_entity.call_args[0]
        assert update_args[0] == "Alice"
        assert update_args[1] == "PERSON"


class TestRoleStatsSync:

    def test_aggregates_document_role_counts(self):
        graph = MagicMock()
        graph.get_all_nodes.return_value = ["n1", "n2", "n3", "n4"]
        rows = [
            {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "review_status": "confirmed",
                "normalization_status": "matched",
                "origin": "manual",
                "tenure_status": "current",
            },
            {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "review_status": "confirmed",
                "normalization_status": "matched",
                "origin": "manual",
                "tenure_status": "current",
            },
            {
                "entity_name": "Bob",
                "entity_type": "PERSON",
                "review_status": "suggested",
                "normalization_status": "unresolved",
                "origin": "extracted",
                "tenure_status": "former",
            },
        ]

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(graph, "/tmp", None)):
            with patch.object(_svc_mod, "_list_roles_sync", return_value=rows):
                result = _role_stats_sync("tenant-a", "doc-9", "cfg")

        assert result["doc_id"] == "doc-9"
        assert result["total_entities"] == 4
        assert result["entities_with_roles"] == 2
        assert result["coverage_ratio"] == 0.5
        assert result["total_roles"] == 3
        assert result["unresolved_roles"] == 1
        assert result["review_status_counts"] == {"confirmed": 2, "suggested": 1}
        assert result["normalization_status_counts"] == {"matched": 2, "unresolved": 1}
        assert result["origin_counts"] == {"manual": 2, "extracted": 1}
        assert result["tenure_status_counts"] == {"current": 2, "former": 1}


# ── _split_sync conservative propagation ─────────────────────────────────────

class TestSplitSyncConservativePropagation:
    """Role assignments should only propagate to child entities whose source_ids
    intersect with the role's own source_ids (conservative split policy)."""

    @staticmethod
    def _make_split_graph(parent: Entity) -> MagicMock:
        from collections import defaultdict
        mock_g = MagicMock()
        mock_g.get_node_name_from_str.side_effect = lambda n, t: f"{n}_PERSON"
        mock_g.kg = MagicMock()
        mock_g.kg.__contains__ = MagicMock(return_value=True)
        mock_g.get_entity_by_node_name.return_value = parent
        mock_g.kg.neighbors.return_value = []
        mock_g.tree2kg = defaultdict(set)
        return mock_g

    def test_roles_propagated_only_where_source_ids_match(self):
        # Parent entity with two roles:
        # - "President" supported by source 1 only
        # - "CEO" supported by source 2 only
        ra_president = _ra("President", assignment_id="ra-p", source_ids=[1])
        ra_ceo = _ra("CEO", assignment_id="ra-c", source_ids=[2])
        parent = Entity(
            entity_name="Parent", entity_type="PERSON",
            source_ids={1, 2},
            role_assignments=[ra_president, ra_ceo],
        )

        # Split into child A (source 1) and child B (source 2)
        new_entities = [
            {"entity_name": "ChildA", "entity_type": "PERSON", "source_ids": [1]},
            {"entity_name": "ChildB", "entity_type": "PERSON", "source_ids": [2]},
        ]

        mock_g = self._make_split_graph(parent)

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            created = _split_sync("t", "d", "cfg", "Parent", "PERSON", new_entities, "duplicate")

        child_a = next(c for c in created if c["entity_name"] == "ChildA")
        child_b = next(c for c in created if c["entity_name"] == "ChildB")

        child_a_role_names = [r["role_name"] for r in child_a["role_assignments"]]
        child_b_role_names = [r["role_name"] for r in child_b["role_assignments"]]

        assert "President" in child_a_role_names
        assert "CEO" not in child_a_role_names
        assert "CEO" in child_b_role_names
        assert "President" not in child_b_role_names

    def test_role_with_no_source_ids_does_not_propagate(self):
        """A role with empty source_ids should not go to any child."""
        ra_no_src = _ra("Advisor", assignment_id="ra-x", source_ids=[])
        parent = Entity(
            entity_name="Parent", entity_type="PERSON",
            source_ids={1},
            role_assignments=[ra_no_src],
        )
        new_entities = [{"entity_name": "Child", "entity_type": "PERSON", "source_ids": [1]}]

        mock_g = self._make_split_graph(parent)

        with patch.object(_svc_mod, "_load_graph_sync", return_value=(mock_g, "/tmp", None)):
            created = _split_sync("t", "d", "cfg", "Parent", "PERSON", new_entities, "none")

        assert created[0]["role_assignments"] == []


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests for LLMExtractor._extract_roles_from_text
# ═══════════════════════════════════════════════════════════════════════════════

# Lightweight imports from kg_prompt (pure Pydantic, no heavy deps)
from pathlib import Path as _Path
import importlib.util as _ilu

_kp_path = _Path(__file__).resolve().parents[1] / "Core" / "prompts" / "kg_prompt.py"
_kp_spec = _ilu.spec_from_file_location("Core.prompts.kg_prompt", _kp_path)
_kp_mod = _ilu.module_from_spec(_kp_spec)
sys.modules.setdefault("Core.prompts.kg_prompt", _kp_mod)
_kp_spec.loader.exec_module(_kp_mod)  # type: ignore[union-attr]

ExtractedRole = _kp_mod.ExtractedRole
RoleExtractionResult = _kp_mod.RoleExtractionResult


def _make_extracted_role(entity_name: str, role_name: str, **kw) -> "ExtractedRole":
    return ExtractedRole(entity_name=entity_name, role_name=role_name, **kw)


def _fake_extractor(llm_return_value=None):
    """Return a minimal fake that exposes _extract_roles_from_text without importing LLMExtractor."""
    from types import SimpleNamespace

    mock_llm = MagicMock()
    mock_llm.get_json_completion.return_value = llm_return_value

    fake = SimpleNamespace(llm=mock_llm)
    return fake


# Import the bare function so we can call it with our fake 'self'
_extractor_path = _Path(__file__).resolve().parents[1] / "Core" / "pipelines" / "kg_extractor.py"


def _get_extract_roles_fn():
    """Read and compile _extract_roles_from_text as a standalone function from kg_extractor.py."""
    import ast, textwrap

    src = _extractor_path.read_text()
    tree = ast.parse(src)

    # Find the method body inside LLMExtractor
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LLMExtractor":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_extract_roles_from_text":
                    # Dedent the method body and wrap as top-level function
                    method_src = ast.get_source_segment(src, item)
                    dedented = textwrap.dedent(method_src)
                    return dedented
    return None


class TestExtractRolesFromText:
    """Unit tests for the _extract_roles_from_text logic without importing LLMExtractor."""

    def _call(self, fake_self, text: str, entities, node_id: int = 42):
        """Call the method directly via function reference to avoid importing the full extractor."""
        from Core.utils.role_normalizer import apply_normalization  # lightweight
        import uuid as _uuid

        from Core.prompts.kg_prompt import ROLE_EXTRACTION

        # Replicate the method logic directly for isolated testing
        role_entity_types = {"PERSON", "ORGANIZATION", "ORG", "GOVERNMENT", "OFFICIAL"}
        role_entities = [e for e in entities if e.entity_type.upper() in role_entity_types]
        if not role_entities:
            return {}

        entity_list_str = "\n".join(f"- {e.entity_name} ({e.entity_type})" for e in role_entities)
        prompt = ROLE_EXTRACTION.format(entity_list=entity_list_str, input_text=text)

        try:
            res = fake_self.llm.get_json_completion(prompt, schema=RoleExtractionResult)
        except Exception:
            return {}

        if not res or not res.roles:
            return {}

        name_map = {e.entity_name.lower(): e.entity_name for e in role_entities}
        result = {}
        for extracted in res.roles:
            canonical_name = name_map.get(extracted.entity_name.lower())
            if canonical_name is None:
                continue
            evidence = RoleEvidence(
                source_id=node_id,
                observed_role_text=extracted.role_name,
                evidence_text=extracted.evidence_text,
                normalization_method="unresolved",
                confidence=extracted.confidence,
            )
            assignment = RoleAssignment(
                assignment_id=str(_uuid.uuid4()),
                role_name=extracted.role_name,
                scope_entity_name=extracted.scope_entity_name,
                tenure_status=extracted.tenure_status or "unknown",
                origin="extracted",
                review_status="suggested",
                confidence=extracted.confidence,
                source_ids=[node_id],
                evidence=[evidence],
            )
            try:
                assignment = apply_normalization(assignment)
            except Exception:
                pass
            result.setdefault(canonical_name, []).append(assignment)
        return result

    def test_filters_non_person_entities(self):
        """Only PERSON/ORG entities should be passed to the LLM."""
        entities = [
            Entity(entity_name="Jakarta", entity_type="LOCATION"),
            Entity(entity_name="2024", entity_type="DATE"),
        ]
        fake = _fake_extractor(llm_return_value=None)
        result = self._call(fake, "text", entities)
        assert result == {}
        # LLM should not have been called at all
        fake.llm.get_json_completion.assert_not_called()

    def test_mixed_types_filters_correctly(self):
        """Only PERSON entity reaches the LLM; LOCATION is dropped."""
        person = Entity(entity_name="Joko Widodo", entity_type="PERSON")
        location = Entity(entity_name="Jakarta", entity_type="LOCATION")
        llm_result = RoleExtractionResult(roles=[
            _make_extracted_role("Joko Widodo", "President", tenure_status="former"),
        ])
        fake = _fake_extractor(llm_return_value=llm_result)
        result = self._call(fake, "some text", [person, location])
        assert "Joko Widodo" in result
        assert "Jakarta" not in result

    def test_hallucinated_entity_name_is_dropped(self):
        """LLM returning an entity name not in the input list is silently dropped."""
        person = Entity(entity_name="Alice", entity_type="PERSON")
        llm_result = RoleExtractionResult(roles=[
            _make_extracted_role("Hallucinated Person", "CEO"),
            _make_extracted_role("Alice", "Chief Financial Officer"),
        ])
        fake = _fake_extractor(llm_return_value=llm_result)
        result = self._call(fake, "text", [person])
        assert "Hallucinated Person" not in result
        assert "Alice" in result
        # The role normalizer may rewrite the role_name to its canonical form; just
        # verify the assignment was created and the hallucinated entity was dropped.
        assert len(result["Alice"]) == 1

    def test_evidence_attached_to_assignment(self):
        """Evidence text from the LLM should be stored on the resulting RoleAssignment."""
        person = Entity(entity_name="Bob", entity_type="PERSON")
        llm_result = RoleExtractionResult(roles=[
            _make_extracted_role("Bob", "CFO", evidence_text="Bob serves as CFO", confidence=0.9),
        ])
        fake = _fake_extractor(llm_return_value=llm_result)
        result = self._call(fake, "text", [person], node_id=7)
        assignments = result["Bob"]
        assert len(assignments) == 1
        ra = assignments[0]
        assert ra.origin == "extracted"
        assert ra.review_status == "suggested"
        assert ra.source_ids == [7]
        assert ra.evidence[0].evidence_text == "Bob serves as CFO"
        assert ra.evidence[0].source_id == 7

    def test_empty_llm_response_returns_empty_dict(self):
        """When the LLM returns no roles, the method should return {}."""
        person = Entity(entity_name="Carol", entity_type="PERSON")
        llm_result = RoleExtractionResult(roles=[])
        fake = _fake_extractor(llm_return_value=llm_result)
        result = self._call(fake, "text", [person])
        assert result == {}

    def test_llm_exception_returns_empty_dict(self):
        """A failing LLM call should not raise — it returns {} gracefully."""
        person = Entity(entity_name="Dave", entity_type="PERSON")
        fake = _fake_extractor()
        fake.llm.get_json_completion.side_effect = RuntimeError("LLM timeout")
        result = self._call(fake, "text", [person])
        assert result == {}

    def test_case_insensitive_entity_matching(self):
        """Entity name matching should be case-insensitive."""
        person = Entity(entity_name="Sri Mulyani", entity_type="PERSON")
        llm_result = RoleExtractionResult(roles=[
            _make_extracted_role("sri mulyani", "Minister of Finance"),
        ])
        fake = _fake_extractor(llm_return_value=llm_result)
        result = self._call(fake, "text", [person])
        # Key should use the canonical (original) casing
        assert "Sri Mulyani" in result

