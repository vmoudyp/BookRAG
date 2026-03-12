import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_helper_namespace():
    source_path = Path(__file__).resolve().parents[1] / "Core" / "rag" / "gbc_rag.py"
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    wanted_assignments = {
        "_ROLE_QUERY_PATTERNS",
        "_EXPLICIT_CURRENT_TENURE_QUERY_PATTERNS",
        "_EXPLICIT_FORMER_TENURE_QUERY_PATTERNS",
        "_CURRENT_TENURE_QUERY_PATTERNS",
        "_FORMER_TENURE_QUERY_PATTERNS",
    }
    wanted_functions = {
        "_is_role_query_text",
        "_infer_role_tenure_intent",
        "_matches_role_tenure_intent",
        "_filter_role_evidence_by_tenure",
    }

    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id in wanted_assignments for target in node.targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            selected.append(node)

    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"re": re, "Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
    exec(compile(module, filename=str(source_path), mode="exec"), namespace)
    return namespace


_HELPERS = _load_helper_namespace()


def test_infers_current_and_former_tenure_intents():
    infer = _HELPERS["_infer_role_tenure_intent"]
    assert infer("Who is the current president of Indonesia?") == "current"
    assert infer("Who was the president of Indonesia before Jokowi?") == "former"


def test_leaves_ambiguous_role_queries_unfiltered():
    is_role_query = _HELPERS["_is_role_query_text"]
    infer = _HELPERS["_infer_role_tenure_intent"]

    query = "Tell me about the president of Indonesia"
    assert is_role_query(query) is True
    assert infer(query) is None


def test_current_intent_accepts_current_like_statuses_only():
    matches = _HELPERS["_matches_role_tenure_intent"]

    assert matches("current", "current") is True
    assert matches("acting", "current") is True
    assert matches("interim", "current") is True
    assert matches("unknown", "current") is True
    assert matches("former", "current") is False
    assert matches("candidate", "current") is False


def test_filters_role_evidence_by_current_and_former_intent():
    filter_evidence = _HELPERS["_filter_role_evidence_by_tenure"]
    evidence = [
        {"entity_name": "Alice", "tenure_status": "current"},
        {"entity_name": "Bob", "tenure_status": "acting"},
        {"entity_name": "Carol", "tenure_status": "unknown"},
        {"entity_name": "Dave", "tenure_status": "former"},
        {"entity_name": "Eve", "tenure_status": "candidate"},
    ]

    current_filtered = filter_evidence(evidence, "current")
    assert [row["entity_name"] for row in current_filtered] == ["Alice", "Bob", "Carol"]

    former_filtered = filter_evidence(evidence, "former")
    assert [row["entity_name"] for row in former_filtered] == ["Dave"]