# Leaf-Text Entity Extraction Patch Plan

## 1. Purpose

This document describes a minimal-risk patch plan for implementing **leaf-text-scoped entity extraction** in BookRAG.

The plan is intentionally conservative:

- change node routing first
- avoid rewriting extractor internals
- preserve backward compatibility where possible

## 2. Implementation Strategy

Apply the change at the **KG orchestration layer** in `Core/pipelines/kg_builder.py`, because that is where the repository currently decides which nodes are passed into:

- `kg_extractor.batch_extract_titles(...)`
- `kg_extractor.batch_extract_kg(...)`

This is the narrowest high-value change because:

- `LocalExtractor` already extracts from `node.meta_info.content`
- `LLMExtractor` already extracts from `node.meta_info.content`
- title extraction already has a dedicated route

## 3. Target Behavior After Patch

With the recommended default configuration:

- `TITLE` nodes continue through `extract_title(...)`
- `TEXT` nodes are included in the main text path only if they are **body-text leaves**
- `IMAGE`, `TABLE`, and `EQUATION` nodes continue through specialized extraction
- provenance and temporary per-node result caching continue to work unchanged

## 4. Files to Change

### 4.1 `Core/configs/graph_config.py`

Add configuration fields for text-node selection. Recommended additions:

- `text_extraction_scope: str = "body_text_leaves"`
- `min_text_chars: int = 1`
- `require_nonempty_text: bool = True`
- optional `allowed_text_source_roles: tuple[str, ...] = ("body_text",)`

Why:

- keeps routing policy configurable
- avoids hard-coded behavior in the builder

### 4.2 `config/graph.yaml`

Add the corresponding runtime config under `graph:`.

Recommended initial values:

- `text_extraction_scope: body_text_leaves`
- `min_text_chars: 1`
- `require_nonempty_text: true`
- `allowed_text_source_roles: [body_text]`

Optional follow-up:

- propagate the same keys to other config presets that rely on KG extraction

### 4.3 `Core/pipelines/kg_builder.py`

This is the main implementation surface.

Add helper functions with narrow responsibilities, for example:

- `_node_has_text_content(node, cfg)`
- `_is_leaf_text_node(node)`
- `_is_eligible_text_node(node, cfg_graph)`

Then update `build_knowledge_graph(...)` so the current loop over `tree.nodes` uses the helper predicate instead of routing all non-title nodes into `batch_nodes`.

Recommended routing logic:

1. skip root
2. route `TITLE` nodes through the title path as today
3. route `TEXT` nodes only if they satisfy the configured text-scope predicate
4. continue routing `IMAGE`, `TABLE`, and `EQUATION` nodes through `batch_nodes`
5. skip unsupported or empty nodes with explicit logging

### 4.4 `Core/pipelines/kg_extractor.py`

No required functional change in the first patch.

Optional low-risk improvements:

- add defensive logging for empty text nodes
- document that text extraction assumes upstream node selection has already filtered scope

The key point is that this file should remain largely unchanged in the first iteration.

### 4.5 `tests/`

Add focused tests for selection and routing behavior.

Recommended new test file:

- `tests/test_kg_builder_leaf_text_selection.py`

## 5. Detailed Patch Steps

### Step 1: add config fields

Update `GraphConfig` so the routing behavior is controlled by config instead of hard-coded conditions.

Expected result:

- the code can choose among `all_text_nodes`, `leaf_text_nodes`, and `body_text_leaves`

### Step 2: update config YAML

Add the new keys to `config/graph.yaml` and any other config presets that should opt in.

Expected result:

- runtime config and Python dataclass stay in sync

### Step 3: add node-selection helpers in `kg_builder.py`

Implement small pure helpers for:

- non-empty text checks
- leaf checks
- source-role checks
- scope dispatch by mode

Expected result:

- routing logic is readable and testable

### Step 4: patch `build_knowledge_graph(...)`

Replace the current broad routing:

- all titles -> title path
- all non-titles -> generic batch path

with scoped routing:

- titles -> title path
- eligible text nodes -> generic batch path
- image/table/equation -> generic batch path
- ineligible text nodes -> skipped

Expected result:

- extraction is narrowed without affecting modality-specific logic

### Step 5: add tests

Add tests covering all supported text scopes.

Expected result:

- future refactors do not accidentally widen extraction scope again

## 6. Recommended Test Cases

### 6.1 `all_text_nodes`

Given a tree with:

- body text node
- title-like text node
- empty text node

Expect:

- all non-empty `TEXT` nodes are selected

### 6.2 `leaf_text_nodes`

Given:

- a parent `TEXT` node with a child
- a terminal `TEXT` node

Expect:

- only the terminal text node is selected

### 6.3 `body_text_leaves`

Given:

- terminal text node with `source_role=body_text`
- terminal text node with `source_role=title`
- non-terminal body text node

Expect:

- only the terminal body-text node is selected

### 6.4 Non-text node preservation

Given:

- image, table, equation nodes

Expect:

- they are still routed to the generic KG extraction batch

### 6.5 Title path preservation

Given:

- one `TITLE` node

Expect:

- it is still routed through `batch_extract_titles(...)`

## 7. Suggested Test Style

Prefer unit-style tests that:

- build a tiny synthetic `DocumentTree`
- avoid model calls
- validate which nodes are selected or routed

If practical, isolate the routing predicate into a helper that can be tested without invoking `LLM`, `VLM`, or actual extractor backends.

## 8. Backward Compatibility Plan

To reduce rollout risk, the patch should preserve behavior for non-text modalities.

Recommended compatibility choices:

- titles remain separate
- image/table/equation behavior unchanged
- only text-node routing narrows

If a fallback path is desired, set:

- `text_extraction_scope: all_text_nodes`

This provides a simple rollback without reverting code.

## 9. Logging Recommendations

Add useful logs in `kg_builder.py`, such as:

- number of title nodes selected
- number of eligible text nodes selected
- number of skipped text nodes
- number of non-text modality nodes selected

This helps validate the scope change on real documents.

## 10. Non-Changes in This Patch

Do not change the following in the first implementation:

- `LocalExtractor` relation rules
- `LLMExtractor` chunking strategy
- ontology alignment
- role normalization via `config/role_vocab.yaml`
- graph refinement behavior

These are intentionally out of scope for the first patch.

## 11. Rollout Order

Recommended rollout order:

1. add config fields
2. implement routing helpers
3. patch `build_knowledge_graph(...)`
4. add tests
5. run targeted tests
6. validate node counts and extraction outputs on one representative document

## 12. Acceptance Criteria

The patch is complete when:

1. `body_text_leaves` works as a configurable extraction scope.
2. Text-node routing is tested.
3. Titles still use the dedicated title path.
4. Image/table/equation routing is unchanged.
5. Existing per-node provenance and temp-result saving still work.
6. The default config reduces noisy structural text in KG extraction inputs.

## 13. Recommended Minimal Patch Shape

If minimizing code churn is the top priority, the best first patch is:

- add a few config fields
- add helper predicates in `Core/pipelines/kg_builder.py`
- keep `Core/pipelines/kg_extractor.py` unchanged
- add one focused routing test file

This gives the project a clean, reversible leaf-text extraction policy without entangling it with multilingual model migration or deeper KG redesign.