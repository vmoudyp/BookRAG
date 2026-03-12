# Leaf-Text Entity Extraction Design for BookRAG

## 1. Purpose

This document defines a practical design for extracting entities from **leaf text** in BookRAG before changing model stacks or rewriting extractor internals.

The main goal is to improve entity quality by choosing the **right node scope** in the `DocumentTree`.

## 2. Decision Summary

The recommended default is:

- extract entities from **terminal `TEXT` nodes** only
- require non-empty `node.meta_info.content`
- restrict the first pass to `source_role == "body_text"`
- keep title extraction as a separate path
- keep table/image/equation extraction as specialized paths
- preserve `node.index_id` as provenance for every extracted result

Architecturally, this should be implemented as a **selection/orchestration change**, not as a rewrite of `LocalExtractor` or `LLMExtractor`.

## 3. Current Repository Findings

### 3.1 Tree representation

The current tree representation already contains the structure needed for leaf-text extraction:

- `Core/Index/Tree.py`
  - `NodeType.TEXT`
  - `TreeNode.children`
  - `TreeNode.meta_info.content`
  - `DocumentTree.get_filtered_nodes(...)`

This means the repository already has:

- a text node type
- explicit parent/child structure
- per-node content storage
- a natural provenance ID via `node.index_id`

### 3.2 Source-role metadata

The tree builders already assign semantic source roles:

- `Core/pipelines/tree_node_builder.py`
  - PDF text nodes use roles such as `title` and `body_text`
- `Core/pipelines/doc_tree_builder.py`
  - normalized HTML/PDF text content is stored as `NodeType.TEXT`
  - `source_role` is preserved on created nodes

This makes it possible to distinguish:

- body prose
- titles/headings
- other text-like structural content

### 3.3 Extraction behavior today

`Core/pipelines/kg_extractor.py` already extracts **per node**:

- `LocalExtractor.extract(node)` reads `node.meta_info.content`
- `LLMExtractor._extract_kg_from_text(node)` reads `node.meta_info.content`
- `LLMExtractor` chunks long node text internally

### 3.4 Orchestration behavior today

`Core/pipelines/kg_builder.py` currently sends:

- all `TITLE` nodes through `extract_title(...)`
- all other non-root nodes through `batch_extract_kg(...)`

So the current pipeline is broader than leaf-text extraction. It effectively processes:

- text nodes
- image nodes
- table nodes
- equation nodes
- and any other non-title nodes routed through the generic batch path

## 4. Definition of “Leaf Text”

For this repository, a **leaf-text node** should mean:

- `node.type == NodeType.TEXT`
- `len(node.children) == 0`
- `node.meta_info.content` is not empty after stripping whitespace

The recommended stricter production definition is **body-text leaf**:

- leaf-text node
- `node.meta_info.source_role == "body_text"`

## 5. Recommended Extraction Modes

The design should support three conceptual modes:

### Mode A: `all_text_nodes`

Eligible nodes:

- all `NodeType.TEXT` nodes with non-empty content

Use when:

- maximizing recall
- debugging coverage

Risk:

- more heading/title noise

### Mode B: `leaf_text_nodes`

Eligible nodes:

- terminal `NodeType.TEXT` nodes with non-empty content

Use when:

- “leaf text” is interpreted literally as terminal text nodes

Risk:

- still includes some title-like text if titles are stored as text

### Mode C: `body_text_leaves` (recommended default)

Eligible nodes:

- terminal `NodeType.TEXT` nodes
- non-empty content
- `source_role == "body_text"`

Use when:

- better entity quality matters more than maximum recall
- the first rollout should minimize structural noise

## 6. Proposed Architecture

### 6.1 Stage 1: node selection

Select eligible nodes from the `DocumentTree` before extraction begins.

This stage should decide:

- whether a node is text-eligible
- whether leaf-ness is required
- whether source role must match `body_text`

### 6.2 Stage 2: per-node extraction

For each eligible node:

- read `node.meta_info.content`
- call `kg_extractor.extract_kg(node)`
- preserve `node.index_id`

The extractor remains responsible for:

- entity extraction
- relation extraction
- any internal chunking for long node text

### 6.3 Stage 3: node-local repair

Keep the current node-local cleanup behavior such as:

- result repair
- name normalization
- local deduplication

### 6.4 Stage 4: cross-node consolidation

After extracting all eligible leaf nodes:

- merge duplicate entities across nodes
- union `source_ids`
- retain role evidence and descriptions where possible

## 7. Separation of Concerns

The design should explicitly separate these concerns:

### Tree selection

Determines the provenance unit.

### In-node chunking

Determines the model inference unit for long text.

These should not be conflated. A node may be a leaf and still be long enough to chunk internally.

## 8. Interaction with Other Node Types

### Titles

Titles should remain a separate extraction path:

- they are useful context
- they often behave differently from body prose
- they should not be mixed into the first leaf-text entity pass by default

### Images, Tables, Equations

These should remain specialized extraction paths because:

- they are not plain prose
- they already use modality-specific logic in `kg_extractor.py`

## 9. Provenance Requirements

Each extracted entity or relationship should remain attributable to the source leaf node.

Minimum provenance to preserve:

- `node.index_id`
- `source_ids`
- optional: `page_idx`
- optional: `source_role`

This supports:

- evidence display
- role extraction trust
- future scoring based on repeated mentions across leaves

## 10. Configuration Recommendations

The extraction scope should be configurable instead of hard-coded.

Recommended settings:

- `graph.text_extraction_scope`
  - values: `all_text_nodes`, `leaf_text_nodes`, `body_text_leaves`
- `graph.min_text_chars`
- `graph.require_nonempty_text`
- optional `graph.allowed_text_source_roles`

Recommended default:

- `text_extraction_scope: body_text_leaves`

## 11. Non-Goals for the First Iteration

The first leaf-text rollout should **not** try to solve all of the following at once:

- replacing spaCy globally
- redesigning relation extraction
- changing title extraction behavior
- cross-paragraph inference beyond node boundaries
- ontology redesign

Those can be addressed after the scope change is validated.

## 12. Expected Benefits

Expected gains from this design:

- lower noise from headings and structural text
- clearer provenance for extracted entities
- better fit for multilingual improvements later
- simpler validation because the change is mostly routing-based

## 13. Risks and Tradeoffs

Main tradeoffs:

- some entities mentioned only in titles may be missed in the primary pass
- some cross-section relations may be lost when extraction becomes more local
- long leaf text may still need chunking for best LLM behavior

These are acceptable tradeoffs for a first production-safe iteration.

## 14. Acceptance Criteria

This design is considered implemented successfully when:

1. KG extraction can be configured to process only body-text leaves.
2. Title extraction remains separate and functional.
3. Image/table/equation extraction remains unchanged.
4. Extracted node results still carry correct `node_idx` and `source_ids`.
5. The default path reduces heading/title noise without breaking existing graph construction.

## 15. Recommended Final Decision

Adopt the following default behavior:

- use **body-text terminal `TEXT` nodes** as the main entity-extraction source
- keep selection logic in the orchestration layer
- keep extractor internals stable in the first iteration
- preserve per-node provenance and perform document-level consolidation afterward

This gives BookRAG a cleaner and more controllable entity extraction path with minimal disruption to the current architecture.