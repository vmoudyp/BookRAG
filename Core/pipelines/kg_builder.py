from Core.Index.Tree import DocumentTree, NodeType
from Core.Index.Graph import Graph
from Core.pipelines.kg_extractor import KGExtractor
from Core.pipelines.kg_refiner import KGRefiner
from Core.configs.system_config import SystemConfig
from Core.utils.ontology_utils import align_entities_to_ontology

from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.TokenTracker import TokenTracker

import logging

log = logging.getLogger(__name__)


def _is_body_text_leaf(node) -> bool:
    if node.type != NodeType.TEXT or not node.is_leaf():
        return False

    meta_info = getattr(node, "meta_info", None)
    if meta_info is None:
        return False

    if getattr(meta_info, "source_role", None) != "body_text":
        return False

    content = getattr(meta_info, "content", None)
    return bool(str(content or "").strip())


def build_knowledge_graph(tree: DocumentTree, cfg: SystemConfig):
    """
    Build a knowledge graph from the given document tree.

    :param tree: DocumentTree object containing the document structure.
    :param graph_config: GraphConfig object containing configuration for the graph.
    :return: A tuple containing the KGExtractor and KGRefiner instances.
    """
    llm = LLM(cfg.llm)
    vlm = VLM(cfg.vlm) if cfg.graph.image_description_force else None

    # try load_the graph if constructed before
    # graph_path = os.path.join(cfg.save_path, Graph._DATA_FILE)
    # if os.path.exists(graph_path):
    #     log.info(f"Loading existing knowledge graph from {graph_path}...")
    #     graph_index = Graph.load_from_dir(cfg.save_path)
    #     return graph_index
    # else:
    #     log.info("No existing knowledge graph found. Creating a new one...")

    if cfg.graph.refine_type == "basic":
        variant = "basic"
    else:
        variant = None

    # Pass FalkorDB config only when tenant/doc IDs are set AND the FalkorDB host env
    # var is configured.  This mirrors the guard in api/services/indexing.py: when
    # BOOKRAG_FALKORDB_HOST is not set, cfg.falkordb may still be a plain dict loaded
    # from the YAML (Pydantic uses Any type), so we must not forward it as a
    # FalkorDBConfig object — doing so causes AttributeError at graph_name_for_doc().
    import os as _os
    _fdb_host = _os.getenv("BOOKRAG_FALKORDB_HOST", "")
    falkordb_cfg = cfg.falkordb if (cfg.tenant_id and cfg.doc_id and _fdb_host) else None
    graph_index = Graph(
        save_path=cfg.save_path,
        variant=variant,
        tenant_id=cfg.tenant_id,
        doc_id=cfg.doc_id,
        falkordb_cfg=falkordb_cfg,
        role_graph_materialization=cfg.graph.role_graph_materialization,
    )

    kg_extractor = KGExtractor(
        cfg_graph=cfg.graph, llm=llm, vlm=vlm, save_path=cfg.save_path
    )
    kg_refiner = KGRefiner(
        llm=llm,
        graph_config=cfg.graph,
        graph_index=graph_index,
        save_path=cfg.save_path,
    )

    kg_extract_res = []

    log.info("Batch processing is enabled for knowledge graph extraction.")
    text_extraction_scope = getattr(cfg.graph, "text_extraction_scope", "legacy")
    batch_title_nodes = []
    batch_title_paths = []
    batch_sibling_nodes = []

    if text_extraction_scope == "body_text_leaves":
        scoped_text_nodes = []
        skipped_text_nodes = 0
        skipped_non_text_nodes = 0

        for node in tree.nodes:
            if node == tree.root_node:
                continue
            if node.type == NodeType.TITLE:
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                batch_title_nodes.append(node)
                batch_title_paths.append(title_path)
                batch_sibling_nodes.append(sibling_nodes)
            elif _is_body_text_leaf(node):
                scoped_text_nodes.append(node)
            elif node.type == NodeType.TEXT:
                skipped_text_nodes += 1
            else:
                skipped_non_text_nodes += 1

        log.info(
            "Tree-aware KG plan: %s title nodes, %s body-text leaf nodes selected, %s text nodes skipped, %s non-text nodes skipped under scope '%s'.",
            len(batch_title_nodes),
            len(scoped_text_nodes),
            skipped_text_nodes,
            skipped_non_text_nodes,
            text_extraction_scope,
        )
    else:
        scoped_text_nodes = []
        text_internal_nodes = []
        non_text_nodes = []
        for node in tree.nodes:
            if node == tree.root_node:
                continue
            if node.type == NodeType.TITLE:
                title_path = tree.get_path_from_root(node.index_id)
                sibling_nodes = tree.get_sibling_nodes(node.index_id)
                batch_title_nodes.append(node)
                batch_title_paths.append(title_path)
                batch_sibling_nodes.append(sibling_nodes)
            elif node.type == NodeType.TEXT:
                if node.is_leaf():
                    scoped_text_nodes.append(node)
                else:
                    text_internal_nodes.append(node)
            else:
                non_text_nodes.append(node)

        log.info(
            "Tree-aware KG plan: %s title nodes, %s text leaf nodes, %s text internal nodes, %s non-text nodes.",
            len(batch_title_nodes),
            len(scoped_text_nodes),
            len(text_internal_nodes),
            len(non_text_nodes),
        )

    if batch_title_nodes:
        log.info("Processing title nodes with tree-first BERT extraction...")
        res_dict = kg_extractor.batch_extract_titles(
            nodes=batch_title_nodes,
            title_paths=batch_title_paths,
            sibling_nodes_list=batch_sibling_nodes,
        )
        kg_extract_res.extend(res_dict)

    if scoped_text_nodes:
        if text_extraction_scope == "body_text_leaves":
            log.info("Processing body-text leaf nodes with BERT extraction...")
        else:
            log.info("Processing text leaf nodes with BERT extraction...")
        res_dict = kg_extractor.batch_extract_kg(nodes=scoped_text_nodes)
        kg_extract_res.extend(res_dict)

    if text_extraction_scope != "body_text_leaves":
        if text_internal_nodes:
            log.info("Processing non-leaf text nodes with BERT extraction...")
            res_dict = kg_extractor.batch_extract_kg(nodes=text_internal_nodes)
            kg_extract_res.extend(res_dict)

        if non_text_nodes:
            log.info("Processing non-text nodes with the structured/visual extractor...")
            res_dict = kg_extractor.batch_extract_kg(nodes=non_text_nodes)
            kg_extract_res.extend(res_dict)

    kg_extract_res.sort(key=lambda x: x.get("node_idx", -1))

    log.info("Knowledge graph extraction completed.")
    log.info(f"Extracted {len(kg_extract_res)} nodes from the document tree.")

    token_tracker = TokenTracker.get_instance()
    kg_extraction_cost = token_tracker.record_stage("kg_extraction")
    log.info(f"Knowledge graph extraction cost: {kg_extraction_cost}")

    for res in kg_extract_res:
        entities, relationships = align_entities_to_ontology(
            entities=res.get("entities", []),
            relationships=res.get("relations", []),
            ontology_cfg=cfg.ontology,
        )
        if cfg.graph.refine_type == "basic":
            log.info("Using basic KG refinement.")
            kg_refiner.basic_kg_refiner(
                entities=entities,
                relationships=relationships,
                source_id=res.get("node_idx", -1),
            )
        elif cfg.graph.refine_type == "advanced":
            kg_refiner.advanced_kg_refiner(
                entities=entities,
                relationships=relationships,
                source_id=res.get("node_idx", -1),
            )

    kg_refiner.refine_entities()
    kg_refiner.refine_relation()

    log.info("Knowledge graph refinement completed.")
    kg_refinement_cost = token_tracker.record_stage("kg_refinement")
    log.info(f"Knowledge graph refinement cost: {kg_refinement_cost}")

    kg_refiner.close()

    return graph_index


if __name__ == "__main__":
    # We test the knowledge graph builder here
    from Core.configs.system_config import load_system_config

    cfg = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/default.yaml")

    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(cfg.save_path))

    token_tracker = TokenTracker.get_instance()
    token_tracker.reset()

    # Build the knowledge graph
    graph_index = build_knowledge_graph(tree_index, cfg)
    graph_index.save_graph()
    print("Knowledge graph built successfully.")
