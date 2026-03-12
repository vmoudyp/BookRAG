from collections import defaultdict
from typing import Any, List, Dict, Optional
import re

from Core.Index.Tree import TreeNode, NodeType
from Core.rag.base_rag import BaseRAG
from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.provider.rerank import TextRerankerProvider
from Core.provider.visual_sidecar import query_visual_sidecar
from Core.configs.rag.gbc_config import GBCRAGConfig
from Core.Index.GBCIndex import GBC
from Core.prompts.gbc_prompt import (
    LLM_EXPANSION_SELECT_PROMPT,
    QuestionEntity,
    QuestionEntityExtraction,
    QUESTION_ENT_PROMPT,
    QUESTION_ENTITY_TYPES,
    SecEXPSelection,
)
from Core.Index.Graph import Entity
from Core.rag.gbc_answer import AnswerAgent
from Core.rag.gbc_plan import TaskPlanner, PlanResult
from Core.rag.gbc_retrieval import Retriever

from Core.rag.gbc_utils import (
    GBCRAGContext,
    SubStep,
    filter_tree_nodes,
)
from Core.utils.ontology_utils import (
    find_best_graph_ontology_node,
    normalize_entity_name,
    normalize_entity_type,
)


import json
import networkx as nx

import logging

log = logging.getLogger(__name__)


_ROLE_QUERY_PATTERNS = re.compile(
    r"\b("
    r"who (is|was|are|were|holds?|held|serves? as|served as)|"
    r"what (role|position|title|post) (does|did|do)|"
    r"which (person|entity|organization|official|minister|director|head)|"
    r"role of|position of|title of|"
    r"minister|president|governor|director|chairman|secretary|ambassador|"
    r"ceo|chief executive|head of|commissioner"
    r")\b",
    re.IGNORECASE,
)

_EXPLICIT_CURRENT_TENURE_QUERY_PATTERNS = re.compile(
    r"\b(current|currently|incumbent|today|now|present-day|present)\b",
    re.IGNORECASE,
)

_EXPLICIT_FORMER_TENURE_QUERY_PATTERNS = re.compile(
    r"\b(former|previous|past|prior|formerly)\b|(?:\bex[- ])",
    re.IGNORECASE,
)

_CURRENT_TENURE_QUERY_PATTERNS = re.compile(
    r"\b(who is|who are|holds?|hold|serves? as|serving as)\b",
    re.IGNORECASE,
)

_FORMER_TENURE_QUERY_PATTERNS = re.compile(
    r"\b(who was|who were|held|served as|used to be)\b",
    re.IGNORECASE,
)


def _is_role_query_text(query: str) -> bool:
    """Return True when *query* appears to ask about a role or title."""
    return bool(_ROLE_QUERY_PATTERNS.search(query or ""))


def _infer_role_tenure_intent(query: str) -> Optional[str]:
    """Infer whether a role-oriented query is asking about current or former holders."""
    if not _is_role_query_text(query):
        return None
    if _EXPLICIT_FORMER_TENURE_QUERY_PATTERNS.search(query or ""):
        return "former"
    if _EXPLICIT_CURRENT_TENURE_QUERY_PATTERNS.search(query or ""):
        return "current"

    has_current = bool(_CURRENT_TENURE_QUERY_PATTERNS.search(query or ""))
    has_former = bool(_FORMER_TENURE_QUERY_PATTERNS.search(query or ""))
    if has_current and not has_former:
        return "current"
    if has_former and not has_current:
        return "former"
    return None


def _matches_role_tenure_intent(
    tenure_status: Optional[str],
    intent: Optional[str],
) -> bool:
    """Return True when a role assignment's tenure is compatible with the query intent."""
    if intent not in {"current", "former"}:
        return True

    normalized = (tenure_status or "").strip().lower()
    if intent == "current":
        if normalized in {"", "unknown"}:
            return True
        return normalized in {"current", "acting", "interim"}

    return normalized in {"former", "past", "previous"}


def _filter_role_evidence_by_tenure(
    role_evidence: List[Dict[str, Any]],
    intent: Optional[str],
) -> List[Dict[str, Any]]:
    """Filter structured role evidence by inferred current/former intent."""
    if not role_evidence or intent not in {"current", "former"}:
        return role_evidence
    return [
        item for item in role_evidence
        if _matches_role_tenure_intent(item.get("tenure_status"), intent)
    ]


class GBCRAG(BaseRAG):
    """
    GBC RAG (Graph-Based Contextual Retrieval Augmented Generation) class.
    This class is designed to handle the retrieval and generation of responses
    based on a graph-based context.
    """

    def __init__(
        self,
        llm: LLM,
        vlm: VLM,
        config: GBCRAGConfig,
        gbc_index: GBC,
        lang: str = "en",
    ):
        super().__init__(
            llm,
            name="GBC RAG",
            description="Graph-Based Contextual Retrieval Augmented Generation",
        )
        self.vlm = vlm
        self.cfg = config
        self.varient = self.cfg.varient
        if not gbc_index:
            raise ValueError("GBC index must be provided for GBCRAG.")
        self.gbc_index = gbc_index
        self.embedder = self.gbc_index.embedder if self.gbc_index else None
        self.reranker = TextRerankerProvider(
            model_name=self.cfg.reranker_config.model_name,
            max_length=self.cfg.reranker_config.max_length,
            device=self.cfg.reranker_config.device,
            backend=self.cfg.reranker_config.backend,
            api_base=self.cfg.reranker_config.api_base,
            api_key=self.cfg.reranker_config.api_key,
        )
        # GBC RAG config
        self.threshold_e = self.cfg.sim_threshold_e
        self.select_depth = self.cfg.select_depth
        self.max_retry = self.cfg.max_retry

        self.lang = lang or "en"

        # Agents
        self.planner = TaskPlanner(llm=self.llm)
        self.answer = AnswerAgent(llm=self.llm, vlm=self.vlm, lang=self.lang)
        self.retriever = Retriever(
            varient=self.varient,
            reranker=self.reranker,
            embedder=self.embedder,
            alpha=self.cfg.alpha,
            topk_ent=self.cfg.topk_ent,
            x_percentile=self.cfg.x_percentile,
            topk=self.cfg.topk,
        )

    def _get_entity_embed_text(self, entity: QuestionEntity) -> str:
        return f"Name: {entity.entity_name}\nType: {entity.entity_type}"

    def _entity_map(
        self, entities: List[QuestionEntity], force_one: bool = False
    ) -> Dict[str, List[str]]:
        """
        Maps entities to their corresponding IDs in the GBC index.
        Use vdb to find the entity in GBC index.
        """
        entities_str = [self._get_entity_embed_text(entity) for entity in entities]
        query_to_gbc_entity_map = defaultdict(list)
        res_list = []
        for ent_str in entities_str:
            query_res = self.gbc_index.entity_vdb.search(query_text=ent_str, top_k=2)
            if not query_res:
                continue
            min_distance = query_res[0]["distance"] if query_res else float("inf")
            metadata = query_res[0].get("metadata") or {}
            retrieve_name = metadata.get("entity_name")
            retrieve_type = metadata.get("entity_type")
            if not retrieve_name:
                continue
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                retrieve_name, retrieve_type
            )
            if min_distance < self.threshold_e:
                query_to_gbc_entity_map[ent_str].append(node_name)
                log.info(f"Entity '{ent_str}' mapped to GBC entity: {node_name}")
            else:
                res_list.append((ent_str, node_name, min_distance))

        if force_one and len(query_to_gbc_entity_map) == 0 and len(res_list) > 0:
            # force map the closest entity if no entity is mapped
            res_list = sorted(res_list, key=lambda x: x[2])
            ent_str, node_name, min_distance = res_list[0]
            query_to_gbc_entity_map[ent_str].append(node_name)
            log.info(f"Force map entity '{ent_str}' to GBC entity: {node_name}")

        return query_to_gbc_entity_map

    def _get_query_entity(self, query: str) -> Dict[str, List[str]]:
        """
        Get the entity mapping for the query.
        """

        # 1. retrieval relevent entities from the query
        retrieval_ents = self.gbc_index.entity_vdb.search(query_text=query, top_k=5)
        retrieval_node_names = set()
        retrieval_nodes = []
        for ent_info in retrieval_ents:
            metadata = ent_info.get("metadata") or {}
            ent_name = metadata.get("entity_name")
            ent_type = metadata.get("entity_type")
            if not ent_name:
                continue
            node_dict = {
                "entity_name": ent_name,
                "entity_type": ent_type,
            }
            node_name = self.gbc_index.GraphIndex.get_node_name_from_str(
                ent_name, ent_type
            )
            if node_name not in retrieval_node_names:
                retrieval_node_names.add(node_name)
                retrieval_nodes.append(node_dict)

        # 2. llm generate and select entities from the query
        prompt = QUESTION_ENT_PROMPT.format(
            input_text=query,
            entity_types=", ".join(QUESTION_ENTITY_TYPES),
            retrieved_entities=json.dumps(retrieval_nodes, ensure_ascii=False),
        )
        res_entities = []
        try:
            res: QuestionEntityExtraction = self.llm.get_json_completion(
                prompt, QuestionEntityExtraction
            )
            if res and res.entities:
                res_entities = res.entities
                entities_name = [entity.entity_name for entity in res_entities]
                log.info(f"Extracted entities: {entities_name}")
            else:
                log.info("No entities extracted from the query.")

        except Exception as e:
            log.error(f"Error during entity extraction: {e}")

        if len(res_entities) == 0:
            # use the retrieval entities if no entity is extracted by llm
            log.info("Use the question as the entity.")
            res_entities = [Entity(entity_name=query, entity_type="Question")]

        query_to_gbc_entity_map = defaultdict(list)
        remain_ents = []
        for res_ent in res_entities:
            res_ent.entity_name = normalize_entity_name(res_ent.entity_name)
            res_ent.entity_type = normalize_entity_type(res_ent.entity_type)
            ent_str = self._get_entity_embed_text(res_ent)

            ontology_cfg = getattr(self.gbc_index.config, "ontology", None)
            ontology_node_name = None
            if ontology_cfg and ontology_cfg.use_query_resolution:
                ontology_node_name = find_best_graph_ontology_node(
                    graph=self.gbc_index.GraphIndex,
                    entity_name=res_ent.entity_name,
                    entity_type=res_ent.entity_type,
                    threshold=ontology_cfg.mapping_threshold,
                )
            if ontology_node_name:
                query_to_gbc_entity_map[ent_str].append(ontology_node_name)
                log.info(
                    f"Entity '{ent_str}' mapped to ontology-backed GBC entity: {ontology_node_name}"
                )
                continue

            ent_node_name = self.gbc_index.GraphIndex.get_node_name_from_entity(res_ent)
            if ent_node_name in retrieval_node_names:
                query_to_gbc_entity_map[ent_str].append(ent_node_name)
                log.info(
                    f"Entity '{ent_node_name}' mapped to GBC entity: {ent_node_name}"
                )
            else:
                remain_ents.append(res_ent)

        should_force_one = len(query_to_gbc_entity_map) == 0
        if remain_ents:
            remain_map = self._entity_map(remain_ents, force_one=should_force_one)
            for k, v in remain_map.items():
                query_to_gbc_entity_map[k].extend(v)

        return query_to_gbc_entity_map

    def link_tree_node(self, entities_map: Dict[str, List[str]]) -> List[dict]:
        """
        Get the tree nodes for the given entities.
        """
        tree_node_cnt = defaultdict(list)
        all_map_nodenames = set()
        for ent_list in entities_map.values():
            for ent in ent_list:
                all_map_nodenames.add(ent)
        all_map_nodenames = list(all_map_nodenames)
        if not all_map_nodenames:
            log.warning("No entities found in the mapping.")
            return []

        for node_name in all_map_nodenames:
            tree_node_set = self.gbc_index.GraphIndex.node_name_to_tree_nodes(node_name)
            for node_id in tree_node_set:
                tree_node_cnt[node_id].append(node_name)

        tree_nodes = [
            {
                "index_id": node_id,
                "map_cnt": len(link_ents),
                "linked_entities": link_ents,
            }
            for node_id, link_ents in sorted(
                tree_node_cnt.items(), key=lambda x: len(x[1]), reverse=True
            )
        ]

        if not tree_nodes:
            log.warning("No tree nodes found for the given entities.")
            return []

        log.info(f"Retrieved {len(tree_nodes)} tree nodes based on entity mapping.")
        return tree_nodes

    def link_section(self, tree_nodes: List[dict]) -> Dict[int, List[str]]:
        """
        Get the linked section TreeNode ids from the tree nodes.
        given the tree nodes, get the linked section TreeNode ids (specific depth).
        return the Dict: section_id --> [linked_entity1, linked_entity2, ...]
        """
        sec_entity_map = defaultdict(list)
        for node in tree_nodes:
            node_idx = node["index_id"]
            ancestor = self.gbc_index.TreeIndex.get_ancestor_at_depth(
                node_idx, self.select_depth
            )
            ancestor_idx = ancestor.index_id if ancestor else None
            node_ents = node["linked_entities"]
            if ancestor_idx:
                sec_entity_map[ancestor_idx].extend(node_ents)

        for sec_id, val in sec_entity_map.items():
            sec_entity_map[sec_id] = list(set(val))

        log.info(
            f"Found {len(sec_entity_map)} linked sections at depth {self.select_depth}."
        )
        return sec_entity_map

    def prep_SecSel_prompt(
        self,
        query,
        link_nodes: List[TreeNode] = None,
        remain_nodes: List[TreeNode] = None,
        sec_entity_map: Optional[Dict[int, List[str]]] = None,
    ) -> str:
        """
        Prepare the prompt for section selection.
        This method should be implemented to prepare the prompt
        """

        def prep_nodes_json(
            nodes: List[TreeNode], sec_entity_map: Optional[Dict[int, List[str]]] = None
        ) -> str:
            node_infos = []
            for node in nodes:
                sec_idx = node.index_id
                section_title = node.meta_info.content
                sec_path = self.gbc_index.TreeIndex.get_path_from_root(sec_idx)
                title_path_obj = [node.meta_info.content for node in sec_path]
                sec_info = {
                    "id": sec_idx,
                    "title": section_title,
                    "path": title_path_obj,
                }
                if sec_entity_map and sec_idx in sec_entity_map:
                    entities_str = ", ".join(sec_entity_map[sec_idx])
                    sec_info["contained_entities"] = entities_str
                node_infos.append(sec_info)

            sec_info_str = json.dumps(node_infos, indent=2, ensure_ascii=False)
            return sec_info_str

        link_sec_str = (
            prep_nodes_json(link_nodes, sec_entity_map=sec_entity_map)
            if link_nodes
            else "[]"
        )
        remain_sec_str = (
            prep_nodes_json(remain_nodes, sec_entity_map=None) if remain_nodes else "[]"
        )
        query_prompt = LLM_EXPANSION_SELECT_PROMPT.format(
            user_question=query,
            primary_candidates_json=link_sec_str,
            remaining_sections_json=remain_sec_str,
        )

        return query_prompt

    def llm_section_selection(
        self,
        query: str,
        tree_nodes: List[dict],
        iter_context: Optional[SubStep] = None,
    ) -> None:
        """
        Use LLM to select the most relevant section based on the query and Section info.
        """
        sec_entity_map = self.link_section(tree_nodes)
        link_section_ids = list(sec_entity_map.keys())

        all_sections = self.gbc_index.TreeIndex.get_nodes_at_depth(self.select_depth)
        link_secs = [sec for sec in all_sections if sec.index_id in link_section_ids]
        remain_secs = [
            sec for sec in all_sections if sec.index_id not in link_section_ids
        ]
        iter_context.linked_section_ids = link_section_ids

        if len(remain_secs) == 0:
            log.info("No remaining sections to select from. Skipping LLM expansion.")
            iter_context.supplementary_ids = []
            iter_context.selected_explanation = (
                "No remaining sections for supplementary selection."
            )
            iter_context.retrieval_sec_ids = link_section_ids
            return

        query_prompt = self.prep_SecSel_prompt(
            query=query,
            link_nodes=link_secs,
            remain_nodes=remain_secs,
            sec_entity_map=sec_entity_map,
        )
        sel_ids = []
        explanation = "Error or no valid response from LLM during section expansion."

        remain_sec_ids_set = {sec.index_id for sec in remain_secs}
        try:
            res: SecEXPSelection = self.llm.get_json_completion(
                query_prompt, SecEXPSelection
            )
            if res:
                explanation = res.explanation
                if res.supplementary_ids:
                    # Validate the IDs returned by the LLM
                    for sup_id in res.supplementary_ids:
                        if sup_id in remain_sec_ids_set:
                            sel_ids.append(sup_id)
                        else:
                            log.warning(
                                f"LLM returned a supplementary ID {sup_id} which is not in the valid list of remaining sections. Ignoring it."
                            )

                    if sel_ids:
                        log.info(f"LLM selected {len(sel_ids)} supplementary sections.")
                    else:
                        log.info("LLM did not select any valid supplementary sections.")
                else:
                    log.info("LLM did not select any supplementary sections.")

        except Exception as e:
            log.error(f"Error occurred during section selection: {e}")

        iter_context.supplementary_ids = sel_ids
        iter_context.selected_explanation = explanation

        retrieval_sec_ids = list(set(link_section_ids + sel_ids))
        iter_context.retrieval_sec_ids = retrieval_sec_ids
        log.info(
            f"LLM selected {len(sel_ids)} supplementary sections, total {len(retrieval_sec_ids)} sections for retrieval."
        )

    def _process_retrieved_nodes(
        self, tree_data: List[Dict[str, Any]], iter_context: SubStep
    ) -> None:
        """Processes and categorizes retrieved nodes into the iteration context."""
        iter_context.retrieval_nodes = tree_data

        image_nodes = [node for node in tree_data if node["type"] == NodeType.IMAGE]
        text_nodes = [node for node in tree_data if node["type"] != NodeType.IMAGE]

        iter_context.iteration_image_nodes = image_nodes
        iter_context.iteration_text_nodes = text_nodes

    def _visual_sidecar_requested(self) -> bool:
        return bool(
            getattr(self.cfg, "visual_sidecar_query_enabled", False)
            or getattr(self.cfg, "visual_sidecar_fusion_enabled", False)
        )

    def _query_visual_sidecar_hits(
        self,
        subtree_nodes: List[TreeNode],
        sub_query: str,
    ) -> List[Dict[str, Any]]:
        if not self._visual_sidecar_requested():
            return []

        system_cfg = getattr(self.gbc_index, "config", None)
        sidecar_cfg = getattr(system_cfg, "visual_sidecar", None) if system_cfg else None
        save_path = getattr(system_cfg, "save_path", None) or getattr(self.gbc_index, "save_dir", None)
        if sidecar_cfg is None or not save_path:
            return []

        try:
            return query_visual_sidecar(
                save_path=save_path,
                sidecar_cfg=sidecar_cfg,
                query_text=sub_query,
                allowed_node_ids=[node.index_id for node in subtree_nodes],
                top_k=max(0, self.cfg.visual_sidecar_query_topk),
            )
        except Exception as exc:
            log.warning("Visual sidecar query failed for sub-query '%s': %s", sub_query, exc)
            return []

    def _build_visual_rerank_res(
        self,
        visual_hits: List[Dict[str, Any]],
    ) -> List[tuple[int, float]]:
        if not getattr(self.cfg, "visual_sidecar_fusion_enabled", False):
            return []

        fusion_weight = float(getattr(self.cfg, "visual_sidecar_fusion_weight", 1.0) or 0.0)
        if fusion_weight <= 0:
            return []

        normalized_hits: List[tuple[int, float]] = []
        seen_ids = set()
        for hit in visual_hits:
            node_id = hit.get("node_id")
            score = hit.get("score")
            try:
                node_id = int(node_id)
                score = float(score)
            except (TypeError, ValueError):
                continue

            if node_id in seen_ids or score <= 0:
                continue

            normalized_hits.append((node_id, score))
            seen_ids.add(node_id)

        if not normalized_hits:
            return []

        score_mode = getattr(self.cfg, "visual_sidecar_fusion_score_mode", "raw") or "raw"
        if score_mode == "max_norm":
            max_score = max(score for _, score in normalized_hits)
            calibrated_hits = [
                (node_id, (score / max_score) if max_score > 0 else 0.0)
                for node_id, score in normalized_hits
            ]
        elif score_mode == "rank":
            calibrated_hits = [
                (node_id, 1.0 / float(rank))
                for rank, (node_id, _) in enumerate(normalized_hits, start=1)
            ]
        else:
            calibrated_hits = normalized_hits

        min_score = float(getattr(self.cfg, "visual_sidecar_fusion_min_score", 0.0) or 0.0)
        visual_rerank_res = [
            (node_id, score * fusion_weight)
            for node_id, score in calibrated_hits
            if score >= min_score
        ]

        return visual_rerank_res

    def _augment_with_visual_sidecar(
        self,
        tree_node_ids: List[int],
        visual_hits: List[Dict[str, Any]],
    ) -> tuple[List[int], List[int]]:
        if not getattr(self.cfg, "visual_sidecar_query_enabled", False):
            return tree_node_ids, []

        if not visual_hits:
            return tree_node_ids, []

        augmented_ids: List[int] = []
        seen_ids = set()
        for node_id in tree_node_ids:
            if node_id in seen_ids:
                continue
            augmented_ids.append(node_id)
            seen_ids.add(node_id)

        supplementary_ids: List[int] = []
        for hit in visual_hits:
            node_id = hit.get("node_id")
            if node_id is None or node_id in seen_ids:
                continue
            augmented_ids.append(node_id)
            supplementary_ids.append(node_id)
            seen_ids.add(node_id)

        if supplementary_ids:
            log.info(
                "Visual sidecar query added %s node(s) to retrieval: %s",
                len(supplementary_ids),
                supplementary_ids,
            )

        return augmented_ids, supplementary_ids

    def get_GBC_info(self, iter_context: SubStep) -> None:
        """
        1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        2. Use Three Layer Reranker to select most relevant TreeNodes in the subtree.
            2.1 PPR to rank the most relevant TreeNodes in the subtree.
            2.2 Rerank with text reranker model.
            2.3 Rerank with Multimodal method.
            Then: Skyline algorithm to select the most relevant TreeNodes.
        3. Combine the connected TreeNodes and Subgraph info to form the final GBC data info.
        """

        # 1. Get subgraph: sel_sec_id --> subtree --> subgraph.
        # Get the subtree rooted at the selected section ID
        if self.varient == "wo_selector":
            log.info("Variant 'wo_selector' selected")
            subtree_nodes = self.gbc_index.TreeIndex.get_nodes(hasRoot=False)
        else:
            log.info(f"Using {self.varient} variant for retrieval.")
            retrieval_sec_ids = iter_context.retrieval_sec_ids
            subtree_nodes = self.gbc_index.TreeIndex.get_subtree_nodes(retrieval_sec_ids)

        subtree_ids = [node.index_id for node in subtree_nodes]

        subgraph: nx.Graph = self.gbc_index.GraphIndex.get_kg_subgraph(subtree_ids)

        start_ent_map = iter_context.gbc_entity_map

        visual_hits = self._query_visual_sidecar_hits(
            subtree_nodes,
            iter_context.sub_query,
        )
        visual_rerank_res = self._build_visual_rerank_res(visual_hits)

        tree_node_ids, res_entities = self.retriever.skyline_filter(
            iter_context.sub_query,
            subtree_nodes,
            subgraph,
            start_ent_map,
            visual_rerank_res=visual_rerank_res,
        )

        tree_node_ids, supplementary_ids = self._augment_with_visual_sidecar(
            tree_node_ids,
            visual_hits,
        )
        iter_context.supplementary_ids = supplementary_ids

        log.info(f"After skyline filtering, select {len(tree_node_ids)} TreeNodes")

        graph_data = self.gbc_index.GraphIndex.get_subgraph_data(res_entities)
        iter_context.iteration_graph_nodes = graph_data.get("nodes", [])

        tree_data = self.gbc_index.TreeIndex.get_nodes_data(tree_node_ids)
        self._process_retrieved_nodes(tree_data, iter_context)

        # Role-aware augmentation: if the sub-query looks role-oriented, inject structured
        # role evidence alongside graph data so the answer agent can reference it.
        if self._is_role_query(iter_context.sub_query):
            iter_context.role_evidence = self._query_role_context(
                iter_context.sub_query, res_entities
            )

    # ------------------------------------------------------------------ #
    #  Role-aware retrieval helpers                                        #
    # ------------------------------------------------------------------ #

    def _is_role_query(self, query: str) -> bool:
        """Return True if the query appears to be asking about a person's role or title."""
        return _is_role_query_text(query)

    def _query_role_context(
        self, sub_query: str, entity_node_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Return structured role evidence for entities relevant to *sub_query*.

        Strategy:
        1. Try a FalkorDB HAS_ROLE Cypher query when the graph is FalkorDB-backed.
        2. Fall back to reading ``role_assignments`` directly from the in-memory graph.

        Returns a list of dicts with keys:
            entity_name, entity_type, role_name, role_id, scope, tenure_status,
            review_status, start_date, end_date, confidence, evidence_text
        """
        evidence: List[Dict[str, Any]] = []
        tenure_intent = _infer_role_tenure_intent(sub_query)

        graph_index = self.gbc_index.GraphIndex

        # ── Strategy 1: FalkorDB ──────────────────────────────────────────
        if graph_index.use_falkordb and entity_node_names:
            try:
                fdb_g = graph_index._get_fdb_graph()
                # Build a small filter of entity node names for the Cypher query
                name_filter = ", ".join(
                    f"'{n.replace(chr(39), '')}'" for n in entity_node_names[:20]
                )
                cypher = (
                    f"MATCH (e:Entity)-[r:HAS_ROLE]->(role:RoleNode) "
                    f"WHERE e.node_name IN [{name_filter}] "
                    f"RETURN e.node_name, e.entity_name, e.entity_type, "
                    f"role.role_name, role.role_id, "
                    f"r.scope_entity_name, r.tenure_status, r.review_status, "
                    f"r.start_date, r.end_date, r.confidence"
                )
                result = fdb_g.query(cypher)
                for row in result.result_set:
                    (
                        node_name, ent_name, ent_type,
                        role_name, role_id,
                        scope, tenure, review,
                        start_d, end_d, conf,
                    ) = row
                    if review in ("rejected",):
                        continue
                    evidence.append({
                        "entity_name": ent_name,
                        "entity_type": ent_type,
                        "role_name": role_name,
                        "role_id": role_id or "",
                        "scope": scope or "",
                        "tenure_status": tenure or "",
                        "review_status": review or "",
                        "start_date": start_d or "",
                        "end_date": end_d or "",
                        "confidence": float(conf or 0.0),
                        "evidence_text": "",
                    })
                filtered = _filter_role_evidence_by_tenure(evidence, tenure_intent)
                log.info(
                    f"Role context: FalkorDB returned {len(filtered)} role record(s) "
                    f"for {len(entity_node_names)} entity node(s)"
                    f" (tenure_intent={tenure_intent or 'any'})."
                )
                return filtered
            except Exception as exc:
                log.warning(f"FalkorDB HAS_ROLE query failed, falling back to in-memory: {exc}")

        # ── Strategy 2: in-memory NetworkX graph ──────────────────────────
        for node_name in entity_node_names:
            try:
                entity = graph_index.get_entity_by_node_name(node_name)
            except KeyError:
                continue
            for ra in entity.role_assignments:
                if ra.review_status == "rejected":
                    continue
                # Pull first evidence text if available
                ev_text = ""
                if ra.evidence:
                    ev_text = ra.evidence[0].evidence_text or ""
                evidence.append({
                    "entity_name": entity.entity_name,
                    "entity_type": entity.entity_type,
                    "role_name": ra.role_name,
                    "role_id": ra.role_id or "",
                    "scope": ra.scope_entity_name or "",
                    "tenure_status": ra.tenure_status,
                    "review_status": ra.review_status,
                    "start_date": ra.start_date or "",
                    "end_date": ra.end_date or "",
                    "confidence": float(ra.confidence or 0.0),
                    "evidence_text": ev_text,
                })

        filtered = _filter_role_evidence_by_tenure(evidence, tenure_intent)
        log.info(
            f"Role context: in-memory graph returned {len(filtered)} role record(s) "
            f"for {len(entity_node_names)} entity node(s)"
            f" (tenure_intent={tenure_intent or 'any'})."
        )
        return filtered

    @staticmethod
    def _enrich_entities_with_roles(
        entity_nodes: List[Dict[str, Any]],
        role_evidence: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Attach a ``role_summary`` string to each entity dict that has matching role evidence.

        The enriched dicts are consumed by ``AnswerAgent.answer_simple_question`` which
        already renders ``entity_name`` and ``entity_type``; any additional key with prefix
        ``role_`` is passed through as supplementary context.
        """
        if not role_evidence:
            return entity_nodes

        # Build a lookup: lower-cased entity_name → list of role dicts
        role_by_name: Dict[str, List[Dict[str, Any]]] = {}
        for r in role_evidence:
            key = (r.get("entity_name") or "").lower()
            role_by_name.setdefault(key, []).append(r)

        enriched: List[Dict[str, Any]] = []
        for node in entity_nodes:
            node = dict(node)  # shallow copy – don't mutate caller's data
            ent_key = (node.get("entity_name") or "").lower()
            roles = role_by_name.get(ent_key, [])
            if roles:
                parts = []
                for r in roles:
                    role_desc = r.get("role_name", "")
                    scope = r.get("scope", "")
                    tenure = r.get("tenure_status", "")
                    start = r.get("start_date", "")
                    end = r.get("end_date", "")
                    ev_text = r.get("evidence_text", "")
                    line = role_desc
                    if scope:
                        line += f" ({scope})"
                    if tenure:
                        line += f" [{tenure}]"
                    if start or end:
                        line += f" {start}–{end}".strip("–").strip()
                    if ev_text:
                        line += f': "{ev_text}"'
                    parts.append(line)
                node["role_summary"] = "; ".join(parts)
            enriched.append(node)
        return enriched

    def _retrieve(
        self,
        query: str,
        iter_context: SubStep = None,
    ) -> None:
        """
        GBC retrieval following the steps:
        1. Extract entities from the query.
        2. Get the section nodes based on the entities.
        3. Use LLM to select the most relevant section based on the query and Section info.
        4. Use graph-based retrieval on the subgraph projected by the subtree (Select Section).

        iter_context: IterationStep, Iteration context for the current step.
        """

        query_to_gbc_entity_map = self._get_query_entity(query)
        iter_context.gbc_entity_map = query_to_gbc_entity_map

        tree_nodes = self.link_tree_node(query_to_gbc_entity_map)
        iter_context.linked_tree_nodes = tree_nodes

        # 3. Use LLM to select the most relevant section or supplementary sections
        if self.varient == "wo_selector":
            log.info("Variant 'wo_selector' selected: Skipping LLM section selection.")
            iter_context.retrieval_sec_ids = [self.gbc_index.TreeIndex.root_node.index_id]
        else:
            self.llm_section_selection(query, tree_nodes, iter_context)

        # 4. Graph-based retrieval on subgraph projected by the subtree (Select Section)
        self.get_GBC_info(iter_context)

    def process_analysis(self, context: GBCRAGContext, query_analysis: PlanResult):
        log.info(f"Query analysis type: {query_analysis.query_type}")

        if query_analysis.query_type == "simple":
            query = query_analysis.original_query
            current_step = SubStep(sub_query=query, sub_number=1)
            self._retrieve(query, current_step)

            entities = self._enrich_entities_with_roles(
                current_step.iteration_graph_nodes, current_step.role_evidence
            )
            final_answer, partial_answers = self.answer.answer_simple_question(
                query=query,
                retrieved_nodes=current_step.retrieval_nodes,
                entities=entities,
            )
            current_step.partial_answers = partial_answers
            current_step.generated_answer = final_answer

            context.iterations.append(current_step)
            context.final_answer = final_answer
        elif query_analysis.query_type == "complex":
            # 1. Separate retrieval tasks from the full plan
            retrieval_tasks = [
                sub_q
                for sub_q in query_analysis.sub_questions
                if sub_q.type == "retrieval"
            ]

            # 2. Execute each retrieval task and collect the results
            sub_question_results = []
            for i, task in enumerate(retrieval_tasks):
                sub_question = task.question
                current_step = SubStep(sub_query=sub_question, sub_number=i + 1)
                self._retrieve(sub_question, current_step)

                entities = self._enrich_entities_with_roles(
                    current_step.iteration_graph_nodes, current_step.role_evidence
                )
                sub_answer, partial_answers = self.answer.answer_simple_question(
                    query=sub_question,
                    retrieved_nodes=current_step.retrieval_nodes,
                    entities=entities,
                )
                current_step.partial_answers = partial_answers
                current_step.generated_answer = sub_answer
                context.iterations.append(current_step)

                sub_question_results.append(
                    {"question": sub_question, "answer": sub_answer}
                )
            final_answer = self.answer.answer_complex_question(
                original_query=query_analysis.original_query,
                sub_question_plan=query_analysis.sub_questions,  # Pass the full plan
                sub_question_results=sub_question_results,  # Pass the results of the retrieval steps
            )
            context.final_answer = final_answer

        elif query_analysis.query_type == "global":
            # Create a step for the global operation
            current_step = SubStep(
                sub_query=query_analysis.original_query, sub_number=1
            )

            # 1. Filter the tree nodes based on the plan's filters
            filtered_nodes: List[TreeNode] = filter_tree_nodes(
                self.gbc_index.TreeIndex, query_analysis.filters
            )
            current_step.retrieval_nodes = filtered_nodes

            filter_nodes_ids = [node.index_id for node in filtered_nodes]
            tree_data = self.gbc_index.TreeIndex.get_nodes_data(filter_nodes_ids)
            self._process_retrieved_nodes(tree_data, current_step)
            log.info(f"Global filter resulted in {len(filtered_nodes)} nodes.")

            operation = query_analysis.operation.upper()

            # 2. Perform the specified operation
            if operation == "COUNT":
                # Direct calculation, no LLM call needed for the final step
                count_result = len(filtered_nodes)
                # You can format this into a more natural sentence if desired
                final_answer = (
                    f"Based on my analysis of the document, I found {count_result} items"
                    f" that answer the question: '{query_analysis.original_query}'"
                )

                current_step.partial_answers = [
                    {"source": "Direct Count", "content": final_answer}
                ]
            else:  # For LIST, SUMMARIZE, ANALYZE
                # Call the dedicated global answer agent method
                final_answer, partials = self.answer.answer_global_question(
                    original_query=query_analysis.original_query,
                    operation=operation,
                    filtered_nodes=current_step.retrieval_nodes,
                )
                current_step.partial_answers = partials

            context.iterations.append(current_step)
            context.final_answer = final_answer
        else:
            log.warning(f"Unknown query type: {query_analysis.query_type}")
            context.final_answer = "I'm sorry, I cannot process this query."

    def _create_augmented_prompt(self, query: str) -> str:
        """Current GBC flow builds prompts via answer agents, so return the raw query."""
        return query

    def _run_query(self, query: str) -> GBCRAGContext:
        context = GBCRAGContext(query=query)

        if self.varient == "wo_plan":
            log.info("Variant 'wo_plan' selected: Skipping LLM planning.")
            query_analysis = PlanResult(
                query_type="simple",
                original_query=query,
            )
        else:
            query_analysis: PlanResult = self.planner.analyze(query)

        context.plan = query_analysis
        self.process_analysis(context, query_analysis)

        return context

    def answer_query(self, query: str) -> str:
        context = self._run_query(query)
        log.info(f"Final answer for query '{query}': {context.final_answer}")
        return context.final_answer

    def generation(self, query: str, query_output_dir: str):
        context = self._run_query(query)

        log.info(f"Final answer for query '{query}': {context.final_answer}")
        retrieval_ids = self._save_retrieval_res(context, query_output_dir)

        return context.final_answer, retrieval_ids

    def _save_retrieval_res(self, context: GBCRAGContext, query_output_dir: str):
        retrieval_ids = []

        # direct save the context to a json file
        retrieval_save_res = query_output_dir / "retrieval_res.json"
        context_dict = context.model_dump()
        with open(retrieval_save_res, "w", encoding="utf-8") as f:
            json.dump(context_dict, f, indent=2, ensure_ascii=False)
        log.info(f"Retrieval results saved to {retrieval_save_res}")

        # use the tree nodes as retrieval ids
        retrieval_ids = []
        for iter_step in context.iterations:
            text_nodes = iter_step.iteration_text_nodes
            if text_nodes:
                for node in text_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)
            image_nodes = iter_step.iteration_image_nodes
            if image_nodes:
                for node in image_nodes:
                    node_id = node.get("index_id")
                    if node_id is not None and node_id not in retrieval_ids:
                        retrieval_ids.append(node_id)

        retrieval_ids = sorted(retrieval_ids)

        return retrieval_ids

    def close(self):
        self.embedder.close()
        self.reranker.close()
        return super().close()
