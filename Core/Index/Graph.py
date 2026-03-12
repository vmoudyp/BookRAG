import networkx as nx
from networkx.readwrite import json_graph
import os
from collections import defaultdict
from typing import Iterable, Union, Set, List, Optional, TYPE_CHECKING  # noqa: F401
from pydantic import BaseModel, Field
import json

import logging

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from Core.configs.falkordb_config import FalkorDBConfig


class RoleEvidence(BaseModel):
    """A single piece of evidence supporting a domain role assignment."""

    source_id: int  # Tree/source node that contains this evidence
    observed_role_text: str  # Literal wording from the document (e.g. "Korsek")
    evidence_text: Optional[str] = None  # Short excerpt supporting the role
    language: Optional[str] = None  # Language code, e.g. "id", "en"
    normalization_method: str = "unresolved"  # alias/abbreviation/fuzzy/embedding/llm/manual/unresolved
    confidence: Optional[float] = None  # Confidence for this evidence item
    provenance: Optional[dict] = None  # Extra metadata for extraction/debugging


class RoleAssignment(BaseModel):
    """A structured domain-level role assertion attached to an entity.

    Distinct from:
    - auth ``role`` (admin/user)
    - ``source_role`` (title/body_text/image/table in document tree)
    - ``entity_role`` (canonical/provisional lifecycle state)
    """

    assignment_id: str  # Stable UUID for this assignment
    role_name: str  # Canonical role label, e.g. "Koordinator Sektor"
    role_id: Optional[str] = None  # Optional stable vocabulary identifier
    scope_entity_name: Optional[str] = None  # Context entity, e.g. "Indonesia"
    scope_entity_type: Optional[str] = None  # Type of scope entity, e.g. "COUNTRY"
    scope_entity_id: Optional[str] = None  # Optional canonical ID for scope entity
    start_date: Optional[str] = None  # Partial ISO-compatible: YYYY, YYYY-MM, or YYYY-MM-DD
    end_date: Optional[str] = None  # Same format as start_date
    tenure_status: str = "unknown"  # current/former/acting/interim/candidate/unknown
    origin: str = "extracted"  # extracted / manual
    review_status: str = "suggested"  # suggested/confirmed/disputed/rejected
    confidence: Optional[float] = None  # Overall assignment confidence
    normalization_status: str = "unresolved"  # matched/ambiguous/unresolved/manual_override
    normalization_confidence: Optional[float] = None  # Confidence of role-name normalization
    source_ids: List[int] = Field(default_factory=list)  # Supporting tree node IDs
    evidence: List[RoleEvidence] = Field(default_factory=list)  # Evidence items
    provenance: Optional[dict] = None  # Free-form trace metadata
    created_by: Optional[str] = None  # Audit: who created this assignment
    updated_by: Optional[str] = None  # Audit: who last updated it
    created_at: Optional[str] = None  # Audit: ISO timestamp of creation
    updated_at: Optional[str] = None  # Audit: ISO timestamp of last update


class Entity(BaseModel):
    entity_name: str  # Primary key for entity
    entity_type: str = Field(default="")  # Entity type
    description: str = Field(default="")  # The description of this entity
    entity_id: str = Field(default="")  # Stable ontology/canonical identifier
    canonical_id: str = Field(default="")  # Points to canonical ontology identifier
    entity_role: str = Field(default="provisional")  # canonical / provisional
    aliases: List[str] = Field(default_factory=list)  # Known aliases for resolution
    mapping_confidence: float = Field(default=0.0)  # Ontology mapping confidence
    ontology_source: str = Field(default="")  # Source of ontology metadata
    source_ids: Set[int] = Field(
        default_factory=set
    )  # Set of source IDs from which this entity is derived
    role_assignments: List[RoleAssignment] = Field(
        default_factory=list
    )  # Domain-level role assertions (e.g. President, CEO)

    def __hash__(self):
        """
        Calculates the hash based on the composite key of entity_name and entity_type.
        """
        return hash((self.entity_name, self.entity_type))

    def __eq__(self, other):
        """
        Defines two Entity objects as equal if their entity_name and entity_type match.
        """
        if isinstance(other, Entity):
            return (self.entity_name, self.entity_type) == (
                other.entity_name,
                other.entity_type,
            )
        return False

    def to_vdb_metadata(self) -> dict:
        aliases = list(dict.fromkeys(self.aliases))
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type,
            "description": self.description,
            "entity_id": self.entity_id,
            "canonical_id": self.canonical_id,
            "entity_role": self.entity_role,
            "mapping_confidence": float(self.mapping_confidence or 0.0),
            "ontology_source": self.ontology_source,
            "aliases_json": json.dumps(aliases, ensure_ascii=False),
            "role_assignments_json": json.dumps(
                [ra.model_dump() for ra in self.role_assignments], ensure_ascii=False
            ),
        }


class Relationship(BaseModel):
    src_entity_name: str  # Name of the entity on the left side of the edge
    tgt_entity_name: str  # Name of the entity on the right side of the edge
    relation_name: str = Field(default="")  # Name of the relation
    weight: float = Field(
        default=0.0
    )  # Weight of the edge, used in GraphRAG and LightRAG
    description: str = Field(
        default=""
    )  # Description of the edge, used in GraphRAG and LightRAG
    source_ids: Set[int] = Field(
        default_factory=set
    )  # Set of source IDs from which this edge is derived


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


class Graph:
    _DATA_FILE = "graph_data.json"  # index data file
    _BASE_FILENAME = "graph_data"

    def __init__(
        self,
        save_path: str = None,
        variant: str = None,
        tenant_id: str = None,
        doc_id: str = None,
        falkordb_cfg=None,  # Optional[FalkorDBConfig]
        role_graph_materialization: bool = False,
    ):
        self.kg = nx.Graph()
        # 节点名采用 "entity_name (entity_type)"，确保唯一性
        self.tree2kg = defaultdict(set)  # Maps tree nodes id (int) to graph entities
        self.save_dir = save_path
        self.variant = variant

        # Multi-tenant FalkorDB support
        self.tenant_id = tenant_id
        self.doc_id = doc_id
        self.falkordb_cfg = falkordb_cfg
        self.use_falkordb = falkordb_cfg is not None and tenant_id is not None and doc_id is not None
        self._fdb_graph = None  # lazy FalkorDB graph handle
        self._fdb_graph_name: Optional[str] = None
        if self.use_falkordb:
            self._fdb_graph_name = falkordb_cfg.graph_name_for_doc(tenant_id, doc_id)

        # Phase 2: promote role_assignments to first-class graph edges when enabled
        self.role_graph_materialization = role_graph_materialization

        # dynamic filename based on variant
        self.data_filename = self._get_filename(variant)

    @classmethod
    def _get_filename(cls, variant: str = None) -> str:
        """内部辅助函数：根据 variant 生成对应的 json 文件名"""
        if variant == "basic":
            log.info("Using 'basic' variant for graph filename.")
            return f"{cls._BASE_FILENAME}_{variant}.json"
        return f"{cls._BASE_FILENAME}.json"

    def get_all_nodes(self) -> Set[str]:
        """Return all node names (entity_name (entity_type)) in the knowledge graph."""
        return set(self.kg.nodes)

    def _debug_check_add_node(self, node_name: str) -> None:
        """Debugging helper to check if a node can be added."""
        if node_name in self.kg.nodes:
            log.warning(
                f"Warning: Node '{node_name}' already exists in the knowledge graph."
            )
            print("warning here")

    def get_node_name_from_entity(self, entity: Entity) -> str:
        """Generate a node name from an Entity object."""
        return self.get_node_name_from_str(entity.entity_name, entity.entity_type)

    def get_node_name_from_str(self, entity_name: str, entity_type: str) -> str:
        return f"Name: {entity_name}\nType: {entity_type}"

    def add_kg_node(self, entity: Entity) -> None:
        """Add an entity/node to the KG with all its attributes."""
        node_name = self.get_node_name_from_entity(entity)

        self.kg.add_node(node_name, **entity.model_dump())

    def add_kg_edge(self, rel: Relationship, src_type: str, tgt_type: str) -> None:
        """Add a relation/edge between two KG entities with all its attributes."""
        src_node_name = self.get_node_name_from_str(rel.src_entity_name, src_type)
        tgt_node_name = self.get_node_name_from_str(rel.tgt_entity_name, tgt_type)
        if src_node_name not in self.kg.nodes:
            raise KeyError(
                f"Source node '{src_node_name}' not found in knowledge graph."
            )
        if tgt_node_name not in self.kg.nodes:
            raise KeyError(
                f"Target node '{tgt_node_name}' not found in knowledge graph."
            )
        # Add the edge with all attributes from the Relationship model
        self.kg.add_edge(src_node_name, tgt_node_name, **rel.model_dump())

    def link(self, tree_node_id: int, entity_name: str, entity_type: str = "") -> None:
        """Create a bidirectional mapping between a tree node and a KG node."""
        node_name = self.get_node_name_from_str(entity_name, entity_type)
        if node_name not in self.kg:
            raise KeyError(f"KG node '{node_name}' not found in knowledge graph.")
        self.tree2kg[tree_node_id].add(node_name)

    def add_and_link(
        self,
        tree_node_id: int,
        entities: Union[Entity, List[Entity]],
    ) -> None:
        """Add one or more Entity nodes and link to tree node."""
        if isinstance(entities, Entity):
            entities = [entities]
        for entity in entities:
            node_name = self.get_node_name_from_entity(entity)
            if node_name not in self.kg:
                self.add_kg_node(entity)
            self.link(tree_node_id, entity.entity_name, entity.entity_type)

    def _rewrite_edge_entity_names(
        self, edge_data: Optional[dict], old_entity_name: str, new_entity_name: str
    ) -> dict:
        updated_edge_data = dict(edge_data or {})
        if updated_edge_data.get("src_entity_name") == old_entity_name:
            updated_edge_data["src_entity_name"] = new_entity_name
        if updated_edge_data.get("tgt_entity_name") == old_entity_name:
            updated_edge_data["tgt_entity_name"] = new_entity_name
        return updated_edge_data

    def update_entity(
        self, old_entity_name: str, old_entity_type: str, new_entity: Entity
    ) -> None:
        """
        if new entity already exists, it will be updated with new attributes.
        Args:
            old_entity_name (str): 需要被更新的实体节点名称。
            old_entity_type (str): 需要被更新的实体类型。
            new_entity (Entity): 新的实体对象。
        Raises:
            KeyError: 如果实体不存在。
        """
        old_node_name = self.get_node_name_from_str(old_entity_name, old_entity_type)
        new_node_name = self.get_node_name_from_entity(new_entity)
        if old_node_name not in self.kg:
            raise KeyError(f"Entity '{old_node_name}' not found in knowledge graph.")
        new_source_ids = new_entity.source_ids
        if new_node_name != old_node_name:
            # 1. Add new node and copy all edges
            self.kg.add_node(new_node_name, **new_entity.model_dump())
            for neighbor in self.kg.neighbors(old_node_name):
                edge_data = self.kg.get_edge_data(old_node_name, neighbor)
                self.kg.add_edge(
                    new_node_name,
                    neighbor,
                    **self._rewrite_edge_entity_names(
                        edge_data=edge_data,
                        old_entity_name=old_entity_name,
                        new_entity_name=new_entity.entity_name,
                    ),
                )
            # 2.1 update tree2kg
            for tree_id in new_source_ids:
                # If the old node is in the tree2kg, remove the old name
                self.tree2kg[tree_id].discard(old_node_name)
                self.tree2kg[tree_id].add(new_node_name)

            # 3. remove old node
            self.kg.remove_node(old_node_name)
        else:
            # only update attributes
            self.kg.nodes[old_node_name].update(new_entity.model_dump())
            # update tree2kg
            for tree_id in new_source_ids:
                self.tree2kg[tree_id].add(new_node_name)

    def get_entity(self, entity_name: str, entity_type: str = "") -> Entity:
        """
        Retrieve an entity from the knowledge graph by its name and type.
        Args:
            entity_name (str): The name of the entity to retrieve.
            entity_type (str): The type of the entity to retrieve.
        Returns:
            Entity: The entity object with all its attributes.
        Raises:
            KeyError: If the entity does not exist in the graph.
        """
        node_name = self.get_node_name_from_str(
            entity_name=entity_name, entity_type=entity_type
        )
        if node_name not in self.kg.nodes:
            raise KeyError(f"Entity '{node_name}' not found in knowledge graph.")
        return Entity(**self.kg.nodes[node_name])

    def get_entity_by_node_name(self, node_name: str) -> Entity:
        """
        Retrieve an entity from the knowledge graph by its node name.
        Args:
            node_name (str): The name of the node to retrieve.
        Returns:
            Entity: The entity object with all its attributes.
        Raises:
            KeyError: If the node does not exist in the graph.
        """
        if node_name not in self.kg.nodes:
            raise KeyError(f"Node '{node_name}' not found in knowledge graph.")
        return Entity(**self.kg.nodes[node_name])

    def get_subgraph_data(self, entities: List[str]) -> dict:
        """Return lightweight node data for the subgraph induced by `entities`."""
        subgraph = self.kg.subgraph(entities)
        data = {"nodes": []}
        for node in subgraph.nodes(data=True):
            node_data = {
                "entity_name": node[1]["entity_name"],
                "entity_type": node[1]["entity_type"],
            }
            data["nodes"].append(node_data)
        return data

    def entities_to_tree_nodes(self, entities: List[Entity]) -> List[int]:
        """
        Given KG node names, return all tree node IDs that link to them.
        """
        result = set()
        for ent in entities:
            result.update(ent.source_ids)
        return sorted(result)

    def entity_to_tree_nodes(self, ent: Entity) -> List[int]:
        """
        Given an Entity object, return all tree node IDs that link to it.
        """
        return sorted(ent.source_ids)

    def node_name_to_tree_nodes(self, node_name: str) -> List[int]:
        """
        Given a node name (entity_name (entity_type)), return all tree node IDs that link to it.
        """
        ent = self.get_entity_by_node_name(node_name)
        return sorted(ent.source_ids)

    def remove_self_loops(self) -> int:
        """
        Returns:
        """
        nodes_with_selfloops = list(nx.nodes_with_selfloops(self.kg))

        if not nodes_with_selfloops:
            log.info("No self-loops found in the graph.")
            return 0

        self_loop_edges = [(node, node) for node in nodes_with_selfloops]

        num_removed = len(self_loop_edges)
        log.info(f"Found {num_removed} self-loops. Removing them...")
        self.kg.remove_edges_from(self_loop_edges)
        log.info("All self-loops have been removed.")

    # ------------------------------------------------------------------ #
    #  FalkorDB helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_fdb_graph(self):
        """Lazy-initialise and return the FalkorDB graph handle."""
        if self._fdb_graph is not None:
            return self._fdb_graph
        try:
            from falkordb import FalkorDB
            cfg = self.falkordb_cfg
            conn_kwargs = {"host": cfg.host, "port": cfg.port}
            if cfg.username:
                conn_kwargs["username"] = cfg.username
            if cfg.password:
                conn_kwargs["password"] = cfg.password
            client = FalkorDB(**conn_kwargs)
            self._fdb_graph = client.select_graph(self._fdb_graph_name)
            log.info(f"Connected to FalkorDB graph '{self._fdb_graph_name}'")
        except Exception as e:
            log.error(f"Failed to connect to FalkorDB: {e}")
            raise
        return self._fdb_graph

    def _save_to_falkordb(self) -> None:
        """Persist the in-memory NetworkX graph to FalkorDB."""
        g = self._get_fdb_graph()

        def _esc(value) -> str:
            return str(value or "").replace("\\", "\\\\").replace("'", "\\'")

        # Clear existing data for idempotent saves
        try:
            g.query("MATCH (n) DETACH DELETE n")
        except Exception:
            pass

        # Write nodes
        for node_name, data in self.kg.nodes(data=True):
            source_ids_list = list(data.get("source_ids", set()))
            desc = _esc(data.get("description", ""))
            ename = _esc(data.get("entity_name", ""))
            etype = _esc(data.get("entity_type", ""))
            entity_id = _esc(data.get("entity_id", ""))
            canonical_id = _esc(data.get("canonical_id", ""))
            entity_role = _esc(data.get("entity_role", "provisional"))
            ontology_source = _esc(data.get("ontology_source", ""))
            aliases_json = _esc(json.dumps(data.get("aliases", []), ensure_ascii=False))
            mapping_confidence = float(data.get("mapping_confidence", 0.0) or 0.0)
            raw_roles = data.get("role_assignments", [])
            role_assignments_json = _esc(
                json.dumps(
                    [ra.model_dump() if isinstance(ra, RoleAssignment) else ra for ra in raw_roles],
                    ensure_ascii=False,
                )
            )
            nname = _esc(node_name)
            cypher = (
                f"CREATE (n:Entity {{"
                f"node_name: '{nname}', "
                f"entity_name: '{ename}', "
                f"entity_type: '{etype}', "
                f"entity_id: '{entity_id}', "
                f"canonical_id: '{canonical_id}', "
                f"entity_role: '{entity_role}', "
                f"aliases_json: '{aliases_json}', "
                f"mapping_confidence: {mapping_confidence}, "
                f"ontology_source: '{ontology_source}', "
                f"description: '{desc}', "
                f"role_assignments_json: '{role_assignments_json}', "
                f"source_ids: {source_ids_list}"
                f"}})"
            )
            g.query(cypher)

        # Write edges
        for src, tgt, data in self.kg.edges(data=True):
            rel_name = data.get("relation_name", "").replace("'", "\\'")
            weight = float(data.get("weight", 0.0))
            desc = data.get("description", "").replace("'", "\\'")
            src_ids = list(data.get("source_ids", set()))
            src_q = src.replace("\\", "\\\\").replace("'", "\\'")
            tgt_q = tgt.replace("\\", "\\\\").replace("'", "\\'")
            cypher = (
                f"MATCH (a:Entity {{node_name: '{src_q}'}}), "
                f"(b:Entity {{node_name: '{tgt_q}'}}) "
                f"CREATE (a)-[:RELATION {{"
                f"relation_name: '{rel_name}', "
                f"weight: {weight}, "
                f"description: '{desc}', "
                f"source_ids: {src_ids}"
                f"}}]->(b)"
            )
            g.query(cypher)

        # Phase 2: materialize role_assignments as first-class graph edges
        if self.role_graph_materialization:
            self._materialize_role_edges(g, _esc)

        log.info(f"Saved graph to FalkorDB '{self._fdb_graph_name}': "
                 f"{self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges.")

    def _materialize_role_edges(self, g, _esc) -> None:
        """Write :RoleNode nodes and (Entity)-[:HAS_ROLE]->(RoleNode) edges to FalkorDB.

        Called from ``_save_to_falkordb`` when ``role_graph_materialization`` is True.
        Uses MERGE on role_id (if present) or role_name to avoid duplicate Role nodes.
        """
        role_node_ids: set = set()

        for node_name, data in self.kg.nodes(data=True):
            raw_roles = data.get("role_assignments", [])
            if not raw_roles:
                continue

            for ra in raw_roles:
                # Accept both RoleAssignment objects and plain dicts (after JSON round-trip)
                if isinstance(ra, RoleAssignment):
                    ra_dict = ra.model_dump()
                else:
                    ra_dict = ra

                assignment_id = _esc(ra_dict.get("assignment_id", ""))
                role_name = _esc(ra_dict.get("role_name", ""))
                role_id = _esc(ra_dict.get("role_id") or "")
                scope = _esc(ra_dict.get("scope_entity_name") or "")
                tenure = _esc(ra_dict.get("tenure_status", "unknown"))
                review = _esc(ra_dict.get("review_status", "suggested"))
                origin = _esc(ra_dict.get("origin", "extracted"))
                confidence = float(ra_dict.get("confidence") or 0.0)
                start_date = _esc(ra_dict.get("start_date") or "")
                end_date = _esc(ra_dict.get("end_date") or "")

                # Stable identity key for Role node: prefer role_id, fall back to role_name
                role_node_key = role_id if role_id else role_name
                if role_node_key not in role_node_ids:
                    cypher_role = (
                        f"MERGE (r:RoleNode {{role_key: '{role_node_key}'}}) "
                        f"ON CREATE SET r.role_name = '{role_name}', r.role_id = '{role_id}'"
                    )
                    try:
                        g.query(cypher_role)
                        role_node_ids.add(role_node_key)
                    except Exception as exc:
                        log.warning(f"Failed to MERGE RoleNode '{role_node_key}': {exc}")
                        continue

                nname_esc = _esc(node_name)
                cypher_edge = (
                    f"MATCH (e:Entity {{node_name: '{nname_esc}'}}), "
                    f"(r:RoleNode {{role_key: '{role_node_key}'}}) "
                    f"CREATE (e)-[:HAS_ROLE {{"
                    f"assignment_id: '{assignment_id}', "
                    f"tenure_status: '{tenure}', "
                    f"review_status: '{review}', "
                    f"origin: '{origin}', "
                    f"scope_entity_name: '{scope}', "
                    f"start_date: '{start_date}', "
                    f"end_date: '{end_date}', "
                    f"confidence: {confidence}"
                    f"}}]->(r)"
                )
                try:
                    g.query(cypher_edge)
                except Exception as exc:
                    log.warning(
                        f"Failed to create HAS_ROLE edge for entity '{node_name}' -> '{role_node_key}': {exc}"
                    )

        log.info(
            f"Role materialization complete: {len(role_node_ids)} unique RoleNode(s) written."
        )

    def _load_from_falkordb(self) -> None:
        """Load graph data from FalkorDB into in-memory NetworkX graph."""
        g = self._get_fdb_graph()
        result = g.query("MATCH (n:Entity) RETURN n")
        for record in result.result_set:
            node = record[0]
            props = node.properties
            source_ids = set(props.get("source_ids", []))
            node_name = props["node_name"]
            aliases_json = props.get("aliases_json", "[]")
            try:
                aliases = json.loads(aliases_json) if isinstance(aliases_json, str) else list(aliases_json or [])
            except json.JSONDecodeError:
                aliases = []
            raw_roles_json = props.get("role_assignments_json", "[]")
            try:
                raw_roles = json.loads(raw_roles_json) if isinstance(raw_roles_json, str) else list(raw_roles_json or [])
                role_assignments = [RoleAssignment(**r) for r in raw_roles]
            except Exception:
                role_assignments = []
            self.kg.add_node(node_name,
                             entity_name=props.get("entity_name", ""),
                             entity_type=props.get("entity_type", ""),
                             entity_id=props.get("entity_id", ""),
                             canonical_id=props.get("canonical_id", ""),
                             entity_role=props.get("entity_role", "provisional"),
                             aliases=aliases,
                             mapping_confidence=float(props.get("mapping_confidence", 0.0) or 0.0),
                             ontology_source=props.get("ontology_source", ""),
                             description=props.get("description", ""),
                             source_ids=source_ids,
                             role_assignments=role_assignments)
            for tid in source_ids:
                self.tree2kg[int(tid)].add(node_name)

        edge_result = g.query(
            "MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
            "RETURN a.node_name, b.node_name, r.relation_name, r.weight, r.description, r.source_ids"
        )
        for rec in edge_result.result_set:
            src_name, tgt_name, rel_name, weight, desc, src_ids = rec
            self.kg.add_edge(
                src_name, tgt_name,
                src_entity_name=self.kg.nodes[src_name].get("entity_name", ""),
                tgt_entity_name=self.kg.nodes[tgt_name].get("entity_name", ""),
                relation_name=rel_name or "",
                weight=float(weight or 0.0),
                description=desc or "",
                source_ids=set(src_ids or []),
            )
        log.info(f"Loaded graph from FalkorDB '{self._fdb_graph_name}': "
                 f"{self.kg.number_of_nodes()} nodes, {self.kg.number_of_edges()} edges.")

    def _get_fdb_subgraph(self, tree_node_ids: Iterable[int]) -> nx.Graph:
        """Query FalkorDB for the subgraph linked to given tree node IDs."""
        # Collect node names from tree2kg (loaded at init)
        kg_nodes = set().union(*(self.tree2kg.get(tid, set()) for tid in tree_node_ids))
        if not kg_nodes:
            return nx.Graph()

        g = self._get_fdb_graph()
        node_list = [n.replace("'", "\\'") for n in kg_nodes]
        node_filter = "['" + "', '".join(node_list) + "']"
        result = g.query(
            f"MATCH (n:Entity) WHERE n.node_name IN {node_filter} RETURN n"
        )
        subgraph = nx.Graph()
        for rec in result.result_set:
            node = rec[0]
            props = node.properties
            node_name = props["node_name"]
            aliases_json = props.get("aliases_json", "[]")
            try:
                aliases = json.loads(aliases_json) if isinstance(aliases_json, str) else list(aliases_json or [])
            except json.JSONDecodeError:
                aliases = []
            raw_roles_json = props.get("role_assignments_json", "[]")
            try:
                raw_roles = json.loads(raw_roles_json) if isinstance(raw_roles_json, str) else list(raw_roles_json or [])
                role_assignments = [RoleAssignment(**r) for r in raw_roles]
            except Exception:
                role_assignments = []
            subgraph.add_node(node_name,
                              entity_name=props.get("entity_name", ""),
                              entity_type=props.get("entity_type", ""),
                              entity_id=props.get("entity_id", ""),
                              canonical_id=props.get("canonical_id", ""),
                              entity_role=props.get("entity_role", "provisional"),
                              aliases=aliases,
                              mapping_confidence=float(props.get("mapping_confidence", 0.0) or 0.0),
                              ontology_source=props.get("ontology_source", ""),
                              description=props.get("description", ""),
                              source_ids=set(props.get("source_ids", [])),
                              role_assignments=role_assignments)

        edge_result = g.query(
            f"MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
            f"WHERE a.node_name IN {node_filter} AND b.node_name IN {node_filter} "
            f"RETURN a.node_name, b.node_name, r.relation_name, r.weight, r.description, r.source_ids"
        )
        for rec in edge_result.result_set:
            src_name, tgt_name, rel_name, weight, desc, src_ids = rec
            if src_name in subgraph and tgt_name in subgraph:
                subgraph.add_edge(src_name, tgt_name,
                                  relation_name=rel_name or "",
                                  weight=float(weight or 0.0),
                                  description=desc or "",
                                  source_ids=set(src_ids or []))
        return subgraph

    def save_graph(self) -> None:

        if not self.save_dir and not self.use_falkordb:
            log.warning("Warning: save_dir is not set. Nothing will be saved.")
            return

        # If FalkorDB is configured, persist there
        if self.use_falkordb:
            self._save_to_falkordb()
            # Also save JSON as backup if save_dir is set
            if self.save_dir:
                os.makedirs(self.save_dir, exist_ok=True)

        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, self.data_filename)
            graph_json_data = json_graph.node_link_data(self.kg, edges="links")
            data_to_save = {
                "graph": graph_json_data,
                "tree2kg": {k: list(v) for k, v in self.tree2kg.items()},
                "variant": self.variant,
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, cls=SetEncoder, indent=4, ensure_ascii=False)
            log.info(f"Graph data successfully saved to: {save_path}")

    def get_kg_subgraph(
        self, tree_node_ids: Iterable[int], copy: bool = True
    ) -> nx.Graph:
        """
        Given one or more tree node IDs, return the induced subgraph of the KG.
        In FalkorDB mode, queries the database directly for only the needed nodes.
        """
        if self.use_falkordb and not self.kg.nodes:
            # FalkorDB-only mode: fetch subgraph from DB
            return self._get_fdb_subgraph(tree_node_ids)
        # Default: in-memory NetworkX subgraph
        kg_nodes = set().union(*(self.tree2kg.get(tid, set()) for tid in tree_node_ids))
        sub = self.kg.subgraph(kg_nodes)
        return sub.copy() if copy else sub

    @classmethod
    def load_from_dir(
        cls,
        load_dir: str,
        variant: str = None,
        tenant_id: str = None,
        doc_id: str = None,
        falkordb_cfg=None,
    ) -> "Graph":
        """Load a Graph from JSON file, or from FalkorDB if configured."""
        # FalkorDB mode: load tree2kg from DB, keep kg empty for lazy subgraph queries
        if falkordb_cfg is not None and tenant_id is not None and doc_id is not None:
            graph_instance = cls(
                save_path=load_dir,
                variant=variant,
                tenant_id=tenant_id,
                doc_id=doc_id,
                falkordb_cfg=falkordb_cfg,
            )
            graph_instance._load_from_falkordb()
            log.info(f"Graph loaded from FalkorDB graph '{graph_instance._fdb_graph_name}'")
            return graph_instance

        # Default: JSON file load
        target_filename = cls._get_filename(variant)
        load_path = os.path.join(load_dir, target_filename)
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Error: Missing graph file: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        graph_instance = cls(save_path=load_dir, variant=variant)
        # Pass edges="links" to match the key used when saving with node_link_data(edges="links")
        graph_instance.kg = json_graph.node_link_graph(loaded_data["graph"], edges="links")

        for _, node_data in graph_instance.kg.nodes(data=True):
            if "source_ids" in node_data and isinstance(node_data["source_ids"], list):
                node_data["source_ids"] = set(node_data["source_ids"])
            node_data.setdefault("entity_id", "")
            node_data.setdefault("canonical_id", "")
            node_data.setdefault("entity_role", "provisional")
            node_data.setdefault("aliases", [])
            node_data.setdefault("mapping_confidence", 0.0)
            node_data.setdefault("ontology_source", "")
            # Deserialize role_assignments stored as raw dicts in JSON
            raw_roles = node_data.get("role_assignments", [])
            if raw_roles and isinstance(raw_roles[0], dict):
                node_data["role_assignments"] = [RoleAssignment(**r) for r in raw_roles]
            else:
                node_data.setdefault("role_assignments", [])

        for _, _, edge_data in graph_instance.kg.edges(data=True):
            if "source_ids" in edge_data and isinstance(edge_data["source_ids"], list):
                edge_data["source_ids"] = set(edge_data["source_ids"])

        graph_instance.tree2kg = defaultdict(
            set, {int(k): set(v) for k, v in loaded_data["tree2kg"].items()}
        )

        log.info(f"Graph data successfully loaded from: {load_path}")
        log.info(
            f"Graph contains {len(graph_instance.kg.nodes)} nodes and {len(graph_instance.kg.edges)} edges."
        )
        return graph_instance

    # ------------------------------------------------------------------ #
    #  Phase 3: Global graph methods                                       #
    # ------------------------------------------------------------------ #

    def save_to_global_graph(self, falkordb_cfg, tenant_id: str) -> None:
        """
        Merge this document's KG nodes into the tenant-level global FalkorDB graph.
        Each entity is stored as a canonical node; a HAS_MENTION edge links the
        canonical node back to its document source.
        """
        try:
            from falkordb import FalkorDB
            cfg = falkordb_cfg
            conn_kwargs = {"host": cfg.host, "port": cfg.port}
            if cfg.username:
                conn_kwargs["username"] = cfg.username
            if cfg.password:
                conn_kwargs["password"] = cfg.password
            client = FalkorDB(**conn_kwargs)
            global_graph = client.select_graph(cfg.graph_name_for_global(tenant_id))
        except Exception as e:
            log.error(f"Failed to connect to FalkorDB global graph: {e}")
            raise

        def _esc(value) -> str:
            return str(value or "").replace("\\", "\\\\").replace("'", "\\'")

        for node_name, data in self.kg.nodes(data=True):
            ename = _esc(data.get("entity_name", ""))
            etype = _esc(data.get("entity_type", ""))
            desc = _esc(data.get("description", ""))
            entity_id = _esc(data.get("entity_id", ""))
            canonical_id = _esc(data.get("canonical_id", ""))
            entity_role = _esc(data.get("entity_role", "provisional"))
            ontology_source = _esc(data.get("ontology_source", ""))
            aliases_json = _esc(json.dumps(data.get("aliases", []), ensure_ascii=False))
            mapping_confidence = float(data.get("mapping_confidence", 0.0) or 0.0)
            nname = _esc(node_name)
            doc_id_esc = _esc(self.doc_id)
            # MERGE canonical entity node
            global_graph.query(
                f"MERGE (n:Entity {{node_name: '{nname}'}}) "
                f"ON CREATE SET n.entity_name='{ename}', n.entity_type='{etype}', n.description='{desc}', "
                f"n.entity_id='{entity_id}', n.canonical_id='{canonical_id}', n.entity_role='{entity_role}', "
                f"n.aliases_json='{aliases_json}', n.mapping_confidence={mapping_confidence}, "
                f"n.ontology_source='{ontology_source}' "
                f"CREATE (n)-[:HAS_MENTION {{doc_id: '{doc_id_esc}'}}]->(n)"
            )
        log.info(f"Merged {self.kg.number_of_nodes()} nodes into global graph for tenant '{tenant_id}'.")

    def get_global_subgraph(
        self,
        falkordb_cfg,
        tenant_id: str,
        accessible_doc_ids: List[str],
    ) -> nx.Graph:
        """
        Fetch a cross-document subgraph from the global FalkorDB graph,
        restricted to documents in accessible_doc_ids.
        """
        try:
            from falkordb import FalkorDB
            cfg = falkordb_cfg
            conn_kwargs = {"host": cfg.host, "port": cfg.port}
            if cfg.username:
                conn_kwargs["username"] = cfg.username
            if cfg.password:
                conn_kwargs["password"] = cfg.password
            client = FalkorDB(**conn_kwargs)
            global_graph = client.select_graph(cfg.graph_name_for_global(tenant_id))
        except Exception as e:
            log.error(f"Failed to connect to FalkorDB global graph: {e}")
            raise

        # Use a relationship variable r to filter by doc_id property on HAS_MENTION edges
        doc_filter = "['" + "', '".join(d.replace("'", "\\'") for d in accessible_doc_ids) + "']"
        result = global_graph.query(
            f"MATCH (n:Entity)-[r:HAS_MENTION]->(n) "
            f"WHERE r.doc_id IN {doc_filter} RETURN DISTINCT n"
        )
        subgraph = nx.Graph()
        for rec in result.result_set:
            node = rec[0]
            props = node.properties
            node_name = props["node_name"]
            aliases_json = props.get("aliases_json", "[]")
            try:
                aliases = json.loads(aliases_json) if isinstance(aliases_json, str) else list(aliases_json or [])
            except json.JSONDecodeError:
                aliases = []
            subgraph.add_node(node_name,
                              entity_name=props.get("entity_name", ""),
                              entity_type=props.get("entity_type", ""),
                              entity_id=props.get("entity_id", ""),
                              canonical_id=props.get("canonical_id", ""),
                              entity_role=props.get("entity_role", "provisional"),
                              aliases=aliases,
                              mapping_confidence=float(props.get("mapping_confidence", 0.0) or 0.0),
                              ontology_source=props.get("ontology_source", ""),
                              description=props.get("description", ""))
        return subgraph


if __name__ == "__main__":
    # Example usage
    tmp_save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    graph = Graph(save_path=tmp_save_path)
    entity1 = Entity(
        entity_name="Entity1",
        entity_type="TypeA",
        description="First entity",
        source_ids={1},
    )
    entity2 = Entity(
        entity_name="Entity2",
        entity_type="TypeB",
        description="Second entity",
        source_ids={2},
    )

    graph.add_and_link(1, entity1)
    graph.add_and_link(2, entity2)

    relationship = Relationship(
        src_entity_name="Entity1", tgt_entity_name="Entity2", relation_name="related_to"
    )
    graph.add_kg_edge(relationship, src_type="TypeA", tgt_type="TypeB")

    graph.save_graph()

    loaded_graph = Graph.load_from_dir(tmp_save_path)
    print(loaded_graph.get_all_nodes())
    print(loaded_graph.get_entity("Entity1", "TypeA"))
    src_node_name = loaded_graph.get_node_name_from_str("Entity1", "TypeA")
    tgt_node_name = loaded_graph.get_node_name_from_str("Entity2", "TypeB")
    print(
        f"relation: {loaded_graph.kg.get_edge_data(src_node_name, tgt_node_name)['relation_name']}"
    )
