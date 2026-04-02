from Core.provider.llm import LLM
from Core.provider.vlm import VLM
from Core.configs.graph_config import GraphConfig
from Core.Index.Graph import Entity, Relationship, RoleAssignment, RoleEvidence, SetEncoder
from Core.Index.Tree import TreeNode, NodeType
from Core.prompts.kg_prompt import (
    DEFAULT_ENTITY_TYPES,
    ENTITY_EXTRACTION,
    ENTITY_CONTINUE_EXTRACTION,
    ENTITY_IF_LOOP_EXTRACTION,
    DEFAULT_TUPLE_DELIMITER,
    DEFAULT_RECORD_DELIMITER,
    DEFAULT_COMPLETION_DELIMITER,
    EQUATION_ENTITY_EXTRACTION,
    IMAGE_ENTITY_EXTRACTION,
    TABLE_DESCRIPTION_EXTRACTION,
    TABLE_BODY_EXTRACTION,
    TABLE_ENTITY_EXTRACTION,
    SECTION_ENTITY_EXTRACTION,
    ExtractionResult,
    EntityExtractionResult,
    FormulaExtractionResult,
    RoleExtractionResult,
    ROLE_EXTRACTION,
    RELATIONSHIP_EXTRACTION_FROM_ENTITIES,
    HYBRID_DOMAIN_ENTITY_TYPES,
)
from Core.Common.Memory import Memory
from Core.Common.Message import Message
from Core.utils.utils import (
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
    TextProcessor,
    num_tokens,
)
from Core.utils.table_utils import (
    create_hierarchical_headers,
    identify_header_rows,
    parse_html_table_to_grid,
)

from abc import ABC, abstractmethod
import spacy
import textacy.extract
from typing import List, Dict, Tuple, Any, Union, Optional, final
import logging
import re
import json
import os
import uuid
from nltk.metrics.distance import edit_distance
from concurrent.futures import ThreadPoolExecutor
import re

logger = logging.getLogger(__name__)


# 定义一个函数，用于通过空格和所有非字母数字的符号来分割字符串
def split_by_space_and_symbols(text):
    # re.split(r'\W+', text) 会按所有非字母数字字符（包括符号和下划线）进行分割
    # 过滤掉分割后可能产生的空字符串
    words = [word for word in re.split(r"\W+", text) if word]
    return words


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, node: TreeNode):
        pass


class LocalExtractor(BaseExtractor):
    """
    Extractor that uses local NLP libraries (mainly spaCy) to extract entities and relations.
    """

    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        Initializes the extractor with GPU support.

        Args:
            model_name (str): The name of the spaCy transformer model.
        """
        try:
            spacy.require_gpu()
            logger.info("GPU activated for spaCy.")
        except Exception as e:
            logger.info(f"GPU activation failed: {e}. Falling back to CPU.")

        logger.info(f"Loading spaCy model '{model_name}'...")
        self.nlp = spacy.load(model_name)
        logger.info("Model loaded successfully.")

    def _extract_entities(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Extracts named entities from the doc and returns a structured list."""
        entities = []
        for ent in doc.ents:
            entities.append(
                {
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
        return entities

    def _extract_relations(self, doc: spacy.tokens.Doc) -> List[Tuple[str, str, str]]:
        """
        Extracts relation triples (subject, relation, object) from the doc.
        This is the core method based on dependency parsing.
        """
        relations = []
        # For efficient lookup, create a mapping from a token's index to its entity span
        token_index_to_entity = {token.i: ent for ent in doc.ents for token in ent}

        for token in doc:
            # Rule 1: Look for verbs as potential relations
            if token.pos_ == "VERB":
                subjects = []
                objects = []

                # Iterate over the verb's children to find subjects and objects
                for child in token.children:
                    # Find subject (nsubj) or passive subject (nsubjpass)
                    if "nsubj" in child.dep_:
                        # Find the full entity span corresponding to the subject token
                        if child.i in token_index_to_entity:
                            subjects.append(token_index_to_entity[child.i])

                    # Find object (dobj), attribute (attr), or object of a preposition (pobj)
                    elif "obj" in child.dep_ or "attr" in child.dep_:
                        # Find the full entity span corresponding to the object token
                        if child.i in token_index_to_entity:
                            # Case A: Direct object (dobj) or attribute (attr)
                            # The relation is the verb lemma itself
                            relation_phrase = token.lemma_
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                        # Case B: Object of a preposition (pobj)
                        elif child.dep_ == "pobj":
                            # 'child' is the object entity, 'child.head' is the preposition
                            preposition = child.head
                            # The relation phrase is the verb's lemma + the preposition's text
                            relation_phrase = f"{token.lemma_} {preposition.text}"
                            objects.append(
                                (relation_phrase, token_index_to_entity[child.i])
                            )

                # Combine all found subjects and objects
                for subj_ent in subjects:
                    for rel, obj_ent in objects:
                        # Avoid duplicates and self-referential relations
                        if subj_ent != obj_ent:
                            relations.append((subj_ent.text, rel, obj_ent.text))

        # Remove duplicates
        return list(set(relations))

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        """
        Executes the full entity and relation extraction pipeline.

        Args:
            text (str): The English text to be processed.

        Returns:
            Dict[str, Any]: A dictionary containing lists of 'entities' and 'relations'.
        """
        text = node.meta_info.content
        node_idx = node.index_id
        doc = self.nlp(text)
        entities = self._extract_entities(doc)
        relations = self._extract_relations(doc)

        return {
            "entities": entities,
            "relations": relations,
            "node_idx": node_idx,
        }

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        return self.extract(node)

    def extract_with_textacy(self, text: str) -> List[Tuple[str, str, str]]:
        """
        A simpler method: Use the Textacy library to directly extract SVO triples.
        This can serve as a quick baseline or a simplified approach.
        """
        doc = self.nlp(text)
        svo_triples = textacy.extract.subject_verb_object_triples(doc)

        # Format Textacy's output into (subject_text, verb_text, object_text)
        relations = []
        for triple in svo_triples:
            subj = " ".join(t.text for t in triple.subject)
            verb = " ".join(t.text for t in triple.verb)
            obj = " ".join(t.text for t in triple.object)
            relations.append((subj, verb, obj))

        return relations


class LLMExtractor(BaseExtractor):
    def __init__(
        self,
        graph_config: GraphConfig,
        llm: LLM,
        vlm: VLM = None,
    ):
        self.llm = llm
        self.max_gleaning = graph_config.max_gleaning
        self.graph_config: GraphConfig = graph_config
        if self.graph_config.image_description_force:
            self.vlm = vlm
        else:
            self.vlm = None

    @classmethod
    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter=DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content,
        )

    def _extract_records_from_text(self, chunk_text: str):
        """
        Extract entity and relationship from chunk, which is used for the GraphRAG.
        Please refer to the following references:
        1. https://github.com/gusye1234/nano-graphrag
        2. https://github.com/HKUDS/LightRAG/tree/main
        """
        context = self._build_context_for_entity_extraction(chunk_text)
        prompt = ENTITY_EXTRACTION.format(**context)

        working_memory = Memory()

        working_memory.add(Message(content=prompt, role="user"))
        final_result = self.llm.get_completion(prompt)
        working_memory.add(Message(content=final_result, role="assistant"))

        for glean_idx in range(self.max_gleaning):
            working_memory.add(Message(content=ENTITY_CONTINUE_EXTRACTION, role="user"))
            glean_result = self.llm.get_completion(working_memory)

            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result

            if glean_idx == self.max_gleaning - 1:
                break

            working_memory.add(Message(content=ENTITY_IF_LOOP_EXTRACTION, role="user"))

            if_loop_result = self.llm.get_completion(working_memory)
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        working_memory.clear()

        return split_string_by_multi_markers(
            final_result, [DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER]
        )

    @classmethod
    def _handle_single_entity_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Entity, None]:

        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1])
        if not entity_name.strip():
            return None

        entity = Entity(
            entity_name=entity_name,
            entity_type=clean_str(record_attributes[2]),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

        return entity

    def _handle_single_relationship_extraction(
        self, record_attributes: list[str], chunk_key: int
    ) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_entity_name=clean_str(record_attributes[1]),
            tgt_entity_name=clean_str(record_attributes[2]),
            weight=(
                float(record_attributes[-1])
                if is_float_regex(record_attributes[-1])
                else 1.0
            ),
            description=clean_str(record_attributes[3]),
            source_ids={chunk_key},
        )

    def _build_graph_from_records(self, records: list[str], chunk_key: int):
        entities_list = []
        relationships_list = []
        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(
                match.group(1), [DEFAULT_TUPLE_DELIMITER]
            )
            entity = self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                entities_list.append(entity)
                continue

            relationship = self._handle_single_relationship_extraction(
                record_attributes, chunk_key
            )

            if relationship is not None:
                relationships_list.append(relationship)

        return entities_list, relationships_list

    def _extract_roles_from_text(
        self,
        text: str,
        entities: List[Entity],
        node_id: int,
    ) -> Dict[str, List[RoleAssignment]]:
        """Run a dedicated LLM role-extraction pass over a text chunk.

        Filters to PERSON/ORGANIZATION entities only, calls the LLM with the
        ROLE_EXTRACTION prompt, and returns a dict mapping canonical entity_name
        (original casing) -> list of RoleAssignment objects with evidence and
        normalization already applied.
        """
        from Core.utils.role_normalizer import apply_normalization

        # Only attempt extraction for person/org entities
        role_entity_types = {"PERSON", "ORGANIZATION", "ORG", "GOVERNMENT", "OFFICIAL"}
        role_entities = [
            e for e in entities if e.entity_type.upper() in role_entity_types
        ]
        if not role_entities:
            return {}

        entity_list_str = "\n".join(
            f"- {e.entity_name} ({e.entity_type})" for e in role_entities
        )
        prompt = ROLE_EXTRACTION.format(entity_list=entity_list_str, input_text=text)

        try:
            res: Optional[RoleExtractionResult] = self.llm.get_json_completion(
                prompt, schema=RoleExtractionResult
            )
        except Exception as exc:
            logger.error(f"Role extraction LLM call failed for node {node_id}: {exc}")
            return {}

        if not res or not res.roles:
            return {}

        # Case-insensitive lookup: lower_name -> canonical entity_name
        name_map: Dict[str, str] = {e.entity_name.lower(): e.entity_name for e in role_entities}

        result: Dict[str, List[RoleAssignment]] = {}
        for extracted in res.roles:
            canonical_name = name_map.get(extracted.entity_name.lower())
            if canonical_name is None:
                logger.debug(
                    f"Role extraction: '{extracted.entity_name}' not in entity list for node {node_id}, skipping."
                )
                continue

            evidence = RoleEvidence(
                source_id=node_id,
                observed_role_text=extracted.role_name,
                evidence_text=extracted.evidence_text,
                normalization_method="unresolved",
                confidence=extracted.confidence,
            )
            assignment = RoleAssignment(
                assignment_id=str(uuid.uuid4()),
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
            except Exception as exc:
                logger.warning(
                    f"Role normalization failed for '{extracted.role_name}': {exc}"
                )

            result.setdefault(canonical_name, []).append(assignment)

        logger.info(
            f"Node {node_id}: role extraction found {sum(len(v) for v in result.values())} "
            f"role assignments across {len(result)} entities."
        )
        return result

    def _extract_kg_from_text(self, node: TreeNode):
        content_texts = node.meta_info.content
        processor = TextProcessor()
        split_tokens = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(ENTITY_CONTINUE_EXTRACTION + ENTITY_EXTRACTION)
        )
        chunks = processor.split_text_into_chunks(
            text=content_texts, max_length=split_tokens
        )
        res_entities = []
        res_relation = []
        for text in chunks:
            records = self._extract_records_from_text(text)
            entities, relations = self._build_graph_from_records(records, node.index_id)

            # Phase 1D: optional role extraction per chunk
            if self.graph_config.role_extraction_enabled and entities:
                role_map = self._extract_roles_from_text(text, entities, node.index_id)
                for entity in entities:
                    extra_roles = role_map.get(entity.entity_name, [])
                    if extra_roles:
                        entity.role_assignments = list(entity.role_assignments) + extra_roles

            res_entities.extend(entities)
            res_relation.extend(relations)
        return res_entities, res_relation

    def _extract_kg_table_step1(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """
        第一步：从描述中提取实体。
        """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()

        column_headers = []
        column_headers = []
        if num_header_rows > 0 and grid:
            column_headers = create_hierarchical_headers(grid, num_header_rows)

        input_data = {"description": description, "column_headers": column_headers}
        input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

        prompt = TABLE_DESCRIPTION_EXTRACTION.format(
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_json=input_json_str,  # 假设 prompt 现在的占位符是 input_text
        )

        llm_entities_raw = []
        try:
            res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                prompt, schema=EntityExtractionResult
            )
            if res and res.entities:
                llm_entities_raw = res.entities
        except Exception as e:
            logger.error(f"Step 1 LLM call failed for node {node.index_id}: {e}")

        # 2. 验证并分离 LLM 的提取结果
        table_entity_from_llm = next(
            (ent for ent in llm_entities_raw if ent.entity_type == "TABLE"), None
        )
        other_entities_from_llm = [
            ent for ent in llm_entities_raw if ent.entity_type != "TABLE"
        ]

        # 3. 决定最终的 TABLE 实体（LLM版本 vs. Fallback版本）
        final_table_entity: Entity
        if table_entity_from_llm:
            # 如果LLM成功提取了TABLE实体，则使用它
            logger.info(
                f"LLM successfully extracted TABLE entity for node {node.index_id}."
            )
            final_table_entity = Entity(
                entity_name=table_entity_from_llm.entity_name,
                entity_type=table_entity_from_llm.entity_type,
                description=table_entity_from_llm.description,
                source_ids={node.index_id},
            )
        else:
            # 如果LLM未能提取，或者没有描述，或者调用失败，则创建我们的 fallback 实体
            logger.warning(
                f"LLM failed to provide a TABLE entity for node {node.index_id}. Creating a fallback."
            )
            if description:
                table_name_prefix = " ".join(description.split()[:8])
                table_desc = f"A data table described as: {description}"
            else:
                table_name_prefix = f"Node {node.index_id}"
                table_desc = "A table with no available description."

            final_table_entity = Entity(
                entity_name=f"Table: {table_name_prefix}...",
                entity_type="TABLE",
                description=table_desc,
                source_ids={node.index_id},
            )

        final_other_entities = [
            Entity(
                entity_name=ent.entity_name,
                entity_type=ent.entity_type,
                description=ent.description,
                source_ids={node.index_id},
            )
            for ent in other_entities_from_llm
        ]

        final_entities = [final_table_entity] + final_other_entities

        return final_entities

    def _split_table_into_batches(
        self,
        grid: List[List[str]],
        num_header_rows: int,
        # description 和 column_headers 的token开销将在主函数中计算
        max_tokens_for_rows: int,
    ) -> List[List[str]]:
        # 1. 提取并转换表格主体为行字符串
        body_rows = grid[num_header_rows:]
        # 我们使用 " | " 作为分隔符，这与我们prompt中的示例一致
        row_strings = [" | ".join(cell.strip() for cell in row) for row in body_rows]

        all_batches = []
        current_batch = []
        current_batch_tokens = 0

        for row_str in row_strings:
            row_tokens = num_tokens(row_str)

            # 如果单个行就超过了限制，将其单独放入一个批次。
            # 这是一个边缘情况，表示批处理大小可能需要调整。
            if row_tokens > max_tokens_for_rows:
                # 如果当前批次有内容，先提交
                if current_batch:
                    all_batches.append(current_batch)
                # 将超长的行单独作为一个批次
                all_batches.append([row_str])
                # 重置当前批次
                current_batch = []
                current_batch_tokens = 0
                continue

            # 如果将当前行加入批次会超出限制
            if current_batch_tokens + row_tokens > max_tokens_for_rows:
                # 提交当前已满的批次
                all_batches.append(current_batch)
                # 用当前行开始一个新的批次
                current_batch = [row_str]
                current_batch_tokens = row_tokens
            else:
                # 将当前行加入批次
                current_batch.append(row_str)
                current_batch_tokens += row_tokens

        # 3. 不要忘记提交最后一个批次
        if current_batch:
            all_batches.append(current_batch)

        return all_batches

    def _create_prompts_from_batches(
        self, batches: List[List[str]], description: str, column_headers: List[str]
    ) -> List[str]:
        """根据批次数据、描述和列表头，创建完整的prompt列表。"""
        prompts = []
        for batch in batches:
            # 1. 构建输入JSON对象
            input_data = {
                "description": description,
                "column_headers": column_headers,
                "rows_batch": batch,  # batch现在是行字符串的列表
            }
            # 2. 序列化为JSON字符串
            input_json_str = json.dumps(input_data, ensure_ascii=False, indent=2)

            # 3. 格式化主Prompt模板
            # 假设 DEFAULT_ENTITY_TYPES 和 TABLE_BODY_EXTRACTION 已定义
            prompt = TABLE_BODY_EXTRACTION.format(
                entity_types=",".join(DEFAULT_ENTITY_TYPES),
                input_json=input_json_str,
            )
            prompts.append(prompt)

        return prompts

    def _extract_kg_table_step2(
        self, node: TreeNode, grid: List[List[str]], num_header_rows: int
    ) -> List[Entity]:
        """ """
        description = (
            (node.meta_info.caption or "") + " " + (node.meta_info.footnote or "")
        )
        description = description.strip()
        column_headers = create_hierarchical_headers(grid, num_header_rows)

        prompt_overhead = num_tokens(description) + num_tokens(
            json.dumps(column_headers)
        )
        max_tokens_for_rows = (
            self.llm.config.max_tokens
            - 400
            - num_tokens(TABLE_BODY_EXTRACTION)
            - prompt_overhead
        )

        batches = self._split_table_into_batches(
            grid, num_header_rows, max_tokens_for_rows
        )
        prompts = self._create_prompts_from_batches(
            batches, description, column_headers
        )

        logger.info(
            f"Node {node.index_id}: Table body split into {len(prompts)} batches for processing."
        )

        all_body_entities = []
        for i, prompt in enumerate(prompts):
            logger.info(
                f"Processing batch {i+1}/{len(prompts)} for node {node.index_id}..."
            )
            try:
                res: Union[None, EntityExtractionResult] = self.llm.get_json_completion(
                    prompt, schema=EntityExtractionResult
                )
                if res and res.entities:
                    batch_entities = [
                        Entity(
                            entity_name=ent.entity_name,
                            entity_type=ent.entity_type,
                            description=ent.description,
                            source_ids={node.index_id},
                        )
                        for ent in res.entities
                    ]
                    all_body_entities.extend(batch_entities)
            except Exception as e:
                logger.error(
                    f"LLM call failed for batch {i+1} of node {node.index_id}: {e}"
                )
                continue

        return all_body_entities

    def _extract_kg_from_table(self, node: TreeNode):

        table_body = node.meta_info.table_body
        grid = parse_html_table_to_grid(table_body)
        num_header_rows = identify_header_rows(grid)

        desc_entities = self._extract_kg_table_step1(node, grid, num_header_rows)
        body_entities = self._extract_kg_table_step2(node, grid, num_header_rows)

        table_entity = desc_entities[0]
        other_step1_entities = desc_entities[1:]

        final_other_entities = []
        seen_entity_names = {ent.entity_name for ent in other_step1_entities}

        for ent in body_entities:
            if ent.entity_name not in seen_entity_names:
                final_other_entities.append(ent)
                seen_entity_names.add(ent.entity_name)

        final_entities = [table_entity] + other_step1_entities + final_other_entities

        final_relations = [
            Relationship(
                src_entity_name=table_entity.entity_name,
                tgt_entity_name=ent.entity_name,
                weight=9.0,
                description=f"Table '{table_entity.entity_name}' contains data about '{ent.entity_name}'.",
                source_ids={node.index_id},
            )
            for ent in final_entities
            if ent.entity_type != "TABLE"
        ]
        logger.info(
            f"Total relations created: {len(final_relations)} in Table node: {node.index_id}"
        )

        return final_entities, final_relations

    def _extract_kg_from_image(self, node: TreeNode):
        if self.graph_config.image_description_force == False:
            logger.warning(
                "Image description force is disabled, skipping image extraction."
            )
            return [], []
        image_description = node.meta_info.content
        image_path = node.meta_info.img_path
        prompt = IMAGE_ENTITY_EXTRACTION.format(
            image_description=image_description,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
        )
        try:
            res: Union[None | EntityExtractionResult] = self.vlm.generate_json(
                prompt_or_memory=prompt,
                images=[image_path],
                schema=EntityExtractionResult,
            )
            if res is None:
                logger.warning("No response from LLM.")
                return [], []

            entities = res.entities if res.entities else []
            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]

            image_entity = next(
                (ent for ent in final_entities if ent.entity_type == "IMAGE"), None
            )
            # check image_entity the first two words in the description or not
            if image_entity is not None:
                if (
                    split_by_space_and_symbols(image_entity.entity_name)[:2]
                    != split_by_space_and_symbols(image_description)[:2]
                ):
                    image_entity = None
                    logger.info(
                        "Image entity name does not match the original image description."
                    )
                    logger.info("Creating a default image entity instead.")

            if image_entity is None:
                # create a default image_entity if not found,
                # use the first 5 words of the description as the name
                # use the description as the description
                logger.warning(
                    "No image entity found in the extracted entities. Creating a default image entity."
                )

                image_name = "Image " + " ".join(image_description.split()[:5])
                description = "Original image description: " + image_description
                image_entity = Entity(
                    entity_name=image_name,
                    entity_type="IMAGE",
                    description=description,
                    source_ids={node.index_id},
                )

            # create relationship from the image entity to all other entities
            final_relations = [
                Relationship(
                    src_entity_name=image_entity.entity_name,
                    tgt_entity_name=ent.entity_name,
                    weight=9.0,  # Default weight
                    description=f"Image entity {image_entity.entity_name} related to {ent.entity_name}",
                    source_ids={node.index_id},
                )
                for ent in final_entities
                if ent.entity_name != image_entity.entity_name
            ]

            return final_entities, final_relations

        except Exception as e:
            logger.exception(f"Error extracting entities and relations from image: {e}")
            return [], []

    def _extract_kg_from_equation(self, node: TreeNode):
        latex_text = node.meta_info.content
        try:
            prompt = EQUATION_ENTITY_EXTRACTION.format(formula_latex_code=latex_text)
            res: Union[None | FormulaExtractionResult] = self.llm.get_json_completion(
                prompt, schema=FormulaExtractionResult
            )
            if res is None:
                logger.info("LLM returned no response for equation extraction.")
                logger.info("Create a default entity for the equation.")

                equation_text = latex_text.split()[:8]
                equation_text = " ".join(equation_text)

                final_entity = Entity(
                    entity_name=f"Equation: {equation_text}",
                    entity_type="EQUATION_OR_FORMULA",
                    description=f"A formula represented by LaTeX code: {latex_text}",
                    source_ids={node.index_id},
                )
                return [final_entity], []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type="EQUATION_OR_FORMULA",
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in res.entities
            ]

            return final_entities, []

        except Exception as e:
            logger.exception(
                f"Error extracting entities and relations from equation: {e}"
            )
            return [], []

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        try:
            if node.type not in NodeType:
                raise ValueError(f"Unsupported node type: {node.type}")
            if node.type == NodeType.TEXT:
                entities, relations = self._extract_kg_from_text(node)
            elif node.type == NodeType.IMAGE:
                entities, relations = self._extract_kg_from_image(node)
            elif node.type == NodeType.TABLE:
                entities, relations = self._extract_kg_from_table(node)
            elif node.type == NodeType.EQUATION:
                entities, relations = self._extract_kg_from_equation(node)
            else:
                raise ValueError(f"Unsupported node type: {node.type}")

            return {
                "entities": entities,
                "relations": relations,
                "node_idx": node.index_id,
            }

        except Exception as e:
            logger.exception(f"Error extracting entities and relation: {e}")
        finally:
            logger.info(
                f"Finish extracting entities and relations in TreeNode {node.index_id}."
            )

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        title_path_obj = [
            {"depth": i, "title": node.meta_info.content}
            for i, node in enumerate(title_path)
        ]
        # find previous and next section title in sibling nodes
        # previou section index_id is small than current node index_id
        # next section index_id is large than current node index_id
        # sibling_nodes is sorted by index_id
        previous_section_title = ""
        next_section_title = ""
        for sibling in sibling_nodes:
            if sibling.index_id < node.index_id and not previous_section_title:
                previous_section_title = sibling.meta_info.content
            elif sibling.index_id > node.index_id and not next_section_title:
                next_section_title = sibling.meta_info.content
                break

        prep_prompt = {
            "context": {
                "previous_section_title": (
                    previous_section_title if previous_section_title else "null"
                ),
                "title_path": title_path_obj,
                "next_section_title": (
                    next_section_title if next_section_title else "null"
                ),
            },
            "title_to_process": node.meta_info.content,
        }
        input_json = json.dumps(prep_prompt, ensure_ascii=False, indent=2)
        prompt = SECTION_ENTITY_EXTRACTION.format(
            input_json=input_json, entity_types=",".join(DEFAULT_ENTITY_TYPES)
        )
        try:
            res = self.llm.get_json_completion(prompt=prompt, schema=ExtractionResult)
            if res is None:
                logger.warning("No response from LLM.")
                return {"entities": [], "relations": [], "node_idx": node.index_id}
            entities = res.entities if res.entities else []
            relations = res.relationships if res.relationships else []

            final_entities = [
                Entity(
                    entity_name=ent.entity_name,
                    entity_type=ent.entity_type,
                    description=ent.description,
                    source_ids={node.index_id},
                )
                for ent in entities
            ]
            final_relations = [
                Relationship(
                    src_entity_name=rel.src_entity_name,
                    tgt_entity_name=rel.tgt_entity_name,
                    weight=rel.weight,
                    description=rel.description,
                    source_ids={node.index_id},
                )
                for rel in relations
            ]
            return {
                "entities": final_entities,
                "relations": final_relations,
                "node_idx": node.index_id,
            }
        except Exception as e:
            logger.exception(f"Error extracting entities and relations from title: {e}")
            return {"entities": [], "relations": [], "node_idx": node.index_id}


class HybridExtractor(BaseExtractor):
    """
    BERT-only NER extractor for Indonesian (Bahasa) documents.

    Uses the Hugging Face token-classification model configured by
    ``graph_config.hybrid_ner_model`` (default:
    ``cahya/bert-base-indonesian-NER``) to extract Indonesian entities
    directly from text without any LLM NER pass.

    Tree-aware extraction policy:
    - TITLE and TEXT nodes → BERT NER
    - IMAGE / TABLE / EQUATION nodes → delegated to LLMExtractor

    Factual precision (exact figures, clause numbers, dates) is delegated to
    the Vector DB (Chroma) layer via embedding retrieval at query time.
    """

    # BERT models have a 512-subword limit.  Use a conservative character
    # budget so we never exceed it even with long Indonesian tokens.
    _BERT_CHAR_LIMIT = 1500

    # Map abbreviated Indonesian NER labels → canonical entity type names.
    _LABEL_MAP: Dict[str, str] = {
        "PER": "PERSON",
        "ORG": "ORGANIZATION",
        "LOC": "LOCATION",
        "GPE": "GEOPOLITICAL_ENTITY",
        "FAC": "FACILITY",
        "EVT": "EVENT",
        "WOA": "WORK_OF_ART",
        "LAW": "LAW",
        "PRO": "PRODUCT",
        "PRD": "PRODUCT",
        "LAN": "LANGUAGE",
        "DAT": "DATE",
        "TIM": "TIME",
        "MON": "MONEY",
        "QTY": "QUANTITY",
        "CRD": "CARDINAL",
        "ORD": "ORDINAL",
        "PRC": "PERCENT",
        "NOR": "POLITICAL_GROUP",
    }

    def __init__(
        self,
        graph_config: GraphConfig,
        llm: LLM,
        vlm: VLM = None,
    ):
        self.graph_config = graph_config
        self.llm = llm
        # Reuse the full LLMExtractor for non-text nodes (tables, images, etc.)
        self._llm_extractor = LLMExtractor(graph_config=graph_config, llm=llm, vlm=vlm)
        self._ner_pipeline = None  # lazy-loaded on first use
        self._ner_model_name = graph_config.hybrid_ner_model
        self._ner_model_ref = self._resolve_ner_model_reference(self._ner_model_name)
        self._confidence_threshold: float = getattr(
            graph_config, "ner_confidence_threshold", 0.5
        )

    # ------------------------------------------------------------------
    # BERT NER helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_ner_model_reference(model_name: str) -> str:
        """Resolve an existing local model directory while leaving Hub IDs untouched."""
        expanded = os.path.abspath(os.path.expanduser(os.path.expandvars(model_name)))
        return expanded if os.path.isdir(expanded) else model_name

    def _get_ner_pipeline(self):
        """Lazy-load the HuggingFace NER pipeline from a Hub ID or local directory."""
        if self._ner_pipeline is not None:
            return self._ner_pipeline
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading HuggingFace NER model '{self._ner_model_ref}' …")
            self._ner_pipeline = hf_pipeline(
                "ner",
                model=self._ner_model_ref,
                device=-1,  # CPU
            )
            logger.info("HuggingFace NER model loaded.")
        except Exception as exc:
            logger.error(f"Failed to load HuggingFace NER model: {exc}")
            self._ner_pipeline = None
        return self._ner_pipeline

    @staticmethod
    def _parse_bio_label(raw_label: str) -> Tuple[str, str]:
        label = (raw_label or "").upper()
        if label in {"", "O"}:
            return "O", "O"
        if "-" in label:
            prefix, base = label.split("-", 1)
            if prefix in {"B", "I"} and base:
                return prefix, base
        return "B", label

    @staticmethod
    def _normalize_entity_surface(surface: str) -> str:
        surface = re.sub(r"\s+", " ", surface).strip()
        surface = re.sub(r"([\(\[\{])\s+", r"\1", surface)
        surface = re.sub(r"\s+([\)\]\},.;:%])", r"\1", surface)
        return surface.strip()

    @classmethod
    def _extend_span_suffix(cls, text: str, start: int, end: int) -> int:
        while end < len(text):
            surface = text[start:end]
            next_char = text[end]
            if surface.count("(") > surface.count(")") and next_char == ")":
                end += 1
                continue
            if surface.count("[") > surface.count("]") and next_char == "]":
                end += 1
                continue
            if surface.count("{") > surface.count("}") and next_char == "}":
                end += 1
                continue
            break
        return end

    @staticmethod
    def _span_join_gap_is_safe(gap_text: str) -> bool:
        return bool(re.fullmatch(r"[\s\-/&.,:'\"()]*", gap_text or ""))

    @classmethod
    def _should_merge_token_into_span(
        cls,
        chunk: str,
        span: Dict[str, Any],
        prefix: str,
        raw_label: str,
        token_start: int,
        token_word: str,
    ) -> bool:
        if token_word.startswith("##") and token_start <= span["end"]:
            return True
        if prefix == "B":
            return False
        gap_text = chunk[span["end"] : token_start] if token_start >= span["end"] else ""
        if not cls._span_join_gap_is_safe(gap_text):
            return False
        dominant_label = max(
            span["label_scores"].items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        return raw_label in {dominant_label, span["last_label"]}

    @classmethod
    def _finalize_span(cls, chunk: str, span: Optional[Dict[str, Any]]):
        if not span:
            return None
        end = cls._extend_span_suffix(chunk, span["start"], span["end"])
        surface = cls._normalize_entity_surface(chunk[span["start"] : end])
        if not surface or not any(char.isalnum() for char in surface):
            return None
        raw_label = max(
            span["label_scores"].items(),
            key=lambda item: (item[1], item[0]),
        )[0]
        score = sum(span["scores"]) / len(span["scores"])
        return surface, raw_label, score

    @staticmethod
    def _split_trailing_corporate_prefix(surface: str) -> Tuple[str, str]:
        match = re.fullmatch(r"(.+?)\s+(PT\.?)", (surface or "").strip())
        if not match:
            return surface, ""
        base_surface = match.group(1).strip()
        prefix = match.group(2).strip()
        if not base_surface:
            return surface, ""
        return base_surface, prefix

    @classmethod
    def _repair_adjacent_corporate_prefix_spans(
        cls, spans: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        repaired: List[Tuple[str, str, float]] = []
        idx = 0
        while idx < len(spans):
            surface, raw_label, score = spans[idx]
            if idx + 1 >= len(spans):
                repaired.append((surface, raw_label, score))
                idx += 1
                continue

            next_surface, next_label, next_score = spans[idx + 1]
            base_surface, prefix = cls._split_trailing_corporate_prefix(surface)
            next_has_prefix = bool(re.match(r"^PT\.?($|\s)", next_surface or ""))
            if raw_label != "ORG" and next_label == "ORG" and prefix and not next_has_prefix:
                if base_surface:
                    repaired.append((base_surface, raw_label, score))
                repaired.append(
                    (
                        cls._normalize_entity_surface(f"{prefix} {next_surface}"),
                        next_label,
                        next_score,
                    )
                )
                idx += 2
                continue

            repaired.append((surface, raw_label, score))
            idx += 1
        return repaired

    @classmethod
    def _reconstruct_entity_spans(
        cls, chunk: str, raw_results: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, float]]:
        spans: List[Tuple[str, str, float]] = []
        current_span: Optional[Dict[str, Any]] = None
        valid_tokens = [
            item
            for item in raw_results
            if isinstance(item, dict)
            and isinstance(item.get("start"), int)
            and isinstance(item.get("end"), int)
            and item.get("end", 0) > item.get("start", 0)
        ]
        valid_tokens.sort(
            key=lambda item: (
                item["start"],
                item["end"],
                item.get("index", 0),
            )
        )

        for item in valid_tokens:
            prefix, raw_label = cls._parse_bio_label(item.get("entity", ""))
            if raw_label == "O":
                finalized = cls._finalize_span(chunk, current_span)
                if finalized is not None:
                    spans.append(finalized)
                current_span = None
                continue

            token_start = item["start"]
            token_end = item["end"]
            token_word = item.get("word", "")
            token_score = float(item.get("score", 0.0))

            if current_span is None:
                current_span = {
                    "start": token_start,
                    "end": token_end,
                    "scores": [token_score],
                    "label_scores": {raw_label: token_score},
                    "last_label": raw_label,
                }
                continue

            if cls._should_merge_token_into_span(
                chunk,
                current_span,
                prefix,
                raw_label,
                token_start,
                token_word,
            ):
                current_span["end"] = max(current_span["end"], token_end)
                current_span["scores"].append(token_score)
                current_span["label_scores"][raw_label] = (
                    current_span["label_scores"].get(raw_label, 0.0) + token_score
                )
                current_span["last_label"] = raw_label
                continue

            finalized = cls._finalize_span(chunk, current_span)
            if finalized is not None:
                spans.append(finalized)
            current_span = {
                "start": token_start,
                "end": token_end,
                "scores": [token_score],
                "label_scores": {raw_label: token_score},
                "last_label": raw_label,
            }

        finalized = cls._finalize_span(chunk, current_span)
        if finalized is not None:
            spans.append(finalized)
        return cls._repair_adjacent_corporate_prefix_spans(spans)

    def _bert_extract_entities(self, text: str, node_id: int) -> List[Entity]:
        """Run BERT NER on *text* (chunked to respect the 512-token limit)."""
        pipeline = self._get_ner_pipeline()
        if pipeline is None:
            return []

        # Split into character-budget chunks; respect sentence boundaries where possible.
        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= self._BERT_CHAR_LIMIT:
                chunks.append(remaining)
                break
            # Try to split at last sentence boundary within the budget.
            split_pos = remaining.rfind(".", 0, self._BERT_CHAR_LIMIT)
            if split_pos == -1:
                split_pos = self._BERT_CHAR_LIMIT
            chunks.append(remaining[: split_pos + 1])
            remaining = remaining[split_pos + 1 :].lstrip()

        entities: List[Entity] = []
        seen_names: set = set()
        for chunk in chunks:
            try:
                raw_results = pipeline(chunk)
            except Exception as exc:
                logger.warning(f"BERT NER failed on chunk for node {node_id}: {exc}")
                continue
            for name, raw_label, score in self._reconstruct_entity_spans(
                chunk, raw_results
            ):
                etype = self._LABEL_MAP.get(raw_label, raw_label)
                if not name or score < self._confidence_threshold:
                    continue
                key = re.sub(r"\s+", " ", name).strip().lower()
                if key in seen_names:
                    continue
                seen_names.add(key)
                entities.append(
                    Entity(
                        entity_name=name,
                        entity_type=etype,
                        description=f"A {etype} entity identified in the text.",
                        source_ids={node_id},
                    )
                )
        logger.info(
            f"Node {node_id}: BERT NER found {len(entities)} entities "
            f"from {len(chunks)} chunk(s)."
        )
        return entities

    # ------------------------------------------------------------------
    # Main extraction entry-point (BERT-only, no LLM for text nodes)
    # ------------------------------------------------------------------

    def _extract_kg_from_text(self, node: TreeNode):
        """
        Extract entities from a content-bearing text/title node using BERT NER.

        The configured Indonesian HF NER model is expected to expose BIO labels
        such as LAW, MON, PRC, and PRODUCT variants like PRO/PRD. Factual
        precision is left to the VDB / Chroma layer.
        """
        content = (node.meta_info.content or "").strip()
        if not content:
            return [], []
        # _bert_extract_entities handles its own chunking to respect the 512-token limit.
        entities = self._bert_extract_entities(content, node.index_id)
        return entities, []  # no relationships from BERT

    @staticmethod
    def _should_use_bert_for_node(node: TreeNode) -> bool:
        return node.type in {NodeType.TEXT, NodeType.TITLE}

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        try:
            if self._should_use_bert_for_node(node):
                entities, relations = self._extract_kg_from_text(node)
            else:
                # For tables, images, equations — delegate to LLMExtractor unchanged.
                return self._llm_extractor.extract(node)

            return {
                "entities": entities,
                "relations": relations,
                "node_idx": node.index_id,
            }
        except Exception as exc:
            logger.exception(f"HybridExtractor error on node {node.index_id}: {exc}")
            return {"entities": [], "relations": [], "node_idx": node.index_id}
        finally:
            logger.info(
                f"HybridExtractor finished node {node.index_id}."
            )

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        entities, relations = self._extract_kg_from_text(node)
        return {
            "entities": entities,
            "relations": relations,
            "node_idx": node.index_id,
        }


class FlairExtractor(BaseExtractor):
    """
    Flair NER extractor for high-quality English (or multilingual) entity extraction.

    Uses a pre-trained Flair SequenceTagger model (e.g. ``flair/ner-english-large``)
    to identify named entities in text nodes without any LLM NER pass.

    Tree-aware extraction policy:
    - TEXT and TITLE nodes → Flair NER
    - IMAGE / TABLE / EQUATION nodes → delegated to LLMExtractor

    Reference: https://github.com/flairNLP/flair
    """

    # Flair models have no hard sub-word limit but keep chunks modest for memory.
    _FLAIR_CHAR_LIMIT = 2000

    # Map Flair CoNLL / OntoNotes labels → canonical entity type names used by
    # the rest of the pipeline.
    _LABEL_MAP: Dict[str, str] = {
        # CoNLL-03 labels (flair/ner-english-large)
        "PER":  "PERSON",
        "ORG":  "ORGANIZATION",
        "LOC":  "LOCATION",
        "MISC": "MISCELLANEOUS",
        # OntoNotes-style labels (flair/ner-english-ontonotes-large)
        "PERSON":       "PERSON",
        "GPE":          "GEOPOLITICAL_ENTITY",
        "DATE":         "DATE",
        "TIME":         "TIME",
        "MONEY":        "MONEY",
        "PERCENT":      "PERCENT",
        "CARDINAL":     "CARDINAL",
        "ORDINAL":      "ORDINAL",
        "NORP":         "NATIONALITY_OR_GROUP",
        "FAC":          "FACILITY",
        "PRODUCT":      "PRODUCT",
        "EVENT":        "EVENT",
        "WORK_OF_ART":  "WORK_OF_ART",
        "LAW":          "LAW",
        "LANGUAGE":     "LANGUAGE",
        "QUANTITY":     "QUANTITY",
    }

    def __init__(
        self,
        graph_config: GraphConfig,
        llm: LLM,
        vlm: VLM = None,
    ):
        self.graph_config = graph_config
        # Reuse LLMExtractor for non-text nodes (tables, images, equations).
        self._llm_extractor = LLMExtractor(graph_config=graph_config, llm=llm, vlm=vlm)
        self._tagger = None  # lazy-loaded on first use
        self._model_name: str = getattr(
            graph_config, "flair_ner_model", "flair/ner-english-large"
        )
        self._confidence_threshold: float = getattr(
            graph_config, "flair_confidence_threshold", 0.5
        )

    # ------------------------------------------------------------------
    # Flair model helpers
    # ------------------------------------------------------------------

    def _get_tagger(self):
        """Lazy-load the Flair SequenceTagger (downloads from Hub on first call)."""
        if self._tagger is not None:
            return self._tagger
        try:
            from flair.models import SequenceTagger
            logger.info(f"Loading Flair NER model '{self._model_name}' …")
            self._tagger = SequenceTagger.load(self._model_name)
            logger.info("Flair NER model loaded.")
        except Exception as exc:
            logger.error(f"Failed to load Flair NER model '{self._model_name}': {exc}")
            self._tagger = None
        return self._tagger

    def _flair_extract_entities(self, text: str, node_id: int) -> List[Entity]:
        """Run Flair NER on *text*, splitting into character-budget chunks."""
        tagger = self._get_tagger()
        if tagger is None:
            return []

        # Chunk by character budget; prefer sentence boundaries.
        chunks: List[str] = []
        remaining = text
        while remaining:
            if len(remaining) <= self._FLAIR_CHAR_LIMIT:
                chunks.append(remaining)
                break
            split_pos = remaining.rfind(".", 0, self._FLAIR_CHAR_LIMIT)
            if split_pos == -1:
                split_pos = self._FLAIR_CHAR_LIMIT
            chunks.append(remaining[: split_pos + 1])
            remaining = remaining[split_pos + 1 :].lstrip()

        entities: List[Entity] = []
        seen_keys: set = set()

        for chunk in chunks:
            try:
                from flair.data import Sentence
                sentence = Sentence(chunk)
                tagger.predict(sentence)
            except Exception as exc:
                logger.warning(
                    f"Flair NER inference failed for node {node_id}: {exc}"
                )
                continue

            for span in sentence.get_spans("ner"):
                name = span.text.strip()
                score: float = span.score           # mean confidence across sub-tokens
                raw_label: str = span.tag.upper()   # e.g. "PER", "ORG"

                if not name or score < self._confidence_threshold:
                    continue

                etype = self._LABEL_MAP.get(raw_label, raw_label)
                dedup_key = re.sub(r"\s+", " ", name).strip().lower()
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                entities.append(
                    Entity(
                        entity_name=name,
                        entity_type=etype,
                        description=f"A {etype} entity identified by Flair NER.",
                        source_ids={node_id},
                    )
                )

        logger.info(
            f"Node {node_id}: Flair NER found {len(entities)} entities "
            f"from {len(chunks)} chunk(s) (threshold={self._confidence_threshold})."
        )
        return entities

    # ------------------------------------------------------------------
    # BaseExtractor interface
    # ------------------------------------------------------------------

    @staticmethod
    def _is_text_node(node: TreeNode) -> bool:
        return node.type in {NodeType.TEXT, NodeType.TITLE}

    def _extract_kg_from_text(self, node: TreeNode) -> Tuple[List[Entity], list]:
        content = (node.meta_info.content or "").strip()
        if not content:
            return [], []
        entities = self._flair_extract_entities(content, node.index_id)
        return entities, []  # Flair produces no relation predictions

    def extract(self, node: TreeNode) -> Dict[str, Any]:
        try:
            if self._is_text_node(node):
                entities, relations = self._extract_kg_from_text(node)
            else:
                # Tables, images, equations → full LLM extraction.
                return self._llm_extractor.extract(node)

            return {
                "entities": entities,
                "relations": relations,
                "node_idx": node.index_id,
            }
        except Exception as exc:
            logger.exception(
                f"FlairExtractor error on node {node.index_id}: {exc}"
            )
            return {"entities": [], "relations": [], "node_idx": node.index_id}
        finally:
            logger.info(f"FlairExtractor finished node {node.index_id}.")

    def extract_title(
        self, node: TreeNode, title_path: List[TreeNode], sibling_nodes: List[TreeNode]
    ):
        entities, relations = self._extract_kg_from_text(node)
        return {
            "entities": entities,
            "relations": relations,
            "node_idx": node.index_id,
        }


class KGExtractor:
    def __init__(
        self,
        cfg_graph: GraphConfig,
        llm: LLM = None,
        vlm: VLM = None,
        save_path: str = None,
        # force_rebuild: bool = True,
        force_rebuild: bool = False,
    ):
        self.cfg_graph = cfg_graph
        extractor_type = cfg_graph.extractor_type
        if extractor_type == "local":
            self.extractor = LocalExtractor(cfg_graph.local_model_name)
        elif extractor_type == "hybrid":
            self.extractor = HybridExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)
        elif extractor_type == "flair":
            self.extractor = FlairExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)
        elif extractor_type == "llm":
            self.extractor = LLMExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)
        else:
            logger.warning(
                f"Unknown extractor_type '{extractor_type}', falling back to 'llm'."
            )
            self.extractor = LLMExtractor(graph_config=cfg_graph, llm=llm, vlm=vlm)

        self.save_path = os.path.join(save_path, "kg_extractor_res")
        os.makedirs(self.save_path, exist_ok=True)
        logger.info(f"KG extraction results will be saved to {self.save_path}")
        self.force_rebuild = force_rebuild

    def res_repair(self, res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert all entity names and types to lower case in the result dictionary.
        This is useful for normalization and comparison purposes.
        """

        def _clean_name(name: str) -> str:
            """
            Normalizes a string by converting to lowercase, stripping whitespace,
            and consolidating multiple spaces into a single space.
            """
            cleaned_name = name.lower()
            cleaned_name = cleaned_name.strip()
            cleaned_name = re.sub(r"\s+", " ", cleaned_name)
            return cleaned_name

        def relation_name_refine(relation_entity_name, entity_name_set, res_obj):
            if relation_entity_name in entity_name_set:
                return relation_entity_name

            # Find the most similar entity name by computing edit distance
            logger.info(
                f"Relation entity '{relation_entity_name}' not found in entities."
            )
            logger.info(f"Available entities: {', '.join(entity_name_set)}")

            # Calculate normalized similarity
            most_similar_entity = None
            if entity_name_set:
                most_similar_entity = min(
                    entity_name_set,
                    key=lambda ent: edit_distance(ent, relation_entity_name),
                )
                distance = edit_distance(most_similar_entity, relation_entity_name)
                max_len = max(len(most_similar_entity), len(relation_entity_name))
                similarity = 1 - (distance / max_len) if max_len > 0 else 1.0
            else:
                # If the entity set was empty, similarity is effectively 0 to force creation
                similarity = 0.0

            if most_similar_entity and similarity >= 0.9:
                logger.info(
                    f"Similarity to '{most_similar_entity}' is {similarity:.2f}. Using it instead."
                )
                return most_similar_entity
            else:
                logger.info(
                    f"Similarity to closest entity is {similarity:.2f} (< 0.9). Creating a new 'UNKNOWN' entity for '{relation_entity_name}'."
                )

                # Create the new entity object
                new_entity = Entity(
                    entity_name=relation_entity_name,
                    entity_type="UNKNOWN",
                    description="",  # Description is empty as requested
                    source_ids={res_obj["node_idx"]},
                )

                # Add the new entity to the main list and the name to the set
                res_obj.get("entities", []).append(new_entity)
                entity_name_set.add(relation_entity_name)

                return relation_entity_name

        entity_name_set = set()
        for entity in res.get("entities", []):
            entity.entity_name = _clean_name(entity.entity_name)
            entity.entity_type = entity.entity_type.upper()
            entity.entity_type = entity.entity_type.replace(" ", "_")
            entity_name_set.add(entity.entity_name)

        # check if have the same name and type entities, merge the description and then delete the duplicate one
        entity_map = {}
        for entity in res.get("entities", []):
            key = (entity.entity_name, entity.entity_type)
            if key not in entity_map:
                entity_map[key] = entity
            else:
                # Merge descriptions.
                # Use the old source_id, since they are extracted from the save node
                existing_entity: Entity = entity_map[key]
                existing_entity.description += " " + entity.description
                # Remove the duplicate entity
                res["entities"].remove(entity)
                logger.info(
                    f"Removed duplicate entity: {entity.entity_name} of type {entity.entity_type}"
                )

        valid_relations = []
        relation_map = {}
        for relation in res.get("relations", []):
            src_name = _clean_name(relation.src_entity_name)
            tgt_name = _clean_name(relation.tgt_entity_name)

            repaired_src_name = relation_name_refine(src_name, entity_name_set, res)
            repaired_tgt_name = relation_name_refine(tgt_name, entity_name_set, res)

            # Check if the relation already exists in the map
            relation_key = (
                repaired_src_name,
                repaired_tgt_name,
                relation.relation_name,
            )
            if relation_key in relation_map:
                # If the relation is same as an existing one, remove the new one
                continue
            else:
                # Add the relation to the map
                relation_map[relation_key] = relation

            # The refine function should now always return a valid name
            relation.src_entity_name = repaired_src_name
            relation.tgt_entity_name = repaired_tgt_name
            valid_relations.append(relation)

        # Replace the old relations list with the filtered and repaired one
        res["relations"] = valid_relations

        return res

    def save_tmp_res(self, res: Dict[str, Any], node_idx: int):
        """
        Save the extracted knowledge graph result to a temporary file.
        This is useful for debugging and intermediate results.
        """

        file_path = os.path.join(self.save_path, f"kg_extractor_res_{node_idx}.json")
        entities = [e.model_dump() for e in res["entities"]]
        relations = [r.model_dump() for r in res["relations"]]
        res_to_save = {
            "entities": entities,
            "relations": relations,
            "node_idx": res["node_idx"],
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                res_to_save,
                f,
                ensure_ascii=False,
                indent=2,
                cls=SetEncoder,
            )
        logger.info(f"Saved temporary KG extraction result to {file_path}.")

    def load_tmp_res(self, node_idx: int) -> Dict[str, Any]:
        """
        Load the extracted knowledge graph result from a temporary file.
        This is useful for debugging and intermediate results.
        """
        file_path = os.path.join(self.save_path, f"kg_extractor_res_{node_idx}.json")
        if os.path.exists(file_path) and not self.force_rebuild:
            logger.info(
                f"Temporary result for node {node_idx} already exists. Loading from {file_path}."
            )
        else:
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        # 还原 Entity/Relationship 对象，并处理 source_ids
        entities = [
            Entity(
                entity_name=e["entity_name"],
                entity_type=e["entity_type"],
                description=e["description"],
                source_ids=set(e.get("source_ids", [])),
            )
            for e in res.get("entities", [])
        ]
        relations = [
            Relationship(
                src_entity_name=r["src_entity_name"],
                tgt_entity_name=r["tgt_entity_name"],
                relation_name=r.get("relation_name", ""),
                weight=r.get("weight", 0.0),
                description=r.get("description", ""),
                source_ids=set(r.get("source_ids", [])),
            )
            for r in res.get("relations", [])
        ]
        res_obj = {
            "entities": entities,
            "relations": relations,
            "node_idx": res.get("node_idx"),
        }

        logger.info(f"Loaded temporary KG extraction result from {file_path}.")
        return res_obj

    def extract_kg(self, node: TreeNode) -> Dict[str, Any]:
        tmp_res = self.load_tmp_res(node.index_id)
        if tmp_res is not None:
            # if node.type == NodeType.IMAGE:
            #     logger.info(f"Image node should be re-extracted")
            #     kg = self.extractor.extract(node)
            # else:
            #     kg = tmp_res
            kg = tmp_res
        else:
            kg = self.extractor.extract(node)

        res = self.res_repair(kg)
        # Save the result to a temporary file
        self.save_tmp_res(res, node.index_id)
        return res

    def extract_title(
        self,
        node: TreeNode,
        title_path: List[TreeNode],
        sibling_nodes: List[TreeNode],
    ) -> Dict[str, Any]:
        tmp_res = self.load_tmp_res(node.index_id)
        if tmp_res is not None:
            kg = tmp_res
        else:
            kg = self.extractor.extract_title(node, title_path, sibling_nodes)
            logger.info(
                f"Extracted title entities and relations for node {node.index_id}."
            )

        res = self.res_repair(kg)
        # Save the result to a temporary file
        self.save_tmp_res(res, node.index_id)
        return res

    def batch_extract_kg(
        self, nodes: List[TreeNode], max_workers: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Batch extract knowledge graphs from a list of nodes.
        This is useful for processing multiple nodes in one go.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_kg, node): node for node in nodes}
            for future in futures:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(
                        f"Error extracting KG for node {futures[future].index_id}: {e}"
                    )
                    results.append(
                        {
                            "entities": [],
                            "relations": [],
                            "node_idx": futures[future].index_id,
                        }
                    )
        return results

    def batch_extract_titles(
        self,
        nodes: List[TreeNode],
        title_paths: List[List[TreeNode]],
        sibling_nodes_list: List[List[TreeNode]],
        max_workers: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Batch extract knowledge graphs from a list of title nodes.
        This is useful for processing multiple title nodes in one go.
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.extract_title, node, title_path, sibling_nodes): (
                    node,
                    title_path,
                    sibling_nodes,
                )
                for node, title_path, sibling_nodes in zip(
                    nodes, title_paths, sibling_nodes_list
                )
            }
            for future in futures:
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    logger.error(
                        f"Error extracting KG for title node {futures[future][0].index_id}: {e}"
                    )
                    results.append(
                        {
                            "entities": [],
                            "relations": [],
                            "node_idx": futures[future][0].index_id,
                        }
                    )
        return results


if __name__ == "__main__":
    # Test
    from Core.configs.system_config import load_system_config

    config = load_system_config("/home/wangshu/multimodal/GBC-RAG/config/gbc.yaml")

    llm = LLM(config.llm)
    vlm = VLM(config.vlm)
    save_path = "/home/wangshu/multimodal/GBC-RAG/test/test_code"
    kg_extractor = KGExtractor(config.graph, llm, vlm, save_path=save_path)

    from Core.Index.Tree import DocumentTree

    tmp_path = "/mnt/data/wangshu/mmrag/MMLongBench-Doc/index/011bb5d3-e95e-5fd2-b9a6-0e0f980e7024"
    tree_index = DocumentTree.load_from_file(DocumentTree.get_save_path(tmp_path))

    # select node for testing
    # 1. one long text node with more than 1000 characters
    # 2. one table node
    # 3. one image node
    # 4. one equation node
    text_node = None
    image_node = None
    table_node = None
    equation_node = None
    title_node = None

    for node in tree_index.nodes:
        if (
            text_node == None
            and node.type == NodeType.TEXT
            and len(node.meta_info.content) > 1000
        ):
            print(
                f"Selected text node: {node.index_id} with content length {len(node.meta_info.content)}"
            )
            text_node = node
        if image_node == None and node.type == NodeType.IMAGE:
            print(
                f"Selected image node: {node.index_id} with image path {node.meta_info.img_path}"
            )
            image_node = node
        if table_node == None and node.type == NodeType.TABLE:
            print(
                f"Selected table node: {node.index_id} with caption '{node.meta_info.caption}'"
            )
            table_node = node
        if equation_node == None and node.type == NodeType.EQUATION:
            print(
                f"Selected equation node: {node.index_id} with content '{node.meta_info.content}'"
            )
            equation_node = node
        if title_node == None and node.type == NodeType.TITLE:
            print(
                f"Selected title node: {node.index_id} with content '{node.meta_info.content}'"
            )
            title_node = node

    def print_entity_name(res: dict):
        for ent in res.get("entities", []):
            print(
                f"Entity: {ent.entity_name}, Type: {ent.entity_type}, Description: {ent.description[:50]}..."
            )

    # text_kg = kg_extractor.extract_kg(text_node)
    # print("Extracted KG from text node:")
    # print(
    #     f"Extract {len(text_kg['entities'])} entities and {len(text_kg['relations'])} relations."
    # )
    # print_entity_name(text_kg)

    image_kg = kg_extractor.extract_kg(image_node)
    print("Extracted KG from image node:")
    print(
        f"Extract {len(image_kg['entities'])} entities and {len(image_kg['relations'])} relations."
    )
    print_entity_name(image_kg)

    # table_kg = kg_extractor.extract_kg(table_node)
    # print("Extracted KG from table node:")
    # print(
    #     f"Extract {len(table_kg['entities'])} entities and {len(table_kg['relations'])} relations."
    # )
    # print_entity_name(table_kg)

    # equation_kg = kg_extractor.extract_kg(equation_node)
    # print("Extracted KG from equation node:")
    # print(
    #     f"Extract {len(equation_kg['entities'])} entities and {len(equation_kg['relations'])} relations."
    # )
    # print_entity_name(equation_kg)

    # title_kg = kg_extractor.extract_title(
    #     title_node,
    #     tree_index.get_path_from_root(title_node.index_id),
    #     tree_index.get_sibling_nodes(title_node.index_id),
    # )
    # print("Extracted KG from title node:")
    # print(
    #     f"Extract {len(title_kg['entities'])} entities and {len(title_kg['relations'])} relations."
    # )
    # print_entity_name(title_kg)
