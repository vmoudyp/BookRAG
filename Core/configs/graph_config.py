from dataclasses import dataclass, field

from Core.configs.embedding_config import EmbeddingConfig
from Core.configs.rerank_config import RerankerConfig


@dataclass
class GraphConfig:
    # KG extraction
    extractor_type: str = "llm"  # Options: "llm", "local", "hybrid"
    local_model_name: str = "en_core_web_sm"
    # Hugging Face model ID or existing local model directory for Indonesian NER.
    hybrid_ner_model: str = "cahya/bert-base-indonesian-NER"
    # Minimum confidence score (0–1) for a BERT NER span to be kept.
    ner_confidence_threshold: float = 0.5
    image_description_force: bool = False
    max_gleaning: int = 0
    text_extraction_scope: str = "body_text_leaves"  # Options: "body_text_leaves", legacy fallbacks

    # KG refinement
    refine_type: str = "advanced"  # Options: "basic", "advanced"

    # Domain role extraction and graph materialization
    role_extraction_enabled: bool = False  # Run LLM role-extraction pass during ingestion
    role_graph_materialization: bool = False  # Write HAS_ROLE edges to FalkorDB (Phase 2)

    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranker_config: RerankerConfig = field(default_factory=RerankerConfig)
