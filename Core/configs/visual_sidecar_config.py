from dataclasses import dataclass
from typing import Optional


@dataclass
class VisualSidecarConfig:
    backend_type: str = "prepared_corpus"
    retriever_family: str = "colqwen2"
    retriever_model: Optional[str] = None
    retriever_version: str = "v1"
    force_rebuild: bool = True
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: Optional[str] = "auto"
    image_batch_size: int = 4