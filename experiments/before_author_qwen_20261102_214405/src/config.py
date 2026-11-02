from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class DatasetConfig:
    name: str = "MultiTQ"
    kg_path: str = "./datasets/MultiTQ/kg/full.txt"
    qa_path: str = "./datasets/MultiTQ/questions/test.json"
    entity_path: str = "./datasets/MultiTQ/kg/entity2id.json"
    relation_path: str = "./datasets/MultiTQ/kg/relation2id.json"
    text_index_path: str = "./artifacts/multitq_text_index.pkl"
    graph_cache_path: str = "./artifacts/multitq_graph_retriever.pkl"
    graph_pattern_prompt: str = "./datasets/MultiTQ/prompt/gp_prompt.txt"
    qa_prompt: str = "./datasets/MultiTQ/prompt/qa_prompt.txt"


@dataclass
class ModelConfig:
    text_retriever_model_name: str = "/path/to/bge-m3"
    device: str = "cuda:0"
    text_top_k: int = 100
    qa_retrieve_cnt: int = 50
    concurrency: int = 64
    random_seed: int = 42
    sample_size: int | None = None


@dataclass
class LLMConfig:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    cache_path: str = "./artifacts/llm_cache.jsonl"


def ensure_parent_dir(path: str) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)