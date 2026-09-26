"""Central application configuration."""

from functools import lru_cache
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str


class ChunkingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_size_tokens: int = Field(gt=0)
    chunk_overlap_pct: float = Field(ge=0, lt=1)
    separators: list[str]


class PathsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_dir: Path
    processed_dir: Path
    vectorstore_dir: Path


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    min_similarity_threshold: float = Field(ge=0)


class RerankerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: EmbeddingConfig
    generation: GenerationConfig
    chunking: ChunkingConfig
    paths: PathsConfig
    retrieval: RetrievalConfig
    reranker: RerankerConfig


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and validate config.yaml once."""

    config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path.resolve()}"
        )

    with config_path.open("r",encoding="utf-8",) as file:
        raw_config = yaml.safe_load(file)

    return AppConfig.model_validate(raw_config)