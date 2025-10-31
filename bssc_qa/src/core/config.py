"""Configuration management for BSSC_QA framework."""
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class LLMProviderConfig(BaseModel):
    api_key: Optional[str] = None
    model: str
    temperature: float = 0.7
    env_var: Optional[str] = None

    def resolve_api_key(self, provider_name: str) -> str:
        """Resolve API key from environment or config while warning on insecure usage."""
        env_var = self.env_var or f"BSSC_QA_{provider_name.upper()}_API_KEY"
        env_value = os.getenv(env_var)
        if env_value:
            return env_value

        config_value = (self.api_key or "").strip()
        if config_value and config_value.lower() != "enabled in environment.":
            warnings.warn(
                f"API key for provider '{provider_name}' was loaded from config. "
                f"Please migrate it to the environment variable '{env_var}'.",
                stacklevel=2,
            )
            os.environ[env_var] = config_value
            return config_value

        raise RuntimeError(
            f"API key for provider '{provider_name}' not found. "
            f"Set the environment variable '{env_var}' to proceed."
        )

class LLMConfig(BaseModel):
    default_provider: str
    providers: Dict[str, LLMProviderConfig]

class VectorStoreConfig(BaseModel):
    type: str
    persist_directory: str
    collection_name: str
    embedding_model: str
    embedding_dimension: Optional[int] = None

class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    auto_adjust: bool = True

class AgentsConfig(BaseModel):
    planner: Dict[str, Any]
    generator: Dict[str, Any]
    synthesis: Dict[str, Any]
    evaluator: Dict[str, Any]

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Allow dict-style access for compatibility with existing code."""
        return getattr(self, key)

class Config(BaseModel):
    llm: LLMConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    agents: AgentsConfig
    bloom_level: Dict[str, Any]
    human_review: Dict[str, Any]
    export: Dict[str, Any]

def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from JSON file."""
    if config_path is None:
        config_path = Path.cwd() / 'config.json'
    
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    return Config(**config_data)
