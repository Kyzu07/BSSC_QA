"""LLM provider factory for BSSC_QA framework."""
from typing import Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint

def create_llm(provider: str, api_key: str, model: str, temperature: float = 0.7) -> Any:
    """Create LLM instance based on provider."""
    
    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature
        )
    
    elif provider in ["deepseek", "mistral"]:
        # Both use OpenAI-compatible API
        base_url = {
            "deepseek": "https://api.deepseek.com/v1",
            "mistral": "https://api.mistral.ai/v1"
        }[provider]
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
    
    elif provider == "huggingface":
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            temperature=temperature
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
