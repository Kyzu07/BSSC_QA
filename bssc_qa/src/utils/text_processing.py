"""Text processing utilities for BSSC_QA framework."""
import re
from typing import List

def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and standardizing format."""
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove multiple newlines (keep max 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def clean_text(text: str) -> str:
    """Clean text by removing special characters while preserving structure."""
    # Remove control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text

def count_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
    return len(text) // 4

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting (works for most cases)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
