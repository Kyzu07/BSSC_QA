"""Intelligent text chunking for BSSC_QA framework."""
from typing import List, Dict, Any
from dataclasses import dataclass
import uuid

@dataclass
class Chunk:
    chunk_id: str
    text: str
    tokens: int
    position: int
    metadata: Dict[str, Any]

def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ≈ 4 chars)."""
    return len(text) // 4

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50,
               metadata: Dict[str, Any] = None) -> List[Chunk]:
    """Split text into overlapping chunks based on token size."""
    
    if metadata is None:
        metadata = {}
    
    chunks = []
    min_chunk_chars = 200
    
    # Convert token size to character estimate
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    
    # Split into chunks
    start = 0
    position = 0
    
    while start < len(text):
        # Get chunk - use different variable name to avoid collision
        end = start + char_size
        current_chunk = text[start:end]
        raw_chunk_len = len(current_chunk)
        
        # Store original length before any modifications
        original_chunk_len = raw_chunk_len

        # Try to break at sentence boundary if possible
        if end < len(text) and len(current_chunk) > 0:
            # Look for sentence ending in last 20% of chunk
            search_start = max(0, len(current_chunk) - int(char_size * 0.2))
            for delimiter in ['. ', '! ', '? ', '।। ', '।।', '। ', '।', '\n\n']:
                last_delim = current_chunk.rfind(delimiter, search_start)
                if last_delim != -1:
                    current_chunk = current_chunk[:last_delim + len(delimiter)]
                    original_chunk_len = len(current_chunk)
                    break
        
        # Strip whitespace for storage
        stripped_chunk = current_chunk.strip()
        
        # Only create chunk if there's actual content
        if stripped_chunk:
            if len(stripped_chunk) < min_chunk_chars and chunks:
                # Merge tiny tail segments into the previous chunk for better context
                previous_chunk = chunks[-1]
                merged_text = f"{previous_chunk.text} {stripped_chunk}".strip()
                previous_chunk.text = merged_text
                previous_chunk.tokens = estimate_tokens(merged_text)
            else:
                chunk = Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=stripped_chunk,
                    tokens=estimate_tokens(stripped_chunk),
                    position=position,
                    metadata=metadata.copy()
                )
                chunks.append(chunk)
                position += 1
        
        # If we've consumed the document, stop processing to avoid zero-length loops
        if end >= len(text):
            break
        
        # Move to next chunk with overlap using original length
        # Ensure we always move forward to prevent infinite loop
        move_amount = max(raw_chunk_len - char_overlap, 1)
        start = start + move_amount
        
        # Safety check: if we're not making progress, break
        if move_amount == 0 or raw_chunk_len == 0:
            break
    
    return chunks
