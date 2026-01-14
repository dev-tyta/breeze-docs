import logging
from typing import List

logger = logging.getLogger(__name__)

def smart_chunk(text: str, max_chars: int = 15000, overlap: int = 500) -> List[str]:
    """
    Splits text into chunks respecting line boundaries and ensuring overlap.
    Ideally used when the source text is too large for the LLM context window.

    Args:
        text: The input text to split.
        max_chars: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks (approximate).

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Simple line-based chunking
    # A improved version would look for specific delimiters like "def " or "class "
    # if it were language aware, but this is a general text utility.
    
    for line in lines:
        line_len = len(line) + 1 # +1 for newline character
        
        if current_length + line_len > max_chars:
            # Current chunk is full
            chunk_text = '\n'.join(current_chunk)
            chunks.append(chunk_text)
            
            # Handle overlap for the next chunk
            # Keep the last few lines that sum up to ~overlap chars
            overlap_buffer = []
            overlap_len = 0
            for k in range(len(current_chunk) - 1, -1, -1):
                prev_line = current_chunk[k]
                if overlap_len + len(prev_line) > overlap:
                    break
                overlap_buffer.insert(0, prev_line)
                overlap_len += len(prev_line) + 1
            
            current_chunk = overlap_buffer
            current_length = overlap_len
        
        current_chunk.append(line)
        current_length += line_len
        
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
        
    return chunks
