"""
src/adv_hedging/nlp/text_processing.py
Text manipulation utilities, specifically for context-aware chunking.
"""
from typing import List, Dict

def create_metadata_header(row: Dict) -> str:
    """
    Creates the metadata string to prepend to every chunk.
    Example: "Title: Apple Inc.\nURL: https://wiki.../Apple\nSector: Tech\n"
    """
    # Adjust fields based on what columns you actually have
    title = row.get('title', 'Unknown')
    url = row.get('URL', 'Unknown')
    sector = row.get('sector', '')
    
    return f"Title: {title}\nURL: {url}\nSector: {sector}\nContent:\n"

def chunk_text_with_metadata(
    text: str, 
    metadata_header: str, 
    chunk_size: int = 500, 
    overlap: int = 100
) -> List[str]:
    """
    Splits text into chunks and prepends metadata to EACH chunk.
    
    Args:
        text: The full Wikipedia content.
        metadata_header: The string to put at the start of each chunk.
        chunk_size: Number of words per chunk.
        overlap: Number of words to overlap between chunks.
    """
    if not text or not isinstance(text, str):
        return []

    words = text.split()
    chunks = []
    
    # Iterate through words with a sliding window
    stride = chunk_size - overlap
    if stride < 1:
        stride = 1
        
    for i in range(0, len(words), stride):
        # Slice the words
        segment_words = words[i : i + chunk_size]
        segment_text = " ".join(segment_words)
        
        # Combine Header + Content
        # This solves the "Context Loss" problem described in the notebook
        full_chunk = f"{metadata_header}{segment_text}"
        chunks.append(full_chunk)
        
        # Break if we've reached the end
        if i + chunk_size >= len(words):
            break
            
    return chunks

def prepare_corpus_for_embedding(df) -> List[str]:
    """
    Takes the main DataFrame and returns a flat list of ALL chunks 
    ready for the embedding model.
    """
    all_chunks = []
    
    # Iterate over every company in the dataframe
    for _, row in df.iterrows():
        meta = create_metadata_header(row)
        content = row.get('content', '')
        
        # Generate chunks for this company
        company_chunks = chunk_text_with_metadata(
            content, 
            meta, 
            chunk_size=500, # Configurable, but 500 is a good default
            overlap=50
        )
        all_chunks.extend(company_chunks)
        
    return all_chunks