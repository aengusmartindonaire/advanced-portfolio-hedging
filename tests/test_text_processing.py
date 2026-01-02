"""
tests/test_text_processing.py
"""
from adv_hedging.nlp.text_processing import chunk_text_with_metadata, create_metadata_header

def test_metadata_persistence(mock_wiki_row):
    """Ensure metadata appears in the second chunk."""
    header = create_metadata_header(mock_wiki_row)
    content = mock_wiki_row['content'] # 1000 words
    
    # Chunk with small size to force multiple chunks
    chunks = chunk_text_with_metadata(content, header, chunk_size=200, overlap=0)
    
    assert len(chunks) >= 5
    
    # Check the SECOND chunk
    chunk_2 = chunks[1]
    
    # It should start with the header
    assert chunk_2.startswith("Title: Test Company")
    assert "Sector: Technology" in chunk_2