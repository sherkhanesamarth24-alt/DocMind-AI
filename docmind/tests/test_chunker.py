from app.ingestion.chunker import chunk_pages


def test_chunks_stay_on_their_page():
    text = " ".join(f"This is sentence number {i} about the topic." for i in range(200))
    chunks = chunk_pages([(3, text)])
    assert len(chunks) > 1
    assert all(c["page"] == 3 for c in chunks)
