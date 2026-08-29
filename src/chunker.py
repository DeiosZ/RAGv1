def create_chunks(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100
):

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks