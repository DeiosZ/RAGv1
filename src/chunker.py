def create_chunks(
    text: str, chunk_size: int = 500, overlap: int = 100
) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("El 'overlap' debe ser menor que el 'chunk_size'.")

    palabras = text.split()
    chunks = []
    start = 0

    while start < len(palabras):
        end = start + chunk_size
        chunk_palabras = palabras[start:end]

        if chunk_palabras:
            chunks.append(" ".join(chunk_palabras))

        if end >= len(palabras):
            break
        start += chunk_size - overlap

    return chunks