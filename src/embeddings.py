from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str):
    """
    Genera el embedding de un único texto.
    Se mantiene para las consultas individuales.
    """
    embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    return embedding


def generate_embeddings(
    texts: list[str],
    batch_size: int = 32
):
  

    if not texts:
        return []

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    return embeddings