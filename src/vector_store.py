from similarity import cosine_similarity


class VectorStore:

    def __init__(self):
        self.documents = []


    def add(
        self,
        filename,
        chunk_id,
        content,
        embedding
    ):

        self.documents.append({
            "filename": filename,
            "chunk_id": chunk_id,
            "content": content,
            "embedding": embedding
        })


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        results = []

        for document in self.documents:

            score = cosine_similarity(
                query_embedding,
                document["embedding"]
            )

            results.append({
                "filename": document["filename"],
                "chunk_id": document["chunk_id"],
                "content": document["content"],
                "score": score
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]