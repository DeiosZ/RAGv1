import numpy as np
from similarity import similitud_coseno


class VectorStore:

    def __init__(self):
        self.documents = []

    def add(self, filename: str, chunk_id: int, content: str, embedding):
        self.documents.append(
            {
                "filename": filename,
                "chunk_id": chunk_id,
                "content": content,
                "embedding": np.array(embedding, dtype=np.float32),
            }
        )

    def search(self, query_embedding, top_k: int = 3) -> list[dict]:
        if not self.documents:
            return []

        matrix = np.array([doc["embedding"] for doc in self.documents])
        query = np.array(query_embedding, dtype=np.float32)

        query_norm = np.linalg.norm(query)
        matrix_norms = np.linalg.norm(matrix, axis=1)
        denom = matrix_norms * query_norm
        denom[denom == 0] = 1e-10

        scores = np.dot(matrix, query) / denom

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "filename": doc["filename"],
                    "chunk_id": doc["chunk_id"],
                    "content": doc["content"],
                    "score": float(scores[idx]),
                }
            )

        return results