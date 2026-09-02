import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "documentos_prueba"


class QdrantStore:

    def __init__(self):

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

    def add(self,point_id: int,filename: str,chunk_id: int,content: str,embedding):

        point = PointStruct(id=point_id,vector=embedding.tolist(),
            payload={
                "filename": filename,
                "chunk_id": chunk_id,
                "content": content
            }
        )

        self.client.upsert(collection_name=COLLECTION_NAME,points=[point]
        )

    def search(self,query_embedding,top_k: int = 3):

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding.tolist(),
            limit=top_k
        )

        results = []

        for result in response.points:

            results.append({
                "filename": result.payload["filename"],
                "chunk_id": result.payload["chunk_id"],
                "content": result.payload["content"],
                "score": float(result.score)
            })

        return results

if __name__ == "__main__":

    from embeddings import generate_embedding

    store = QdrantStore()

    texto = "La teoría de grafos estudia nodos y aristas."

    embedding = generate_embedding(texto)

    store.add(point_id=1,filename="prueba.txt",chunk_id=0,content=texto,embedding=embedding)

    print("Embedding insertado correctamente.")