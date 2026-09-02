"""
crear/recrear la colección;
insertar puntos;
buscar puntos.
"""
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams
)

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "documentos_prueba"

VECTOR_SIZE = 384


class QdrantStore:

    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

    def recreate_collection(self):

        if self.client.collection_exists(COLLECTION_NAME):
            print(f"Eliminando colección: {COLLECTION_NAME}")

            self.client.delete_collection(
                collection_name=COLLECTION_NAME
            )

        print(f"Creando colección: {COLLECTION_NAME}")

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

    def add(
        self,
        point_id: str,
        filename: str,
        chunk_id: int,
        content: str,
        embedding
    ):
        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "filename": filename,
                "chunk_id": chunk_id,
                "content": content
            }
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

    def search(
        self,
        query_embedding,
        top_k: int = 3
    ):
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