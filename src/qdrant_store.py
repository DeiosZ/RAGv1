import os
import time

from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (PointStruct, Filter, FieldCondition, MatchValue)


load_dotenv()


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "documentos_prueba"

VECTOR_SIZE = 384


class QdrantStore:

    def __init__(self):

        if not QDRANT_URL:
            raise ValueError(
                "No se encontró QDRANT_URL en el archivo .env"
            )

        if not QDRANT_API_KEY:
            raise ValueError(
                "No se encontró QDRANT_API_KEY en el archivo .env"
            )

        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )


    def ensure_collection(self):

        collections = self.client.get_collections()

        exists = any(
            collection.name == COLLECTION_NAME
            for collection in collections.collections
        )

        if exists:
            return

        from qdrant_client.models import VectorParams, Distance

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print(
            f"Colección '{COLLECTION_NAME}' creada correctamente."
        )


    def recreate_collection(self):

        from qdrant_client.models import VectorParams, Distance

        collections = self.client.get_collections()

        exists = any(
            collection.name == COLLECTION_NAME
            for collection in collections.collections
        )

        if exists:

            print(
                f"Eliminando colección '{COLLECTION_NAME}'..."
            )

            self.client.delete_collection(
                collection_name=COLLECTION_NAME
            )

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )

        print(
            f"Colección '{COLLECTION_NAME}' recreada correctamente."
        )


    def add(
        self,
        point_id: str,
        document_id: str,
        filename: str,
        file_hash: str,
        chunk_id: int,
        content: str,
        embedding
    ):

        point = PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "document_id": document_id,
                "filename": filename,
                "file_hash": file_hash,
                "chunk_id": chunk_id,
                "content": content
            }
        )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )



    def add_batch(
        self,
        points: list[PointStruct],
        batch_size: int = 64,
        max_retries: int = 3
    ):
        

        if not points:
            return

        total = len(points)

        total_batches = (
            total + batch_size - 1
        ) // batch_size

        print(
            f"Insertando {total} puntos "
            f"en {total_batches} batches..."
        )

        for batch_number, start in enumerate(
            range(0, total, batch_size),
            start=1
        ):

            batch = points[
                start:start + batch_size
            ]

            print(
                f"  → Batch "
                f"{batch_number}/{total_batches} "
                f"({len(batch)} puntos)"
            )

            self._upsert_with_retry(
                batch=batch,
                max_retries=max_retries
            )


    def _upsert_with_retry(
        self,
        batch: list[PointStruct],
        max_retries: int = 3
    ):

        for attempt in range(
            1,
            max_retries + 1
        ):

            try:

                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )

                return

            except Exception as error:

                print(
                    f" Error en Qdrant "
                    f"(intento {attempt}/{max_retries})"
                )

                print(
                    f"    {type(error).__name__}: {error}"
                )

                if attempt == max_retries:

                    raise

                wait_time = 2 ** attempt

                print(
                    f"    Reintentando en "
                    f"{wait_time} segundos..."
                )

                time.sleep(wait_time)


    def get_documents_metadata(self):

        documents = {}

        offset = None

        while True:

            points, next_offset = self.client.scroll(
                collection_name=COLLECTION_NAME,
                offset=offset,
                limit=100,
                with_payload=[
                    "document_id",
                    "filename",
                    "file_hash"
                ],
                with_vectors=False
            )

            for point in points:

                payload = point.payload or {}

                document_id = payload.get(
                    "document_id"
                )

                filename = payload.get(
                    "filename"
                )

                file_hash = payload.get(
                    "file_hash"
                )

                if not document_id or not filename:
                    continue

                documents[filename] = {
                    "document_id": document_id,
                    "filename": filename,
                    "file_hash": file_hash
                }

            if next_offset is None:
                break

            offset = next_offset

        return documents



    def delete_document(
        self,
        document_id: str
    ):

        print(
            f"Eliminando documento "
            f"{document_id}..."
        )

        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        )
                    )
                ]
            )
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

            payload = result.payload or {}

            results.append({
                "filename": payload.get(
                    "filename",
                    "desconocido"
                ),

                "chunk_id": payload.get(
                    "chunk_id",
                    -1
                ),

                "content": payload.get(
                    "content",
                    ""
                ),

                "score": float(
                    result.score
                )
            })

        return results
