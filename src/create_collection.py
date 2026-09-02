import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

COLLECTION_NAME = "documentos_prueba"

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
    ),
)

print(f"Colección '{COLLECTION_NAME}' creada correctamente.")