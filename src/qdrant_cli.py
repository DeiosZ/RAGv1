import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
print("URL:", QDRANT_URL)
print("API KEY:", "OK" if QDRANT_API_KEY else "NO ENCONTRADA")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print(client.get_collections())