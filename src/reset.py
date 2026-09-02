from qdrant_store import QdrantStore


store = QdrantStore()

store.recreate_collection()

print(
    "Colección reiniciada correctamente."
)