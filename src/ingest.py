from uuid import uuid4

from loader import cargar_documents
from chunker import create_chunks
from embeddings import generate_embedding
from qdrant_store import QdrantStore


DOCUMENTS_PATH = "data/documents"


def ingest_documents():

    print("=" * 60)
    print("INGESTA DE DOCUMENTOS")
    print("=" * 60)


    qdrant_store = QdrantStore()


    print()
    print("Preparando colección...")

    qdrant_store.recreate_collection()


    print()
    print("Cargando documentos...")

    documents = cargar_documents(DOCUMENTS_PATH)

    print(f"Documentos encontrados: {len(documents)}")


    total_chunks = 0

    for document in documents:

        filename = document["filename"]
        content = document["content"]

        print()
        print(f"Documento: {filename}")

        chunks = create_chunks(content, chunk_size=100,overlap=20)

        print(f"Chunks generados: {len(chunks)}")

        for chunk_id, chunk in enumerate(chunks):

            print(
                f"  => Procesando chunk {chunk_id}"
            )

            embedding = generate_embedding(chunk)

            point_id = str(uuid4())


            qdrant_store.add(point_id=point_id, filename=filename, chunk_id=chunk_id, content=chunk, embedding=embedding)

            total_chunks += 1

    print()
    print("=" * 60)
    print("INGESTA FINALIZADA")
    print("=" * 60)

    print(f"Documentos procesados: {len(documents)}")
    print(f"Chunks almacenados: {total_chunks}")


if __name__ == "__main__":
    ingest_documents()