import hashlib
from uuid import uuid4
from pathlib import Path
from loader import cargar_documents
from chunker import create_chunks
from embeddings import generate_embeddings

from qdrant_store import QdrantStore
from qdrant_client.models import PointStruct


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIRECTORY = BASE_DIR / "data" / "documents"

CHUNK_SIZE = 100
CHUNK_OVERLAP = 20

EMBEDDING_BATCH_SIZE = 32

QDRANT_BATCH_SIZE = 64


def calcular_hash(content: str) -> str:

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()



def insertar_documento(
    qdrant_store: QdrantStore,
    document: dict
):

    filename = document["filename"]
    content = document["content"]

    print(
        f"\n[INSERT] {filename}"
    )

    file_hash = calcular_hash(content)

    document_id = str(uuid4())

    chunks = create_chunks(
        content,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(
        f"Chunks generados: {len(chunks)}"
    )

    if not chunks:

        print(
            "Documento sin contenido."
        )

        return 0

    print(
        f"Generando embeddings "
        f"en batches de {EMBEDDING_BATCH_SIZE}..."
    )

    embeddings = generate_embeddings(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE
    )

    points = []

    for chunk_id, (
        chunk,
        embedding
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        point = PointStruct(

            id=str(uuid4()),

            vector=embedding.tolist(),

            payload={

                "document_id": document_id,

                "filename": filename,

                "file_hash": file_hash,

                "chunk_id": chunk_id,

                "content": chunk
            }
        )

        points.append(point)

    qdrant_store.add_batch(
        points=points,
        batch_size=QDRANT_BATCH_SIZE
    )

    print(
        f"✓ Documento insertado: "
        f"{len(points)} chunks"
    )

    return len(points)



def actualizar_documento(
    qdrant_store: QdrantStore,
    document: dict,
    document_id: str
):

    filename = document["filename"]
    content = document["content"]

    print(
        f"\n[UPDATE] {filename}"
    )

    print(
        "Eliminando chunks anteriores..."
    )


    qdrant_store.delete_document(
        document_id
    )

    file_hash = calcular_hash(content)


    chunks = create_chunks(
        content,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    print(
        f"Chunks generados: {len(chunks)}"
    )

    if not chunks:

        print(
            "Documento sin contenido."
        )

        return 0

    print(
        f"Generando embeddings "
        f"en batches de {EMBEDDING_BATCH_SIZE}..."
    )

    embeddings = generate_embeddings(
        chunks,
        batch_size=EMBEDDING_BATCH_SIZE
    )

    points = []

    for chunk_id, (
        chunk,
        embedding
    ) in enumerate(
        zip(chunks, embeddings)
    ):

        point = PointStruct(

            id=str(uuid4()),

            vector=embedding.tolist(),

            payload={

                "document_id": document_id,

                "filename": filename,

                "file_hash": file_hash,

                "chunk_id": chunk_id,

                "content": chunk
            }
        )

        points.append(point)

 

    qdrant_store.add_batch(
        points=points,
        batch_size=QDRANT_BATCH_SIZE
    )

    print(
        f"✓ Documento actualizado: "
        f"{len(points)} chunks"
    )

    return len(points)


def ingest_documents():

    print("=" * 60)
    print("INGESTA INCREMENTAL")
    print("=" * 60)


    qdrant_store = QdrantStore()

    qdrant_store.ensure_collection()



    print("\nLeyendo documentos locales...")

    documents = cargar_documents(
        DOCUMENTS_DIRECTORY
    )

    print(
        f"Documentos locales: "
        f"{len(documents)}"
    )


    print(
        "\nConsultando documentos "
        "existentes en Qdrant..."
    )

    existing_documents = (
        qdrant_store.get_documents_metadata()
    )

    print(
        f"Documentos en Qdrant: "
        f"{len(existing_documents)}"
    )


    inserted = 0
    updated = 0
    skipped = 0
    deleted = 0

    local_filenames = set()


    for document in documents:

        filename = document["filename"]

        local_filenames.add(filename)

        content = document["content"]

        file_hash = calcular_hash(
            content
        )

 

        if filename not in existing_documents:

            chunks = insertar_documento(
                qdrant_store,
                document
            )

            inserted += 1

            continue


        existing = existing_documents[
            filename
        ]

        existing_hash = existing[
            "file_hash"
        ]



        if existing_hash == file_hash:

            print(
                f"\n[SKIP] {filename}"
            )

            print(
                "Sin cambios."
            )

            skipped += 1

            continue


        actualizar_documento(
            qdrant_store,
            document,
            existing["document_id"]
        )

        updated += 1



    print(
        "\nComprobando documentos eliminados..."
    )

    for filename, existing in (
        existing_documents.items()
    ):

        if filename not in local_filenames:

            print(
                f"\n[DELETE] {filename}"
            )

            qdrant_store.delete_document(
                existing["document_id"]
            )

            deleted += 1


    print("\n" + "=" * 60)
    print("RESUMEN DE INGESTA")
    print("=" * 60)

    print(
        f"Insertados : {inserted}"
    )

    print(
        f"Actualizados: {updated}"
    )

    print(
        f"Sin cambios : {skipped}"
    )

    print(
        f"Eliminados  : {deleted}"
    )

    print("=" * 60)



if __name__ == "__main__":

    ingest_documents()