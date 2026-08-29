from loader import cargar_documents
from chunker import create_chunks
from embeddings import generate_embedding
from vector_store import VectorStore


DOCUMENTS_PATH = "data/documents"


def build_vector_store():

    documents = cargar_documents(DOCUMENTS_PATH)

    vector_store = VectorStore()

    for document in documents:

        chunks = create_chunks(
            document["content"],
            chunk_size=100,
            overlap=20
        )

        for index, chunk in enumerate(chunks):

            embedding = generate_embedding(chunk)

            vector_store.add(
                filename=document["filename"],
                chunk_id=index,
                content=chunk,
                embedding=embedding
            )

    return vector_store


def print_results(results):

    print()
    print("=" * 50)
    print("RESULTADOS")
    print("=" * 50)

    for index, result in enumerate(results, start=1):

        print()
        print(f"[{index}] Score: {result['score']:.4f}")

        print(f"Documento: {result['filename']}")

        print(f"Chunk: {result['chunk_id']}")

        print()

        print(f"\"{result['content']}\"")

        print()
        print("-" * 50)


def main():

    print("Construyendo base vectorial...")

    vector_store = build_vector_store()

    print()
    print("Base vectorial creada.")
    print(
        f"Chunks almacenados: "
        f"{len(vector_store.documents)}"
    )

    while True:

        print()
        print("=" * 50)
        print("CONSULTA")
        print("=" * 50)

        query = input(
            "\nEscribe tu pregunta "
            "(o 'salir'): "
        )

        if query.lower() == "salir":
            break

        query_embedding = generate_embedding(query)

        results = vector_store.search(
            query_embedding,
            top_k=3
        )

        print_results(results)

        print()
        print("=" * 50)
        print("CONTEXTO RECUPERADO")
        print("=" * 50)

        for result in results:

            print()
            print(result["content"])


if __name__ == "__main__":
    main()