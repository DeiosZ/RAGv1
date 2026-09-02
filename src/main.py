from embeddings import generate_embedding
from qdrant_store import QdrantStore

def print_results(results):

    print()

    print("=" * 50)

    print("RESULTADOS")

    print("=" * 50)

    if not results:

        print(
            "No se encontraron resultados."
        )

        return

    for index, result in enumerate(
        results,
        start=1
    ):

        print()

        print(
            f"[{index}] "
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Documento: "
            f"{result['filename']}"
        )

        print(
            f"Chunk: "
            f"{result['chunk_id']}"
        )

        print()

        print(
            f"\"{result['content']}\""
        )

        print()

        print("-" * 50)



def main():

    print("=" * 50)

    print("RAG + QDRANT")

    print("=" * 50)

    print()

    qdrant_store = QdrantStore()

    while True:

        query = input(
            "Escribe tu pregunta "
            "(o 'salir'): "
        )

        if query.lower() == "salir":

            print(
                "Finalizando..."
            )

            break

        if not query.strip():

            continue


        query_embedding = (
            generate_embedding(query)
        )


        results = qdrant_store.search(

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

            print(
                result["content"]
            )


if __name__ == "__main__":

    main()