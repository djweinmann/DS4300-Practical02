import ollama
import argparse

from utils.parse_args import get_database


VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


def search_embeddings(client, query, top_k=3):
    try:
        top_results = client.retreive(query)[:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, model):
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama
    response = ollama.chat(
        model=f"{model}:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search(embedder, model="mistral"):
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    db = get_database()

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(db, query)

        # Generate RAG response
        response = generate_rag_response(query, context_results, model)

        print("\n--- Response ---")
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameters to search")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="LLM Model to use",
    )

    args = parser.parse_args()

    model = args.model

    interactive_search(model)
