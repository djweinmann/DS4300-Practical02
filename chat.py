import ollama
from utils.parse_args import get_database, get_model, get_verbose, get_query


def search_embeddings(client, query, top_k=3, verbose=False):
    try:
        top_results = client.retreive(query)[:top_k]

        if verbose:
            for result in top_results:
                print(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
                )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results, model, verbose=False):
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    if verbose:
        print(f"context_str: {context_str}")

    prompt = f"""You are an uptight rizzer rapper by the name drippidy d. You also happen to
    be a computer science pro that knowns all you know ya know. Use the following context to answer the query as accurately as possible
    while bringing in the drip. If the context is not relevant to the query, say 'I don't know'.

    You should respond in the tone of the true jayz or emenem to honor your rapping legacy.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=f"{model}:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search(model, db, verbose=False):
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(db, query, verbose=verbose)

        # Generate RAG response
        response = generate_rag_response(query, context_results, model, verbose)

        print("\n--- Response ---")
        print(response)


def main():
    db = get_database()
    model = get_model()
    verbose = get_verbose()

    # If provided, process and return the provided query
    query = get_query()
    if query:
        context_results = search_embeddings(db, query)
        response = generate_rag_response(query, context_results, model)
        print(response)
        return

    interactive_search(model, db, verbose)


if __name__ == "__main__":
    main()
