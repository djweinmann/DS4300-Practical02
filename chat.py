from dbs.database import VDatabase
from utils.parse_args import get_database, get_model, get_verbose, get_query
import asyncio
from ollama import AsyncClient, generate

SYSTEM_MSG = """\
You are an uptight rizzer rapper by the name drippidy d. You also happen to \
be a computer science pro that knowns all you know ya know. Use the following context to answer the query as accurately as possible \
while bringing in the drip. If the context is not relevant to the query, say 'I don't know'.

You should respond in the tone of the true jayz or emenem to honor your rapping legacy.
"""

HELP_MSG = """\
Available commands:
    :exit      Exit the chat
    :clear     Clear the chatlog
    :help      Shows this help page
"""


def generate_prompt(query, ctx):
    return f"""\
Only respond to the query with the provided context or the chat history.

<Context>
{ctx}
</Context>

<Query>
{query}
</Query>
"""


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


def generate_rag_response(context_results, verbose=False):
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

    return context_str


async def chat(model: str, chatlog: list) -> None:
    """
    send the query to the LLM and stream in the response
    :model: name of the model to use
    :chatlog: chatlog for the current conversation
    """
    async for part in await AsyncClient().chat(
        model=f"{model}:latest", messages=chatlog, stream=True
    ):
        print(part["message"]["content"], end="", flush=True)


def interactive_chat(model: str, db: VDatabase, verbose=False) -> None:
    """
    start an interactive chat session
    :model: name of the model to use
    :db: vector database to use
    :verbose: enable verbose logging
    """
    print("🔍 RAG Search Interface")
    print("Type ':help' for ")

    chatlog = [{"role": "system", "content": SYSTEM_MSG}]

    while True:
        query = input("\n>>> ")

        if query[0] == ":":
            match query.lower():
                case ":help":
                    print(HELP_MSG)
                case ":clear":
                    chatlog = [{"role": "system", "content": SYSTEM_MSG}]
                    print("Chatlog cleared")
                case ":exit":
                    break
                case _:
                    print("unknown command " + query)
            continue

        # Search for relevant embeddings
        context_results = search_embeddings(db, query, verbose=verbose)

        # Generate RAG response
        ctx_str = generate_rag_response(context_results, verbose)

        chatlog.append({"role": "user", "content": generate_prompt(query, ctx_str)})
        asyncio.run(chat(model, chatlog))


def main():
    db = get_database()
    model = get_model()
    verbose = get_verbose()

    # If provided, process and return the provided query
    query = get_query()
    if query:
        ctx_response = search_embeddings(db, query)
        ctx = generate_rag_response(ctx_response)
        res = generate(
            model=f"{model}:latest",
            system=SYSTEM_MSG,
            prompt=generate_prompt(query, ctx),
        )
        print(res["response"])
        return

    # Otherwise start up an interactive session
    interactive_chat(model, db, verbose)


if __name__ == "__main__":
    main()
