""" """

from dbs.database import VDatabase
from utils.parse_args import get_database, get_model, get_verbose, get_prompt
import asyncio
from ollama import AsyncClient, generate
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit import prompt

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


def chat_input(prompt_text: str) -> str:
    """
    a wrapped and configured input element which has additional features compared to
    the native python implementation
    :param prompt_text: text to prompt the user with for input
    :returns: the user inputted text
    """
    prompt_style = Style.from_dict(
        {
            "prompt": "ansipurple bold",
            "input": "",
        }
    )

    placeholder = HTML(
        "<ansibrightblack>send message (:help for help)</ansibrightblack>"
    )

    return prompt(
        prompt_text,
        placeholder=placeholder,
        default="",
        style=prompt_style,
        mouse_support=True,
        cursor=CursorShape.BLINKING_BEAM,
    ).lower()


def search_embeddings(client, query, top_k=3, verbose=False):
    try:
        top_results = client.retreive(query)[:top_k]

        if verbose:
            for result in top_results:
                print(
                    f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
                )

            print("\n")

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
        print(f"---> Context prompt:\n{context_str}")
        print("\n")

    return context_str


async def chat(model: str, chatlog: list) -> None:
    """
    send the query to the LLM and stream in the response
    :param model: name of the model to use
    :param chatlog: chatlog for the current conversation
    """
    async for part in await AsyncClient().chat(
        model=f"{model}:latest", messages=chatlog, stream=True
    ):
        print(part["message"]["content"], end="", flush=True)


def interactive_chat(model: str, db: VDatabase, verbose=False) -> None:
    """
    start an interactive chat session
    :param model: name of the model to use
    :param db: vector database to use
    :param verbose: enable verbose logging
    """
    chatlog = [{"role": "system", "content": SYSTEM_MSG}]

    while True:
        query = chat_input(">>> ")

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

        print("\n")


def main():
    db = get_database()
    model = get_model()
    verbose = get_verbose()

    # If provided, process and return the provided query
    prompt = get_prompt()
    if prompt:
        ctx_response = search_embeddings(db, prompt)
        ctx = generate_rag_response(ctx_response)
        res = generate(
            model=f"{model}:latest",
            system=SYSTEM_MSG,
            prompt=generate_prompt(prompt, ctx),
        )
        print(res["response"])
        return

    # Otherwise start up an interactive session
    interactive_chat(model, db, verbose)


if __name__ == "__main__":
    main()
