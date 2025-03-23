""" """

from dbs.database import VDatabase
from utils.parse_args import get_database, get_model, get_verbose, get_prompt
import asyncio
from ollama import AsyncClient, generate
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit import prompt

# SYSTEM_MSG = """\
# You are an uptight rizzer rapper by the name drippidy d. You also happen to \
# be a computer science pro that knowns all you know ya know. Use the following context to answer the query as accurately as possible \
# while bringing in the drip. If the context is not relevant to the query, say 'I don't know'.
#
# You should respond in the tone of the true jayz or emenem to honor your rapping legacy.
# """

# SYSTEM_MSG = """\
# You are basketball superstar LeBron James, aka the GOAT, aka King James. You also are a computer science pro \
# that knows about information and data storage systems. If the context is not relevant to the query, say 'I don't know'. \
# You should respond by lying a lot and acting like a basketball superstar and the GOAT.
# """

SYSTEM_MSG = """\
You are a helpful assistant. Use the provided context to help generate your response. If you do not know, say 'I do not know'
"""


HELP_MSG = """\
Available commands:
    :exit      Exit the chat
    :clear     Clear the chatlog
    :help      Shows this help page
"""


def generate_prompt(query, ctx):
    return f"""\
Respond to the query using the context to guide your response

<Context>
{ctx}
</Context>

<Query>
{query}
</Query>
"""


def chat_input(prompt_text: str, **kwargs) -> str:
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
        cursor=CursorShape.BLINKING_BEAM,
        **kwargs,
    ).lower()


def generate_ctx_string(context_results):
    context_str = "\n".join(
        [
            f"""\
<Document>
Filename: {result.get("file", "Unknown file")}
Page: {result.get("page", "Unknown page")}
Similarity: {float(result.get("similarity", 0)):.2f}
<Text>
{result.get("chunk", "Unknown chunk")}
</Text>
</Document>
            """
            for result in context_results
        ]
    )

    return context_str


def search_embeddings(client, query, top_k=3, verbose=False):
    try:
        top_results = client.retreive(query)[:top_k]

        if verbose:
            print("---> Matching documents (file | page | chunk)")
            for result in top_results:
                print(f"{result['file']} | {result['page']} | {result['chunk']}")

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


async def chat(model: str, chatlog: list, db) -> None:
    """
    send the query to the LLM and stream in the response
    :param model: name of the model to use
    :param chatlog: chatlog for the current conversation
    """
    client = AsyncClient().chat(
        model=model,
        messages=chatlog,
        stream=True,
        # tools=[
        #     {
        #         "type": "function",
        #         "function": {
        #             "name": "fetch_data",
        #             "description": "Search documentation",
        #             "parameters": {
        #                 "type": "object",
        #                 "properties": {
        #                     "query": {
        #                         "type": "string",
        #                         "description": "Query to search",
        #                     },
        #                 },
        #                 "required": ["query"],
        #             },
        #         },
        #     }
        # ],
    )

    res = []
    message = ""
    async for part in await client:
        # print(part)
        # if res is None and not part.message.content:
        #     res = part
        if token := part.message.content:
            print(token, end="", flush=True)
            message = message + token
        elif not part.done:
            res.append(part)

    # print("\n", res)
    # print("\n", message)

    # available_functions = {
    #     "fetch_data": lambda **kwargs: search_embeddings(db, kwargs["query"])
    # }

    # if res is None:
    #     return
    #
    # if res.message.tool_calls:
    #     # There may be multiple tool calls in the response
    #     for tool in res.message.tool_calls:
    #         # Ensure the function is available, and then call it
    #         if function_to_call := available_functions.get(tool.function.name):
    #             print("Calling function:", tool.function.name)
    #             print("Arguments:", tool.function.arguments)
    #             output = function_to_call(**tool.function.arguments)
    #             print("Function output:", output)
    #         else:
    #             print("Function", tool.function.name, "not found")
    #
    #     # # Only needed to chat with the model using the tool call results
    #     # if res.message.tool_calls:
    #     #     # Add the function response to messages for the model to use
    #     #     chatlog.append(response.message)
    #     #     chatlog.append({'role': 'tool', 'content': str(output), 'name': tool.function.name})
    #     #
    #     #     # Get final response from model with function outputs
    #     #     final_response = await client.chat('llama3.1', messages=messages)
    #     #     print('Final response:', final_response.message.content)
    #     #
    #     #   else:
    #     #     print('No tool calls returned from model')

    if message:
        chatlog.append({"role": "assistant", "content": message})


def interactive_chat(model: str, db: VDatabase, verbose=False) -> None:
    """
    start an interactive chat session
    :param model: name of the model to use
    :param db: vector database to use
    :param verbose: enable verbose logging
    """
    chatlog = [{"role": "system", "content": SYSTEM_MSG}]
    multiline = False
    mouse_support = False

    while True:
        query = chat_input(">>> ", multiline=multiline, mouse_support=mouse_support)

        if query[0] == ":":
            match query.lower():
                case ":help":
                    print(HELP_MSG)
                case ":clear":
                    chatlog = [{"role": "system", "content": SYSTEM_MSG}]
                    print("Chatlog cleared")
                case ":multiline":
                    multiline = True
                case ":mouse":
                    mouse_support = True
                case ":exit":
                    break
                case _:
                    print("unknown command " + query)
            continue

        context_results = search_embeddings(db, query, verbose=verbose)
        ctx = generate_ctx_string(context_results)

        if verbose:
            print("---> Full prompt")
            print(generate_prompt(prompt, ctx))

        chatlog.append({"role": "user", "content": generate_prompt(query, ctx)})
        asyncio.run(chat(model, chatlog, db))

        print("\n")


def main():
    db = get_database()
    model = get_model()
    verbose = get_verbose()

    # If provided, process and return the provided query
    prompt = get_prompt()
    if prompt:
        ctx_response = search_embeddings(db, prompt, verbose=verbose)
        ctx = generate_ctx_string(ctx_response)
        if verbose:
            print("---> Full prompt")
            print(generate_prompt(prompt, ctx))

        res = generate(
            model=model,
            system=SYSTEM_MSG,
            prompt=generate_prompt(prompt, ctx),
        )

        print(res["response"])
        return

    # Otherwise start up an interactive session
    interactive_chat(model, db, verbose)


if __name__ == "__main__":
    main()
