# DS4300-Practical02

## Instructions

1. Run `docker compose up -d` to spin up the vector databases
2. Run `python ingest.py` to clear and ingest the data to **all** the databases
3. Run `python chat.py` to chat with the model

## Chat Commands

In an interactive chat session, there are some commands that can be run. All
commands are prefiexed with a colon `:`.

- `:help` - show the help page which lists all available commands
- `:exit` - exits the chat session
- `:clear` - clears the chat log (creating a new conversation)

## CLI Documentation

There are a variety of CI flags designed to allow control over the ingestion and
chat behaviors as well as automate testing for different parameters.

### Ingest

- `-d`, `--database` - vector database to ingest documents. Defaults to `redisstack`
- `-e`, `--embedder` - embedding model to use. Defaults to `nomic-embed-text`
  served through Ollama
- `--indexname` - name of the index to store the embedded vectors. Defaults to `embedding_idx`
- `--prefix` - prefix for the documents in the database. Defaults to `doc:`
- `--metric` - similarity metric to use. Defaults to `COSINE` (cossine similarity)
- `--chunksize` - size of text chunks in characters to store in the database. This
  only affects ingesting the documents. Defaults to `300`
- `--overlap` - overlap between text chunks in characters. This only affects ingesting
  the documents. Defaults to `50`
- `-v`, `--verbose` - enable verbose logging

### Chat

- `-m`, `--model` - model to use. Any valid model id from
  [Ollama](https://ollama.com/search), so long as the model is locally available.
  Defaults to `mistral:latest`
- `-p`, `--prompt` - prompt to generate a RAG response against. This only affects
  chatting and will return just the llm response on `STOUT`
- `-d`, `--database` - vector database to use. Defaults to `redisstack`
- `-e`, `--embedder` - embedding model to use. Defaults to `nomic-embed-text`
- `--indexname` - name of the index to store the embedded vectors. Defaults to `embedding_idx`
- `--prefix` - prefix for the documents in the database. Defaults to `doc:`
- `--metric` - similarity metric to use. Defaults to `COSINE` (cossine similarity)
- `-v`, `--verbose` - enable verbose logging

## Available Configurtions

Currently, the following databases are supported:

- [Redis Stack](https://redis.io/) (`redisstack`)
- [Chromadb](https://www.trychroma.com/) (`chroma`)
- [Qdrant](https://qdrant.tech/) (`qdrant`)

Addionally, the following embedders are supported:

- [Nomic Embed Text](https://huggingface.co/nomic-ai/nomic-embed-text-v1) (`nomic-embed-text`)
- [All MiniLM L6 v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (`all-minilm`)
- [MixedBread Embed Large v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) (`mxbai-embed-large`)
