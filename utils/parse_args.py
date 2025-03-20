import argparse

from dbs.chroma import Chroma
from dbs.redis_stack import RedisStack
from embeddings.nomic_embed_text import NomicEmbedText
from dbs.qdrant import Qdrant

parser = argparse.ArgumentParser(description="Parameters to search")

parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="mistral:latest",
    help="LLM Model to use",
)

parser.add_argument(
    "-e",
    "--embedder",
    type=str,
    default="nomic-embed-text",
    help="embedding model to use",
)
parser.add_argument(
    "-d",
    "--database",
    type=str,
    default="redisstack",
    help="vector database to use",
)

parser.add_argument(
    "--vectordim", type=int, default=768, help="dimension of the vector"
)
parser.add_argument(
    "--indexname",
    type=str,
    default="embedding_idx",
    help="name of the vector index",
)
parser.add_argument("--prefix", type=str, default="doc:", help="document prefix")
parser.add_argument(
    "--metric",
    type=str,
    default="COSINE",
    help="distance metric to use",
)
parser.add_argument("--chunksize", type=int, default=300, help="size to chunk text")
parser.add_argument("--overlap", type=int, default=50, help="overlap in chunks")

parser.add_argument(
    "-p", "--prompt", type=str, help="chat prompt to run. will return only the response"
)

parser.add_argument(
    "-v", "--verbose", action="store_true", help="include debug logging"
)

args = parser.parse_args()


def get_embedder():
    match args.embedder:
        case "nomic-embed-text":
            return NomicEmbedText()

    raise TypeError("unknown embedding model " + args.embedder)


def get_database():
    database = args.database
    embedder = get_embedder()

    dim = args.vectordim
    name = args.indexname
    prefix = args.prefix
    metric = args.metric

    match database:
        case "redisstack":
            return RedisStack(embedder, dim, name, prefix, metric)
        case "chroma":
            return Chroma(embedder, dim, name, prefix, metric)
        case "qdrant":
            return Qdrant(embedder, dim, name, prefix, metric)

    raise TypeError("unknown database " + args.database)


def get_model():
    return args.model


def get_verbose():
    return args.verbose


def get_ingestion():
    return args.chunksize, args.overlap


def get_prompt():
    return args.prompt
