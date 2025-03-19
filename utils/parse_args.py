import argparse

from dbs.chroma import Chroma
from dbs.redis_stack import RedisStack
from embeddings.nomic_embed_text import NomicEmbedText

parser = argparse.ArgumentParser(description="Parameters to search")

parser.add_argument(
    "-e",
    "--embedder",
    type=str,
    required=False,
    default="nomic-embed-text",
    help="embedding model to use",
)
parser.add_argument(
    "-d",
    "--database",
    type=str,
    required=False,
    default="redisstack",
    help="vector database to use",
)

parser.add_argument("--vectordim", type=int, required=False, default=768, help="")
parser.add_argument(
    "--indexname", type=str, required=False, default="embedding_idx", help=""
)
parser.add_argument("--prefix", type=str, required=False, default="doc:", help="")
parser.add_argument("--metric", type=str, required=False, default="COSINE", help="")


def get_embedder():
    args = parser.parse_args()
    match args.embedder:
        case "nomic-embed-text":
            return NomicEmbedText()

    raise TypeError("unknown embedding model " + args.embedder)


def get_database():
    args = parser.parse_args(["database"])
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


parser.add_argument(
    "-m",
    "--model",
    type=str,
    required=True,
    help="LLM Model to use",
)
parser.add_argument("-t", "--trials", type=int, required=True, help="Number of trials")

args = parser.parse_args()
