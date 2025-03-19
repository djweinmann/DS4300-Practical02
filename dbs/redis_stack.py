from redis.commands.search.query import Query
from dbs.database import VDatabase
import redis
import numpy as np

from embeddings.embedder import Embedder


class RedisStack(VDatabase):
    """
    Redis Stack vector database
    """

    def __init__(
        self, embedder: Embedder, dim: int, name: str, prefix: str, metric: str
    ) -> None:
        self.dim = dim
        self.name = name
        self.prefix = prefix
        self.metric = metric
        self.embedder = embedder

        self.client = redis.Redis(host="localhost", port=6379, db=0)

    def clear(self):
        """ """
        self.client.flushdb()

        try:
            self.client.execute_command(f"FT.DROPINDEX {self.name} DD")
        except redis.ResponseError:
            pass

        self.client.execute_command(
            f"""
            FT.CREATE {self.name} ON HASH PREFIX 1 {self.prefix}
            SCHEMA text TEXT
            embedding VECTOR HNSW 6 DIM {self.dim} TYPE FLOAT32 DISTANCE_METRIC {self.metric}
            """
        )

    def store(self, file, page, chunk):
        """ """
        embedding = self.embedder(chunk)

        key = f"{self.prefix}:{file}_page_{page}_chunk_{chunk}"
        self.client.hset(
            key,
            mapping={
                "file": file,
                "page": page,
                "chunk": chunk,
                "embedding": np.array(
                    embedding, dtype=np.float32
                ).tobytes(),  # Store as byte array
            },
        )

    def retreive(self, prompt):
        """ """

        embedding = self.embedder(prompt)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        res = self.client.ft(self.name).search(
            q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )

        results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in res.docs
        ]

        return results
