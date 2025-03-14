from redis.commands.search.query import Query
from dbs.database import VDatabase
import redis
import numpy as np


class RedisStack(VDatabase):
    """
    Redis Stack vector database
    """

    def __init__(self, dim: int, name: str, prefix: str, metric: str) -> None:
        self.dim = dim
        self.name = name
        self.prefix = prefix
        self.metric = metric

        self.client = redis.Redis(host="localhost", port=6380, db=0)

        # Clear Redis
        self.client.flushdb()

        # Create an HNSW index in Redis
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

    def store(self, file, page, chunk, embedding):
        """ """
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

    def retreive(self, embedding):
        """ """
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "vector_distance")
            .dialect(2)
        )

        res = self.client.ft(self.name).search(
            q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
        )

        return res.docs
