from dbs.database import VDatabase
import chromadb
from datetime import datetime
from chromadb import Documents, EmbeddingFunction, Embeddings

from embeddings.embedder import Embedder


class Chroma(VDatabase):
    """"""

    def __init__(self, embedder: Embedder, name: str, prefix: str, metric: str) -> None:
        self.embedder = embedder
        self.dim = embedder.vector_dim

        self.name = name
        self.prefix = prefix
        self.metric = metric

        self.client = chromadb.HttpClient(host="localhost", port=8000)

        class MyEmbeddingFunction(EmbeddingFunction):
            def __call__(self, input: Documents) -> Embeddings:
                return embedder(input[0])

        self.chromaEmbedder = MyEmbeddingFunction()

    def clear(self) -> None:
        """"""
        try:
            self.client.delete_collection(name=self.name)
        except:
            pass

        self.collection = self.client.create_collection(
            name=self.name,
            embedding_function=self.chromaEmbedder,
            metadata={
                "created": str(datetime.now()),
                "hnsw:space": "cosine",
                "hnsw:search_ef": 100,  # TODO: Investigate this value
            },
        )

    def store(self, file: str, page: str, chunk: str) -> None:
        """"""
        key = f"{self.prefix}:{file}_page_{page}_chunk_{chunk}"

        self.collection.add(
            ids=[key], documents=[chunk], metadatas=[{"file": file, "page": page}]
        )

    def retreive(self, prompt) -> list:
        """"""
        self.collection = self.client.get_collection(
            name=self.name, embedding_function=self.chromaEmbedder
        )
        res = self.collection.query(
            query_texts=[prompt],
            n_results=10,
        )

        results = []
        for i in range(len(res["ids"][0])):
            results.append(
                {
                    "file": res["metadatas"][0][i]["file"],
                    "page": res["metadatas"][0][i]["file"],
                    "chunk": res["documents"][0][i],
                    "similarity": res["distances"][0][i],
                }
            )

        return results
