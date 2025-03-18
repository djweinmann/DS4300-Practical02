from dbs.database import VDatabase
import chromadb


class Chroma(VDatabase):
    """"""

    def __init__(self, dim: int, name: str, prefix: str, metric: str) -> None:
        self.dim = dim
        self.name = name
        self.prefix = prefix
        self.metric = metric

        self.client = chromadb.HttpClient(host="localhost", port=8000)

    def clear(self) -> None:
        """"""
        pass

    def store(self, file: str, page: str, chunk: str, embedding: list) -> None:
        """"""
        pass

    def retreive(self, embedding: list) -> list:
        """"""
        pass
