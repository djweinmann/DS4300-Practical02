from abc import ABC, abstractmethod


class VDatabase(ABC):
    """"""

    @abstractmethod
    def __init__(self, dim: int, name: str, prefix: str, metric: str) -> None:
        pass

    @abstractmethod
    def store(self, file: str, page: str, chunk: str, embedding: list) -> None:
        """"""
        pass

    @abstractmethod
    def retreive(self, embedding: list) -> list:
        """"""
        pass
