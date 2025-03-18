from abc import ABC, abstractmethod


class VDatabase(ABC):
    """"""

    @abstractmethod
    def __init__(self, dim: int, name: str, prefix: str, metric: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        """"""
        pass

    @abstractmethod
    def store(self, file: str, page: str, chunk: str) -> None:
        """"""
        pass

    @abstractmethod
    def retreive(self, prompt: str) -> list:
        """"""
        pass
