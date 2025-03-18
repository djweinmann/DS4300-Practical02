from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def __call__(self, text) -> list:
        """ """
        pass
