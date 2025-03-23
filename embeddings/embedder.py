from abc import ABC, abstractmethod


class Embedder(ABC):
    vector_dim = 0

    @abstractmethod
    def __call__(self, text) -> list:
        """ """
        pass
