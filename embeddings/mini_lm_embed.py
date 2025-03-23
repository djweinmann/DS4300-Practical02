from embeddings.embedder import Embedder
import ollama


class MiniLMEmbedText(Embedder):
    vector_dim = 384

    def __call__(self, text):
        response = ollama.embeddings(model="all-minilm", prompt=text)
        return response["embedding"]
