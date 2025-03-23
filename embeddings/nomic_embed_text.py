from embeddings.embedder import Embedder
import ollama


class NomicEmbedText(Embedder):
    vector_dim = 768

    def __call__(self, text):
        response = ollama.embeddings(model="nomic-embed-text", prompt=text)
        return response["embedding"]
