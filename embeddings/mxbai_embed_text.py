from embeddings.embedder import Embedder
import ollama


class MxbaiEmbedText(Embedder):
    vector_dim = 1024

    def __call__(self, text):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response["embedding"]
