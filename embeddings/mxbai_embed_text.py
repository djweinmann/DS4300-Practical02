from embeddings.embedder import Embedder
import ollama


class MxbaiEmbedText(Embedder):
    def __call__(self, text):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response["embedding"]
