from dbs.database import VDatabase
from embeddings.embedder import Embedder
import qdrant_client as qdrant
from qdrant_client import models
from qdrant_client.models import PointStruct


class Qdrant(VDatabase):
    """
    Qdrant vector database
    """
    
    def __init__(
        self, embedder: Embedder, dim: int, name: str, prefix: str, metric: str
    ) -> None:
        self.dim = dim
        self.name = name
        self.prefix = prefix
        self.metric = metric
        self.embedder = embedder
        
        self.client = qdrant.QdrantClient(host="localhost", port=6333)
        
    def clear(self) -> None:
        """"""
        try:
            self.client.delete_collection(collection_name=self.name)
        except:
            pass
        
        self.collection = self.client.create_collection(
            collection_name=self.name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE)
        )
       
    
    def store(self, file: str, page: str, chunk: str) -> None:
        """"""
        embedding = self.embedder(chunk)
        
        key = f"{self.prefix}:{file}_page_{page}_chunk_{chunk}"
        self.client.upsert(
            collection_name=self.name,
            wait=True,
            points=[
                PointStruct(id=abs(hash(key)), vector=embedding, payload={"file": file, "page":page, "chunk": chunk})
            ]
        )
        
    
    def retreive(self, prompt):
        """ """

        embedding = self.embedder(prompt)
        res = self.client.search(
            collection_name=self.name,
            query_vector=embedding,
            with_payload=True,
            limit=10
            )
        
        results = [
            {
                "file": result.payload["file"],
                "page": result.payload["page"],
                "chunk": result.payload["chunk"],
                "similarity": result.score
            }
            for result in res
        ]
        
        return results