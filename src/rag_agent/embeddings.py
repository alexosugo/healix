import os
from typing import List
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingManager:
    def __init__(self):
        # Initialize embedding model
        self.model = SentenceTransformer(os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'))
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=os.getenv('CHROMA_DB_PATH', './chroma_db'))
        
    def create_collection(self, collection_name: str):
        """
        Create a new ChromaDB collection for storing embeddings
        """
        return self.chroma_client.create_collection(name=collection_name)
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        """
        return self.model.encode(documents).tolist()
    
    def add_documents(self, collection, documents: List[str], ids: List[str] = None):
        """
        Add documents to a ChromaDB collection
        """
        # Generate embeddings
        embeddings = self.embed_documents(documents)
        
        # If no ids provided, generate unique ids
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=ids
        )
    
    def query_collection(self, collection, query: str, n_results: int = 3):
        """
        Query the collection and retrieve most similar documents
        """
        query_embedding = self.embed_documents([query])[0]
        
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
