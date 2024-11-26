from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from .embeddings import EmbeddingManager
from .generators import create_generator

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="A Retrieval-Augmented Generation API for intelligent document querying",
    version="0.1.0"
)

# Initialize managers
embedding_manager = EmbeddingManager()
rag_generator = create_generator(
    provider=os.getenv('MODEL_PROVIDER', 'openai'),
    model=None  # Will use default model for the provider
)

# Request models
class DocumentRequest(BaseModel):
    documents: List[str]
    collection_name: str

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    n_results: int = 3

class ResponseModel(BaseModel):
    query: str
    context_documents: List[str]
    response: str

@app.post("/create_collection")
def create_collection(request: DocumentRequest):
    """
    Create a new document collection and add documents
    """
    try:
        collection = embedding_manager.create_collection(request.collection_name)
        embedding_manager.add_documents(collection, request.documents)
        return {"status": "success", "message": f"Collection {request.collection_name} created with {len(request.documents)} documents"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query", response_model=ResponseModel)
def query_collection(request: QueryRequest):
    """
    Query a document collection and generate a response
    """
    try:
        # Get collection
        collection = embedding_manager.chroma_client.get_collection(request.collection_name)
        
        # Retrieve context documents
        query_result = embedding_manager.query_collection(
            collection, 
            request.query, 
            n_results=request.n_results
        )
        
        # Extract documents from query result
        context_documents = query_result['documents'][0]
        
        # Generate response
        response = rag_generator.generate_response(request.query, context_documents)
        
        return {
            "query": request.query,
            "context_documents": context_documents,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Optional: Add a health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}
