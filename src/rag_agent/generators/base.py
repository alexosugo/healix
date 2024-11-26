from abc import ABC, abstractmethod
from typing import List

class BaseGenerator(ABC):
    """Base class for RAG response generators"""
    
    @abstractmethod
    def __init__(self, model: str = None):
        """Initialize the generator with a specific model"""
        pass
    
    @abstractmethod
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a response using retrieved context and query
        
        Args:
            query (str): User's input query
            context_docs (List[str]): Retrieved context documents
        
        Returns:
            str: Generated response
        """
        pass
    
    def _create_prompt(self, query: str, context_docs: List[str]) -> str:
        """
        Create a standardized prompt format for all generators
        
        Args:
            query (str): User's input query
            context_docs (List[str]): Retrieved context documents
            
        Returns:
            str: Formatted prompt
        """
        return f"""
        Context Documents:
        {' '.join(context_docs)}
        
        Query: {query}
        
        Based on the context, provide a comprehensive and precise answer to the query.
        """
