import os
from typing import List
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

from .base import BaseGenerator

# Load environment variables
load_dotenv()

class MistralGenerator(BaseGenerator):
    """Mistral-specific implementation of the RAG generator"""
    
    def __init__(self, model: str = None):
        """
        Initialize Mistral generator
        
        Args:
            model (str, optional): Mistral model name. Defaults to "mistral-tiny"
        """
        self.client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
        self.model = model or "mistral-tiny"
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a response using Mistral's chat completion
        
        Args:
            query (str): User's input query
            context_docs (List[str]): Retrieved context documents
        
        Returns:
            str: Generated response
        """
        try:
            prompt = self._create_prompt(query, context_docs)
            
            messages = [
                ChatMessage(role="system", content="You are a helpful AI assistant that generates answers based on provided context."),
                ChatMessage(role="user", content=prompt)
            ]
            
            response = self.client.chat(
                model=self.model,
                messages=messages
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"An error occurred while generating response with Mistral: {str(e)}"
