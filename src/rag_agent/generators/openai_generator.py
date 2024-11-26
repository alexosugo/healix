import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseGenerator

# Load environment variables
load_dotenv()

class OpenAIGenerator(BaseGenerator):
    """OpenAI-specific implementation of the RAG generator"""
    
    def __init__(self, model: str = None):
        """
        Initialize OpenAI generator
        
        Args:
            model (str, optional): OpenAI model name. Defaults to "gpt-3.5-turbo"
        """
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = model or "gpt-3.5-turbo"
    
    def generate_response(self, query: str, context_docs: List[str]) -> str:
        """
        Generate a response using OpenAI's chat completion
        
        Args:
            query (str): User's input query
            context_docs (List[str]): Retrieved context documents
        
        Returns:
            str: Generated response
        """
        try:
            prompt = self._create_prompt(query, context_docs)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that generates answers based on provided context."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"An error occurred while generating response with OpenAI: {str(e)}"
