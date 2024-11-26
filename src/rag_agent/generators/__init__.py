from .base import BaseGenerator
from .openai_generator import OpenAIGenerator
from .mistral_generator import MistralGenerator

def create_generator(provider: str = "openai", model: str = None) -> BaseGenerator:
    """
    Factory function to create the appropriate generator based on provider
    
    Args:
        provider (str): The model provider ('openai' or 'mistral')
        model (str, optional): Specific model name. Defaults to None.
    
    Returns:
        BaseGenerator: An instance of the appropriate generator
    """
    if provider == "openai":
        return OpenAIGenerator(model)
    elif provider == "mistral":
        return MistralGenerator(model)
    else:
        raise ValueError(f"Unsupported model provider: {provider}")
