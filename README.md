# RAG Agent API

## Overview
This is a Retrieval-Augmented Generation (RAG) agent API built with FastAPI, leveraging OpenAI's language models and ChromaDB for document embedding and retrieval.

## Features
- Create document collections
- Semantic search across document collections
- AI-powered response generation using context retrieval

## Prerequisites
- Python 3.9+
- OpenAI API Key

## Installation
1. Clone the repository
2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
- Copy `.env.example` to `.env`
- Add your OpenAI API Key

## Running the Application
```bash
uvicorn src.rag_agent.main:app --reload
```

## API Endpoints
- `/create_collection`: Add documents to a collection
- `/query`: Retrieve context and generate AI responses
- `/health`: Check application health

## Usage Example
```python
import requests

# Create a collection
create_response = requests.post('http://localhost:8000/create_collection', json={
    'collection_name': 'my_documents',
    'documents': ['Sample document 1', 'Sample document 2']
})

# Query the collection
query_response = requests.post('http://localhost:8000/query', json={
    'collection_name': 'my_documents',
    'query': 'What is the main topic?'
})
```

## Configuration
Customize settings in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key
- `MISTRAL_API_KEY`: Your Mistral AI API key
- `MODEL_PROVIDER`: Choose between 'openai' or 'mistral'
- `EMBEDDING_MODEL`: Sentence transformer model
- `CHROMA_DB_PATH`: Vector database storage path

## Model Support
The RAG agent supports two LLM providers:
1. OpenAI
   - Default model: gpt-3.5-turbo
   - Requires OPENAI_API_KEY

2. Mistral AI
   - Default model: mistral-tiny
   - Other options: mistral-small, mistral-medium
   - Requires MISTRAL_API_KEY

To switch between providers, set MODEL_PROVIDER in .env to either 'openai' or 'mistral'.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
