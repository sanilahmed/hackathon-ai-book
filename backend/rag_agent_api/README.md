# RAG Agent and API Layer

A FastAPI-based question-answering system that uses OpenAI Agents and Qdrant retrieval to generate grounded responses based on book content.

## Overview

The RAG Agent and API Layer provides a question-answering API that retrieves relevant content from Qdrant and uses an OpenAI agent to generate accurate, source-grounded responses. The system ensures that all answers are based only on the provided context to prevent hallucinations.

## Architecture

The system consists of several key components:

- **FastAPI Application**: Main entry point for the question-answering API
- **OpenAI Agent**: Generates responses based on retrieved context
- **Qdrant Retriever**: Retrieves relevant content chunks from Qdrant database
- **Configuration Manager**: Handles environment variables and settings
- **Data Models**: Pydantic models for API requests/responses
- **Utility Functions**: Common helpers and validation utilities

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key
- Qdrant Cloud instance with book content embeddings
- Cohere API key (for query embeddings)

### Installation

1. Install dependencies:
   ```bash
   pip install -e .
   # or if using uv:
   uv sync
   ```

2. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your API keys and configuration:
   ```env
   OPENAI_API_KEY=your-openai-api-key-here
   QDRANT_URL=your-qdrant-instance-url
   QDRANT_API_KEY=your-qdrant-api-key
   QDRANT_COLLECTION_NAME=rag_embedding
   COHERE_API_KEY=your-cohere-api-key-here
   ```

## Usage

### Starting the Server

```bash
cd backend
uvicorn rag_agent_api.main:app --reload --port 8000
```

### API Endpoints

#### POST `/ask`
Main question-answering endpoint.

**Request Body**:
```json
{
  "query": "What is the transformer architecture?",
  "context_window": 5,
  "include_sources": true,
  "temperature": 0.1
}
```

**Response**:
```json
{
  "id": "resp_abc123",
  "query": "What is the transformer architecture?",
  "answer": "The transformer architecture is a neural network architecture introduced in the 'Attention Is All You Need' paper...",
  "sources": [
    {
      "id": "chunk_123",
      "url": "https://book.example.com/chapter-3",
      "title": "Transformer Models",
      "content": "The transformer model uses self-attention mechanisms...",
      "similarity_score": 0.87,
      "chunk_index": 2
    }
  ],
  "confidence": 0.92,
  "timestamp": "2025-12-16T10:30:00Z",
  "model_used": "gpt-4-turbo"
}
```

#### GET `/health`
Health check endpoint.

#### GET `/`
Root endpoint with API information.

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: Your Qdrant API key
- `QDRANT_COLLECTION_NAME`: Name of the collection with book embeddings (default: `rag_embedding`)
- `COHERE_API_KEY`: Your Cohere API key for query embeddings
- `DEFAULT_CONTEXT_WINDOW`: Number of chunks to retrieve (default: 5)
- `DEFAULT_TEMPERATURE`: Default temperature for agent responses (default: 0.1)
- `LOG_LEVEL`: Logging level (default: `INFO`)

## Components

### Configuration (`config.py`)
Handles environment variable loading and validation.

### Models (`models.py`)
Pydantic models for API request/response schemas.

### Schemas (`schemas.py`)
Additional schemas for internal data structures.

### Agent (`agent.py`)
OpenAI agent implementation with context injection and response validation.

### Retrieval (`retrieval.py`)
Qdrant integration for content retrieval with semantic search.

### Utilities (`utils.py`)
Helper functions for logging, validation, and common operations.

### Main Application (`main.py`)
FastAPI application with API endpoints and request handling.

## Features

- **Grounded Responses**: All responses are generated based only on retrieved context
- **Source Citations**: Responses include citations to the original content
- **Confidence Scoring**: Each response includes a confidence score
- **Error Handling**: Graceful handling of various error conditions
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **Health Checks**: Endpoints to monitor system status

## Security

- Input sanitization to prevent injection attacks
- Proper secrets management via environment variables
- Rate limiting to prevent abuse
- Validation of all user inputs

## Testing

Run the tests with:
```bash
# Run all tests
pytest

# Run specific test files
pytest tests/test_api.py
pytest tests/test_agent.py
pytest tests/test_retrieval.py
```

## Performance

- Asynchronous processing for high concurrency
- Caching for frequently accessed content
- Optimized vector similarity searches
- Proper connection pooling for Qdrant