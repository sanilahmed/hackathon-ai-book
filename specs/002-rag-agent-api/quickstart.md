# Quickstart: RAG Agent and API Layer

## Overview
Quick setup and usage guide for the RAG Agent and API Layer using OpenAI Agents SDK and FastAPI for question-answering with grounded responses based on book content.

## Prerequisites

- Python 3.9 or higher
- OpenAI API key
- Qdrant Cloud instance with book content embeddings (from Spec-1)
- Validated retrieval pipeline (from Spec-2)

## Setup

### 1. Clone and Navigate
```bash
cd /path/to/your/project
cd backend
```

### 2. Install Dependencies
```bash
pip install -e .
# Or if using uv:
uv sync
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your API keys and configuration:
```env
OPENAI_API_KEY=your-openai-api-key-here
QDRANT_URL=your-qdrant-instance-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION_NAME=rag_embedding
```

### 4. Verify Retrieval Pipeline
Ensure the retrieval pipeline from Spec-2 is accessible and working:
```bash
# Test that you can retrieve content from Qdrant
python -m verify_retrieval.main --query "test query"
```

## Basic Usage

### 1. Start the API Server
```bash
cd backend
uvicorn rag_agent_api.main:app --reload --port 8000
```

### 2. Test the API
```bash
# Send a question to the API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the transformer architecture?",
    "context_window": 5,
    "include_sources": true
  }'
```

### 3. Example Response
```json
{
  "id": "resp_abc123",
  "query": "What is the transformer architecture?",
  "answer": "The transformer architecture is a neural network architecture introduced in the 'Attention Is All You Need' paper. It relies entirely on attention mechanisms...",
  "sources": [
    {
      "id": "chunk_123",
      "url": "https://book.example.com/chapter-3",
      "title": "Transformer Models",
      "content": "The transformer model uses self-attention mechanisms to weigh the importance of different parts of the input sequence...",
      "similarity_score": 0.87,
      "chunk_index": 2
    }
  ],
  "confidence": 0.92,
  "timestamp": "2025-12-16T10:30:00Z",
  "model_used": "gpt-4-turbo"
}
```

## Advanced Usage

### Customizing Response Parameters
```bash
# Adjust temperature for more creative responses
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain attention mechanisms",
    "temperature": 0.7,
    "context_window": 10
  }'
```

### Handling Different Query Types
```bash
# Factual query (low temperature for accuracy)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key components of a transformer?",
    "temperature": 0.1
  }'

# Analytical query (higher temperature for synthesis)
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare transformer and RNN architectures",
    "temperature": 0.5
  }'
```

## API Endpoints

### POST /ask
**Description**: Main question-answering endpoint
**Request Body**:
- `query` (string, required): The question to answer
- `context_window` (integer, optional): Number of chunks to retrieve (default: 5)
- `include_sources` (boolean, optional): Include source citations (default: true)
- `temperature` (float, optional): Response randomness (default: 0.1)

**Response**: API Response Model with answer and sources

### GET /health
**Description**: Health check endpoint
**Response**: Status of the API and its dependencies

### GET /docs
**Description**: Interactive API documentation (Swagger UI)

## Configuration Options

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: Your Qdrant API key
- `QDRANT_COLLECTION_NAME`: Name of the collection with book embeddings
- `AGENT_TEMPERATURE`: Default temperature for agent responses
- `CONTEXT_WINDOW_SIZE`: Default number of chunks to retrieve
- `MAX_QUERY_LENGTH`: Maximum allowed query length in characters

### Performance Tuning
- Adjust `CONTEXT_WINDOW_SIZE` based on your token limits
- Set appropriate `AGENT_TEMPERATURE` for your use case
- Configure connection pooling for Qdrant

## Troubleshooting

### Common Issues

**Q: API returns "No relevant content found"**
- A: Verify that your Qdrant collection has the expected embeddings
- A: Check that the retrieval pipeline from Spec-2 is working correctly
- A: Try with a broader query or different keywords

**Q: Responses contain hallucinations**
- A: Lower the temperature parameter
- A: Implement stricter grounding validation
- A: Verify that retrieved context is relevant to the query

**Q: Slow response times**
- A: Check Qdrant connection and performance
- A: Verify OpenAI API availability
- A: Consider implementing caching for frequent queries

**Q: Rate limit errors from OpenAI**
- A: Implement retry logic with exponential backoff
- A: Consider caching responses for common queries
- A: Check your OpenAI rate limit plan

## Next Steps

1. **Production Deployment**: Set up proper monitoring and logging
2. **Caching**: Implement Redis or similar for frequently asked questions
3. **Monitoring**: Add metrics collection for response times and success rates
4. **Testing**: Create comprehensive test suite for your specific use case
5. **Security**: Implement authentication and rate limiting for production use