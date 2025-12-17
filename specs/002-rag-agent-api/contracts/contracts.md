# API Contracts: RAG Agent and API Layer

## Overview
This document defines the API contracts for the RAG Agent and API Layer system, including request/response schemas, error handling, and integration contracts with external services.

## API Endpoints

### Question-Answering Endpoint
**URL**: `POST /ask`
**Purpose**: Main endpoint for submitting questions and receiving AI-generated answers based on book content

#### Request Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "The user's question or query",
      "minLength": 1,
      "maxLength": 1000
    },
    "context_window": {
      "type": "integer",
      "description": "Number of chunks to retrieve",
      "minimum": 1,
      "maximum": 20,
      "default": 5
    },
    "include_sources": {
      "type": "boolean",
      "description": "Whether to include source citations",
      "default": true
    },
    "temperature": {
      "type": "number",
      "description": "Response randomness",
      "minimum": 0.0,
      "maximum": 1.0,
      "default": 0.1
    }
  },
  "required": ["query"]
}
```

#### Response Schema
```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Unique identifier for the response"
    },
    "query": {
      "type": "string",
      "description": "The original user query"
    },
    "answer": {
      "type": "string",
      "description": "The AI-generated answer"
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "id": {
            "type": "string",
            "description": "Unique identifier for the source chunk"
          },
          "url": {
            "type": "string",
            "description": "URL of the original content",
            "format": "uri"
          },
          "title": {
            "type": "string",
            "description": "Title of the source document"
          },
          "content": {
            "type": "string",
            "description": "The content chunk used"
          },
          "similarity_score": {
            "type": "number",
            "description": "Similarity score to the query",
            "minimum": 0.0,
            "maximum": 1.0
          },
          "chunk_index": {
            "type": "integer",
            "description": "Index of the chunk in the original document"
          }
        },
        "required": ["id", "url", "title", "content", "similarity_score", "chunk_index"]
      }
    },
    "confidence": {
      "type": "number",
      "description": "Confidence level of the response",
      "minimum": 0.0,
      "maximum": 1.0
    },
    "timestamp": {
      "type": "string",
      "description": "ISO 8601 timestamp of the response",
      "format": "date-time"
    },
    "model_used": {
      "type": "string",
      "description": "Name of the model used for generation"
    }
  },
  "required": ["id", "query", "answer", "sources", "confidence", "timestamp", "model_used"]
}
```

### Health Check Endpoint
**URL**: `GET /health`
**Purpose**: Check the health status of the API and its dependencies

#### Response Schema
```json
{
  "type": "object",
  "properties": {
    "status": {
      "type": "string",
      "enum": ["healthy", "degraded", "unhealthy"]
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "services": {
      "type": "object",
      "properties": {
        "openai": {
          "type": "string",
          "enum": ["up", "down", "degraded"]
        },
        "qdrant": {
          "type": "string",
          "enum": ["up", "down", "degraded"]
        },
        "agent": {
          "type": "string",
          "enum": ["up", "down", "degraded"]
        }
      }
    }
  },
  "required": ["status", "timestamp", "services"]
}
```

## External Service Contracts

### OpenAI API Contract
**Service**: OpenAI Agents API
**Purpose**: Generate responses based on provided context

#### Expected Input Format
- System message with context and instructions
- User query
- Context chunks retrieved from Qdrant

#### Expected Output Format
- Text response
- Token usage information
- Potential error responses

#### Error Conditions
- Authentication failures
- Rate limit exceeded
- Context window exceeded
- Service unavailable

### Qdrant API Contract
**Service**: Qdrant Vector Database
**Purpose**: Retrieve relevant content chunks based on semantic similarity

#### Expected Query Format
- Query vector (embedded query text)
- Collection name
- Number of results to return
- Optional filters

#### Expected Response Format
- Array of content chunks with similarity scores
- Metadata for each chunk (URL, title, etc.)
- Payload with source information

#### Error Conditions
- Connection failures
- Authentication failures
- Collection not found
- Query parsing errors

## Internal Component Contracts

### Retrieval Service Contract
**Interface**: `retrieve_context(query: str, top_k: int) -> List[RetrievedChunk]`
**Purpose**: Retrieve relevant content from Qdrant using Spec-2 logic

#### Input Parameters
- `query`: User's question string
- `top_k`: Number of chunks to retrieve

#### Output Format
- List of retrieved chunks with metadata and similarity scores

#### Error Handling
- Raise exceptions for connection issues
- Return empty list if no relevant content found
- Log retrieval failures

### Agent Service Contract
**Interface**: `generate_response(query: str, context: List[Chunk]) -> AgentResponse`
**Purpose**: Generate grounded response using OpenAI agent

#### Input Parameters
- `query`: User's original question
- `context`: List of retrieved content chunks

#### Output Format
- Response text
- Source citations
- Confidence score
- Validation results

#### Error Handling
- Handle OpenAI API errors gracefully
- Return appropriate error responses
- Validate response grounding

### Response Formatter Contract
**Interface**: `format_response(query: str, agent_response: AgentResponse) -> APIResponse`
**Purpose**: Format agent response into API response structure

#### Input Parameters
- `query`: Original user query
- `agent_response`: Response from agent service

#### Output Format
- Complete API response with all required fields
- Properly structured sources and metadata

## Data Contracts

### Content Chunk Contract
**Purpose**: Standard format for content chunks passed between components

```json
{
  "id": "unique identifier",
  "content": "text content of the chunk",
  "similarity_score": "0.0-1.0 similarity to query",
  "metadata": {
    "url": "source URL",
    "title": "document title",
    "chunk_index": "position in document",
    "total_chunks": "total chunks in document"
  }
}
```

### Error Response Contract
**Purpose**: Standardized error responses across all endpoints

```json
{
  "error": {
    "code": "error code",
    "message": "human-readable error message",
    "details": "optional additional details"
  },
  "timestamp": "ISO 8601 timestamp"
}
```

## Versioning Strategy

### API Versioning
- Use URI versioning: `/v1/ask`, `/v2/ask`, etc.
- Maintain backward compatibility for 12 months after deprecation
- Provide migration guides for breaking changes

### Schema Evolution
- Add new optional fields without breaking changes
- Mark fields as deprecated before removal
- Use major version increments for breaking changes

## Security Contracts

### Authentication
- API endpoints may require authentication tokens
- Rate limiting based on API keys or IP addresses
- Input validation to prevent injection attacks

### Data Privacy
- Query logs should not store PII
- Responses should not contain sensitive information
- Proper handling of copyrighted content

## Performance Contracts

### Response Time SLAs
- 95th percentile response time: < 5 seconds
- 99th percentile response time: < 10 seconds
- Health check response time: < 100ms

### Availability SLAs
- 99.9% uptime for question-answering endpoint
- 99.5% uptime for health check endpoint
- Degraded service when external dependencies are unavailable

## Error Handling Contracts

### HTTP Status Codes
- `200`: Successful response
- `400`: Bad request (invalid input)
- `401`: Unauthorized (missing/invalid auth)
- `429`: Rate limited
- `500`: Internal server error
- `502`: Bad gateway (external service error)
- `503`: Service unavailable

### Error Code Definitions
- `QUERY_TOO_LONG`: Query exceeds maximum length
- `INVALID_QUERY`: Query format is invalid
- `RETRIEVAL_FAILED`: Qdrant retrieval failed
- `AGENT_ERROR`: OpenAI agent processing failed
- `NO_RELEVANT_CONTENT`: No relevant content found
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `SERVICE_UNAVAILABLE`: External service unavailable