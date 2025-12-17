# Data Model: RAG Agent and API Layer

## Overview
This document defines the data structures and models for the RAG Agent and API Layer system, including API request/response schemas, internal data structures, and metadata formats.

## API Request/Response Models

### Query Request Model
**Purpose**: Structure for incoming user queries to the question-answering API

```json
{
  "query": "string (required) - The user's question or query",
  "context_window": "integer (optional) - Number of chunks to retrieve (default: 5)",
  "include_sources": "boolean (optional) - Whether to include source citations (default: true)",
  "temperature": "float (optional) - Response randomness (default: 0.1)"
}
```

**Validation Rules**:
- Query must be 1-1000 characters
- Context_window must be between 1-20
- Temperature must be between 0.0-1.0

### API Response Model
**Purpose**: Structure for responses returned by the question-answering API

```json
{
  "id": "string - Unique identifier for the response",
  "query": "string - The original user query",
  "answer": "string - The AI-generated answer",
  "sources": [
    {
      "id": "string - Unique identifier for the source chunk",
      "url": "string - URL of the original content",
      "title": "string - Title of the source document",
      "content": "string - The content chunk used",
      "similarity_score": "float - Similarity score to the query (0.0-1.0)",
      "chunk_index": "integer - Index of the chunk in the original document"
    }
  ],
  "confidence": "float - Confidence level of the response (0.0-1.0)",
  "timestamp": "string - ISO 8601 timestamp of the response",
  "model_used": "string - Name of the model used for generation"
}
```

### Error Response Model
**Purpose**: Standardized error responses

```json
{
  "error": {
    "code": "string - Error code (e.g., 'RETRIEVAL_FAILED', 'AGENT_ERROR')",
    "message": "string - Human-readable error message",
    "details": "object - Additional error details (optional)"
  },
  "timestamp": "string - ISO 8601 timestamp of the error"
}
```

## Internal Data Models

### Retrieved Context Chunk
**Purpose**: Structure for content chunks retrieved from Qdrant

```json
{
  "id": "string - Unique identifier for the chunk",
  "content": "string - The text content of the chunk",
  "similarity_score": "float - Similarity score to the query (0.0-1.0)",
  "metadata": {
    "url": "string - URL of the original document",
    "title": "string - Title of the source document",
    "chunk_index": "integer - Position of this chunk in the document",
    "total_chunks": "integer - Total number of chunks in the document",
    "source_hash": "string - Hash of the original content for verification"
  },
  "retrieved_at": "string - Timestamp when chunk was retrieved"
}
```

### Agent Context Model
**Purpose**: Structure for context passed to the OpenAI agent

```json
{
  "query": "string - The original user query",
  "retrieved_chunks": "array of Retrieved Context Chunk objects",
  "max_context_length": "integer - Maximum allowed context length in tokens",
  "source_policy": "string - Policy for source citation ('strict', 'permissive', 'none')"
}
```

### Agent Response Model
**Purpose**: Structure for responses from the OpenAI agent before API formatting

```json
{
  "raw_response": "string - The raw response from the agent",
  "used_sources": "array of strings - IDs of source chunks used in the response",
  "confidence_score": "float - Agent's confidence in the response (0.0-1.0)",
  "grounding_validation": {
    "is_valid": "boolean - Whether the response is grounded in provided context",
    "validation_details": "string - Details about grounding validation",
    "unsupported_claims": "array of strings - Claims not supported by context"
  }
}
```

## Data Flow

### Query Processing Flow
1. User submits Query Request
2. System validates request parameters
3. Retrieves relevant chunks from Qdrant (Retrieved Context Chunk)
4. Constructs Agent Context with retrieved chunks
5. Processes with OpenAI agent (Agent Response Model)
6. Validates grounding in provided context
7. Formats into API Response Model
8. Returns to user

### Error Handling Flow
1. System detects error condition
2. Constructs Error Response Model
3. Logs error with details
4. Returns appropriate error response to user

## Validation Rules

### Content Validation
- All text fields should be sanitized to prevent injection
- Content length should be validated against token limits
- URLs should be validated for proper format

### Metadata Validation
- Source URLs should be verified as part of the original book content
- Chunk indices should be consistent with document structure
- Similarity scores should be in the 0.0-1.0 range

### Response Validation
- All responses must be grounded in provided context chunks
- Source citations must reference actual retrieved chunks
- Confidence scores should be calculated based on context quality

## Schema Evolution

### Versioning Strategy
- Use semantic versioning for schema changes
- Maintain backward compatibility where possible
- Provide migration paths for breaking changes

### Extension Points
- Allow additional fields in metadata for future expansion
- Design flexible error codes to accommodate new error types
- Support optional fields that can be added without breaking changes

## Integration Points

### With Spec-2 Retrieval Pipeline
- Retrieved Context Chunk structure must match Spec-2 output
- Metadata format must be consistent with existing embeddings
- Similarity scoring approach should align with Spec-2

### With Frontend Applications
- API Response Model should provide all necessary information for UI
- Source citations should be structured for easy display
- Confidence scores should be suitable for UI indicators