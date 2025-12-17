# API Contracts: Frontend Integration of RAG Chatbot with Docusaurus Book

**Feature**: Frontend Integration of RAG Chatbot with Docusaurus Book
**Created**: 2025-12-16
**Status**: Draft

## Overview

This document defines the API contracts between the Docusaurus frontend chatbot component and the FastAPI RAG backend. These contracts ensure consistent communication and data exchange between the frontend and backend systems.

## Backend API Endpoints

### 1. Question Answering Endpoint

**Endpoint**: `POST /ask`
**Purpose**: Submit user questions and receive AI-generated responses with source metadata

#### Request
```
POST /ask
Content-Type: application/json
```

**Request Body**:
```json
{
  "query": "string (required) - User's question",
  "context_window": "integer (optional) - Number of context chunks to retrieve (default: 5)",
  "include_sources": "boolean (optional) - Whether to include source information (default: true)",
  "temperature": "number (optional) - AI response randomness (0.0 to 1.0, default: 0.1)"
}
```

**Example Request**:
```json
{
  "query": "What are the key concepts of machine learning?",
  "context_window": 5,
  "include_sources": true,
  "temperature": 0.1
}
```

#### Response
**Success Response (200 OK)**:
```json
{
  "id": "string - Unique response identifier",
  "query": "string - Original user query",
  "answer": "string - AI-generated answer",
  "sources": [
    {
      "id": "string - Chunk identifier",
      "url": "string - URL of source content",
      "title": "string - Title of source document",
      "content": "string - Content snippet",
      "similarity_score": "number - Relevance score (0.0 to 1.0)",
      "chunk_index": "integer - Position in original document"
    }
  ],
  "confidence": "number - Confidence score (0.0 to 1.0)",
  "timestamp": "string - ISO 8601 formatted timestamp",
  "model_used": "string - Name of AI model used"
}
```

**Error Response (400 Bad Request)**:
```json
{
  "id": "string - Error identifier",
  "error_code": "string - Machine-readable error code",
  "message": "string - Human-readable error message",
  "timestamp": "string - ISO 8601 formatted timestamp"
}
```

### 2. Health Check Endpoint

**Endpoint**: `GET /health`
**Purpose**: Check the health status of the backend service

#### Response
**Success Response (200 OK)**:
```json
{
  "status": "string - Overall health status (healthy, degraded, unhealthy)",
  "timestamp": "string - ISO 8601 formatted timestamp",
  "services": {
    "openai": "string - Status of OpenAI service (up, down, degraded)",
    "qdrant": "string - Status of Qdrant service (up, down, degraded)",
    "agent": "string - Status of agent service (up, down, degraded)"
  }
}
```

### 3. Readiness Check Endpoint

**Endpoint**: `GET /ready`
**Purpose**: Check if the backend is ready to process requests

#### Response
**Ready Response (200 OK)**:
```json
{
  "status": "string - Readiness status (ready, not ready)"
}
```

**Not Ready Response (503 Service Unavailable)**:
```json
{
  "detail": "string - Error message explaining why service is not ready"
}
```

## Frontend API Contracts

### 1. Chatbot Component Interface

**Component Name**: `ChatbotComponent`
**Purpose**: Provide user interface for asking questions and displaying responses

#### Props Interface
```typescript
interface ChatbotProps {
  position?: 'bottom-right' | 'bottom-left' | 'side' | 'embedded';
  theme?: 'light' | 'dark';
  maxHistory?: number;
  enableHistory?: boolean;
  placeholder?: string;
  backendUrl?: string;
  onQuestionSubmit?: (question: string) => void;
  onResponseReceived?: (response: APIResponse) => void;
  onError?: (error: Error) => void;
}
```

#### Internal State Interface
```typescript
interface ChatbotState {
  question: string;
  responses: Array<{
    question: string;
    answer: string;
    sources: Array<SourceChunk>;
    timestamp: Date;
  }>;
  isLoading: boolean;
  error: string | null;
  backendStatus: 'unknown' | 'healthy' | 'unhealthy';
}
```

### 2. Data Transformation Contracts

#### Frontend Request Transformation
The frontend must transform user input before sending to backend:
- Trim whitespace from questions
- Validate question length (minimum 3 characters, maximum 1000 characters)
- Set default values for optional parameters

#### Backend Response Transformation
The frontend must process backend responses for display:
- Format source links as clickable elements
- Sanitize HTML content to prevent XSS
- Format timestamps for user-friendly display
- Extract and organize source metadata for attribution

## Error Handling Contracts

### Frontend Error Handling
The frontend component must handle these error scenarios:

1. **Network Errors**: Handle connection failures to backend
2. **Backend Errors**: Display user-friendly messages for backend errors
3. **Validation Errors**: Validate user input before sending to backend
4. **Timeout Errors**: Handle requests that exceed expected response time

### Backend Error Responses
The backend must return standardized error responses:

```typescript
interface ErrorResponse {
  id: string;           // Unique error identifier
  error_code: string;   // Machine-readable error code
  message: string;      // Human-readable error message
  timestamp: string;    // ISO 8601 timestamp
  details?: any;        // Optional additional error details
}
```

## Performance Contracts

### Response Time Requirements
- Health check: < 100ms
- Question answering: < 10 seconds for 90% of requests
- Error responses: < 500ms

### Data Size Limits
- Maximum question length: 1000 characters
- Maximum response length: 5000 characters
- Maximum number of sources: 10 per response

## Security Contracts

### Input Validation
- All user inputs must be validated and sanitized
- No executable code in user queries
- Proper encoding for special characters

### Output Sanitization
- All AI responses must be sanitized before display
- Source URLs must be validated
- No script execution from AI-generated content

## Compatibility Contracts

### Browser Support
- Modern browsers (Chrome 80+, Firefox 75+, Safari 13+, Edge 80+)
- Mobile browsers with JavaScript enabled
- Screen readers for accessibility

### API Versioning
- Backend API version: v1 (current)
- Frontend must handle API version negotiation
- Backward compatibility for at least 6 months