# Data Models: Frontend Integration of RAG Chatbot with Docusaurus Book

**Feature**: Frontend Integration of RAG Chatbot with Docusaurus Book
**Created**: 2025-12-16
**Status**: Draft

## Overview

This document defines the data models used in the RAG chatbot frontend integration. These models represent the structure of data exchanged between the frontend and backend, as well as internal data structures used within the frontend component.

## External Data Models (Backend API)

### 1. QueryRequest Model

**Source**: FastAPI backend (rag_agent_api.models.QueryRequest)
**Purpose**: Represents a user's question and query parameters

```json
{
  "query": "string (required) - The user's question",
  "context_window": "integer (optional, default: 5) - Number of context chunks to retrieve",
  "include_sources": "boolean (optional, default: true) - Whether to include source information",
  "temperature": "number (optional, default: 0.1) - AI response randomness (0.0 to 1.0)"
}
```

**Validation Rules**:
- `query`: Required, minimum 3 characters, maximum 1000 characters
- `context_window`: Minimum 1, maximum 20
- `temperature`: Minimum 0.0, maximum 1.0

### 2. APIResponse Model

**Source**: FastAPI backend (rag_agent_api.models.APIResponse)
**Purpose**: Represents the backend's response to a query

```json
{
  "id": "string - Unique response identifier",
  "query": "string - Original user query",
  "answer": "string - AI-generated answer",
  "sources": "array - List of source chunks used to generate the answer",
  "confidence": "number - Confidence score (0.0 to 1.0)",
  "timestamp": "string - ISO 8601 formatted timestamp",
  "model_used": "string - Name of AI model used"
}
```

### 3. SourceChunk Model

**Source**: FastAPI backend (rag_agent_api.models.SourceChunk)
**Purpose**: Represents a content chunk from the book that was used to generate the response

```json
{
  "id": "string - Unique identifier for the chunk",
  "url": "string - URL of the source content",
  "title": "string - Title of the source document",
  "content": "string - Content snippet",
  "similarity_score": "number - Relevance score (0.0 to 1.0)",
  "chunk_index": "integer - Position in original document"
}
```

### 4. ErrorResponse Model

**Source**: FastAPI backend (rag_agent_api.models.ErrorResponse)
**Purpose**: Represents error responses from the backend

```json
{
  "id": "string - Error identifier",
  "error_code": "string - Machine-readable error code",
  "message": "string - Human-readable error message",
  "timestamp": "string - ISO 8601 formatted timestamp"
}
```

## Internal Frontend Data Models

### 1. FrontendConversation Model

**Purpose**: Represents a conversation thread in the frontend component

```typescript
interface FrontendConversation {
  id: string;                           // Unique conversation identifier
  createdAt: Date;                      // When the conversation started
  questions: FrontendQuestion[];        // Array of questions in this conversation
  isActive: boolean;                    // Whether this conversation is currently active
}
```

### 2. FrontendQuestion Model

**Purpose**: Represents a single question in the frontend conversation

```typescript
interface FrontendQuestion {
  id: string;                           // Unique question identifier
  question: string;                     // The user's question
  timestamp: Date;                      // When the question was asked
  status: 'pending' | 'processing' | 'completed' | 'error'; // Current status
  response?: FrontendResponse;          // The corresponding response (if available)
}
```

### 3. FrontendResponse Model

**Purpose**: Represents a single response in the frontend conversation

```typescript
interface FrontendResponse {
  id: string;                           // Unique response identifier
  answer: string;                       // The AI-generated answer
  sources: FrontendSource[];            // Array of sources used
  confidence: number;                   // Confidence score (0.0 to 1.0)
  timestamp: Date;                      // When the response was received
  modelUsed: string;                    // Name of AI model used
  backendResponse: APIResponse;         // Original backend response object
}
```

### 4. FrontendSource Model

**Purpose**: Represents a source in the frontend, with additional UI properties

```typescript
interface FrontendSource {
  id: string;                           // Source chunk identifier
  url: string;                          // URL of the source content
  title: string;                        // Title of the source document
  content: string;                      // Content snippet
  similarityScore: number;              // Relevance score (0.0 to 1.0)
  chunkIndex: number;                   // Position in original document
  displayText: string;                  // Formatted text for display
  linkable: boolean;                    // Whether the source can be linked to
}
```

### 5. ChatbotState Model

**Purpose**: Represents the internal state of the chatbot component

```typescript
interface ChatbotState {
  currentQuestion: string;              // The current question being typed
  conversationHistory: FrontendConversation[]; // All conversations
  isLoading: boolean;                   // Whether a request is in progress
  error: string | null;                 // Current error message
  backendStatus: 'unknown' | 'healthy' | 'unhealthy'; // Backend health status
  isChatOpen: boolean;                  // Whether the chat interface is open
  settings: ChatbotSettings;            // Current component settings
}
```

### 6. ChatbotSettings Model

**Purpose**: Represents configurable settings for the chatbot component

```typescript
interface ChatbotSettings {
  position: 'bottom-right' | 'bottom-left' | 'side' | 'embedded';
  theme: 'light' | 'dark';
  maxHistory: number;                   // Maximum number of conversations to store
  enableHistory: boolean;               // Whether to store conversation history
  placeholder: string;                  // Input field placeholder text
  backendUrl: string;                   // URL of the backend API
  maxQuestionLength: number;            // Maximum length of user questions
  responseTimeout: number;              // Timeout for API responses in milliseconds
}
```

## Data Transformation Mappings

### Backend to Frontend Transformation

When receiving data from the backend, the frontend performs these transformations:

1. **APIResponse → FrontendResponse**:
   - Convert ISO timestamp string to Date object
   - Map SourceChunk[] to FrontendSource[]
   - Add display formatting to source content

2. **SourceChunk → FrontendSource**:
   - Add linkable property based on URL validity
   - Create displayText with proper formatting
   - Convert similarity_score to similarityScore (camelCase)

### Frontend to Backend Transformation

When sending data to the backend, the frontend performs these transformations:

1. **FrontendQuestion → QueryRequest**:
   - Trim whitespace from question text
   - Apply default values for optional parameters
   - Validate question length

## Data Validation Rules

### Frontend Validation
- Question length: 3-1000 characters
- URL format validation for source links
- HTML sanitization of responses
- Rate limiting of requests (max 1 per 2 seconds)

### Backend Validation
- QueryRequest validation (per backend model)
- Response content validation
- Source attribution verification

## Data Flow Patterns

### 1. Question Submission Flow
```
User Input → FrontendValidation → QueryRequest → Backend API → APIResponse → FrontendResponse → UI Update
```

### 2. Source Attribution Flow
```
Backend SourceChunk → FrontendSource → Linkable Format → UI Display → Source Attribution
```

### 3. Error Handling Flow
```
Backend ErrorResponse → Frontend Error State → User-Friendly Message → UI Feedback
```

## Performance Considerations

### Data Size Limits
- Maximum conversation history: 50 question-response pairs
- Maximum source chunks per response: 10
- Maximum response content size: 5KB per response
- Maximum source content snippet: 500 characters

### Caching Strategy
- Conversation history stored in browser localStorage
- Source metadata cached for 24 hours
- Backend health status cached for 5 minutes