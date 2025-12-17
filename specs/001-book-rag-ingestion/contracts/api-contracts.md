# API Contracts: Book Content Ingestion System

## Overview
This document defines the contracts and interfaces for the book content ingestion system that extracts content from Docusaurus-based book websites, chunks and embeds it using Cohere, and stores embeddings in Qdrant Cloud.

## Internal Function Contracts

### 1. get_all_urls Function
**Signature**: `get_all_urls(base_url: str) -> List[str]`

**Purpose**: Discover and validate all URLs from a Docusaurus book website.

**Parameters**:
- `base_url` (str): The base URL of the Docusaurus book (e.g., "https://sanilahmed.github.io/hackathon-ai-book/")

**Returns**:
- List of valid URLs found on the site, all from the same domain as the base URL

**Preconditions**:
- `base_url` must be a valid, accessible URL
- Network connectivity must be available

**Postconditions**:
- Returns only URLs from the same domain as the base URL
- Returns a list of unique URLs
- Returns URLs in a discoverable format

**Error Conditions**:
- Returns empty list if base URL is inaccessible
- May raise exceptions for network errors (to be handled by caller)

### 2. extract_text_from_url Function
**Signature**: `extract_text_from_url(url: str) -> Dict[str, Any]`

**Purpose**: Extract and clean textual content from a single URL.

**Parameters**:
- `url` (str): The URL to extract content from

**Returns**:
- Dictionary with keys: 'url', 'title', 'text', 'length'
  - 'url': The source URL
  - 'title': Page title
  - 'text': Clean extracted text content
  - 'length': Length of extracted text

**Preconditions**:
- URL must be accessible
- URL should contain HTML content

**Postconditions**:
- Returns clean text without HTML tags, navigation, or styling elements
- Preserves main content of the page
- Title is extracted from the page

**Error Conditions**:
- Returns dictionary with empty text if content extraction fails
- Logs error but doesn't raise exception

### 3. chunk_text Function
**Signature**: `chunk_text(text: str, chunk_size: int = 1000) -> List[str]`

**Purpose**: Split text into fixed-size segments suitable for embedding generation.

**Parameters**:
- `text` (str): The text to chunk
- `chunk_size` (int): Maximum size of each chunk in characters (default: 1000)

**Returns**:
- List of text chunks, each under the specified size
- Empty list if input text is empty

**Preconditions**:
- Input text should be clean and processed

**Postconditions**:
- Each chunk is at most `chunk_size` characters
- Chunks preserve sentence or paragraph boundaries when possible
- No empty chunks in the returned list

**Error Conditions**:
- Returns empty list for empty input
- May split mid-sentence if no good breaking points found

### 4. embed Function
**Signature**: `embed(texts: List[str]) -> List[List[float]]`

**Purpose**: Generate embeddings using Cohere API.

**Parameters**:
- `texts` (List[str]): List of text chunks to generate embeddings for

**Returns**:
- List of embeddings, where each embedding is a list of floats
- Length of returned list matches length of input list

**Preconditions**:
- `COHERE_API_KEY` environment variable must be set
- Valid Cohere API credentials
- Each text in the list should be within Cohere's size limits

**Postconditions**:
- Returns embeddings with correct dimensions (1024 for Cohere)
- Each embedding corresponds positionally to input text

**Error Conditions**:
- Raises ValueError if API key is not set
- Raises Exception for API errors, rate limits, or network issues

### 5. create_collection Function
**Signature**: `create_collection(client: QdrantClient, collection_name: str = "rag_embedding") -> None`

**Purpose**: Create a collection in Qdrant for storing embeddings.

**Parameters**:
- `client` (QdrantClient): Initialized Qdrant client instance
- `collection_name` (str): Name of the collection to create (default: "rag_embedding")

**Returns**:
- None

**Preconditions**:
- Valid Qdrant client with proper credentials
- Client must have permissions to create collections

**Postconditions**:
- Collection exists in Qdrant with appropriate vector configuration
- If collection already exists, no error is raised
- Vector configuration is set for 1024-dimension embeddings with cosine distance

**Error Conditions**:
- Raises Exception for connection errors or permission issues

### 6. save_chunk_to_qdrant Function
**Signature**:
```python
save_chunk_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunk: str,
    embedding: List[float],
    metadata: Dict[str, Any]
) -> None
```

**Purpose**: Save a text chunk with its embedding to Qdrant.

**Parameters**:
- `client` (QdrantClient): Initialized Qdrant client instance
- `collection_name` (str): Name of the collection to save to
- `chunk` (str): The text chunk to save
- `embedding` (List[float]): The embedding vector
- `metadata` (Dict[str, Any]): Additional metadata with keys: 'url', 'title', 'source', etc.

**Returns**:
- None

**Preconditions**:
- Valid Qdrant client connection
- Collection must exist
- Embedding must have correct dimensions (1024 for Cohere)

**Postconditions**:
- Chunk and embedding are stored in Qdrant with metadata
- Unique ID is generated for the stored point
- Point is searchable using vector similarity

**Error Conditions**:
- Raises Exception for connection errors, invalid data, or storage failures

### 7. main Function
**Signature**: `main() -> None`

**Purpose**: Execute the complete book content ingestion pipeline.

**Parameters**:
- None (reads configuration from environment variables)

**Returns**:
- None

**Preconditions**:
- Environment variables set: `COHERE_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
- Target book URL (https://sanilahmed.github.io/hackathon-ai-book/) is accessible
- Network connectivity available

**Postconditions**:
- All content from the book site is processed
- Embeddings are stored in Qdrant collection named "rag_embedding"
- Proper logging of progress and errors
- Graceful handling of partial failures

**Error Conditions**:
- Raises ValueError if required environment variables are missing
- Raises other exceptions for various processing failures

## External API Contracts

### Cohere API Integration
**Service**: Cohere Embed API
**Model**: embed-english-v3.0 (or equivalent)
**Dimensions**: 1024
**Rate Limits**: As per Cohere account tier
**Input Type**: "search_document" for document embeddings

### Qdrant Cloud API Integration
**Collection Vector Configuration**:
- Size: 1024
- Distance: Cosine
- Datatype: Float

**Payload Schema**:
- text: String (original text chunk)
- url: String (source URL)
- title: String (document title)
- source: String (value: "book_ingestion")
- timestamp: Float (unix timestamp)

## Error Handling Contracts

### Network Errors
- All network operations have timeout configurations
- Retry mechanisms with exponential backoff for transient failures
- Graceful degradation when services are unavailable

### Data Validation
- Input validation at function boundaries
- Type checking where appropriate
- Range validation for parameters like chunk_size

### Logging Standards
- All functions log significant operations
- Error conditions are logged with appropriate context
- Progress updates are provided for long-running operations