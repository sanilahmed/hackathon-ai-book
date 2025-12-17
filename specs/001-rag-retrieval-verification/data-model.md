# Data Model: RAG Retrieval Pipeline Verification

## Overview
Data structures for the RAG retrieval verification system, defining entities for content chunks, metadata, queries, and verification results.

## Entity Definitions

### Retrieved Content Chunk
**Description**: Represents a segment of book content returned by semantic search, including the text content, similarity score, and associated metadata.

**Attributes**:
- `content` (string): The actual text content of the chunk
- `similarity_score` (float): The semantic similarity score (0.0 to 1.0) to the query
- `vector_id` (string): Unique identifier for the vector in Qdrant
- `metadata` (object): Associated metadata record for the chunk

**Constraints**:
- Similarity score must be between 0.0 and 1.0
- Content must not be empty
- Vector ID must be unique within the collection

### Metadata Record
**Description**: Contains information about the original source of content (URL, title, chunk index, etc.) that helps provide context for retrieved results.

**Attributes**:
- `url` (string): The source URL of the original content
- `title` (string): The title of the source document/page
- `chunk_index` (integer): The index of this chunk within the original content
- `total_chunks` (integer): Total number of chunks from the same source document
- `source_document_id` (string): Identifier for the source document

**Constraints**:
- URL must be a valid URL format
- Chunk index must be non-negative
- Total chunks must be greater than 0 and >= chunk_index

### Query Request
**Description**: Represents a semantic search request from a user, containing the search terms and parameters for retrieval.

**Attributes**:
- `query_text` (string): The text to search for using semantic similarity
- `top_k` (integer): Number of results to return (default: 5)
- `min_similarity` (float): Minimum similarity threshold for results (default: 0.5)
- `filters` (object): Optional filters to apply during search

**Constraints**:
- Query text must not be empty
- Top_k must be between 1 and 100
- Min_similarity must be between 0.0 and 1.0

### Verification Result
**Description**: Contains the outcome of pipeline validation, including success/failure status, error details, and execution metrics.

**Attributes**:
- `status` (enum): Verification status (SUCCESS, PARTIAL_SUCCESS, FAILURE)
- `timestamp` (datetime): When the verification was performed
- `query` (Query Request): The original query that was verified
- `retrieved_chunks` (array): List of Retrieved Content Chunks returned
- `metadata_accuracy` (float): Accuracy score for metadata validation (0.0 to 1.0)
- `content_relevance` (float): Relevance score for content validation (0.0 to 1.0)
- `execution_time_ms` (float): Time taken to execute the verification
- `errors` (array): List of any errors encountered during verification

**Constraints**:
- Status must be one of the defined enum values
- Metadata and content accuracy scores must be between 0.0 and 1.0
- Execution time must be non-negative

## Relationships

```
Query Request 1 -- * Verification Result
Verification Result 1 -- * Retrieved Content Chunk
Retrieved Content Chunk 1 -- 1 Metadata Record
```

## Validation Rules

### For Retrieved Content Chunk:
- Content similarity score must exceed minimum threshold to be considered valid
- Vector ID must correspond to an actual vector in Qdrant
- Metadata must be present and valid

### For Metadata Record:
- URL must match the original source document
- Chunk index and total chunks must form a valid sequence
- All metadata fields must be present and non-empty

### For Verification Result:
- Status must accurately reflect the outcome of the verification
- Retrieved chunks count must match the query's top_k parameter (or less if fewer results available)
- Execution time must be recorded for performance monitoring

## Schema Examples

### Sample Retrieved Content Chunk:
```json
{
  "content": "The transformer architecture revolutionized natural language processing...",
  "similarity_score": 0.85,
  "vector_id": "chunk_12345",
  "metadata": {
    "url": "https://example.com/book/chapter-1",
    "title": "Introduction to Transformers",
    "chunk_index": 2,
    "total_chunks": 10,
    "source_document_id": "doc_67890"
  }
}
```

### Sample Verification Result:
```json
{
  "status": "SUCCESS",
  "timestamp": "2025-12-16T10:30:00Z",
  "query": {
    "query_text": "transformer architecture in NLP",
    "top_k": 5,
    "min_similarity": 0.7
  },
  "retrieved_chunks": [/* array of Retrieved Content Chunks */],
  "metadata_accuracy": 1.0,
  "content_relevance": 0.88,
  "execution_time_ms": 125.5,
  "errors": []
}
```