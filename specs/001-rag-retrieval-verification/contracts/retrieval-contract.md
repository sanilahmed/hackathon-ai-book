# API Contract: RAG Retrieval Verification

## Overview
API contract for the RAG retrieval verification system, defining the interfaces for querying, validation, and reporting functions.

## Qdrant Integration Contract

### Function: `query_qdrant_for_chunks(query_text, top_k=5, min_similarity=0.5)`

**Purpose**: Query Qdrant vector database using semantic search to retrieve relevant content chunks.

**Input Parameters**:
- `query_text` (string, required): The semantic query text to search for
- `top_k` (integer, optional): Number of results to return, default: 5, range: 1-100
- `min_similarity` (float, optional): Minimum similarity threshold, default: 0.5, range: 0.0-1.0

**Returns**:
- `results` (array): Array of Retrieved Content Chunk objects
- `query_vector` (array): The vector representation of the query
- `execution_time_ms` (float): Time taken for the query execution

**Success Criteria**:
- Returns up to `top_k` results that meet the `min_similarity` threshold
- All results include content, similarity score, and metadata
- Execution completes within performance goals

**Error Conditions**:
- Qdrant connection failure: Returns error with connection details
- Invalid parameters: Returns error with parameter validation details
- Empty result set: Returns empty array with appropriate status

### Function: `validate_metadata_consistency(results)`

**Purpose**: Validate that metadata in retrieved results matches expected source content metadata.

**Input Parameters**:
- `results` (array, required): Array of Retrieved Content Chunk objects from query

**Returns**:
- `validation_report` (object): Report with validation results
  - `accuracy_percentage` (float): Percentage of metadata fields that match
  - `errors` (array): List of validation errors found
  - `details` (array): Per-result validation details

**Success Criteria**:
- Returns accuracy percentage between 0.0 and 1.0
- Identifies any metadata inconsistencies
- Provides detailed validation information

## Verification Pipeline Contract

### Function: `run_verification_pipeline(queries, options)`

**Purpose**: Execute complete verification pipeline with multiple queries and validation checks.

**Input Parameters**:
- `queries` (array, required): Array of query strings to test
- `options` (object, optional): Configuration options
  - `collection_name` (string): Qdrant collection to query, default: "rag_embedding"
  - `top_k` (integer): Number of results per query, default: 5
  - `min_similarity` (float): Minimum similarity threshold, default: 0.5

**Returns**:
- `verification_result` (object): Complete verification result
  - `status` (string): Overall verification status
  - `summary` (object): Summary statistics
    - `total_queries` (integer): Number of queries executed
    - `successful_queries` (integer): Number of successful queries
    - `metadata_accuracy` (float): Overall metadata accuracy
    - `content_relevance` (float): Average content relevance
  - `detailed_results` (array): Per-query results
  - `execution_time_ms` (float): Total execution time
  - `errors` (array): Any errors encountered

**Success Criteria**:
- Executes all queries successfully
- Returns comprehensive validation results
- Completes within performance goals
- Maintains idempotent behavior

## Data Contracts

### Retrieved Content Chunk Schema
```json
{
  "content": {
    "type": "string",
    "required": true,
    "description": "The actual text content of the chunk"
  },
  "similarity_score": {
    "type": "number",
    "required": true,
    "min": 0.0,
    "max": 1.0,
    "description": "Semantic similarity score between 0.0 and 1.0"
  },
  "vector_id": {
    "type": "string",
    "required": true,
    "description": "Unique identifier for the vector in Qdrant"
  },
  "metadata": {
    "type": "object",
    "required": true,
    "properties": {
      "url": {
        "type": "string",
        "required": true,
        "format": "uri",
        "description": "Source URL of the original content"
      },
      "title": {
        "type": "string",
        "required": true,
        "description": "Title of the source document/page"
      },
      "chunk_index": {
        "type": "integer",
        "required": true,
        "minimum": 0,
        "description": "Index of this chunk within the source"
      },
      "total_chunks": {
        "type": "integer",
        "required": true,
        "minimum": 1,
        "description": "Total number of chunks from same source"
      }
    }
  }
}
```

## Error Handling Contract

All functions must follow this error handling pattern:

```json
{
  "success": false,
  "error": {
    "type": "string",
    "message": "string",
    "details": "object or null",
    "timestamp": "ISO 8601 datetime"
  }
}
```

## Performance Contract

- Query response time: Under 2 seconds for typical queries
- Metadata validation: Under 100ms per result
- Full verification pipeline: Under 30 seconds for standard test suite
- Memory usage: Under 500MB for typical execution

## Security Contract

- API keys must be loaded from environment variables, never hardcoded
- No sensitive information should be logged
- Connection to Qdrant must use secure protocols (HTTPS)
- Input validation must prevent injection attacks