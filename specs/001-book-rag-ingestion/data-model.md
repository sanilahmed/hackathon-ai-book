# Data Model: Book Content Ingestion System

## Overview
This document defines the data structures and models used in the book content ingestion system that extracts content from Docusaurus-based book websites, chunks and embeds it using Cohere, and stores embeddings in Qdrant Cloud.

## Core Data Entities

### Book Content Chunk
The primary data entity representing a processed segment of book content.

**Fields:**
- `id` (string): Unique identifier for the chunk in Qdrant (auto-generated)
- `text` (string): The actual text content of the chunk
- `url` (string): Source URL where the content was extracted from
- `title` (string): Title of the source page
- `source` (string): Identifier for the source system ("book_ingestion")
- `chunk_index` (integer): Position of this chunk within the original document
- `total_chunks` (integer): Total number of chunks from the same document
- `timestamp` (float): Unix timestamp when the chunk was stored
- `embedding` (array[float]): Vector representation of the text (1024 dimensions for Cohere)

### URL Collection
Represents the discovered URLs from the Docusaurus book site.

**Fields:**
- `url` (string): The discovered URL
- `title` (string): Page title (if available)
- `text_length` (integer): Length of extracted text
- `is_processed` (boolean): Whether the URL has been processed

### Embedding Vector
The vector representation of text content for semantic search.

**Fields:**
- `vector` (array[float]): The embedding values (1024-dimensional for Cohere)
- `model_name` (string): Name of the embedding model used
- `input_text_hash` (string): Hash of the original text for reference

## Qdrant Collection Schema

### Collection: "rag_embedding"
**Vector Configuration:**
- Size: 1024 (for Cohere embeddings)
- Distance: Cosine
- Datatype: Float

**Payload Fields:**
- `text`: String (the original text chunk)
- `url`: String (source URL)
- `title`: String (document title)
- `source`: String (always "book_ingestion")
- `timestamp`: Float (unix timestamp of when the chunk was stored)

## Data Relationships

### Content Hierarchy
- One book URL can produce multiple content chunks
- Each chunk maintains its position within the original document via `chunk_index` and `total_chunks`

### Processing Flow
1. **Source URL** → **Extracted Content** → **Chunked Content** → **Embedded Vector** → **Stored in Qdrant**

## Data Validation Rules

### Content Extraction
- Text must be non-empty after cleaning
- URL must be from the same domain as the base book URL
- Title must be captured from the page

### Chunking
- Each chunk should be between 100 and 1000 characters
- Chunks should preserve sentence or paragraph boundaries where possible
- Metadata must be preserved for each chunk

### Embedding
- Embedding vectors must have exactly 1024 dimensions (for Cohere)
- Text chunks must be valid for embedding generation
- API rate limits must be respected

## Data Lifecycle

### Creation
1. URL discovered from book site
2. Content extracted and cleaned
3. Content chunked into segments
4. Embeddings generated
5. Data stored in Qdrant with metadata

### Access
- Vector similarity search through Qdrant API
- Metadata retrieval for context
- Original text access for RAG applications

### Retention
- Data stored permanently in Qdrant Cloud until explicitly deleted
- No automatic cleanup unless implemented separately