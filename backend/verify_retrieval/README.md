# RAG Retrieval Verification

This module provides functionality to verify the RAG (Retrieval-Augmented Generation) pipeline by testing the retrieval of stored vector embeddings from Qdrant and validating their accuracy and metadata consistency.

## Overview

The RAG Retrieval Verification system:
- Loads vectors and metadata stored in Qdrant from the original ingestion pipeline
- Implements retrieval functions to query Qdrant using sample keywords or phrases
- Validates that retrieved chunks are accurate and relevant
- Checks that metadata (URL, title, chunk_id) matches source content
- Logs results and confirms the pipeline executes end-to-end without errors

## Usage

Run the verification pipeline:
```bash
cd backend
python -m verify_retrieval.main
```

Or run with specific query:
```bash
python -m verify_retrieval.main --query "your search query here"
```

## Configuration

The verification system requires the following environment variables to be set:

- `QDRANT_URL`: URL of your Qdrant instance
- `QDRANT_API_KEY`: API key for Qdrant access
- `QDRANT_COLLECTION_NAME`: Name of the collection containing embeddings (default: "rag_embedding")

## Components

- `qdrant_client.py`: Handles all Qdrant interactions and queries
- `validators.py`: Contains functions to validate chunk accuracy and metadata consistency
- `reporters.py`: Generates verification reports and logs
- `config.py`: Manages configuration and environment variables
- `main.py`: Main entry point with CLI interface

## Success Criteria

- All book pages from the deployed website can be retrieved via sample semantic queries
- Retrieval returns relevant chunks based on keyword or phrase search (similarity > 0.7)
- Metadata accuracy is 100% - all URL, title, and chunk index fields match source documents
- Pipeline execution completes without exceptions
- Pipeline is repeatable and idempotent