# Quickstart: RAG Retrieval Pipeline Verification

## Overview
Quick setup and usage guide for the RAG retrieval verification system that validates vector embeddings and semantic search functionality.

## Prerequisites
- Python 3.10 or higher
- Access to Qdrant Cloud instance with RAG embeddings stored
- API keys for Qdrant access

## Setup

### 1. Clone and Navigate
```bash
cd /path/to/your/project
cd backend
```

### 2. Install Dependencies
```bash
uv sync
# or if using pip
pip install -e .
```

### 3. Configure Environment
```bash
cp .env.example .env
```

Edit `.env` with your Qdrant configuration:
```env
QDRANT_URL=https://your-instance.qdrant.tech:6333
QDRANT_API_KEY=your-api-key-here
QDRANT_COLLECTION_NAME=rag_embedding
```

## Basic Usage

### Run Verification Pipeline
```bash
# Run complete verification pipeline
cd backend
python -m verify_retrieval.main

# Run with specific query
python -m verify_retrieval.main --query "transformer architecture in NLP"

# Run with specific number of results
python -m verify_retrieval.main --top-k 10

# Run with minimum similarity threshold
python -m verify_retrieval.main --min-similarity 0.7

# Run idempotency check with multiple runs
python -m verify_retrieval.main --idempotency-check --runs 3
```

## Verification Results

The verification will output:
- **Status**: SUCCESS, PARTIAL_SUCCESS, or FAILURE
- **Metadata Accuracy**: Percentage of metadata fields that match source content
- **Content Relevance**: Average similarity score of retrieved chunks
- **Execution Time**: Time taken to complete verification
- **Detailed Report**: Per-query breakdown of results

## Sample Output
```
RAG Retrieval Verification Results
=================================
Query: "transformer architecture in NLP"
Status: SUCCESS
Metadata Accuracy: 100.0%
Content Relevance: 88.5%
Execution Time: 125.5ms
Retrieved Chunks: 5/5 met minimum similarity threshold

Detailed Results:
- Chunk 1: Similarity=0.85, URL: https://book.com/chapter-3
- Chunk 2: Similarity=0.78, URL: https://book.com/chapter-5
...
```

## Troubleshooting

### Common Issues

**Qdrant Connection Errors**:
- Verify QDRANT_URL and QDRANT_API_KEY are correct
- Check network connectivity to Qdrant instance
- Ensure the collection name is correct

**No Results Returned**:
- Check if embeddings exist in Qdrant collection
- Verify minimum similarity threshold is not too high
- Confirm the query text is semantically meaningful

**Low Metadata Accuracy**:
- Verify that the original ingestion pipeline stored correct metadata
- Check for any changes in metadata format between ingestion and verification

## Next Steps

1. **Custom Queries**: Create your own verification queries specific to your content
2. **Integration Testing**: Integrate verification into your CI/CD pipeline
3. **Performance Monitoring**: Set up regular verification runs to monitor pipeline health
4. **Threshold Tuning**: Adjust similarity thresholds based on your specific requirements