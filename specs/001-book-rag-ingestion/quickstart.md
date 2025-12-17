# Quickstart Guide: Book Content Ingestion System

## Overview
This guide will help you quickly set up and run the book content ingestion system that extracts content from Docusaurus-based book websites, chunks and embeds it using Cohere, and stores embeddings in Qdrant Cloud.

## Prerequisites
- Python 3.10 or higher
- uv package manager installed
- Cohere API key
- Qdrant Cloud account and API credentials

## Setup Steps

### 1. Navigate to the Backend Directory
```bash
cd backend
```

### 2. Install Dependencies with uv
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# OR
.venv\Scripts\activate  # On Windows

# Install project dependencies
uv sync
```

### 3. Configure Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your actual credentials
nano .env  # or use your preferred editor
```

Set these values in your `.env` file:
```
COHERE_API_KEY=your_cohere_api_key_here
QDRANT_URL=your_qdrant_cloud_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

## Running the Ingestion Pipeline

### Execute the Complete Pipeline
```bash
uv run python main.py
```

This command will:
1. Discover all URLs in https://sanilahmed.github.io/hackathon-ai-book/
2. Extract clean text content from each URL
3. Chunk content into fixed-size segments (max 1000 characters)
4. Generate embeddings using Cohere
5. Store embeddings with metadata in Qdrant Cloud collection named "rag_embedding"

## Expected Output
The system will display progress logs showing:
- URL discovery progress
- Content extraction status
- Embedding generation progress
- Storage confirmation messages
- Final summary of processed content

## Verification
After completion, verify that:
1. The "rag_embedding" collection exists in your Qdrant Cloud instance
2. The collection contains the expected number of points (embeddings)
3. Each point has the correct metadata (text, URL, title, etc.)

## Troubleshooting

### Common Issues
1. **API Key Errors**: Verify your API keys are correctly set in `.env`
2. **Network Issues**: Ensure the target book URL is accessible
3. **Rate Limiting**: The system includes delays to respect API limits

### Quick Checks
```bash
# Verify environment variables are loaded
python -c "import os; print('COHERE_API_KEY available:', bool(os.getenv('COHERE_API_KEY')))"
python -c "import os; print('QDRANT credentials available:', bool(os.getenv('QDRANT_URL') and os.getenv('QDRANT_API_KEY')))"
```

## Next Steps
After successful ingestion:
1. Use the stored embeddings for RAG applications
2. Query the Qdrant collection for semantic search
3. Build applications that leverage the embedded book content