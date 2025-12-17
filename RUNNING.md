# RAG Chatbot - Frontend and Backend Setup

This guide will help you set up and run both the Docusaurus frontend and FastAPI backend for the RAG Chatbot system.

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- uv package manager (optional but recommended)

## Quick Start

### Method 1: Using the setup script (Recommended)

```bash
# Make sure you're in the project root directory
cd /mnt/d/Hackathon/Book

# Run the setup script to start both servers
./setup_and_start.sh
```

This will:
1. Install backend dependencies
2. Start the FastAPI backend server on port 8000
3. Install frontend dependencies
4. Start the Docusaurus frontend on port 3000

### Method 2: Manual setup

#### 1. Start the Backend Server

```bash
# Navigate to the backend directory
cd backend

# Activate virtual environment if needed
source .venv/bin/activate  # if you have a .venv

# Install dependencies
uv sync  # or pip install -e .

# Start the FastAPI server
uv run uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
# OR
python -m rag_agent_api.main
```

The backend should be available at: http://localhost:8000

#### 2. Start the Frontend Server

In a new terminal, navigate to the project root:

```bash
cd /mnt/d/Hackathon/Book

# Install dependencies
npm install

# Start the Docusaurus development server
npm start
```

The frontend should be available at: http://localhost:3000

## API Endpoints

### Backend (FastAPI server on port 8000)
- `GET /` - API information
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `POST /ask` - Main question-answering endpoint
- `GET /docs` - API documentation (Swagger UI)
- `GET /redoc` - API documentation (Redoc)

### Frontend (Docusaurus on port 3000)
- `http://localhost:3000` - Main book interface with integrated chatbot

## Environment Variables

The backend uses the following environment variables (already configured in `.env`):
- `OPENAI_API_KEY` - Your OpenAI API key
- `QDRANT_URL` - Your Qdrant Cloud URL
- `QDRANT_API_KEY` - Your Qdrant API key
- `COHERE_API_KEY` - Your Cohere API key
- `QDRANT_COLLECTION_NAME` - Name of the Qdrant collection (default: "rag_embedding")

## Troubleshooting

### Backend Issues
1. Make sure all API keys are valid in the `.env` file
2. Check that Qdrant is accessible and the collection exists
3. Verify that required Python packages are installed

### Frontend Issues
1. Make sure the backend server is running on port 8000
2. Check that the chatbot component is properly integrated
3. Verify CORS settings if accessing from different domains

### Common Commands
```bash
# Check backend health
curl http://localhost:8000/health

# Test the question-answering endpoint
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this book about?"}'

# Check if frontend is running
curl http://localhost:3000
```

## Stopping the Servers

To stop the servers:
1. Press `Ctrl+C` in the terminal where each server is running
2. If running in background, use `kill` command with the process ID

## Testing the Integration

Once both servers are running:
1. Open http://localhost:3000 in your browser
2. Look for the chatbot interface (usually at bottom-right corner)
3. Enter a question related to the book content
4. You should receive an AI-generated answer with source attribution