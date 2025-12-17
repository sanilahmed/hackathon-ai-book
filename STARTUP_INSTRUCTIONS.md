# 🚀 Startup Instructions for RAG Chatbot

Complete guide to get both frontend and backend running together.

## Quick Start (Recommended)

```bash
# From the project root directory
cd /mnt/d/Hackathon/Book

# Run the setup script to start both servers
./setup_and_start.sh
```

This will:
1. Install backend dependencies with pip
2. Start the FastAPI backend server on port 8000
3. Install frontend dependencies
4. Start the Docusaurus frontend on port 3000

## Manual Setup (Alternative)

### 1. Start Backend Server

```bash
# Open a new terminal and navigate to backend
cd /mnt/d/Hackathon/Book/backend

# Install dependencies
pip install -e .

# Start the FastAPI server
uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
```

### 2. Start Frontend Server

```bash
# Open another terminal and navigate to project root
cd /mnt/d/Hackathon/Book

# Install dependencies
npm install

# Start Docusaurus development server
npm start
```

## What You'll Get

- **Backend API**: http://localhost:8000
  - `/health` - Health check
  - `/ask` - Question answering endpoint
  - `/docs` - API documentation

- **Frontend UI**: http://localhost:3000
  - Docusaurus book interface with integrated chatbot
  - Ask questions about book content and get AI-generated answers

## Troubleshooting

### If the backend doesn't start:
- Check that all API keys in `backend/.env` are valid
- Verify Qdrant connection details
- Check `backend/backend.log` for error details

### If the frontend doesn't start:
- Make sure Node.js 18+ is installed
- Verify the backend server is running on port 8000
- Check that you're in the correct directory (`/mnt/d/Hackathon/Book`)

### Quick Checks:
```bash
# Check backend health
curl http://localhost:8000/health

# Check frontend
open http://localhost:3000  # or visit in browser
```

## Environment Configuration

Your `backend/.env` file already contains all required API keys:
- OpenAI API Key
- Qdrant URL and API Key
- Cohere API Key
- Server configuration (port 8000)

No additional configuration is needed!

## Next Steps

Once both servers are running:
1. Visit http://localhost:3000 in your browser
2. Look for the chatbot interface (usually bottom-right)
3. Ask questions about the book content
4. Receive AI-generated answers with source citations