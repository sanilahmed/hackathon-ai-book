#!/bin/bash
# Script to start the backend server

echo "Starting backend server..."

# Change to backend directory
cd /mnt/d/Hackathon/Book/backend

# Install dependencies if needed
pip install -e . > /dev/null 2>&1 || echo "Dependencies already installed or error"

# Start the FastAPI server
echo "Starting FastAPI server on port 8000..."
uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000 --reload --log-level warning