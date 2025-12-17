#!/bin/bash

# Script to start both frontend and backend for the RAG chatbot application

set -e  # Exit on any error

echo "Starting RAG Chatbot Application..."

# Function to start backend
start_backend() {
    echo "Starting backend server..."
    cd backend
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
}

# Function to start frontend
start_frontend() {
    echo "Starting frontend server..."
    cd /mnt/d/Hackathon/Book  # Change to your project directory
    npm run start
}

# Function to start both
start_both() {
    echo "Starting both frontend and backend..."

    # Start backend in background
    cd backend
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi
    uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"

    # Wait a moment for backend to start
    sleep 3

    # Start frontend in foreground
    cd /mnt/d/Hackathon/Book
    npm run start

    # Cleanup on exit
    echo "Stopping backend server..."
    kill $BACKEND_PID
}

# Parse command line arguments
case "$1" in
    "backend")
        start_backend
        ;;
    "frontend")
        start_frontend
        ;;
    "both"|"")
        start_both
        ;;
    *)
        echo "Usage: $0 [backend|frontend|both]"
        echo "  backend  - Start only the backend server"
        echo "  frontend - Start only the frontend server"
        echo "  both     - Start both backend and frontend (default)"
        exit 1
        ;;
esac