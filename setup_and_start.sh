#!/bin/bash
# Setup script for RAG Chatbot - Frontend and Backend

echo "Setting up RAG Chatbot - Frontend and Backend..."

# Function to start backend
start_backend() {
    echo "Starting backend server..."
    cd backend

    # Install dependencies with pip (since uv had memory issues)
    echo "Installing backend dependencies with pip..."
    pip install -e . 2>/dev/null || echo "Could not install backend dependencies with pip"

    # Start the FastAPI server
    echo "Starting FastAPI server on port 8000..."
    if command -v uvicorn &> /dev/null; then
        uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
    else
        python -m rag_agent_api.main
    fi
}

# Function to start frontend
start_frontend() {
    echo "Starting Docusaurus frontend..."
    cd ..

    # Install frontend dependencies
    echo "Installing Docusaurus dependencies..."
    npm install

    # Start the Docusaurus development server
    echo "Starting Docusaurus server on port 3000..."
    npm start
}

# Check if both servers should be started
if [ "$1" = "backend" ]; then
    start_backend
elif [ "$1" = "frontend" ]; then
    start_frontend
else
    echo "Starting both backend and frontend..."

    # Start backend in background
    echo "Starting backend server in background..."
    cd backend

    # Install dependencies with pip
    echo "Installing backend dependencies with pip..."
    pip install -e . 2>/dev/null || echo "Could not install backend dependencies"

    # Start backend server in background
    if command -v uvicorn &> /dev/null; then
        uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
    else
        python -m rag_agent_api.main > backend.log 2>&1 &
    fi

    BACKEND_PID=$!
    echo "Backend started with PID $BACKEND_PID"

    # Wait a moment for backend to start
    sleep 8

    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✓ Backend is running on http://localhost:8000"
    else
        echo "⚠ Warning: Backend may not be running properly. Check backend.log for details."
    fi

    # Start frontend
    cd ..
    echo "Starting frontend..."

    # Install frontend dependencies
    npm install 2>/dev/null || echo "npm install failed, continuing..."

    # Start the Docusaurus development server
    echo "Starting Docusaurus server on http://localhost:3000..."
    echo "✓ Docusaurus server will be available at http://localhost:3000"
    npm start
fi