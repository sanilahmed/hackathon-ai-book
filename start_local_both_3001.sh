#!/bin/bash
# Local development startup script for RAG Chatbot - Both Frontend and Backend on Port 3001
# Use this script to start both systems with frontend on port 3001

echo "🚀 Starting RAG Chatbot - Full Local Development Mode (Frontend on Port 3001)"
echo "💡 This script runs the frontend on port 3001 and backend on port 8000"

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn" 2>/dev/null
pkill -f "docusaurus" 2>/dev/null
pkill -f "npm start" 2>/dev/null

# Start backend
echo "📦 Starting Backend Server..."
cd /mnt/d/Hackathon/book/backend

# Install dependencies
echo "Installing backend dependencies..."
pip install -e . > /dev/null 2>&1 || echo "Backend dependencies already installed"

# Start backend server
echo "Starting FastAPI server on port 8000..."
nohup uvicorn rag_agent_api.main:app --host 127.0.0.1 --port 8000 --log-level warning > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 10

# Verify backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend running: http://localhost:8000"
else
    echo "❌ Backend failed to start. Check backend.log"
    exit 1
fi

# Start frontend
echo "🌐 Building and Starting Frontend Server on Port 3001..."
cd /mnt/d/Hackathon/book

# Install frontend dependencies
echo "Installing frontend dependencies..."
npm install > /dev/null 2>&1 || echo "Frontend dependencies already installed"

# Build the site
echo "Building Docusaurus site..."
npm run build > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Check for errors."
    exit 1
fi

echo "✅ Build successful!"

# Start the development server on port 3001
echo "Starting Docusaurus server on port 3001..."
nohup npx docusaurus start --host 127.0.0.1 --port 3001 --no-open > frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

# Wait for frontend to start
sleep 8

echo ""
echo "🎉 FULL DEVELOPMENT SETUP COMPLETE!"
echo ""
echo "📍 Backend API: http://localhost:8000"
echo "📍 Frontend UI: http://localhost:3001"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 The site is now available at: http://localhost:3001"
echo ""
echo "📝 Logs:"
echo "   - backend.log (backend logs)"
echo "   - frontend.log (frontend logs)"
echo ""
echo "🔄 To stop servers: pkill -f 'uvicorn\|docusaurus'"
echo ""
echo "🔍 To verify everything is working:"
echo "   - Check backend: curl http://localhost:8000/health"
echo "   - Frontend should be accessible at the URL above"