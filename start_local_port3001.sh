#!/bin/bash
# Local development startup script for RAG Chatbot - Port 3001
# Use this script when port 3000 is already in use

echo "🚀 Starting RAG Chatbot - Local Development Mode (Port 3001)"
echo "💡 This script runs the frontend on port 3001 instead of 3000"

# Clean up any existing processes
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn" 2>/dev/null
pkill -f "docusaurus" 2>/dev/null
pkill -f "npm start" 2>/dev/null

# Start backend
echo "📦 Starting Backend Server..."
cd /mnt/d/Hackathon/Book/backend

# Install dependencies
pip install -e . > /dev/null 2>&1 || echo "Backend dependencies already installed"

# Start backend server
nohup uvicorn rag_agent_api.main:app --host 127.0.0.1 --port 8000 --log-level warning > backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait for backend to start
sleep 8

# Verify backend is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend running: http://localhost:8000"
else
    echo "❌ Backend failed to start. Check backend.log"
    exit 1
fi

# Start frontend in PRODUCTION mode (serve built files) instead of development mode
echo "🌐 Building and Starting Frontend Server on Port 3001..."
cd /mnt/d/Hackathon/Book

# Clean install frontend dependencies
npm install > /dev/null 2>&1

# Build the site
echo "Building Docusaurus site..."
npm run build > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "❌ Build failed. Check for errors."
    exit 1
fi

echo "✅ Build successful!"

# Serve the built site on port 3001 (this is more reliable than 'npm start' for local access)
nohup npx docusaurus serve --host 127.0.0.1 --port 3001 > frontend_serve.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend served with PID: $FRONTEND_PID"

# Wait for frontend to start
sleep 5

echo ""
echo "🎉 LOCAL DEVELOPMENT SETUP COMPLETE!"
echo ""
echo "📍 Backend API: http://localhost:8000"
echo "📍 Frontend UI: http://localhost:3001/hackathon-ai-book/"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "💡 The site is now available at: http://localhost:3001/hackathon-ai-book/"
echo ""
echo "📝 Logs:"
echo "   - backend.log (backend logs)"
echo "   - frontend_serve.log (frontend serve logs)"
echo ""
echo "🔄 To stop servers: pkill -f 'uvicorn\|docusaurus serve'"
echo ""
echo "🔍 To verify everything is working:"
echo "   - Check backend: curl http://localhost:8000/health"
echo "   - Frontend should be accessible at the URL above"