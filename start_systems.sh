#!/bin/bash
# Comprehensive startup script for RAG Chatbot - Backend and Frontend

echo "🚀 Starting RAG Chatbot Systems..."

# Function to start backend
start_backend() {
    echo "📦 Starting Backend Server..."
    cd /mnt/d/Hackathon/Book/backend

    # Install dependencies
    echo "Installing backend dependencies..."
    pip install -e . > /dev/null 2>&1 || echo "Dependencies may already be installed"

    # Start backend server in background
    echo "Starting FastAPI server on port 8000..."
    nohup uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000 --log-level warning > backend.log 2>&1 &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"

    # Wait for backend to start
    echo "Waiting for backend to start..."
    sleep 10

    # Check if backend is running
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend is running: http://localhost:8000"
        echo "Health check: $(curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo 'Health endpoint accessible')"
    else
        echo "❌ Backend failed to start. Check backend.log for details."
        cat backend.log
        return 1
    fi
}

# Function to start frontend
start_frontend() {
    echo "🌐 Starting Frontend Server..."
    cd /mnt/d/Hackathon/Book

    # Install frontend dependencies
    echo "Installing frontend dependencies..."
    npm install > /dev/null 2>&1 || echo "Frontend dependencies may already be installed"

    # Build the site first to make sure there are no errors
    echo "Building Docusaurus site..."
    npm run build > /dev/null 2>&1 || echo "Build had warnings, continuing..."

    # Start the development server with explicit host binding
    echo "Starting Docusaurus server..."
    nohup npx docusaurus start --host 0.0.0.0 > frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"

    echo "✅ Frontend will be running. In cloud environments, you may need to:"
    echo "   1. Use the 'Ports' tab to forward port 3000"
    echo "   2. Or use a web preview feature"
    echo "   3. Access through your development platform's interface"
    echo ""
    echo "📝 Check frontend.log for any startup details."
}

# Main execution
if [ "$1" = "backend" ]; then
    start_backend
elif [ "$1" = "frontend" ]; then
    start_frontend
else
    # Start both systems
    echo "Starting both Backend and Frontend..."

    # Start backend first
    start_backend
    if [ $? -ne 0 ]; then
        echo "❌ Failed to start backend. Aborting."
        exit 1
    fi

    # Small delay before starting frontend
    sleep 3

    # Start frontend
    start_frontend

    echo ""
    echo "🎉 Both systems are now running!"
    echo ""
    echo "📍 Backend API: http://localhost:8000"
    echo "📍 API Docs: http://localhost:8000/docs"
    echo ""
    echo "🌐 Frontend Access (Docusaurus):"
    echo "   In local environments: http://localhost:3000/hackathon-ai-book/"
    echo "   In cloud environments (VS Code, Codespaces, etc.):"
    echo "   - Look for port forwarding options"
    echo "   - Use the 'Ports' tab to forward port 3000"
    echo "   - Or use your platform's web preview feature"
    echo ""
    echo "💡 The chatbot should be available on all pages of the book interface."
    echo "   Look for the chat interface (usually bottom-right corner)."
    echo ""
    echo "📝 Logs are available in:"
    echo "   - backend.log (backend errors/infos)"
    echo "   - frontend.log (frontend errors/infos)"
    echo ""
    echo "🔄 To stop servers: pkill -f 'uvicorn\|docusaurus' "
    echo ""
    echo "📋 For cloud environments, make sure to:"
    echo "   1. Open the ports tab in your IDE"
    echo "   2. Forward port 3000 (set to 'Public' if needed)"
    echo "   3. Access the URL provided by your cloud environment"
fi