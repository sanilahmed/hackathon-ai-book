# Quickstart Guide: Frontend Integration of RAG Chatbot with Docusaurus Book

**Feature**: Frontend Integration of RAG Chatbot with Docusaurus Book
**Created**: 2025-12-16
**Status**: Draft

## Overview

This guide provides developers with the essential steps to set up, configure, and run the RAG chatbot frontend integration with the Docusaurus book. The integration allows users to ask questions about book content and receive AI-generated answers sourced from the book's content.

## Prerequisites

- Node.js 18+ installed
- npm or yarn package manager
- Access to running FastAPI RAG backend (from Spec-3)
- Docusaurus project set up and running
- Git for version control

## Setup Instructions

### 1. Environment Preparation

First, ensure your Docusaurus project is set up and the FastAPI RAG backend is running:

```bash
# Clone or navigate to your Docusaurus project
cd your-docusaurus-project

# Install required dependencies
npm install

# Verify the FastAPI backend is running (typically on port 8000)
curl http://localhost:8000/health
```

### 2. Backend Connection Configuration

Create or update your environment configuration to connect to the FastAPI backend:

```bash
# In your Docusaurus project root, create/edit .env file
echo "REACT_APP_RAG_API_URL=http://localhost:8000" >> .env
```

### 3. Component Installation

Install the chatbot component in your Docusaurus project:

```bash
# Create the chatbot component directory
mkdir src/components/rag-chatbot

# The component files will be created during implementation
# See the implementation plan for file structure details
```

## Basic Usage

### 1. Component Integration

To integrate the chatbot into your Docusaurus site, add the component to your layout or specific pages:

```jsx
// Example: Adding to a specific page
import ChatbotComponent from '@site/src/components/rag-chatbot';

function MyPage() {
  return (
    <div>
      <h1>My Book Content</h1>
      <ChatbotComponent />
    </div>
  );
}
```

### 2. Global Integration

To make the chatbot available across all pages, modify your theme layout:

```jsx
// In src/theme/Layout/index.js
import ChatbotComponent from '@site/src/components/rag-chatbot';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <ChatbotComponent />
    </>
  );
}
```

## Development Workflow

### 1. Starting the Development Server

```bash
# Start Docusaurus development server
npm run start

# The chatbot should now be accessible on all pages
```

### 2. Testing the Integration

1. Navigate to any page in your Docusaurus book
2. Locate the chatbot interface (typically at the bottom or side)
3. Enter a question related to your book content
4. Verify that the response includes source metadata
5. Test error handling by temporarily stopping the backend

### 3. Backend API Endpoints

The chatbot communicates with these FastAPI endpoints:

- `POST /ask` - Submit questions and receive AI responses
- `GET /health` - Check backend health status
- `GET /ready` - Check if backend is ready to process requests

## Configuration Options

### Environment Variables

- `REACT_APP_RAG_API_URL` - Base URL for the FastAPI backend
- `REACT_APP_CHATBOT_ENABLED` - Enable/disable chatbot functionality (default: true)
- `REACT_APP_MAX_HISTORY` - Maximum number of conversation items to store (default: 10)

### Component Props

The ChatbotComponent supports these configuration options:

```jsx
<ChatbotComponent
  position="bottom-right"        // Position on screen
  theme="light"                  // Color theme
  maxHistory={20}                // Maximum history items
  enableHistory={true}           // Enable conversation history
  placeholder="Ask about..."     // Input field placeholder
/>
```

## Troubleshooting

### Common Issues

1. **Backend Connection Errors**
   - Verify FastAPI backend is running
   - Check CORS configuration in FastAPI
   - Confirm API URL in environment variables

2. **Component Not Appearing**
   - Verify component is properly imported
   - Check Docusaurus version compatibility
   - Ensure component files are in correct location

3. **Response Formatting Issues**
   - Check response format from backend
   - Verify source metadata structure
   - Review CSS styling conflicts

### Debugging Commands

```bash
# Check backend connectivity
curl http://localhost:8000/health

# Test question endpoint directly
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "test question", "context_window": 5}'

# Check Docusaurus build
npm run build
```

## Next Steps

1. Complete the full implementation following the detailed plan
2. Test across different devices and browsers
3. Validate source metadata display functionality
4. Implement error handling and fallback responses
5. Conduct user acceptance testing