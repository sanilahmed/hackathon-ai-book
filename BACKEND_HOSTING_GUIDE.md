# Backend API Hosting for GitHub Pages RAG Chatbot

The RAG chatbot requires a backend API to function, but GitHub Pages is a static hosting service that cannot run backend services. This document explains how to properly deploy your RAG chatbot with GitHub Pages.

## Architecture Overview

- **Frontend**: Hosted on GitHub Pages (`https://sanilahmed.github.io/hackathon-ai-book/`)
- **Backend API**: Must be hosted separately (e.g., on a cloud service)

## Option 1: Deploy Backend to Cloud Platform (Recommended)

### Deploy to Render.com

1. Create a `render.yaml` file in your repository:

```yaml
services:
  - type: web
    name: rag-agent-api
    env: python
    region: frankfurt  # or your preferred region
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn rag_agent_api.main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.10
      - key: QDRANT_URL
        value: https://your-qdrant-instance.com:6334  # Update with your Qdrant URL
      - key: QDRANT_API_KEY
        value: your-qdrant-api-key
      - key: GEMINI_API_KEY
        value: your-gemini-api-key
```

2. Push the `render.yaml` file to your repository
3. Connect your GitHub repository to Render
4. Update your GitHub Pages configuration to point to the Render backend URL

### Deploy to Heroku

1. Create a `Procfile` in your backend directory:

```
web: uvicorn rag_agent_api.main:app --host 0.0.0.0 --port $PORT
```

2. Create a `requirements.txt` file with your Python dependencies
3. Deploy to Heroku using the Heroku CLI or GitHub integration

## Option 2: Temporary Solution - Disable Chatbot for GitHub Pages

If you want to deploy the site immediately without backend hosting:

1. Update the chatbot config to disable it for production:

```javascript
// In docusaurus.config.js
customFields: {
  // For local development: 'http://localhost:8000'
  // For GitHub Pages without backend: null or comment out
  RAG_API_URL: null, // This will disable the chatbot on GitHub Pages
},
```

2. Update the chatbot config to handle null URLs:

```javascript
// In src/components/rag-chatbot/config.js
const getBackendUrl = () => {
  // Check if RAG_API_URL is available from Docusaurus config
  if (typeof window !== 'undefined' && window.RAG_API_URL && window.RAG_API_URL !== 'null') {
    return window.RAG_API_URL;
  }
  // Disable chatbot if no backend URL is available
  if (typeof window !== 'undefined' && window.location.hostname.includes('github.io')) {
    return null; // Disable for GitHub Pages without backend
  }
  return getConfigValue('BACKEND_URL', 'http://localhost:8000');
};
```

## Option 3: Self-Host Backend on VPS

1. Deploy the backend to a cloud server (AWS, GCP, Azure, DigitalOcean, etc.)
2. Configure SSL/TLS certificates
3. Set up reverse proxy with Nginx
4. Update the RAG API URL to point to your server

## CORS Configuration

When hosting the backend separately, ensure CORS is properly configured in your backend:

```python
# In your backend FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://sanilahmed.github.io"],  # Add your GitHub Pages URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Updating Your Configuration

After deploying your backend API to a public URL, update your `docusaurus.config.js`:

```javascript
customFields: {
  RAG_API_URL: 'https://your-deployed-backend.onrender.com', // Update with your actual URL
},
```

## Testing Before Deployment

1. Test the backend API endpoint directly
2. Verify CORS headers are properly set
3. Test the API from a browser console to ensure cross-origin requests work
4. Test the full integration locally before deploying to GitHub Pages

## Alternative: Full-Stack Hosting Solutions

Consider using hosting platforms that support both frontend and backend:

- **Vercel**: Can host both frontend and backend (with serverless functions)
- **Netlify**: Can host frontend with Netlify Functions for backend
- **Firebase**: Hosting for frontend with Cloud Functions for backend
- **AWS Amplify**: Full stack hosting with backend services

This approach would allow you to have both frontend and backend in one place, simplifying deployment.