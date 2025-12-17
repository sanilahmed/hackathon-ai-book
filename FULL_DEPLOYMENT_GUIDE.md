# Running Backend Server and Deploying with Full RAG Chatbot

This guide explains how to run the backend server and deploy the complete RAG chatbot functionality.

## Running the Backend Server Locally

### Prerequisites
- Python 3.10+
- uv (or pip) package manager
- Environment variables set in `.env` file

### Step 1: Set up Environment Variables

Make sure your `.env` file in the backend directory contains:

```env
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key
```

### Step 2: Install Dependencies

```bash
cd backend
uv venv  # or create virtual environment with python -m venv .venv && source .venv/bin/activate
uv pip sync requirements.txt  # or pip install -r requirements.txt
```

### Step 3: Run the Backend Server

```bash
cd backend
uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
```

Or use the npm script:

```bash
npm run backend
```

## Making the Backend Publicly Accessible

### Option 1: Using ngrok (Recommended for testing)

1. Install ngrok:
   ```bash
   # Download from https://ngrok.com/
   # Or install via package manager
   ```

2. Start your backend server:
   ```bash
   cd backend
   uvicorn rag_agent_api.main:app --host 0.0.0.0 --port 8000
   ```

3. In another terminal, expose your local server:
   ```bash
   ngrok http 8000
   ```

4. Note the public URL (e.g., `https://xxxx-xx-xxx-xxx-xxx.ngrok.io`)

### Option 2: Deploy Backend to Cloud Platform

#### Deploy to Render.com

1. Create a `render.yaml` file:

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
        value: your-qdrant-url
      - key: QDRANT_API_KEY
        value: your-qdrant-api-key
      - key: GEMINI_API_KEY
        value: your-gemini-api-key
      - key: COHERE_API_KEY
        value: your-cohere-api-key
```

2. Push to GitHub and connect to Render

#### Deploy to Heroku

1. Create a `Procfile`:

```
web: uvicorn rag_agent_api.main:app --host 0.0.0.0 --port $PORT
```

2. Create `requirements.txt` with all dependencies

3. Deploy to Heroku

## Updating Frontend Configuration

Before deploying to GitHub Pages, update `docusaurus.config.js`:

```javascript
customFields: {
  RAG_API_URL: 'https://your-deployed-backend-url.com', // Replace with your actual URL
},
```

## Deployment Process

### For Full Functionality (Frontend + Backend)

1. Deploy backend to a cloud platform (Render, Heroku, etc.)
2. Update `RAG_API_URL` in `docusaurus.config.js` to your deployed backend URL
3. Deploy frontend to GitHub Pages:
   ```bash
   npm run build
   export GIT_USER=your-username
   export GIT_PASS=your-personal-access-token
   npm run deploy
   ```

### For Local Development

1. Run backend server:
   ```bash
   npm run backend
   ```

2. Run frontend:
   ```bash
   npm run start
   ```

## Testing the RAG Chatbot

1. Ensure backend server is running
2. Visit your frontend application
3. Open the chatbot and ask questions about the book content
4. The chatbot should retrieve information from your Qdrant database and provide answers via the Gemini API

## Troubleshooting

- **CORS errors**: Make sure your backend allows requests from your frontend domain
- **API key errors**: Verify all API keys are correctly set in environment variables
- **Qdrant connection errors**: Check that Qdrant URL and API key are correct
- **Empty responses**: Ensure the book content has been ingested into Qdrant

## Ingesting Book Content

Before the RAG chatbot can answer questions, you need to ingest the book content:

```bash
cd backend
python main.py  # This will crawl your book site and ingest content into Qdrant
```