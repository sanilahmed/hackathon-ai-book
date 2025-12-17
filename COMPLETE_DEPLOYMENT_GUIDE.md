# Complete RAG Chatbot Deployment Guide

This guide provides step-by-step instructions to deploy your Docusaurus site with the fully functional RAG chatbot to GitHub Pages.

## Overview

Your site includes:
- Docusaurus documentation site
- Fully integrated RAG chatbot
- Backend API for question answering
- Qdrant vector database for content retrieval
- Gemini AI for generating responses

## Deployment Steps

### Step 1: Deploy the Backend API

The RAG chatbot requires a backend API to function. GitHub Pages only hosts static files, so you need to deploy the backend separately.

#### Option A: Deploy to Render.com (Recommended)

1. Create an account at [Render.com](https://render.com)
2. Create a new Web Service
3. Connect to your GitHub repository
4. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn rag_agent_api.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables:
   ```
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   GEMINI_API_KEY=your_gemini_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```
6. Deploy and note the generated URL (e.g., `https://your-app.onrender.com`)

#### Option B: Deploy to Heroku

1. Create an account at [Heroku](https://heroku.com)
2. Create a new app
3. Add a `Procfile` with content: `web: uvicorn rag_agent_api.main:app --host 0.0.0.0 --port $PORT`
4. Deploy using Git or GitHub integration
5. Set environment variables in Settings > Config Vars

#### Option C: Use ngrok for Testing

1. Install ngrok from [ngrok.com](https://ngrok.com)
2. Run your backend locally: `npm run backend`
3. In another terminal: `ngrok http 8000`
4. Note the public URL (e.g., `https://xxxx-xx-xxx-xxx-xxx.ngrok.io`)

### Step 2: Update Frontend Configuration

Update `docusaurus.config.js` with your backend URL:

```javascript
customFields: {
  RAG_API_URL: 'https://your-deployed-backend-url.com', // Replace with your actual backend URL
},
```

### Step 3: Ingest Book Content

Before deploying, ensure your book content is in the Qdrant database:

```bash
cd backend
python main.py  # This ingests your book content into Qdrant
```

### Step 4: Deploy Frontend to GitHub Pages

1. Create a GitHub Personal Access Token with `repo` permissions
2. Set environment variables:
   ```bash
   export GIT_USER=your-github-username
   export GIT_PASS=your-personal-access-token
   ```
3. Build and deploy:
   ```bash
   npm run build
   npm run deploy
   ```

## Running Locally for Development

To run both frontend and backend locally:

```bash
# Terminal 1: Start backend
npm run backend

# Terminal 2: Start frontend
npm run start
```

Or use the combined script:
```bash
npm run start-app
```

## Alternative: Deploy Without Backend (Frontend Only)

If you want to deploy the frontend only (with chatbot showing informative messages):

1. Keep the default configuration in `docusaurus.config.js`:
   ```javascript
   RAG_API_URL: 'http://localhost:8000', // This will show "backend unavailable" message on GitHub Pages
   ```

2. Deploy as usual:
   ```bash
   npm run build
   export GIT_USER=your-username
   export GIT_PASS=your-token
   npm run deploy
   ```

## Verifying the Deployment

After deployment, your site will be available at:
`https://your-username.github.io/project-name/`

The RAG chatbot will be fully functional if:
- Backend API is deployed and accessible
- Qdrant database contains the book content
- All API keys are correctly configured

## Troubleshooting

### Chatbot Not Working
- Verify backend API is running and accessible
- Check browser console for CORS errors
- Ensure RAG_API_URL is correctly set

### Content Not Found
- Verify book content was ingested into Qdrant
- Check Qdrant connection settings

### API Key Errors
- Verify all API keys are set in backend environment
- Check that Qdrant and Gemini API keys are valid

## Scripts Available

- `npm run start` - Start frontend development server
- `npm run backend` - Start backend API server
- `npm run start-app` - Start both frontend and backend
- `npm run build` - Build static site
- `npm run deploy` - Deploy to GitHub Pages
- `npm run serve` - Serve built site locally

## Files Created/Modified

- `FULL_DEPLOYMENT_GUIDE.md` - Complete deployment instructions
- `start_app.sh` - Script to start frontend and backend
- Updated `docusaurus.config.js` - Configuration for backend API
- Updated `package.json` - Additional npm scripts
- Updated chatbot configuration - Production-ready behavior

Your RAG chatbot is now ready for deployment! Follow the steps above to get your fully functional AI-powered documentation site live on GitHub Pages.