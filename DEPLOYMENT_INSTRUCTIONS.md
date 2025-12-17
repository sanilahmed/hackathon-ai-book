# GitHub Pages Deployment Instructions

This document explains how to deploy your Docusaurus site with the RAG chatbot to GitHub Pages.

## Prerequisites

1. A GitHub repository named `hackathon-ai-book` under the `sanilahmed` organization/user
2. Proper GitHub authentication (Personal Access Token or SSH keys)
3. The repository must be set up to allow GitHub Pages deployment
4. **Important**: A publicly accessible backend API for the RAG chatbot (see Backend Configuration below)

## Configuration

Your `docusaurus.config.js` is already configured correctly:

```javascript
// GitHub pages deployment config
organizationName: 'sanilahmed',
projectName: 'hackathon-ai-book',
deploymentBranch: 'gh-pages',
baseUrl: '/hackathon-ai-book',
```

### Backend Configuration (Critical)

**Important Note**: GitHub Pages is a static hosting service and cannot run backend services. The RAG chatbot requires a backend API to function. You must host the backend API separately.

In `docusaurus.config.js`:
```javascript
customFields: {
  // Update this to your publicly hosted backend API URL
  RAG_API_URL: 'https://your-backend-api-url.com', // Replace with your actual backend URL
},
```

For more information about backend hosting options, see [BACKEND_HOSTING_GUIDE.md](./BACKEND_HOSTING_GUIDE.md).

## Deployment Steps

### Option 1: Using Personal Access Token (Recommended)

1. Create a Personal Access Token in GitHub:
   - Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
   - Click "Generate new token"
   - Select the scopes: `repo` (full control of private repositories)
   - Copy the generated token

2. Set the environment variables:
   ```bash
   export GIT_USER=sanilahmed
   export GIT_PASS=<your-personal-access-token>
   ```

3. Update your RAG_API_URL in `docusaurus.config.js` to point to your hosted backend API
4. Run the deployment command:
   ```bash
   npm run deploy
   ```

### Option 2: Using SSH Keys

1. Set up SSH keys with GitHub (if not already done):
   ```bash
   # Generate SSH key if you don't have one
   ssh-keygen -t ed25519 -C "your_email@example.com"

   # Add the SSH key to your GitHub account
   # Copy the public key content
   cat ~/.ssh/id_ed25519.pub
   ```

2. Clone your repository using SSH:
   ```bash
   git remote set-url origin git@github.com:sanilahmed/hackathon-ai-book.git
   ```

3. Set the environment variable and deploy:
   ```bash
   export GIT_USER=sanilahmed
   npm run deploy
   ```

## Alternative: Manual Deployment

If the automated deployment doesn't work, you can manually create the `gh-pages` branch:

1. Update your RAG_API_URL in `docusaurus.config.js` to point to your hosted backend API
2. Build the site:
   ```bash
   npm run build
   ```

3. Navigate to the build folder:
   ```bash
   cd build
   ```

4. Initialize a new git repository and push to gh-pages:
   ```bash
   git init
   git remote add origin https://github.com/sanilahmed/hackathon-ai-book.git
   git add .
   git commit -m "Deploy website"
   git push -f origin main:gh-pages
   ```

## Enable GitHub Pages

1. Go to your GitHub repository page
2. Click on "Settings"
3. Scroll down to the "Pages" section
4. Under "Source", select "Deploy from a branch"
5. Choose "gh-pages" and "/ (root)" as the folder
6. Click "Save"

## Backend API Deployment Options

Before your chatbot will work on GitHub Pages, you need to deploy your backend API separately:

1. **Render.com** (Recommended): Easy deployment with free tier
2. **Heroku**: Popular platform with free tier
3. **Self-hosting**: On a VPS or cloud server
4. **Alternative hosting**: Consider Vercel, Netlify Functions, or AWS Amplify for full-stack hosting

See [BACKEND_HOSTING_GUIDE.md](./BACKEND_HOSTING_GUIDE.md) for detailed instructions on each option.

## Troubleshooting

- If you get authentication errors, make sure your Personal Access Token has the correct permissions
- If the gh-pages branch doesn't exist, it will be created automatically during the first deployment
- Make sure your `baseUrl` in `docusaurus.config.js` matches your GitHub Pages URL structure
- Check that your repository settings allow GitHub Pages
- Ensure your backend API is publicly accessible and CORS is properly configured
- The chatbot will not work if the RAG_API_URL is set to localhost

## Verification

After deployment, your site will be available at:
https://sanilahmed.github.io/hackathon-ai-book

The RAG chatbot will function only if you have a properly configured and accessible backend API.