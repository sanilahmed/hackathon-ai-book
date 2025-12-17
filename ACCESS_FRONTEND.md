# 🌐 Accessing the Frontend in Different Environments

## Issue: Frontend Not Loading at http://localhost:3000/hackathon-ai-book/

This is a common issue when running Docusaurus in cloud development environments like VS Code Dev Containers, GitHub Codespaces, or other remote development platforms.

## Solutions by Environment

### 🟦 VS Code (Remote Development)
1. After running `./start_systems.sh`, go to the **Ports** tab in VS Code
2. Look for port `3000` in the list
3. Click the **"Forward"** or **"Open in Browser"** button
4. Or click the globe icon to make it public
5. Access the URL provided by VS Code (not localhost)

### 🔵 GitHub Codespaces
1. After running `./start_systems.sh`, look for a notification about port 3000
2. Click **"Open in browser"** in the notification
3. Or go to the **Ports** tab and click the eye icon to open in browser
4. The URL will be something like: `https://[workspace-name]-3000.app.github.dev`

### 🟨 Other Cloud Environments
- Look for **port forwarding** or **web preview** features
- Access through the platform's web interface, not localhost

### 🟢 Local Development (if applicable)
- Access at: http://localhost:3000/hackathon-ai-book/

## Quick Check Commands
```bash
# Verify frontend server is running
ps aux | grep docusaurus

# Check frontend logs
cat frontend.log

# Verify backend is working (should respond)
curl http://localhost:8000/health
```

## Port Forwarding Instructions

### In VS Code:
1. Open **Ports** tab (next to Terminal tab)
2. Find port `3000`
3. Click the **globe icon** to make it Public
4. Click **"Open in Browser"**

### Command Line Port Forwarding (if SSH access):
```bash
# Forward local port 3000 to access the Docusaurus server
ssh -L 3000:localhost:3000 [your-server]
```

## Verification Steps
1. Run: `./start_systems.sh`
2. Check that both servers are running: `ps aux | grep -E "(uvicorn|docusaurus)"`
3. Verify backend: `curl http://localhost:8000/health`
4. Access frontend through your development environment's web interface (not localhost)

## Troubleshooting
- If you still can't access, check `frontend.log` for errors
- Make sure port 3000 is not blocked by firewall
- Ensure you're accessing through the cloud environment's interface, not localhost

## Once Accessible
- The chatbot should appear on all pages (usually bottom-right corner)
- You can ask questions about the book content
- Responses will include source citations