#!/bin/bash

# Script to build and deploy the Docusaurus site to GitHub Pages

echo "Building the Docusaurus site..."
npm run build

if [ $? -eq 0 ]; then
    echo "Build successful. Deploying to GitHub Pages..."

    # Set git user to trigger deployment
    export GIT_USER=sanilahmed

    # Run the deploy command
    npm run deploy

    if [ $? -eq 0 ]; then
        echo "Deployment successful!"
        echo "Your site should be available at: https://sanilahmed.github.io/hackathon-ai-book"
        echo "The chatbot should now be visible as a toggle button on the site."
    else
        echo "Deployment failed!"
        exit 1
    fi
else
    echo "Build failed!"
    exit 1
fi