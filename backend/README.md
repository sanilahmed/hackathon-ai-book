---
title: Backend Deploy
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# RAG Agent and API Layer

This is a FastAPI application that provides a question-answering API using Gemini agents and Qdrant retrieval for RAG (Retrieval Augmented Generation) functionality.

## API Endpoints

- `GET /` - Root endpoint with API information
- `POST /ask` - Main question-answering endpoint
- `GET /health` - Health check endpoint
- `GET /ready` - Readiness check endpoint
- `/docs` - API documentation (Swagger UI)
- `/redoc` - API documentation (Redoc)

## Configuration

The application requires the following environment variables:
- `GEMINI_API_KEY` - API key for Google Gemini
- `QDRANT_URL` - URL for Qdrant vector database
- `QDRANT_API_KEY` - API key for Qdrant database

## Deployment

This application is configured for deployment on Hugging Face Spaces using Docker.