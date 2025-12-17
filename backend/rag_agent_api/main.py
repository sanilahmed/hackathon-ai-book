"""
Main module for the RAG Agent and API Layer system.

This module provides the FastAPI application with endpoints for question-answering.
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import asyncio
import logging
from .config import validate_config, get_config
from .models import QueryRequest, APIResponse, ErrorResponse, HealthResponse
from .schemas import AgentResponse, AgentContext
from .utils import setup_logging, generate_response_id, format_timestamp, create_error_response
from .agent import GeminiAgent
from .retrieval import QdrantRetriever


# Initialize the FastAPI application
app = FastAPI(
    title="RAG Agent and API Layer",
    description="Question-answering API using OpenAI Agents and Qdrant retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
config = get_config()
setup_logging(config.log_level)

# Global instances
retriever: Optional[QdrantRetriever] = None
agent: Optional[GeminiAgent] = None


@app.on_event("startup")
async def startup_event():
    """Initialize components on application startup."""
    global retriever, agent

    # Validate configuration
    if not validate_config():
        logging.error("Configuration validation failed")
        raise RuntimeError("Configuration validation failed")

    # Initialize agent first (this doesn't require async initialization)
    try:
        agent = GeminiAgent()
        logging.info("Google Gemini agent initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Google Gemini agent: {e}")
        raise

    # Initialize retriever (async operations will be handled in the methods themselves)
    try:
        retriever = QdrantRetriever()
        logging.info("Qdrant retriever initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Qdrant retriever: {e}")
        raise

    logging.info("Application startup completed")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify the status of the API and its dependencies.

    Returns:
        HealthResponse with status of services
    """
    # Check if all required components are initialized
    gemini_status = "up" if agent else "down"
    qdrant_status = "up" if retriever else "down"
    agent_status = "up" if agent else "down"

    # Determine overall status
    overall_status = "healthy"
    if gemini_status == "down" or qdrant_status == "down":
        overall_status = "unhealthy"
    elif gemini_status == "degraded" or qdrant_status == "degraded":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        timestamp=format_timestamp(),
        services={
            "gemini": gemini_status,
            "qdrant": qdrant_status,
            "agent": agent_status
        }
    )


@app.post("/ask", response_model=APIResponse)
async def ask_question(request: QueryRequest) -> APIResponse:
    """
    Main question-answering endpoint that accepts user queries and returns AI-generated answers
    based on book content retrieved from Qdrant.

    Args:
        request: QueryRequest containing the user's question and parameters

    Returns:
        APIResponse with the answer and source citations
    """
    try:
        # Validate components are initialized
        if not retriever or not agent:
            raise HTTPException(status_code=500, detail="Service not properly initialized")

        # Generate response ID
        response_id = generate_response_id()

        # Log the incoming query
        logging.info(f"Processing query: {request.query[:100]}...")

        # Step 1: Retrieve relevant content chunks from Qdrant
        logging.info("Step 1: Retrieving relevant content from Qdrant...")
        retrieved_chunks = await retriever.retrieve_context(
            query=request.query,
            top_k=request.context_window
        )

        logging.info(f"Retrieved {len(retrieved_chunks)} chunks from Qdrant")

        # Step 2: Create agent context with retrieved chunks
        agent_context = AgentContext(
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            max_context_length=4000,  # Typical token limit consideration
            source_policy="strict"  # Ensure responses are grounded in provided context
        )

        # Step 3: Generate response using the OpenAI agent
        logging.info("Step 2: Generating response with OpenAI agent...")
        agent_response = await agent.generate_response(agent_context)

        # Step 4: Format the response according to API specification
        logging.info("Step 3: Formatting response...")

        # Extract source information from agent response
        sources = []
        for chunk in retrieved_chunks:
            if hasattr(agent_response, 'used_sources') and chunk.id in agent_response.used_sources:
                sources.append(chunk)

        # Create the final API response
        api_response = APIResponse(
            id=response_id,
            query=request.query,
            answer=agent_response.raw_response if hasattr(agent_response, 'raw_response') else agent_response.answer,
            sources=sources,
            confidence=agent_response.confidence_score if hasattr(agent_response, 'confidence_score') else 0.0,
            timestamp=format_timestamp(),
            model_used=agent.model_name if hasattr(agent, 'model_name') else "unknown"  # Assuming agent has this attribute
        )

        logging.info(f"Query processed successfully, response ID: {response_id}")
        return api_response

    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        error_resp = create_error_response(
            error_code="PROCESSING_ERROR",
            message=f"Error processing your query: {str(e)}"
        )
        raise HTTPException(status_code=500, detail=error_resp.dict())


@app.get("/")
async def root() -> Dict[str, Any]:
    """
    Root endpoint providing basic information about the API.

    Returns:
        Dictionary with API information
    """
    return {
        "message": "RAG Agent and API Layer",
        "version": "1.0.0",
        "description": "Question-answering API using OpenAI Agents and Qdrant retrieval",
        "endpoints": {
            "POST /ask": "Main question-answering endpoint",
            "GET /health": "Health check endpoint",
            "/docs": "API documentation (Swagger UI)",
            "/redoc": "API documentation (Redoc)"
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    error_resp = create_error_response(
        error_code="ENDPOINT_NOT_FOUND",
        message="The requested endpoint was not found"
    )
    return JSONResponse(
        status_code=404,
        content=error_resp.dict()
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors."""
    error_resp = create_error_response(
        error_code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred"
    )
    return JSONResponse(
        status_code=500,
        content=error_resp.dict()
    )


# Additional utility endpoints if needed
@app.get("/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness check endpoint to verify the application is ready to serve traffic.

    Returns:
        Dictionary with readiness status
    """
    if retriever and agent:
        return {"status": "ready"}
    else:
        raise HTTPException(status_code=503, detail="Service not ready")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)