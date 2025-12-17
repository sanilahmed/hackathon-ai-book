"""
Pydantic models for the RAG Agent and API Layer system.

This module defines the data structures for API requests and responses.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class QueryRequest(BaseModel):
    """
    Model for incoming user queries to the question-answering API.
    """
    query: str = Field(
        ...,
        description="The user's question or query",
        min_length=1,
        max_length=1000
    )
    context_window: int = Field(
        5,
        description="Number of chunks to retrieve",
        ge=1,
        le=20
    )
    include_sources: bool = Field(
        True,
        description="Whether to include source citations"
    )
    temperature: float = Field(
        0.1,
        description="Response randomness",
        ge=0.0,
        le=1.0
    )


class SourceChunk(BaseModel):
    """
    Model for source chunks in the response.
    """
    id: str = Field(..., description="Unique identifier for the source chunk")
    url: str = Field(..., description="URL of the original content", format="uri")
    title: str = Field(..., description="Title of the source document")
    content: str = Field(..., description="The content chunk used")
    similarity_score: float = Field(
        ...,
        description="Similarity score to the query",
        ge=0.0,
        le=1.0
    )
    chunk_index: int = Field(..., description="Index of the chunk in the original document")


class APIResponse(BaseModel):
    """
    Model for responses returned by the question-answering API.
    """
    id: str = Field(..., description="Unique identifier for the response")
    query: str = Field(..., description="The original user query")
    answer: str = Field(..., description="The AI-generated answer")
    sources: List[SourceChunk] = Field(..., description="List of source chunks used")
    confidence: float = Field(
        ...,
        description="Confidence level of the response",
        ge=0.0,
        le=1.0
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of the response")
    model_used: str = Field(..., description="Name of the model used for generation")

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"


class ErrorResponse(BaseModel):
    """
    Model for standardized error responses.
    """
    error: Dict[str, Any] = Field(
        ...,
        description="Error information with code, message and details"
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of the error")


class HealthResponse(BaseModel):
    """
    Model for health check endpoint response.
    """
    status: str = Field(
        ...,
        description="Overall health status",
        enum=["healthy", "degraded", "unhealthy"]
    )
    timestamp: str = Field(..., description="ISO 8601 timestamp of the response")
    services: Dict[str, str] = Field(
        ...,
        description="Status of individual services",
        example={"openai": "up", "qdrant": "up", "agent": "up"}
    )