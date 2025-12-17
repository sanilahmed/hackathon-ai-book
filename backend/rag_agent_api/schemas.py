"""
API request/response schemas for the RAG Agent and API Layer system.

This module contains additional schemas for the API endpoints.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from .models import QueryRequest, APIResponse, ErrorResponse, HealthResponse, SourceChunk


# Re-exporting models as schemas for consistency
QueryRequestSchema = QueryRequest
APIResponseSchema = APIResponse
ErrorResponseSchema = ErrorResponse
HealthResponseSchema = HealthResponse
SourceChunkSchema = SourceChunk


class AgentContext(BaseModel):
    """
    Model for context passed to the OpenAI agent.
    """
    query: str = Field(..., description="The original user query")
    retrieved_chunks: List[SourceChunkSchema] = Field(
        ...,
        description="Array of retrieved context chunks"
    )
    max_context_length: int = Field(
        4000,
        description="Maximum allowed context length in tokens"
    )
    source_policy: str = Field(
        "strict",
        description="Policy for source citation ('strict', 'permissive', 'none')",
        enum=["strict", "permissive", "none"]
    )


class AgentResponse(BaseModel):
    """
    Model for responses from the OpenAI agent before API formatting.
    """
    raw_response: str = Field(..., description="The raw response from the agent")
    used_sources: List[str] = Field(
        ...,
        description="IDs of source chunks used in the response"
    )
    confidence_score: float = Field(
        ...,
        description="Agent's confidence in the response",
        ge=0.0,
        le=1.0
    )
    is_valid: bool = Field(
        ...,
        description="Whether the response is grounded in provided context"
    )
    validation_details: str = Field(
        "",
        description="Details about grounding validation"
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims not supported by context"
    )


class RetrievedContextChunk(BaseModel):
    """
    Model for content chunks retrieved from Qdrant.
    """
    id: str = Field(..., description="Unique identifier for the chunk")
    content: str = Field(..., description="The text content of the chunk")
    similarity_score: float = Field(
        ...,
        description="Similarity score to the query",
        ge=0.0,
        le=1.0
    )
    metadata: dict = Field(..., description="Additional metadata for the chunk")
    retrieved_at: str = Field(..., description="Timestamp when chunk was retrieved")