"""
Data models for the RAG Retrieval Verification system.

This module defines the core data structures based on the specification.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class MetadataRecord:
    """
    Contains information about the original source of content (URL, title, chunk index, etc.)
    that helps provide context for retrieved results.
    """
    url: str
    title: str
    chunk_index: int
    total_chunks: int
    source_document_id: Optional[str] = None

    def __post_init__(self):
        """Validate the metadata record after initialization."""
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.total_chunks <= 0:
            raise ValueError("total_chunks must be greater than 0")
        if self.chunk_index >= self.total_chunks:
            raise ValueError("chunk_index must be less than total_chunks")


@dataclass
class RetrievedContentChunk:
    """
    Represents a segment of book content returned by semantic search,
    including the text content, similarity score, and associated metadata.
    """
    content: str
    similarity_score: float
    vector_id: str
    metadata: MetadataRecord

    def __post_init__(self):
        """Validate the retrieved content chunk after initialization."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("similarity_score must be between 0.0 and 1.0")
        if not self.content:
            raise ValueError("content must not be empty")
        if not self.vector_id:
            raise ValueError("vector_id must not be empty")


@dataclass
class QueryRequest:
    """
    Represents a semantic search request from a user, containing the search terms
    and parameters for retrieval.
    """
    query_text: str
    top_k: int = 5
    min_similarity: float = 0.5
    filters: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate the query request after initialization."""
        if not self.query_text:
            raise ValueError("query_text must not be empty")
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0.0 and 1.0")


@dataclass
class VerificationResult:
    """
    Contains the outcome of pipeline validation, including success/failure status,
    error details, and execution metrics.
    """
    status: str  # SUCCESS, PARTIAL_SUCCESS, FAILURE
    timestamp: datetime
    query: QueryRequest
    retrieved_chunks: List[RetrievedContentChunk]
    metadata_accuracy: float
    content_relevance: float
    execution_time_ms: float
    errors: List[str]

    def __post_init__(self):
        """Validate the verification result after initialization."""
        valid_statuses = ["SUCCESS", "PARTIAL_SUCCESS", "FAILURE"]
        if self.status not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        if not 0.0 <= self.metadata_accuracy <= 1.0:
            raise ValueError("metadata_accuracy must be between 0.0 and 1.0")
        if not 0.0 <= self.content_relevance <= 1.0:
            raise ValueError("content_relevance must be between 0.0 and 1.0")
        if self.execution_time_ms < 0:
            raise ValueError("execution_time_ms must be non-negative")


# Type aliases for convenience
VerificationStatus = str
QueryResult = List[RetrievedContentChunk]