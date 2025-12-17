"""
Utility functions for the RAG Agent and API Layer system.

This module contains helper functions, logging setup, and common utilities.
"""
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from .models import ErrorResponse


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging configuration for the application.

    Args:
        level: Logging level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def generate_response_id() -> str:
    """
    Generate a unique identifier for API responses.

    Returns:
        String identifier in UUID format
    """
    return f"resp_{uuid.uuid4().hex[:8]}"


def format_timestamp() -> str:
    """
    Generate an ISO 8601 formatted timestamp.

    Returns:
        ISO 8601 formatted timestamp string
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def create_error_response(error_code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """
    Create a standardized error response.

    Args:
        error_code: Error code string
        message: Human-readable error message
        details: Optional additional error details

    Returns:
        ErrorResponse instance with standardized format
    """
    error_info = {
        "code": error_code,
        "message": message
    }

    if details:
        error_info["details"] = details

    return ErrorResponse(
        error=error_info,
        timestamp=format_timestamp()
    )


def sanitize_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: Input text to sanitize

    Returns:
        Sanitized text with potentially harmful characters escaped
    """
    # Basic sanitization - in a real implementation, you might want more sophisticated
    # sanitization depending on your specific security requirements
    if not isinstance(text, str):
        return ""

    # Remove or escape potentially dangerous characters
    sanitized = text.replace("<script", "&lt;script").replace("javascript:", "javascript_")
    return sanitized


def validate_url(url: str) -> bool:
    """
    Validate that a string is a properly formatted URL.

    Args:
        url: URL string to validate

    Returns:
        True if URL is valid, False otherwise
    """
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None


def format_confidence_score(score: float) -> float:
    """
    Format a confidence score to be within valid bounds.

    Args:
        score: Raw confidence score

    Returns:
        Confidence score normalized to 0.0-1.0 range
    """
    return max(0.0, min(1.0, score))


def extract_content_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to split into chunks
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        start = end - overlap

        # Handle the last chunk properly
        if end >= len(text):
            break

    return chunks


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate a basic similarity score between two texts using a simple approach.
    NOTE: This is a simplified implementation. In a real system, you would use
    vector embeddings and cosine similarity or other advanced methods.

    Args:
        text1: First text for comparison
        text2: Second text for comparison

    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Simple word overlap approach (case-insensitive)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    intersection = words1.intersection(words2)
    union = words1.union(words2)

    if not union:
        return 0.0

    return len(intersection) / len(union)


class RateLimiter:
    """
    Simple rate limiter class to control API request frequency.
    """
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        """
        Initialize the rate limiter.

        Args:
            max_requests: Maximum number of requests allowed per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # Dictionary to track requests per identifier

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if a request from the given identifier is allowed.

        Args:
            identifier: Identifier for the request (e.g., IP address, user ID)

        Returns:
            True if request is allowed, False if rate limit exceeded
        """
        current_time = datetime.utcnow().timestamp()

        if identifier not in self.requests:
            self.requests[identifier] = []

        # Clean old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        # Check if we're under the limit
        if len(self.requests[identifier]) < self.max_requests:
            self.requests[identifier].append(current_time)
            return True

        return False