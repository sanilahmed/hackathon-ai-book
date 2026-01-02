"""
Qdrant retrieval module for the RAG Agent and API Layer system.

This module provides functionality for retrieving relevant content chunks from Qdrant.
"""
import asyncio
import logging
import os
from typing import List, Dict, Any, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import cohere
from .config import get_config
from .schemas import RetrievedContextChunk, SourceChunkSchema
from .utils import format_confidence_score, validate_url


class QdrantRetriever:
    """
    A class to manage retrieval of content chunks from Qdrant.
    """
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the Qdrant retriever with configuration.

        Args:
            collection_name: Name of the Qdrant collection to query (uses config default if None)
        """
        config = get_config()

        qdrant_url = config.qdrant_url
        qdrant_api_key = config.qdrant_api_key
        self.collection_name = collection_name or config.qdrant_collection_name

        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable not set")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable not set")

        # Initialize Qdrant client
        self.client = AsyncQdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30
        )

        # Initialize Cohere client for query embedding
        # Get Cohere API key from config
        config = get_config()
        cohere_api_key = config.cohere_api_key

        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY environment variable not set")

        # Initialize the Cohere client
        self.cohere_client = cohere.AsyncClient(api_key=cohere_api_key)

        # Note: Collection validation is deferred to async methods since we can't await in __init__
        logging.info(f"Initialized Qdrant retriever for collection: {self.collection_name}")

    async def retrieve_context(self, query: str, top_k: int = 5) -> List[SourceChunkSchema]:
        """
        Retrieve relevant content chunks from Qdrant based on the query.

        Args:
            query: The user's query string
            top_k: Number of results to return (default: 5)

        Returns:
            List of SourceChunkSchema objects containing relevant content
        """
        try:
            logging.info(f"Retrieving context for query: '{query[:50]}{'...' if len(query) > 50 else ''}' from collection: {self.collection_name}")

            # Embed the query using Cohere
            query_embedding = await self._embed_query(query)

            # Check if we got a zero vector fallback (indicating embedding service failure)
            is_zero_vector = all(x == 0.0 for x in query_embedding)

            if is_zero_vector:
                # If we have a zero vector, try a different approach - keyword search
                logging.warning("Zero vector detected, attempting keyword-based fallback search")
                retrieved_chunks = await self._keyword_search_fallback(query, top_k)
                logging.info(f"Keyword fallback search retrieved {len(retrieved_chunks)} chunks from Qdrant")
                return retrieved_chunks

            # Perform semantic search in Qdrant
            search_results = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            # Convert search results to SourceChunkSchema objects
            retrieved_chunks = []
            # The query_points method returns a named tuple or object with points
            # Need to access the points attribute
            points = search_results.points if hasattr(search_results, 'points') else search_results
            for result in points:
                # Extract metadata from payload
                payload = result.payload if hasattr(result, 'payload') else result

                # Create a SourceChunkSchema object
                chunk = SourceChunkSchema(
                    id=result.id if hasattr(result, 'id') else getattr(result, 'point_id', None),
                    url=payload.get('url', '') if isinstance(payload, dict) else getattr(payload, 'url', ''),
                    title=payload.get('title', '') if isinstance(payload, dict) else getattr(payload, 'title', ''),
                    content=payload.get('text', '') if isinstance(payload, dict) else getattr(payload, 'text', ''),  # Assuming 'text' is the key for content
                    similarity_score=result.score if hasattr(result, 'score') else getattr(result, 'similarity', 0),
                    chunk_index=payload.get('chunk_index', 0) if isinstance(payload, dict) else getattr(payload, 'chunk_index', 0)
                )

                # Validate the chunk before adding it
                if self._validate_chunk(chunk):
                    retrieved_chunks.append(chunk)

            logging.info(f"Retrieved {len(retrieved_chunks)} valid chunks from Qdrant")
            return retrieved_chunks

        except Exception as e:
            logging.error(f"Error retrieving context from Qdrant: {e}", exc_info=True)
            # Return empty list instead of raising exception to allow graceful handling
            return []

    async def _keyword_search_fallback(self, query: str, top_k: int = 5) -> List[SourceChunkSchema]:
        """
        Fallback method to search using keyword matching when embedding service is unavailable.

        Args:
            query: The user's query string
            top_k: Number of results to return (default: 5)

        Returns:
            List of SourceChunkSchema objects containing relevant content
        """
        try:
            # Use Qdrant's full-text search capability or filter-based approach
            # For now, we'll use a scroll + filter approach to find relevant chunks
            from qdrant_client.http import models

            # Simple approach: get all points and filter based on keyword matching
            # In a production system, you'd want to use Qdrant's text indexing capabilities
            all_points = await self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Get up to 10000 points (or as many as exist)
                with_payload=True,
                with_vectors=False
            )

            # Extract points from the result (structure may vary depending on Qdrant client version)
            points = all_points[0] if isinstance(all_points, tuple) else all_points

            # Score points based on keyword matching
            scored_chunks = []
            query_lower = query.lower()
            query_words = set(query_lower.split())

            for point in points:
                payload = point.payload if hasattr(point, 'payload') else point
                content = payload.get('text', '') if isinstance(payload, dict) else getattr(payload, 'text', '')
                content_lower = content.lower()

                # Calculate a simple keyword match score
                content_words = set(content_lower.split())
                overlap = query_words.intersection(content_words)
                score = len(overlap) / len(query_words) if query_words else 0  # Jaccard similarity

                if score > 0 or query_lower in content_lower:  # Only include if there's some match
                    chunk = SourceChunkSchema(
                        id=point.id if hasattr(point, 'id') else getattr(point, 'point_id', None),
                        url=payload.get('url', '') if isinstance(payload, dict) else getattr(payload, 'url', ''),
                        title=payload.get('title', '') if isinstance(payload, dict) else getattr(payload, 'title', ''),
                        content=content,
                        similarity_score=score,
                        chunk_index=payload.get('chunk_index', 0) if isinstance(payload, dict) else getattr(payload, 'chunk_index', 0)
                    )

                    if self._validate_chunk(chunk):
                        scored_chunks.append((chunk, score))

            # Sort by score and return top_k
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [chunk for chunk, score in scored_chunks[:top_k]]

            return top_chunks

        except Exception as e:
            logging.error(f"Error in keyword fallback search: {e}", exc_info=True)
            return []

    async def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query using Cohere to prepare for semantic search with retry logic for rate limits.

        Args:
            query: The query string to embed

        Returns:
            List of floats representing the query embedding
        """
        import time
        import random
        from cohere.errors.too_many_requests_error import TooManyRequestsError

        # Try Cohere with retry logic for rate limits
        for attempt in range(3):  # Try up to 3 times
            try:
                # Use Cohere to embed the query
                # The original book content was likely embedded with Cohere embed-english-v3.0
                response = await self.cohere_client.embed(
                    texts=[query],
                    model="embed-english-v3.0",  # 1024-dimensional embedding model
                    input_type="search_query"  # Specify this is a search query
                )

                # Extract the embedding from the response
                embedding = response.embeddings[0]  # Get the first (and only) embedding
                return embedding
            except TooManyRequestsError as e:
                if attempt < 2:  # Don't wait after the last attempt
                    # Exponential backoff with jitter
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"Cohere rate limited (attempt {attempt + 1}), waiting {wait_time:.2f}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"Cohere rate limited after {attempt + 1} attempts: {e}")
            except Exception as e:
                logging.error(f"Error embedding query with Cohere: {e}", exc_info=True)
                break  # Don't retry for other types of errors

        # If Cohere fails, try using OpenAI embeddings as fallback if available
        try:
            from openai import OpenAI
            from .config import get_config
            config = get_config()

            if config.openai_api_key:
                client = OpenAI(api_key=config.openai_api_key)
                response = client.embeddings.create(
                    input=query,
                    model="text-embedding-ada-002"
                )
                embedding = response.data[0].embedding
                logging.info("Successfully used OpenAI embedding as fallback")
                return embedding
        except Exception as openai_error:
            logging.warning(f"OpenAI fallback also failed: {openai_error}")

        # If all fail, return a zero vector of the correct size (1024) as a last resort
        # This will result in poor semantic matches but won't crash the system
        logging.warning("Using zero vector as final fallback for query embedding")
        return [0.0] * 1024

    def _validate_chunk(self, chunk: SourceChunkSchema) -> bool:
        """
        Validate that a retrieved chunk has valid content.

        Args:
            chunk: SourceChunkSchema object to validate

        Returns:
            True if chunk is valid, False otherwise
        """
        # Check if required fields are present and not empty
        if not chunk.id or not chunk.content:
            return False

        # Validate URL format if present
        if chunk.url and not validate_url(chunk.url):
            logging.warning(f"Invalid URL in chunk: {chunk.url}")
            return False

        # Check content length
        if len(chunk.content.strip()) == 0:
            return False

        # Check similarity score range
        if chunk.similarity_score < 0.0 or chunk.similarity_score > 1.0:
            return False

        return True

    async def validate_collection_exists(self) -> bool:
        """
        Validate that the configured collection exists in Qdrant.

        Returns:
            True if collection exists, False otherwise
        """
        try:
            await self.client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    async def get_total_points(self) -> int:
        """
        Get the total number of points in the collection.

        Returns:
            Total number of points in the collection
        """
        try:
            collection_info = await self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logging.error(f"Error getting total points from Qdrant: {e}")
            return 0

    async def search_with_filters(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[SourceChunkSchema]:
        """
        Search with additional filters applied to the results.

        Args:
            query: The user's query string
            top_k: Number of results to return
            filters: Optional dictionary of filters to apply

        Returns:
            List of filtered SourceChunkSchema objects
        """
        try:
            # Embed the query using OpenAI
            query_embedding = await self._embed_query(query)

            # Create Qdrant filter if provided
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    conditions.append(models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    ))

                if conditions:
                    qdrant_filter = models.Filter(must=conditions)

            # Perform semantic search in Qdrant with filters
            search_results = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                query_filter=qdrant_filter
            )

            # Convert search results to SourceChunkSchema objects
            retrieved_chunks = []
            # The query_points method returns a named tuple or object with points
            # Need to access the points attribute
            points = search_results.points if hasattr(search_results, 'points') else search_results
            for result in points:
                # Extract metadata from payload
                payload = result.payload if hasattr(result, 'payload') else result

                # Create a SourceChunkSchema object
                chunk = SourceChunkSchema(
                    id=result.id if hasattr(result, 'id') else getattr(result, 'point_id', None),
                    url=payload.get('url', '') if isinstance(payload, dict) else getattr(payload, 'url', ''),
                    title=payload.get('title', '') if isinstance(payload, dict) else getattr(payload, 'title', ''),
                    content=payload.get('text', '') if isinstance(payload, dict) else getattr(payload, 'text', ''),
                    similarity_score=result.score if hasattr(result, 'score') else getattr(result, 'similarity', 0),
                    chunk_index=payload.get('chunk_index', 0) if isinstance(payload, dict) else getattr(payload, 'chunk_index', 0)
                )

                if self._validate_chunk(chunk):
                    retrieved_chunks.append(chunk)

            return retrieved_chunks

        except Exception as e:
            logging.error(f"Error searching with filters in Qdrant: {e}", exc_info=True)
            return []

    async def close(self):
        """
        Close the connection to Qdrant.
        """
        if hasattr(self.client, 'close'):
            await self.client.close()


# Helper function to create a retriever instance with error handling
async def create_qdrant_retriever(collection_name: Optional[str] = None) -> Optional[QdrantRetriever]:
    """
    Factory function to create a QdrantRetriever instance with proper error handling.

    Args:
        collection_name: Name of the collection to use (optional)

    Returns:
        QdrantRetriever instance or None if initialization fails
    """
    try:
        return QdrantRetriever(collection_name=collection_name)
    except Exception as e:
        logging.error(f"Failed to create QdrantRetriever: {e}")
        return None