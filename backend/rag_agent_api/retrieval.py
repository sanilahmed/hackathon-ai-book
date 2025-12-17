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

    async def _embed_query(self, query: str) -> List[float]:
        """
        Embed the query using Cohere to prepare for semantic search.

        Args:
            query: The query string to embed

        Returns:
            List of floats representing the query embedding
        """
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
        except Exception as e:
            logging.error(f"Error embedding query with Cohere: {e}", exc_info=True)

            # Try using OpenAI embeddings as fallback if available
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

            # If both fail, return a zero vector of the correct size (1024) as a last resort
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