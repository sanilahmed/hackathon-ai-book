"""
Qdrant client module for the RAG Retrieval Verification system.

This module provides functions to connect to Qdrant, load vectors and metadata,
and perform semantic searches for verification purposes.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import numpy as np
import cohere

logger = logging.getLogger(__name__)

class QdrantVerificationClient:
    """
    A client class for Qdrant operations specific to RAG retrieval verification.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None,
                 collection_name: str = "rag_embedding"):
        """
        Initialize the Qdrant client with configuration.

        Args:
            url: Qdrant instance URL (defaults to QDRANT_URL environment variable)
            api_key: Qdrant API key (defaults to QDRANT_API_KEY environment variable)
            collection_name: Name of the collection to operate on
        """
        self.collection_name = collection_name

        # Get configuration from environment or parameters
        qdrant_url = url or os.getenv('QDRANT_URL')
        qdrant_api_key = api_key or os.getenv('QDRANT_API_KEY')

        if not qdrant_url:
            raise ValueError("QDRANT_URL environment variable not set")
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable not set")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30
        )

        logger.info(f"Initialized Qdrant client for collection: {collection_name}")

    def load_qdrant_vectors(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load vectors and metadata from Qdrant collection.

        Args:
            collection_name: Name of collection to load from (uses instance default if not provided)

        Returns:
            List of dictionaries containing vector data and metadata
        """
        collection_name = collection_name or self.collection_name

        try:
            # Get the total count of points in the collection
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count

            logger.info(f"Loading {total_points} vectors from collection: {collection_name}")

            # For verification, we'll retrieve points in batches
            all_points = []
            limit = 1000  # Process in batches to avoid memory issues

            for offset in range(0, total_points, limit):
                points = self.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )

                for point in points[0]:  # points is a tuple (records, next_page_offset)
                    all_points.append({
                        'id': point.id,
                        'vector': point.vector,
                        'payload': point.payload
                    })

                logger.info(f"Loaded {min(offset + limit, total_points)}/{total_points} points")

            logger.info(f"Successfully loaded {len(all_points)} vectors from Qdrant")
            return all_points

        except Exception as e:
            logger.error(f"Error loading vectors from Qdrant: {e}")
            raise

    def query_qdrant_for_chunks(self, query_text: str, top_k: int = 5,
                                collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Query Qdrant using semantic search to retrieve relevant content chunks.

        Args:
            query_text: The semantic query text to search for
            top_k: Number of results to return (default: 5)
            collection_name: Name of collection to query (uses instance default if not provided)

        Returns:
            List of dictionaries containing retrieved chunks with similarity scores
        """
        # Input validation
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string")

        if not isinstance(top_k, int) or top_k <= 0 or top_k > 100:
            raise ValueError("top_k must be a positive integer between 1 and 100")

        collection_name = collection_name or self.collection_name
        if not collection_name or not isinstance(collection_name, str):
            raise ValueError("collection_name must be a non-empty string")

        # Validate collection name for allowed characters to prevent injection
        import re
        if not re.match(r'^[a-zA-Z0-9._-]+$', collection_name):
            raise ValueError("collection_name contains invalid characters")

        try:
            # First, we need to embed the query text using the same model used during ingestion
            # For now, we'll use Cohere to embed the query text
            import cohere
            cohere_api_key = os.getenv('COHERE_API_KEY')
            if not cohere_api_key:
                raise ValueError("COHERE_API_KEY environment variable not set")

            # Limit query text length for security and performance
            if len(query_text) > 1000:
                logger.warning(f"Query text is very long ({len(query_text)} chars), truncating to 1000 chars")
                query_text = query_text[:1000]

            co = cohere.Client(cohere_api_key)
            response = co.embed([query_text], model="embed-english-v3.0", input_type="search_query")
            query_embedding = response.embeddings[0]

            logger.info(f"Querying for: '{query_text}' in collection: {collection_name}")

            # Perform semantic search in Qdrant using the embedded vector
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'content': result.payload.get('text', ''),
                    'similarity_score': result.score,
                    'metadata': {
                        'url': result.payload.get('url', ''),
                        'title': result.payload.get('title', ''),
                        'chunk_index': result.payload.get('chunk_index', 0),
                        'total_chunks': result.payload.get('total_chunks', 1)
                    }
                })

            logger.info(f"Retrieved {len(results)} results for query: '{query_text}'")
            return results

        except Exception as e:
            logger.error(f"Error querying Qdrant: {e}")
            raise

    def validate_collection_exists(self, collection_name: Optional[str] = None) -> bool:
        """
        Validate that the specified collection exists in Qdrant.

        Args:
            collection_name: Name of collection to check (uses instance default if not provided)

        Returns:
            True if collection exists, False otherwise
        """
        collection_name = collection_name or self.collection_name

        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False


def load_qdrant_vectors(collection_name: str = "rag_embedding") -> List[Dict[str, Any]]:
    """
    Load vectors and metadata from Qdrant collection.

    Args:
        collection_name: Name of the collection to load from

    Returns:
        List of dictionaries containing vector data and metadata
    """
    client = QdrantVerificationClient(collection_name=collection_name)
    return client.load_qdrant_vectors()


def load_qdrant_vectors_with_metadata(collection_name: str = "rag_embedding") -> List[Dict[str, Any]]:
    """
    Load vectors and metadata from Qdrant collection with additional verification-specific functionality.

    Args:
        collection_name: Name of the collection to load from

    Returns:
        List of dictionaries containing vector data, metadata, and verification-specific fields
    """
    client = QdrantVerificationClient(collection_name=collection_name)
    points = client.load_qdrant_vectors()

    # Enhance with additional verification-specific data
    enhanced_points = []
    for point in points:
        enhanced_point = {
            'id': point['id'],
            'vector': point['vector'],
            'payload': point['payload'],
            'similarity_score': 0.0,  # Will be set during queries
            'retrieved_at': None      # Will be set during queries
        }
        enhanced_points.append(enhanced_point)

    return enhanced_points


def query_qdrant_for_chunks(query_text: str, top_k: int = 5,
                           collection_name: str = "rag_embedding") -> List[Dict[str, Any]]:
    """
    Query Qdrant using semantic search to retrieve relevant content chunks.

    Args:
        query_text: The semantic query text to search for
        top_k: Number of results to return (default: 5)
        collection_name: Name of the collection to query

    Returns:
        List of dictionaries containing retrieved chunks with similarity scores
    """
    client = QdrantVerificationClient(collection_name=collection_name)
    return client.query_qdrant_for_chunks(query_text, top_k)


def verify_embedding_retrieval(query_text: str, top_k: int = 5,
                              collection_name: str = "rag_embedding",
                              min_similarity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Verify embedding retrieval using semantic search with specific validation criteria.

    Args:
        query_text: The semantic query text to search for
        top_k: Number of results to return (default: 5)
        collection_name: Name of the collection to query
        min_similarity: Minimum similarity threshold for valid results (default: 0.7)

    Returns:
        List of dictionaries containing verified retrieved chunks with similarity scores
    """
    client = QdrantVerificationClient(collection_name=collection_name)
    results = client.query_qdrant_for_chunks(query_text, top_k)

    # Filter results based on minimum similarity threshold
    verified_results = []
    for result in results:
        if result['similarity_score'] >= min_similarity:
            result['passed_verification'] = True
            verified_results.append(result)
        else:
            result['passed_verification'] = False
            # Still include below-threshold results but mark them as not verified

    return verified_results


def query_qdrant_for_chunks_batch(queries: List[str], top_k: int = 5,
                                 collection_name: str = "rag_embedding",
                                 min_similarity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Query Qdrant in batch mode for multiple queries to optimize performance.

    Args:
        queries: List of query strings to search for
        top_k: Number of results to return per query (default: 5)
        collection_name: Name of the collection to query
        min_similarity: Minimum similarity threshold for valid results (default: 0.7)

    Returns:
        List of dictionaries containing all verified results from all queries
    """
    all_results = []

    for query in queries:
        try:
            query_results = query_qdrant_for_chunks(query, top_k, collection_name)

            # Filter results based on minimum similarity threshold
            for result in query_results:
                if result['similarity_score'] >= min_similarity:
                    result['passed_verification'] = True
                else:
                    result['passed_verification'] = False
                result['query_source'] = query  # Track which query produced this result
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            continue

    return all_results