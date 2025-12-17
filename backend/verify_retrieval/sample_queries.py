"""
Sample queries for testing the RAG Retrieval Verification system.

This module provides sample queries that can be used to test the retrieval functionality.
"""
from typing import List, Dict, Any


def get_sample_queries() -> List[str]:
    """
    Get a list of sample queries for testing semantic search functionality.

    Returns:
        List of sample query strings
    """
    return [
        "What is the transformer architecture?",
        "Explain attention mechanism in neural networks",
        "How does semantic search work?",
        "What are vector embeddings?",
        "RAG pipeline architecture explained",
        "Natural language processing fundamentals",
        "Machine learning model training process",
        "Neural network layers and functions",
        "How to implement semantic similarity?",
        "Vector database storage and retrieval",
        "Cohere embedding model usage",
        "Qdrant vector database setup",
        "Retrieval augmented generation",
        "Embedding dimension and similarity",
        "Content chunking for RAG systems"
    ]


def get_topic_based_queries() -> List[str]:
    """
    Get a list of topic-based queries for testing semantic search functionality.

    Returns:
        List of topic-based query strings
    """
    return [
        "machine learning algorithms",
        "neural network architectures",
        "natural language processing",
        "vector embeddings and similarity",
        "semantic search techniques",
        "RAG system components",
        "AI model training",
        "deep learning fundamentals",
        "attention mechanisms",
        "transformer models"
    ]


def get_keyword_variations() -> List[Dict[str, Any]]:
    """
    Get a list of keyword variations for testing query robustness.

    Returns:
        List of dictionaries with query variations and expected results
    """
    return [
        {
            "query": "transformer architecture",
            "variations": [
                "transformer model architecture",
                "what is transformer architecture",
                "transformer neural network structure",
                "how transformers work"
            ],
            "expected_topic": "Transformer models and architecture"
        },
        {
            "query": "vector embeddings",
            "variations": [
                "word embeddings",
                "sentence embeddings",
                "vector representations",
                "semantic embeddings"
            ],
            "expected_topic": "Embedding techniques and representations"
        },
        {
            "query": "semantic search",
            "variations": [
                "meaning-based search",
                "conceptual search",
                "semantic similarity search",
                "context-aware search"
            ],
            "expected_topic": "Semantic search methodologies"
        },
        {
            "query": "RAG pipeline",
            "variations": [
                "retrieval augmented generation",
                "RAG system",
                "retrieval pipeline",
                "augmented generation system"
            ],
            "expected_topic": "RAG system architecture and implementation"
        }
    ]


def get_direct_content_queries() -> List[str]:
    """
    Get queries that should match specific content directly from the book.

    Returns:
        List of direct content query strings
    """
    return [
        "Docusaurus-based book content extraction",
        "Cohere embedding model implementation",
        "Qdrant Cloud vector storage",
        "Book content ingestion pipeline",
        "Vector database indexing strategy",
        "Embedding generation workflow",
        "Content chunking algorithm",
        "Metadata storage in Qdrant",
        "Semantic similarity threshold",
        "RAG chatbot architecture"
    ]


def get_comprehensive_test_suite() -> Dict[str, List[str]]:
    """
    Get a comprehensive test suite with different types of queries.

    Returns:
        Dictionary mapping test category to list of queries
    """
    return {
        "basic_semantic": get_sample_queries(),
        "topic_based": get_topic_based_queries(),
        "direct_content": get_direct_content_queries(),
        "variations": [item for sublist in [v["variations"] for v in get_keyword_variations()] for item in sublist]
    }


def get_validation_queries(min_similarity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Get queries specifically designed to validate the retrieval system against success criteria.

    Args:
        min_similarity: Expected minimum similarity threshold for validation

    Returns:
        List of query configurations for validation testing
    """
    return [
        {
            "query": "transformer architecture in NLP",
            "min_similarity": min_similarity,
            "expected_high_similarity": True,
            "category": "architecture"
        },
        {
            "query": "vector embeddings for semantic search",
            "min_similarity": min_similarity,
            "expected_high_similarity": True,
            "category": "embeddings"
        },
        {
            "query": "RAG pipeline implementation",
            "min_similarity": min_similarity,
            "expected_high_similarity": True,
            "category": "pipeline"
        },
        {
            "query": "nonexistent concept that should return no results",
            "min_similarity": min_similarity,
            "expected_high_similarity": False,
            "category": "edge_case"
        }
    ]