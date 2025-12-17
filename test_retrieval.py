#!/usr/bin/env python3
"""
Test script to verify RAG system is working with OpenAI embeddings
"""
import asyncio
import sys
import os

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from backend.rag_agent_api.retrieval import QdrantRetriever, create_qdrant_retriever
from backend.rag_agent_api.config import get_config

async def test_retrieval():
    print("Testing RAG system with OpenAI embeddings...")

    # Get configuration
    config = get_config()
    print(f"Using Qdrant URL: {config.qdrant_url}")
    print(f"Using collection: {config.qdrant_collection_name}")

    # Create retriever instance
    print("\nCreating Qdrant retriever...")
    retriever = await create_qdrant_retriever()

    if retriever is None:
        print("❌ Failed to create Qdrant retriever")
        return False

    print("✅ Qdrant retriever created successfully")

    # Test collection exists
    print("\nChecking if collection exists...")
    collection_exists = await retriever.validate_collection_exists()
    if collection_exists:
        print("✅ Collection exists")
    else:
        print("❌ Collection does not exist")
        return False

    # Test total points
    print("\nGetting total points in collection...")
    total_points = await retriever.get_total_points()
    print(f"Total points in collection: {total_points}")

    if total_points == 0:
        print("❌ No points in collection - book content may not be indexed")
        return False

    # Test retrieval with a sample query
    print("\nTesting retrieval with sample query...")
    sample_query = "What is Physical AI?"
    results = await retriever.retrieve_context(query=sample_query, top_k=3)

    print(f"Retrieved {len(results)} chunks for query: '{sample_query}'")

    if len(results) > 0:
        print("✅ Retrieval successful!")
        print("\nFirst result details:")
        first_result = results[0]
        print(f"  ID: {first_result.id}")
        print(f"  Title: {first_result.title}")
        print(f"  URL: {first_result.url}")
        print(f"  Content preview: {first_result.content[:200]}...")
        print(f"  Similarity score: {first_result.similarity_score}")
        print(f"  Chunk index: {first_result.chunk_index}")
    else:
        print("❌ No results retrieved - check if book content is properly indexed in Qdrant")
        return False

    # Test embedding function directly
    print("\nTesting OpenAI embedding function...")
    try:
        sample_embedding = await retriever._embed_query("test query for embedding")
        print(f"✅ Embedding successful! Embedding length: {len(sample_embedding)}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False

    # Test search with filters
    print("\nTesting search with filters...")
    filtered_results = await retriever.search_with_filters(
        query="Physical AI",
        top_k=2,
        filters={"title": "Physical AI & Humanoid Robotics"}
    )
    print(f"Filtered results count: {len(filtered_results)}")

    print("\n" + "="*50)
    print("🎉 RAG SYSTEM TEST COMPLETED SUCCESSFULLY!")
    print("✅ OpenAI embeddings are working")
    print("✅ Qdrant connection is established")
    print("✅ Data retrieval is functional")
    print("✅ Book content is available in Qdrant")
    print("="*50)

    return True

if __name__ == "__main__":
    success = asyncio.run(test_retrieval())
    if success:
        print("\n✅ All tests passed! The RAG system is working correctly with OpenAI embeddings.")
    else:
        print("\n❌ Tests failed! There are issues with the RAG system.")
        sys.exit(1)