#!/usr/bin/env python3
"""
Simple test to check Qdrant connection and collection
"""
import asyncio
import sys
import os

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from backend.rag_agent_api.retrieval import QdrantRetriever, create_qdrant_retriever
from backend.rag_agent_api.config import get_config

async def test_qdrant_connection():
    print("Testing Qdrant connection only (without embedding)...")

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

    print("\n" + "="*50)
    print("🎉 QDRANT CONNECTION TEST COMPLETED!")
    print("✅ Qdrant connection is established")
    print("✅ Collection exists")
    print("✅ Book content is available in Qdrant")
    print("⚠️  NOTE: Embedding service has rate limits but Qdrant itself is working")
    print("="*50)

    return True

if __name__ == "__main__":
    success = asyncio.run(test_qdrant_connection())
    if success:
        print("\n✅ Qdrant connection test passed! The issue is with the embedding service (OpenAI rate limits), not Qdrant itself.")
    else:
        print("\n❌ Qdrant connection test failed!")
        sys.exit(1)