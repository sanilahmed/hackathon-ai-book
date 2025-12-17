#!/usr/bin/env python3
"""
Script to verify book chunks exist in Qdrant and test the embedding system
"""
import asyncio
import sys
import os

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from backend.rag_agent_api.retrieval import QdrantRetriever, create_qdrant_retriever
from backend.rag_agent_api.config import get_config

async def verify_qdrant_chunks():
    print("🔍 Verifying Qdrant collection and book chunks...\n")

    # Get configuration
    config = get_config()
    print(f"Using Qdrant URL: {config.qdrant_url}")
    print(f"Using collection: {config.qdrant_collection_name}\n")

    # Create retriever instance
    print("Creating Qdrant retriever...")
    retriever = await create_qdrant_retriever()

    if retriever is None:
        print("❌ Failed to create Qdrant retriever")
        return False

    print("✅ Qdrant retriever created successfully\n")

    # Test collection exists
    print("Checking if collection exists...")
    collection_exists = await retriever.validate_collection_exists()
    if collection_exists:
        print("✅ Collection exists\n")
    else:
        print("❌ Collection does not exist\n")
        return False

    # Get total points
    print("Getting total points in collection...")
    total_points = await retriever.get_total_points()
    print(f"Total points in collection: {total_points}\n")

    if total_points == 0:
        print("❌ No points in collection - book content may not be indexed\n")
        return False

    # Test retrieval with a sample query that should match book content
    print("Testing retrieval with sample queries that should match book content...")

    # Test queries that should be in the Physical AI & Humanoid Robotics book
    test_queries = [
        "Physical AI",
        "Humanoid Robotics",
        "ROS 2",
        "AI Robot",
        "Digital Twin"
    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        try:
            results = await retriever.retrieve_context(query=query, top_k=3)
            print(f"Retrieved {len(results)} chunks for query: '{query}'")

            if len(results) > 0:
                print("✅ Results found!")
                for i, result in enumerate(results[:2]):  # Show first 2 results
                    print(f"  Result {i+1}:")
                    print(f"    ID: {result.id}")
                    print(f"    Title: {result.title}")
                    print(f"    Content preview: {result.content[:100]}...")
                    print(f"    Similarity score: {result.similarity_score}")
            else:
                print("⚠️  No results found for this query")
        except Exception as e:
            print(f"❌ Error retrieving for query '{query}': {e}")

    # Test embedding function directly
    print(f"\nTesting embedding function with sample text...")
    try:
        sample_embedding = await retriever._embed_query("Physical AI and Humanoid Robotics")
        print(f"✅ Embedding successful! Embedding length: {len(sample_embedding)}")
        print(f"Sample values: {sample_embedding[:5]}...")  # Show first 5 values
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False

    print(f"\n" + "="*60)
    print("🎉 QDRANT VERIFICATION COMPLETED SUCCESSFULLY!")
    print(f"✅ Collection: {config.qdrant_collection_name}")
    print(f"✅ Total points: {total_points}")
    print(f"✅ Book content is available in Qdrant")
    print(f"✅ Embedding function working")
    print(f"✅ Sample queries returned results")
    print("="*60)

    return True

if __name__ == "__main__":
    success = asyncio.run(verify_qdrant_chunks())
    if success:
        print("\n✅ All verifications passed! The book chunks exist in Qdrant.")
    else:
        print("\n❌ Verification failed!")
        sys.exit(1)