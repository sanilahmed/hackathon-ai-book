#!/usr/bin/env python3
"""
Test script to directly test the Qdrant retrieval functionality
"""
import asyncio
import os
from dotenv import load_dotenv
from rag_agent_api.retrieval import QdrantRetriever
from rag_agent_api.config import get_config

# Load environment variables
load_dotenv()

async def test_retrieval():
    print("Testing Qdrant retrieval functionality...")

    # Create a QdrantRetriever instance
    retriever = QdrantRetriever()

    print("1. Testing collection existence...")
    exists = await retriever.validate_collection_exists()
    print(f"   Collection exists: {exists}")

    if exists:
        print("2. Getting total points in collection...")
        total_points = await retriever.get_total_points()
        print(f"   Total points: {total_points}")

    print("3. Testing query embedding...")
    try:
        query = "what about this book?"
        embedding = await retriever._embed_query(query)
        print(f"   Query embedding successful, length: {len(embedding)}")
    except Exception as e:
        print(f"   Query embedding failed: {e}")
        return

    print("4. Testing direct search...")
    try:
        results = await retriever.retrieve_context(query, top_k=5)
        print(f"   Retrieved {len(results)} results")

        if results:
            print("   Sample results:")
            for i, result in enumerate(results[:2]):  # Show first 2 results
                print(f"     Result {i+1}:")
                print(f"       ID: {result.id}")
                print(f"       Title: {result.title}")
                print(f"       Content preview: {result.content[:100]}...")
                print(f"       Similarity: {result.similarity_score}")
                print(f"       URL: {result.url}")
        else:
            print("   No results retrieved - this indicates the main issue")
    except Exception as e:
        print(f"   Direct search failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_retrieval())