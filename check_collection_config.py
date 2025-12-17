#!/usr/bin/env python3
"""
Script to check Qdrant collection configuration to understand the embedding setup
"""
import asyncio
import sys
import os
import requests
from qdrant_client import AsyncQdrantClient
from backend.rag_agent_api.config import get_config

async def check_collection_config():
    print("🔍 Checking Qdrant collection configuration...\n")

    # Get configuration
    config = get_config()
    print(f"Using Qdrant URL: {config.qdrant_url}")
    print(f"Using collection: {config.qdrant_collection_name}\n")

    # Initialize Qdrant client directly to check collection info
    client = AsyncQdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
        timeout=30
    )

    try:
        # Get collection info
        collection_info = await client.get_collection(config.qdrant_collection_name)
        print(f"Collection info:")
        print(f"  Status: {collection_info.status}")
        print(f"  Optimizer status: {collection_info.optimizer_status}")
        print(f"  Total points: {collection_info.points_count}")

        # Check vector configuration
        print(f"\nVector configuration:")
        if hasattr(collection_info, 'config') and collection_info.config:
            vector_params = collection_info.config.params.vectors
            print(f"  Vector size: {vector_params.size}")
            print(f"  Distance: {vector_params.distance}")
        else:
            print("  Could not retrieve vector configuration from collection info")

        # Get a sample point to see its structure
        print(f"\nGetting a sample point to check structure...")
        sample_points = await client.scroll(
            collection_name=config.qdrant_collection_name,
            limit=1,
            with_payload=True,
            with_vectors=False
        )

        if sample_points[0]:
            point = sample_points[0][0]  # Get the first point
            print(f"Sample point ID: {point.id}")
            print(f"Sample point payload keys: {list(point.payload.keys()) if point.payload else 'None'}")
            if point.payload:
                print(f"  - Title: {point.payload.get('title', 'N/A')[:50]}...")
                print(f"  - URL: {point.payload.get('url', 'N/A')}")
                print(f"  - Text length: {len(point.payload.get('text', ''))}")

        print(f"\n✅ Collection configuration checked successfully!")
        return collection_info

    except Exception as e:
        print(f"❌ Error checking collection configuration: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    collection_info = asyncio.run(check_collection_config())