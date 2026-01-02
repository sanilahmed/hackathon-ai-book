#!/usr/bin/env python3
"""
Script to check if Qdrant collection exists and has data.
"""
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
qdrant_url = os.getenv('QDRANT_URL')
qdrant_api_key = os.getenv('QDRANT_API_KEY')

if not qdrant_url or not qdrant_api_key:
    print("Error: QDRANT_URL or QDRANT_API_KEY not found in environment variables")
    exit(1)

# Initialize Qdrant client
client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    timeout=30
)

try:
    # List all collections
    collections = client.get_collections()
    print("Available collections:")
    for collection in collections.collections:
        # For newer Qdrant versions, get the collection info to get point count
        collection_info = client.get_collection(collection.name)
        print(f"  - {collection.name} (points: {collection_info.points_count})")

    # Check specifically for the rag_embedding collection
    try:
        collection_info = client.get_collection("rag_embedding")
        print(f"\nCollection 'rag_embedding' exists with {collection_info.points_count} points")

        if collection_info.points_count > 0:
            # Get a sample point to verify data exists
            points = client.scroll(
                collection_name="rag_embedding",
                limit=1
            )
            if len(points[0]) > 0:
                sample_point = points[0][0]
                print(f"Sample point ID: {sample_point.id}")
                print(f"Sample point payload keys: {list(sample_point.payload.keys())}")
                print(f"Sample text preview: {sample_point.payload.get('text', '')[:100]}...")
        else:
            print("Collection 'rag_embedding' exists but is empty")

    except Exception as e:
        print(f"\nCollection 'rag_embedding' does not exist: {e}")

except Exception as e:
    print(f"Error connecting to Qdrant: {e}")