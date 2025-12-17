#!/usr/bin/env python3
"""
Test script to verify Google Gemini API integration
"""
import asyncio
import os
import sys

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import google.generativeai as genai
from backend.rag_agent_api.config import get_config

async def test_gemini_embeddings():
    print("Testing Google Gemini API integration...")

    # Get configuration
    config = get_config()

    if not config.gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return False

    # Configure the API
    genai.configure(api_key=config.gemini_api_key)

    print("✅ Gemini API key loaded successfully")

    try:
        # Test embedding functionality
        print("\nTesting embedding functionality...")
        result = await genai.embed_content_async(
            model="models/embedding-001",
            content=["Hello, world! This is a test query."],
            task_type="RETRIEVAL_QUERY"
        )

        print(f"✅ Embedding successful!")
        print(f"Embedding type: {type(result)}")

        # Check the structure of the result
        if hasattr(result, 'embedding'):
            print(f"Embedding attribute exists: {type(result.embedding)}")
            if isinstance(result.embedding, list) and len(result.embedding) > 0:
                print(f"First embedding length: {len(result.embedding[0])}")
            else:
                print(f"Embedding length: {len(result.embedding)}")
        elif 'embedding' in result:
            print(f"Embedding key exists: {type(result['embedding'])}")
        else:
            print(f"Result structure: {result}")

        return True

    except Exception as e:
        print(f"❌ Error testing Gemini embeddings: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gemini_text_generation():
    print("\nTesting Gemini text generation...")

    try:
        # Get configuration
        config = get_config()
        genai.configure(api_key=config.gemini_api_key)

        # Create a generative model instance
        model = genai.GenerativeModel('gemini-pro')

        # Test text generation
        response = await model.generate_content_async("What is machine learning?")

        print(f"✅ Text generation successful!")
        print(f"Response type: {type(response)}")
        if hasattr(response, 'text'):
            print(f"Response preview: {response.text[:100]}...")
        else:
            print(f"Response: {response}")

        return True

    except Exception as e:
        print(f"❌ Error testing Gemini text generation: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🔍 Testing Google Gemini API Integration\n")

    success1 = await test_gemini_embeddings()
    success2 = await test_gemini_text_generation()

    if success1 and success2:
        print("\n🎉 All Gemini API tests passed!")
        return True
    else:
        print("\n❌ Some Gemini API tests failed!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)