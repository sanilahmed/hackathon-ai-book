#!/usr/bin/env python3
"""
Test script to verify Gemini API responses work properly
"""
import asyncio
import sys
import os

# Add the backend directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from backend.rag_agent_api.agent import GeminiAgent
from backend.rag_agent_api.schemas import AgentContext, SourceChunkSchema

async def test_gemini_response():
    print("🔍 Testing Gemini API response functionality...\n")

    try:
        # Create agent instance
        print("Creating Gemini agent...")
        agent = GeminiAgent()
        print("✅ Gemini agent created successfully\n")

        # Create a simple test context with minimal content to avoid rate limits on embedding
        test_chunks = [
            SourceChunkSchema(
                id="test_chunk_1",
                url="https://example.com",
                title="Test Content",
                content="The Physical AI & Humanoid Robotics book covers advanced robotics concepts.",
                similarity_score=0.8,
                chunk_index=0
            )
        ]

        test_context = AgentContext(
            query="What is this book about?",
            retrieved_chunks=test_chunks,
            max_context_length=4000,
            source_policy="strict"
        )

        print("Testing response generation with context...")
        response = await agent.generate_response(test_context)

        print(f"✅ Response generated successfully!")
        print(f"Response: {response.raw_response}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Is valid: {response.is_valid}")

        if response.raw_response and len(response.raw_response) > 0:
            print("✅ Gemini API is working properly and generating responses")
            return True
        else:
            print("⚠️  Gemini API returned empty response")
            return False

    except Exception as e:
        print(f"❌ Error testing Gemini response: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_gemini_without_context():
    print("\n🔍 Testing Gemini API response without context (triggering hard guard)...\n")

    try:
        # Create agent instance
        print("Creating Gemini agent...")
        agent = GeminiAgent()
        print("✅ Gemini agent created successfully\n")

        # Create a context with no chunks to test the hard guard
        test_context = AgentContext(
            query="What is the meaning of life?",
            retrieved_chunks=[],  # Empty chunks to trigger hard guard
            max_context_length=4000,
            source_policy="strict"
        )

        print("Testing response generation without context (should trigger hard guard)...")
        response = await agent.generate_response(test_context)

        print(f"✅ Response generated successfully!")
        print(f"Response: {response.raw_response}")
        print(f"Confidence: {response.confidence_score}")
        print(f"Is valid: {response.is_valid}")

        if "I could not find this information in the book" in response.raw_response:
            print("✅ Hard guard working properly - returned book-only message")
            return True
        else:
            print("⚠️  Hard guard may not be working as expected")
            return False

    except Exception as e:
        print(f"❌ Error testing Gemini response without context: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    print("🚀 Testing Gemini API Response Functionality\n")

    success1 = await test_gemini_response()
    success2 = await test_gemini_without_context()

    if success1 and success2:
        print(f"\n" + "="*60)
        print("🎉 GEMINI API TESTS COMPLETED SUCCESSFULLY!")
        print("✅ Gemini agent properly configured")
        print("✅ Response generation working")
        print("✅ Hard guard functionality verified")
        print("✅ System properly handles both with/without context")
        print("="*60)
        return True
    else:
        print(f"\n❌ Some Gemini API tests failed!")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if not success:
        sys.exit(1)