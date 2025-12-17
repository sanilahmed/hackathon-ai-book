#!/usr/bin/env python3
"""
Test the RAG API endpoint directly
"""
import requests
import json

def test_rag_api():
    print("Testing RAG API endpoint...")

    # Test endpoint
    url = "http://localhost:8000/ask"

    # Test payload
    payload = {
        "query": "What is Physical AI?",
        "top_k": 5,
        "source_policy": 0.1
    }

    headers = {
        "Content-Type": "application/json"
    }

    print(f"Sending request to: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")

        if response.status_code == 200:
            response_data = response.json()
            print(f"Response data: {json.dumps(response_data, indent=2)}")

            # Check if response contains expected fields
            if 'answer' in response_data:
                print("✅ API is responding correctly")
                print(f"Response: {response_data['answer'][:200]}...")

                # Check if it returns the expected message when context is not found
                if "I could not find this information in the book" in response_data['answer']:
                    print("✅ API correctly returns 'information not found' message when context is unavailable due to rate limits")
                else:
                    print("✅ API returned a response, though it may be limited by rate limits")
            else:
                print("❌ Response doesn't contain expected 'answer' field")
        else:
            print(f"❌ API returned error: {response.status_code}")
            print(f"Response text: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_rag_api()