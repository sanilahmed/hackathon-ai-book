#!/usr/bin/env python3
"""
Script to list available Gemini models
"""
import google.generativeai as genai
from backend.rag_agent_api.config import get_config

def list_models():
    print("🔍 Listing available Gemini models...\n")

    # Get configuration
    config = get_config()

    if not config.gemini_api_key:
        print("❌ GEMINI_API_KEY not found in environment")
        return

    # Configure the API
    genai.configure(api_key=config.gemini_api_key)

    try:
        # List all available models
        print("Available models:")
        for model in genai.list_models():
            print(f"  - {model.name}")
            # Check if this model supports the functionality we need
            if 'generateContent' in model.supported_generation_methods:
                print(f"    → Supports generateContent")
            if 'embedContent' in model.supported_generation_methods:
                print(f"    → Supports embedContent")
            if hasattr(model, 'input_token_limit'):
                print(f"    → Input token limit: {model.input_token_limit}")
            print()

    except Exception as e:
        print(f"❌ Error listing models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    list_models()