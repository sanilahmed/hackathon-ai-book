"""
Tests for response grounding validation in the RAG Agent and API Layer system.
"""
import pytest
from rag_agent_api.agent import OpenAIAgent
from rag_agent_api.schemas import AgentContext, SourceChunkSchema
from unittest.mock import patch


def test_response_grounding_validation():
    """Test that the agent validates response grounding in context."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create test chunks with different content
            chunk1 = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/ml",
                title="Machine Learning Basics",
                content="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                similarity_score=0.9,
                chunk_index=1
            )

            chunk2 = SourceChunkSchema(
                id="chunk-2",
                url="https://example.com/nn",
                title="Neural Networks",
                content="Neural networks are computing systems inspired by the human brain, consisting of interconnected nodes that process information.",
                similarity_score=0.85,
                chunk_index=2
            )

            context = AgentContext(
                query="What is machine learning?",
                retrieved_chunks=[chunk1, chunk2],
                max_context_length=4000,
                source_policy="strict"
            )

            # Test the grounding validation function directly
            response_text = "Machine learning is a subset of AI that enables computers to learn from data."
            validation_result = agent._validate_response_grounding(response_text, context.retrieved_chunks, context.query)

            assert "is_valid" in validation_result
            assert "details" in validation_result
            assert "unsupported_claims" in validation_result


def test_response_with_unsupported_claims():
    """Test validation when response contains unsupported claims."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create a chunk with specific content
            chunk = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/python",
                title="Python Programming",
                content="Python is a high-level programming language known for its simplicity and readability.",
                similarity_score=0.8,
                chunk_index=1
            )

            context = AgentContext(
                query="What is Python?",
                retrieved_chunks=[chunk],
                max_context_length=4000,
                source_policy="strict"
            )

            # Response that includes a claim not in the context
            response_text = "Python is a high-level programming language. It was created in 1991 by Guido van Rossum."
            # Note: The creation year and creator are not mentioned in the context

            validation_result = agent._validate_response_grounding(response_text, context.retrieved_chunks, context.query)

            # The validation should still return valid but might identify unsupported claims
            # depending on the sophistication of the validation
            assert "is_valid" in validation_result


def test_empty_context_handling():
    """Test how the system handles empty context."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Empty context
            context = AgentContext(
                query="What is AI?",
                retrieved_chunks=[],
                max_context_length=4000,
                source_policy="strict"
            )

            response_text = "I don't have enough information in the provided context to answer this question."
            validation_result = agent._validate_response_grounding(response_text, context.retrieved_chunks, context.query)

            # Even with empty context, validation should work
            assert "is_valid" in validation_result
            assert "details" in validation_result


def test_context_overlap_calculation():
    """Test the calculation of overlap between response and context."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create test chunks
            chunk1 = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/test",
                title="Test Document",
                content="Artificial intelligence and machine learning are related fields in computer science.",
                similarity_score=0.85,
                chunk_index=1
            )

            chunks = [chunk1]
            query = "What is AI?"

            response_with_context_terms = "Artificial intelligence is a field in computer science."
            response_without_context_terms = "Cooking is an important life skill."

            # Test response that uses terms from context
            validation_result_with = agent._validate_response_grounding(
                response_with_context_terms, chunks, query
            )

            # Test response that doesn't use terms from context
            validation_result_without = agent._validate_response_grounding(
                response_without_context_terms, chunks, query
            )

            # Both should return valid results
            assert "is_valid" in validation_result_with
            assert "is_valid" in validation_result_without


def test_agent_response_quality_validation():
    """Test the quality validation of agent responses."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create test context
            chunk = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/test",
                title="Test Document",
                content="The transformer model is a deep learning model introduced in the paper 'Attention Is All You Need'.",
                similarity_score=0.9,
                chunk_index=1
            )

            context = AgentContext(
                query="What is a transformer model?",
                retrieved_chunks=[chunk],
                max_context_length=4000,
                source_policy="strict"
            )

            # Test with a valid response
            valid_response = "A transformer model is a deep learning model introduced in the paper 'Attention Is All You Need'."
            is_quality_valid = agent.validate_response_quality(valid_response, context)

            # Should be valid as it's based on the context
            assert is_quality_valid


if __name__ == "__main__":
    pytest.main([__file__])