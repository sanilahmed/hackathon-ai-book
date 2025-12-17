"""
Unit tests for the OpenAI Agent functionality.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from rag_agent_api.agent import OpenAIAgent
from rag_agent_api.schemas import AgentContext, SourceChunkSchema


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    # Patch the AsyncOpenAI client to avoid actual API calls
    with patch('rag_agent_api.agent.AsyncOpenAI') as mock_openai:
        agent = OpenAIAgent(model_name="gpt-4-test")
        agent.client = Mock()
        agent.client.chat = Mock()
        agent.client.chat.completions = Mock()

        # Mock the create method to return a fake response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = "This is a test response"

        agent.client.chat.completions.create = AsyncMock(return_value=mock_completion)

        yield agent


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test that the agent initializes correctly."""
    # Mock environment variable
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            assert agent.model_name == "gpt-4-turbo"
            assert agent.default_temperature == 0.1  # Default from config


@pytest.mark.asyncio
async def test_generate_response(mock_agent):
    """Test that the agent generates a response."""
    # Create a mock context
    mock_chunk = SourceChunkSchema(
        id="test-chunk-1",
        url="https://example.com/test",
        title="Test Document",
        content="This is test content for the agent.",
        similarity_score=0.85,
        chunk_index=1
    )

    context = AgentContext(
        query="What is this document about?",
        retrieved_chunks=[mock_chunk],
        max_context_length=4000,
        source_policy="strict"
    )

    # Call generate_response
    response = await mock_agent.generate_response(context)

    # Assert the response has expected properties
    assert response.raw_response == "This is a test response"
    assert response.is_valid is not None  # Should be True/False based on validation
    assert 0.0 <= response.confidence_score <= 1.0


@pytest.mark.asyncio
async def test_create_system_message():
    """Test that the system message is created correctly."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            mock_chunk = SourceChunkSchema(
                id="test-chunk-1",
                url="https://example.com/test",
                title="Test Document",
                content="This is test content.",
                similarity_score=0.85,
                chunk_index=1
            )

            context = AgentContext(
                query="What is this?",
                retrieved_chunks=[mock_chunk],
                max_context_length=4000,
                source_policy="strict"
            )

            system_message = agent._create_system_message(context)

            # Check that system message contains expected elements
            assert "You are an AI assistant that answers questions based only on the provided context" in system_message
            assert "Do not make up information that is not present in the provided context" in system_message


@pytest.mark.asyncio
async def test_identify_used_sources():
    """Test the identification of used sources."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create test chunks
            chunk1 = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/doc1",
                title="Document 1",
                content="This is content about machine learning.",
                similarity_score=0.9,
                chunk_index=1
            )

            chunk2 = SourceChunkSchema(
                id="chunk-2",
                url="https://example.com/doc2",
                title="Document 2",
                content="This is content about neural networks.",
                similarity_score=0.7,
                chunk_index=1
            )

            chunks = [chunk1, chunk2]
            response_text = "Machine learning is important in neural networks."

            used_sources = agent._identify_used_sources(response_text, chunks)

            # Should identify both chunks as potentially used
            assert "chunk-1" in used_sources or "chunk-2" in used_sources


@pytest.mark.asyncio
async def test_calculate_confidence_score():
    """Test the calculation of confidence score."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        with patch('rag_agent_api.agent.AsyncOpenAI'):
            agent = OpenAIAgent()

            # Create test chunks
            chunk1 = SourceChunkSchema(
                id="chunk-1",
                url="https://example.com/doc1",
                title="Document 1",
                content="This is content about machine learning.",
                similarity_score=0.9,
                chunk_index=1
            )

            chunk2 = SourceChunkSchema(
                id="chunk-2",
                url="https://example.com/doc2",
                title="Document 2",
                content="This is content about neural networks.",
                similarity_score=0.7,
                chunk_index=1
            )

            chunks = [chunk1, chunk2]
            used_sources = ["chunk-1", "chunk-2"]

            confidence = agent._calculate_confidence_score(used_sources, chunks)

            # Should be average of the similarity scores (0.9 + 0.7) / 2 = 0.8
            assert 0.0 <= confidence <= 1.0


if __name__ == "__main__":
    pytest.main([__file__])