"""
Unit tests for the Qdrant Retrieval functionality.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from rag_agent_api.retrieval import QdrantRetriever
from rag_agent_api.schemas import SourceChunkSchema


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    with patch.dict('os.environ', {
        'QDRANT_URL': 'http://test-qdrant:6333',
        'QDRANT_API_KEY': 'test-api-key',
        'COHERE_API_KEY': 'test-cohere-key'
    }):
        with patch('rag_agent_api.retrieval.AsyncQdrantClient') as mock_qdrant_client:
            with patch('rag_agent_api.retrieval.cohere.Client') as mock_cohere_client:
                # Mock the Qdrant client
                mock_qdrant_instance = Mock()
                mock_qdrant_client.return_value = mock_qdrant_instance

                # Mock the Cohere client
                mock_cohere_instance = Mock()
                mock_cohere_client.return_value = mock_cohere_instance
                mock_cohere_instance.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])

                # Mock the get_collection method
                mock_qdrant_instance.get_collection.return_value = Mock(points_count=100)

                retriever = QdrantRetriever(collection_name="test_collection")

                yield retriever, mock_qdrant_instance, mock_cohere_instance


def test_retriever_initialization():
    """Test that the retriever initializes correctly."""
    with patch.dict('os.environ', {
        'QDRANT_URL': 'http://test-qdrant:6333',
        'QDRANT_API_KEY': 'test-api-key',
        'COHERE_API_KEY': 'test-cohere-key'
    }):
        with patch('rag_agent_api.retrieval.AsyncQdrantClient') as mock_qdrant_client:
            with patch('rag_agent_api.retrieval.cohere.Client') as mock_cohere_client:
                # Mock the Qdrant client
                mock_qdrant_instance = Mock()
                mock_qdrant_client.return_value = mock_qdrant_instance
                mock_qdrant_instance.get_collection.return_value = Mock(points_count=100)

                # Mock the Cohere client
                mock_cohere_instance = Mock()
                mock_cohere_client.return_value = mock_cohere_instance
                mock_cohere_instance.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])

                retriever = QdrantRetriever(collection_name="test_collection")

                assert retriever.collection_name == "test_collection"
                assert mock_qdrant_client.called


@pytest.mark.asyncio
async def test_retrieve_context_empty_query(mock_retriever):
    """Test retrieving context with an empty query."""
    retriever, mock_qdrant, mock_cohere = mock_retriever

    # Mock the search method to return empty results
    mock_qdrant.search.return_value = []

    chunks = await retriever.retrieve_context("", top_k=5)

    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_context_successful(mock_retriever):
    """Test successful context retrieval."""
    retriever, mock_qdrant, mock_cohere = mock_retriever

    # Mock a search result
    mock_result = Mock()
    mock_result.id = "test-id"
    mock_result.score = 0.85
    mock_result.payload = {
        'url': 'https://example.com/test',
        'title': 'Test Document',
        'text': 'This is test content',
        'chunk_index': 1
    }

    mock_qdrant.search.return_value = [mock_result]

    chunks = await retriever.retrieve_context("test query", top_k=1)

    # Should return a list with one SourceChunkSchema
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.id == "test-id"
    assert chunk.url == "https://example.com/test"
    assert chunk.title == "Test Document"
    assert chunk.content == "This is test content"
    assert chunk.similarity_score == 0.85
    assert chunk.chunk_index == 1


def test_validate_chunk_valid():
    """Test validation of a valid chunk."""
    with patch.dict('os.environ', {
        'QDRANT_URL': 'http://test-qdrant:6333',
        'QDRANT_API_KEY': 'test-api-key',
        'COHERE_API_KEY': 'test-cohere-key'
    }):
        with patch('rag_agent_api.retrieval.AsyncQdrantClient') as mock_qdrant_client:
            with patch('rag_agent_api.retrieval.cohere.Client') as mock_cohere_client:
                # Mock the Qdrant client
                mock_qdrant_instance = Mock()
                mock_qdrant_client.return_value = mock_qdrant_instance
                mock_qdrant_instance.get_collection.return_value = Mock(points_count=100)

                # Mock the Cohere client
                mock_cohere_instance = Mock()
                mock_cohere_client.return_value = mock_cohere_instance
                mock_cohere_instance.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])

                retriever = QdrantRetriever(collection_name="test_collection")

                # Create a valid chunk
                valid_chunk = SourceChunkSchema(
                    id="test-id",
                    url="https://example.com/test",
                    title="Test Document",
                    content="This is test content",
                    similarity_score=0.85,
                    chunk_index=1
                )

                is_valid = retriever._validate_chunk(valid_chunk)
                assert is_valid is True


def test_validate_chunk_invalid():
    """Test validation of an invalid chunk."""
    with patch.dict('os.environ', {
        'QDRANT_URL': 'http://test-qdrant:6333',
        'QDRANT_API_KEY': 'test-api-key',
        'COHERE_API_KEY': 'test-cohere-key'
    }):
        with patch('rag_agent_api.retrieval.AsyncQdrantClient') as mock_qdrant_client:
            with patch('rag_agent_api.retrieval.cohere.Client') as mock_cohere_client:
                # Mock the Qdrant client
                mock_qdrant_instance = Mock()
                mock_qdrant_client.return_value = mock_qdrant_instance
                mock_qdrant_instance.get_collection.return_value = Mock(points_count=100)

                # Mock the Cohere client
                mock_cohere_instance = Mock()
                mock_cohere_client.return_value = mock_cohere_instance
                mock_cohere_instance.embed.return_value = Mock(embeddings=[[0.1, 0.2, 0.3]])

                retriever = QdrantRetriever(collection_name="test_collection")

                # Create an invalid chunk (empty content)
                invalid_chunk = SourceChunkSchema(
                    id="",
                    url="https://example.com/test",
                    title="Test Document",
                    content="",
                    similarity_score=0.85,
                    chunk_index=1
                )

                is_valid = retriever._validate_chunk(invalid_chunk)
                assert is_valid is False


@pytest.mark.asyncio
async def test_get_total_points(mock_retriever):
    """Test getting total points in the collection."""
    retriever, mock_qdrant, mock_cohere = mock_retriever

    # Mock collection info
    mock_collection_info = Mock()
    mock_collection_info.points_count = 150
    mock_qdrant.get_collection.return_value = mock_collection_info

    total_points = await retriever.get_total_points()

    assert total_points == 150


@pytest.mark.asyncio
async def test_validate_collection_exists(mock_retriever):
    """Test validating collection exists."""
    retriever, mock_qdrant, mock_cohere = mock_retriever

    # Mock the get_collection method to simulate successful collection retrieval
    mock_qdrant.get_collection.return_value = Mock(points_count=100)

    exists = await retriever.validate_collection_exists()

    assert exists is True


if __name__ == "__main__":
    pytest.main([__file__])