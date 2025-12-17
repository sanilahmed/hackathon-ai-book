"""
Unit tests for the RAG Retrieval Verification system.

This module contains unit tests for the verification functions and classes.
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from verify_retrieval.qdrant_client import QdrantVerificationClient
from verify_retrieval.validators import (
    validate_metadata_consistency,
    validate_retrieved_chunks,
    validate_similarity_scores,
    validate_metadata_accuracy
)
from verify_retrieval.models import MetadataRecord, RetrievedContentChunk, QueryRequest, VerificationResult


class TestQdrantVerificationClient:
    """Test cases for QdrantVerificationClient class."""

    @patch('verify_retrieval.qdrant_client.QdrantClient')
    def test_client_initialization(self, mock_qdrant_client):
        """Test that the client initializes with correct configuration."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://fake-qdrant:6333',
            'QDRANT_API_KEY': 'fake-api-key',
            'QDRANT_COLLECTION_NAME': 'rag_embedding'
        }):
            # Mock the QdrantClient constructor to avoid actual connection
            mock_qdrant_client.return_value = Mock()

            client = QdrantVerificationClient(collection_name="test_collection")
            assert client.collection_name == "test_collection"

    @patch('verify_retrieval.qdrant_client.QdrantClient')
    def test_validate_collection_exists(self, mock_qdrant_client):
        """Test collection existence validation."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://fake-qdrant:6333',
            'QDRANT_API_KEY': 'fake-api-key',
            'QDRANT_COLLECTION_NAME': 'rag_embedding'
        }):
            mock_client = Mock()
            mock_qdrant_client.return_value = mock_client
            mock_client.get_collection.return_value = True

            client = QdrantVerificationClient(collection_name="test_collection")
            result = client.validate_collection_exists()

            assert result is True
            mock_client.get_collection.assert_called_once_with("test_collection")

    @patch('verify_retrieval.qdrant_client.QdrantClient')
    def test_query_qdrant_for_chunks(self, mock_qdrant_client):
        """Test querying Qdrant for content chunks."""
        with patch.dict(os.environ, {
            'QDRANT_URL': 'http://fake-qdrant:6333',
            'QDRANT_API_KEY': 'fake-api-key',
            'QDRANT_COLLECTION_NAME': 'rag_embedding',
            'COHERE_API_KEY': 'fake-cohere-key'
        }):
            # Since the real Cohere API might have different signature,
            # let's just test the logic flow by mocking the embed call internally
            with patch('verify_retrieval.qdrant_client.cohere') as mock_cohere_module:
                mock_cohere_client = Mock()
                mock_cohere_module.Client.return_value = mock_cohere_client

                # Mock the embed method response correctly
                mock_embed_response = Mock()
                mock_embed_response.embeddings = [[0.1, 0.2, 0.3]]
                mock_cohere_client.embed.return_value = mock_embed_response

                mock_client = Mock()
                mock_qdrant_client.return_value = mock_client
                mock_client.search.return_value = [
                    Mock(
                        id="test_id",
                        score=0.8,
                        payload={
                            "url": "https://example.com/test",
                            "title": "Test Title",
                            "content": "Test content",
                            "chunk_index": 1
                        }
                    )
                ]

                client = QdrantVerificationClient(collection_name="test_collection")
                results = client.query_qdrant_for_chunks("test query", top_k=5)

                assert len(results) == 1
                assert results[0]['similarity_score'] == 0.8
                assert results[0]['url'] == "https://example.com/test"
                assert results[0]['title'] == "Test Title"

                # Verify that embed was called with correct parameters
                mock_cohere_client.embed.assert_called_once_with(
                    ["test query"],
                    model="embed-english-v3.0",
                    input_type="search_query"
                )


class TestValidators:
    """Test cases for validation functions."""

    def test_validate_metadata_consistency_valid(self):
        """Test metadata consistency validation with valid data."""
        results = [
            {
                'id': 'test_id_1',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1,
                'content': 'Test content 1',
                'similarity_score': 0.8
            },
            {
                'id': 'test_id_2',
                'url': 'https://example.com/test2',
                'title': 'Test Title 2',
                'chunk_id': 2,
                'content': 'Test content 2',
                'similarity_score': 0.75
            }
        ]

        validation_result = validate_metadata_consistency(results)

        assert validation_result['is_valid'] is True
        assert validation_result['total_errors'] == 0
        assert validation_result['accuracy_percentage'] == 100.0

    def test_validate_metadata_consistency_invalid(self):
        """Test metadata consistency validation with invalid data."""
        results = [
            {
                'id': 'test_id_1',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1,
                'content': 'Test content 1',
                'similarity_score': 0.8
            },
            {
                'id': 'test_id_2',
                'url': '',  # Missing URL
                'title': 'Test Title 2',
                'chunk_id': 2,
                'content': 'Test content 2',
                'similarity_score': 0.75
            }
        ]

        validation_result = validate_metadata_consistency(results)

        assert validation_result['is_valid'] is False
        assert validation_result['total_errors'] == 1
        assert validation_result['accuracy_percentage'] == 50.0

    def test_validate_retrieved_chunks(self):
        """Test validation of retrieved chunks."""
        query = "test query"
        results = [
            {
                'id': 'test_id_1',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1,
                'content': 'Test content related to query',
                'similarity_score': 0.8
            }
        ]

        validation_result = validate_retrieved_chunks(query, results)

        assert validation_result['query'] == query
        assert validation_result['total_chunks'] == 1
        assert validation_result['valid_chunks'] == 1
        assert validation_result['has_content'] is True

    def test_validate_similarity_scores_above_threshold(self):
        """Test similarity score validation with scores above threshold."""
        results = [
            {
                'id': 'test_id_1',
                'similarity_score': 0.8,
                'content': 'Test content',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1
            },
            {
                'id': 'test_id_2',
                'similarity_score': 0.75,
                'content': 'Test content',
                'url': 'https://example.com/test2',
                'title': 'Test Title 2',
                'chunk_id': 2
            }
        ]

        validation_result = validate_similarity_scores(results, min_threshold=0.7)

        assert validation_result['threshold_compliance'] == 100.0
        assert validation_result['below_threshold'] == 0
        assert validation_result['average_score'] == 0.775

    def test_validate_similarity_scores_below_threshold(self):
        """Test similarity score validation with some scores below threshold."""
        results = [
            {
                'id': 'test_id_1',
                'similarity_score': 0.8,
                'content': 'Test content',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1
            },
            {
                'id': 'test_id_2',
                'similarity_score': 0.5,
                'content': 'Test content',
                'url': 'https://example.com/test2',
                'title': 'Test Title 2',
                'chunk_id': 2
            }
        ]

        validation_result = validate_similarity_scores(results, min_threshold=0.7)

        assert validation_result['threshold_compliance'] == 50.0
        assert validation_result['below_threshold'] == 1
        assert validation_result['average_score'] == 0.65

    def test_validate_metadata_accuracy(self):
        """Test metadata accuracy validation."""
        results = [
            {
                'id': 'test_id_1',
                'url': 'https://example.com/test1',
                'title': 'Test Title 1',
                'chunk_id': 1,
                'content': 'Test content 1',
                'similarity_score': 0.8
            },
            {
                'id': 'test_id_2',
                'url': 'https://example.com/test2',
                'title': 'Test Title 2',
                'chunk_id': 2,
                'content': 'Test content 2',
                'similarity_score': 0.75
            }
        ]

        accuracy = validate_metadata_accuracy(results)

        assert accuracy == 100.0


class TestModels:
    """Test cases for data models."""

    def test_metadata_record_creation(self):
        """Test creation of MetadataRecord with valid data."""
        record = MetadataRecord(
            url="https://example.com/test",
            title="Test Title",
            chunk_id=1
        )

        assert record.url == "https://example.com/test"
        assert record.title == "Test Title"
        assert record.chunk_id == 1

    def test_metadata_record_validation(self):
        """Test validation of MetadataRecord with invalid data."""
        with pytest.raises(ValueError):
            MetadataRecord(
                url="",  # Invalid - empty URL
                title="Test Title",
                chunk_id=1
            )

    def test_retrieved_content_chunk_creation(self):
        """Test creation of RetrievedContentChunk with valid data."""
        chunk = RetrievedContentChunk(
            id="test_id",
            content="Test content",
            similarity_score=0.8,
            metadata=MetadataRecord(
                url="https://example.com/test",
                title="Test Title",
                chunk_id=1
            )
        )

        assert chunk.id == "test_id"
        assert chunk.content == "Test content"
        assert chunk.similarity_score == 0.8
        assert chunk.metadata.url == "https://example.com/test"

    def test_query_request_creation(self):
        """Test creation of QueryRequest with valid data."""
        request = QueryRequest(
            query="test query",
            top_k=5,
            min_similarity=0.7
        )

        assert request.query == "test query"
        assert request.top_k == 5
        assert request.min_similarity == 0.7

    def test_verification_result_creation(self):
        """Test creation of VerificationResult with valid data."""
        result = VerificationResult(
            query="test query",
            results=[],
            success=True,
            metrics={"accuracy": 100.0}
        )

        assert result.query == "test query"
        assert result.success is True
        assert result.metrics["accuracy"] == 100.0


def test_end_to_end_verification_pipeline():
    """End-to-end test of the verification pipeline."""
    # This would test the full pipeline with mocked Qdrant client
    # For now, just a placeholder for the concept
    assert True  # Placeholder - would implement with proper mocking


if __name__ == "__main__":
    pytest.main([__file__])