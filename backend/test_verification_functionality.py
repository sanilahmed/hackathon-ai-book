"""
Test script to validate the verification functionality without requiring actual Qdrant data.
This will test the validation logic and report generation components.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from verify_retrieval.validators import (
    validate_metadata_consistency,
    validate_retrieved_chunks,
    validate_similarity_scores,
    validate_metadata_accuracy
)
from verify_retrieval.reporters import generate_verification_report
from verify_retrieval.models import MetadataRecord, RetrievedContentChunk, QueryRequest, VerificationResult


def test_validation_functions():
    """Test the validation functions with sample data."""
    print("Testing validation functions...")

    # Test data simulating Qdrant results
    sample_results = [
        {
            'id': 'test_id_1',
            'content': 'This is a sample content chunk about transformer architecture.',
            'similarity_score': 0.85,
            'metadata': {
                'url': 'https://example.com/transformers',
                'title': 'Transformer Models',
                'chunk_index': 1,
                'total_chunks': 3
            },
            'payload': {
                'url': 'https://example.com/transformers',
                'title': 'Transformer Models',
                'chunk_index': 1,
                'total_chunks': 3
            }
        },
        {
            'id': 'test_id_2',
            'content': 'Another content chunk about neural networks.',
            'similarity_score': 0.72,
            'metadata': {
                'url': 'https://example.com/neural-networks',
                'title': 'Neural Networks',
                'chunk_index': 2,
                'total_chunks': 3
            },
            'payload': {
                'url': 'https://example.com/neural-networks',
                'title': 'Neural Networks',
                'chunk_index': 2,
                'total_chunks': 3
            }
        },
        {
            'id': 'test_id_3',
            'content': 'Content about semantic search techniques.',
            'similarity_score': 0.65,
            'metadata': {
                'url': 'https://example.com/semantic-search',
                'title': 'Semantic Search',
                'chunk_index': 3,
                'total_chunks': 3
            },
            'payload': {
                'url': 'https://example.com/semantic-search',
                'title': 'Semantic Search',
                'chunk_index': 3,
                'total_chunks': 3
            }
        }
    ]

    # Test metadata consistency validation
    metadata_validation = validate_metadata_consistency(sample_results)
    print(f"Metadata validation: {metadata_validation}")

    # Test retrieved chunks validation
    chunk_validation = validate_retrieved_chunks("transformer architecture", sample_results)
    print(f"Chunk validation: {chunk_validation}")

    # Test similarity scores validation
    similarity_validation = validate_similarity_scores(sample_results, min_threshold=0.7)
    print(f"Similarity validation: {similarity_validation}")

    # Test metadata accuracy
    metadata_accuracy = validate_metadata_accuracy(sample_results)
    print(f"Metadata accuracy: {metadata_accuracy}%")

    # Check if the validation was successful based on actual return values
    all_metadata_valid = len(metadata_validation['errors']) == 0
    chunks_have_content = chunk_validation['total_chunks'] > 0
    avg_similarity_good = similarity_validation['average_score'] > 0.7
    metadata_accuracy_good = metadata_accuracy >= 90.0  # Using 90% as a reasonable threshold

    return all([
        all_metadata_valid,
        chunks_have_content,
        avg_similarity_good,
        metadata_accuracy_good
    ])


def test_report_generation():
    """Test the report generation functionality."""
    print("\nTesting report generation...")

    sample_results = [
        {
            'id': 'test_id_1',
            'content': 'This is a sample content chunk about transformer architecture.',
            'similarity_score': 0.85,
            'metadata': {
                'url': 'https://example.com/transformers',
                'title': 'Transformer Models',
                'chunk_index': 1,
                'total_chunks': 2
            },
            'payload': {
                'url': 'https://example.com/transformers',
                'title': 'Transformer Models',
                'chunk_index': 1,
                'total_chunks': 2
            }
        },
        {
            'id': 'test_id_2',
            'content': 'Another content chunk about neural networks.',
            'similarity_score': 0.72,
            'metadata': {
                'url': 'https://example.com/neural-networks',
                'title': 'Neural Networks',
                'chunk_index': 2,
                'total_chunks': 2
            },
            'payload': {
                'url': 'https://example.com/neural-networks',
                'title': 'Neural Networks',
                'chunk_index': 2,
                'total_chunks': 2
            }
        }
    ]

    config = {
        'top_k': 5,
        'min_similarity': 0.7,
        'collection_name': 'rag_embedding'
    }

    report = generate_verification_report(
        query="transformer architecture",
        results=sample_results,
        config=config
    )

    print(f"Report generated successfully: {report['status']}")
    print(f"Metrics: {report['metrics']}")

    return report['status'] in ['SUCCESS', 'PARTIAL_SUCCESS']


def main():
    """Main test function."""
    print("Running verification system tests...\n")

    # Test validation functions
    validation_success = test_validation_functions()
    print(f"Validation functions test: {'PASSED' if validation_success else 'FAILED'}\n")

    # Test report generation
    report_success = test_report_generation()
    print(f"Report generation test: {'PASSED' if report_success else 'FAILED'}\n")

    overall_success = validation_success and report_success
    print(f"Overall test result: {'PASSED' if overall_success else 'FAILED'}")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)