"""
Validation module for the RAG Retrieval Verification system.

This module provides functions to validate chunk accuracy and metadata consistency.
"""
from typing import List, Dict, Any, Tuple
from .models import RetrievedContentChunk, MetadataRecord
import logging

logger = logging.getLogger(__name__)


def validate_metadata_consistency(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate that metadata in retrieved results matches expected source content metadata.

    Args:
        results: Array of Retrieved Content Chunk objects from query

    Returns:
        Report with validation results including accuracy percentage and errors
    """
    if not results:
        logger.warning("No results to validate metadata consistency")
        return {
            'accuracy_percentage': 0.0,
            'errors': [],
            'details': [],
            'total_validated': 0,
            'total_errors': 0
        }

    total_chunks = len(results)
    errors = []
    details = []

    for i, result in enumerate(results):
        chunk_errors = []

        # Extract metadata from result
        metadata = result.get('metadata', {})
        payload = result.get('payload', {})

        # Validate URL consistency
        result_url = metadata.get('url', '')
        payload_url = payload.get('url', '')

        if result_url and payload_url and result_url != payload_url:
            chunk_errors.append(f"URL mismatch: result='{result_url}', payload='{payload_url}'")

        # Validate title consistency
        result_title = metadata.get('title', '')
        payload_title = payload.get('title', '')

        if result_title and payload_title and result_title != payload_title:
            chunk_errors.append(f"Title mismatch: result='{result_title}', payload='{payload_title}'")

        # Validate chunk index consistency
        result_chunk_idx = metadata.get('chunk_index', -1)
        payload_chunk_idx = payload.get('chunk_index', -1)

        if result_chunk_idx != payload_chunk_idx:
            chunk_errors.append(f"Chunk index mismatch: result={result_chunk_idx}, payload={payload_chunk_idx}")

        # Validate total chunks consistency
        result_total_chunks = metadata.get('total_chunks', -1)
        payload_total_chunks = payload.get('total_chunks', -1)

        if result_total_chunks != payload_total_chunks:
            chunk_errors.append(f"Total chunks mismatch: result={result_total_chunks}, payload={payload_total_chunks}")

        # Add to errors if any inconsistencies found
        if chunk_errors:
            errors.extend(chunk_errors)
            details.append({
                'chunk_index': i,
                'chunk_id': result.get('id', 'unknown'),
                'errors': chunk_errors
            })
        else:
            details.append({
                'chunk_index': i,
                'chunk_id': result.get('id', 'unknown'),
                'errors': []
            })

    total_errors = len(errors)
    accuracy_percentage = ((total_chunks - (total_errors / max(len(details), 1))) / total_chunks) * 100 if total_chunks > 0 else 0.0

    validation_report = {
        'accuracy_percentage': round(accuracy_percentage, 2),
        'errors': errors,
        'details': details,
        'total_validated': total_chunks,
        'total_errors': total_errors
    }

    logger.info(f"Metadata validation completed: {validation_report['accuracy_percentage']:.2f}% accuracy, {total_errors} errors found")
    return validation_report


def validate_retrieved_chunks(query: str, results: List[Dict[str, Any]],
                            expected_keywords: List[str] = None) -> Dict[str, Any]:
    """
    Validate chunk relevance based on query and expected keywords.

    Args:
        query: The original query that produced these results
        results: List of retrieved chunk results
        expected_keywords: Optional list of keywords that should appear in relevant chunks

    Returns:
        Dictionary with validation results
    """
    if not results:
        logger.warning("No results to validate chunk relevance")
        return {
            'relevance_score': 0.0,
            'relevant_chunks': 0,
            'total_chunks': 0,
            'keyword_matches': 0,
            'errors': []
        }

    total_chunks = len(results)
    relevant_chunks = 0
    keyword_matches = 0
    errors = []

    for result in results:
        content = result.get('content', '').lower()
        similarity_score = result.get('similarity_score', 0.0)

        # Count as relevant if similarity is above threshold
        if similarity_score >= 0.5:  # This could be configurable
            relevant_chunks += 1

        # Check for expected keywords if provided
        if expected_keywords:
            keyword_match_count = 0
            for keyword in expected_keywords:
                if keyword.lower() in content:
                    keyword_match_count += 1

            if keyword_match_count > 0:
                keyword_matches += 1

    relevance_score = (relevant_chunks / total_chunks) * 100 if total_chunks > 0 else 0.0
    keyword_score = (keyword_matches / total_chunks) * 100 if total_chunks > 0 else 0.0

    validation_result = {
        'relevance_score': round(relevance_score, 2),
        'relevant_chunks': relevant_chunks,
        'total_chunks': total_chunks,
        'keyword_matches': keyword_matches,
        'keyword_score': round(keyword_score, 2),
        'errors': errors
    }

    logger.info(f"Chunk validation completed: {validation_result['relevance_score']:.2f}% relevance, {relevant_chunks}/{total_chunks} relevant chunks")
    return validation_result


def validate_similarity_scores(results: List[Dict[str, Any]], min_threshold: float = 0.7) -> Dict[str, Any]:
    """
    Validate that similarity scores meet the minimum threshold requirements.

    Args:
        results: List of retrieved chunk results with similarity scores
        min_threshold: Minimum acceptable similarity score (default: 0.7)

    Returns:
        Dictionary with validation results
    """
    if not results:
        logger.warning("No results to validate similarity scores")
        return {
            'above_threshold': 0,
            'below_threshold': 0,
            'average_score': 0.0,
            'threshold_compliance': 0.0,
            'errors': []
        }

    total_chunks = len(results)
    above_threshold = 0
    below_threshold = 0
    total_score = 0.0
    errors = []

    for result in results:
        similarity_score = result.get('similarity_score', 0.0)
        total_score += similarity_score

        if similarity_score >= min_threshold:
            above_threshold += 1
        else:
            below_threshold += 1
            errors.append(f"Chunk with ID {result.get('id', 'unknown')} has low similarity: {similarity_score:.3f}")

    average_score = total_score / total_chunks if total_chunks > 0 else 0.0
    threshold_compliance = (above_threshold / total_chunks) * 100 if total_chunks > 0 else 0.0

    validation_result = {
        'above_threshold': above_threshold,
        'below_threshold': below_threshold,
        'average_score': round(average_score, 3),
        'threshold_compliance': round(threshold_compliance, 2),
        'min_threshold': min_threshold,
        'errors': errors
    }

    logger.info(f"Similarity validation completed: {validation_result['threshold_compliance']:.2f}% compliance with threshold {min_threshold}")
    return validation_result


def validate_no_results_case(query: str, results: List[Dict[str, Any]]) -> bool:
    """
    Validate behavior when no results are returned for a query.

    Args:
        query: The query that returned no results
        results: The empty results list

    Returns:
        True if the no-results case is handled appropriately
    """
    if len(results) == 0:
        logger.warning(f"Query '{query}' returned no results")
        # This is valid behavior - sometimes queries legitimately return no results
        return True
    else:
        # If there are results, this validation doesn't apply
        return True


def validate_metadata_accuracy(results: List[Dict[str, Any]], required_fields: List[str] = None) -> float:
    """
    Calculate metadata accuracy percentage based on required fields.

    Args:
        results: List of retrieved results to validate
        required_fields: List of required metadata fields (default: ['url', 'title', 'chunk_index'])

    Returns:
        Accuracy percentage as a float between 0.0 and 100.0
    """
    if not results:
        return 0.0

    if required_fields is None:
        required_fields = ['url', 'title', 'chunk_index']

    total_checks = len(results) * len(required_fields)
    if total_checks == 0:
        return 100.0

    valid_checks = 0

    for result in results:
        metadata = result.get('metadata', {})
        payload = result.get('payload', {})

        for field in required_fields:
            # Check if field exists in either metadata or payload
            meta_value = metadata.get(field)
            payload_value = payload.get(field)

            # Field is valid if it exists and is not empty/non-null
            if meta_value is not None and meta_value != "":
                valid_checks += 1
            elif payload_value is not None and payload_value != "":
                valid_checks += 1

    accuracy = (valid_checks / total_checks) * 100
    logger.info(f"Metadata accuracy: {accuracy:.2f}% ({valid_checks}/{total_checks} fields valid)")

    return round(accuracy, 2)