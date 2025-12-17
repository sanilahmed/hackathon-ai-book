"""
Main module for the RAG Retrieval Verification system.

This module provides the main entry point for running the verification pipeline.
"""
import argparse
import sys
import time
import logging
from typing import List, Dict, Any, Optional
from .qdrant_client import QdrantVerificationClient, verify_embedding_retrieval
from .validators import validate_metadata_consistency, validate_retrieved_chunks, validate_similarity_scores, validate_metadata_accuracy
from .reporters import generate_verification_report, print_verification_summary, save_verification_report
from .sample_queries import get_sample_queries, get_validation_queries
from .config import validate_config, get_config, get_verification_config
from .logging_config import setup_logging

logger = logging.getLogger(__name__)


def run_verification_pipeline(queries: Optional[List[str]] = None,
                             collection_name: str = "rag_embedding",
                             top_k: int = 5,
                             min_similarity: float = 0.7) -> Dict[str, Any]:
    """
    Execute the complete verification pipeline with multiple queries and validation checks.

    Args:
        queries: List of query strings to test (uses default sample queries if None)
        collection_name: Name of the Qdrant collection to query
        top_k: Number of results to return per query
        min_similarity: Minimum similarity threshold for validation

    Returns:
        Dictionary containing comprehensive verification results
    """
    start_time = time.time()
    logger.info(f"Starting verification pipeline for collection: {collection_name}")

    # Validate configuration first
    if not validate_config():
        raise ValueError("Invalid configuration - please check environment variables")

    # Use default sample queries if none provided
    if queries is None:
        queries = get_sample_queries()[:5]  # Use first 5 sample queries for default run
        logger.info(f"Using default sample queries: {len(queries)} queries")

    # Initialize Qdrant client
    client = QdrantVerificationClient(collection_name=collection_name)

    # Verify collection exists
    if not client.validate_collection_exists():
        raise ValueError(f"Collection '{collection_name}' does not exist in Qdrant")

    logger.info(f"Found collection '{collection_name}', starting verification...")

    # Initialize results tracking
    all_results = []
    all_reports = []
    total_errors = []
    verification_details = []

    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        try:
            # Query Qdrant for chunks
            query_results = client.query_qdrant_for_chunks(query, top_k)

            # Perform validations
            metadata_validation = validate_metadata_consistency(query_results)
            chunk_validation = validate_retrieved_chunks(query, query_results)
            similarity_validation = validate_similarity_scores(query_results, min_similarity)
            metadata_accuracy = validate_metadata_accuracy(query_results)

            # Generate individual report for this query
            report = generate_verification_report(
                query=query,
                results=query_results,
                config={'top_k': top_k, 'min_similarity': min_similarity, 'collection_name': collection_name}
            )

            # Track verification details for this query
            query_detail = {
                'query_index': i,
                'query': query,
                'num_results': len(query_results),
                'metadata_accuracy': metadata_accuracy,
                'similarity_compliance': similarity_validation['threshold_compliance'],
                'status': report['status'],
                'errors': metadata_validation['errors'] + similarity_validation['errors']
            }

            all_results.extend(query_results)
            all_reports.append(report)
            verification_details.append(query_detail)

            # Track errors
            total_errors.extend([
                f"Query '{query}': {error}"
                for error in metadata_validation['errors'] + similarity_validation['errors']
            ])

            logger.info(f"Query {i+1} completed: {len(query_results)} results, status: {report['status']}")

        except Exception as e:
            error_msg = f"Error processing query '{query}': {str(e)}"
            logger.error(error_msg)
            total_errors.append(error_msg)
            verification_details.append({
                'query_index': i,
                'query': query,
                'num_results': 0,
                'metadata_accuracy': 0.0,
                'similarity_compliance': 0.0,
                'status': 'FAILURE',
                'errors': [error_msg]
            })

    # Calculate final metrics
    total_time = time.time() - start_time
    total_queries = len(queries)
    successful_queries = len([vd for vd in verification_details if vd['status'] == 'SUCCESS'])
    failed_queries = len([vd for vd in verification_details if vd['status'] == 'FAILURE'])

    # Calculate aggregate metrics
    if verification_details:
        avg_metadata_accuracy = sum(vd['metadata_accuracy'] for vd in verification_details) / len(verification_details)
        avg_similarity_compliance = sum(vd['similarity_compliance'] for vd in verification_details) / len(verification_details)
    else:
        avg_metadata_accuracy = 0.0
        avg_similarity_compliance = 0.0

    # Determine overall status
    if failed_queries == 0 and successful_queries > 0:
        overall_status = "SUCCESS"
    elif failed_queries == total_queries:
        overall_status = "FAILURE"
    else:
        overall_status = "PARTIAL_SUCCESS"

    # Create final verification result
    verification_result = {
        'status': overall_status,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'execution_time_seconds': round(total_time, 2),
        'total_queries': total_queries,
        'successful_queries': successful_queries,
        'failed_queries': failed_queries,
        'aggregate_metrics': {
            'avg_metadata_accuracy': round(avg_metadata_accuracy, 2),
            'avg_similarity_compliance': round(avg_similarity_compliance, 2),
            'total_results': len(all_results),
            'total_errors': len(total_errors)
        },
        'queries': verification_details,
        'errors': total_errors,
        'reports': all_reports,
        'summary': {
            'success_rate': round((successful_queries / total_queries * 100) if total_queries > 0 else 0, 2),
            'error_rate': round((failed_queries / total_queries * 100) if total_queries > 0 else 0, 2),
            'performance': f"{round(total_time, 2)}s for {total_queries} queries"
        }
    }

    logger.info(f"Verification pipeline completed: {overall_status}, {total_time:.2f}s, {successful_queries}/{total_queries} successful")
    return verification_result


def run_idempotency_check(queries: List[str],
                         collection_name: str = "rag_embedding",
                         runs: int = 2) -> Dict[str, Any]:
    """
    Run multiple verification runs to check for idempotency and consistency.

    Args:
        queries: List of query strings to test
        collection_name: Name of the Qdrant collection to query
        runs: Number of times to run the verification (default: 2)

    Returns:
        Dictionary containing idempotency check results
    """
    logger.info(f"Running idempotency check with {runs} runs")

    results = []
    for run_num in range(1, runs + 1):
        logger.info(f"Running verification #{run_num}")
        result = run_verification_pipeline(queries, collection_name)
        results.append(result)

        # Check for consistency between runs
        if run_num > 1:
            prev_result = results[run_num - 2]
            current_result = results[run_num - 1]

            # Compare key metrics for consistency
            metrics_match = (
                prev_result['aggregate_metrics']['avg_metadata_accuracy'] == current_result['aggregate_metrics']['avg_metadata_accuracy'] and
                prev_result['aggregate_metrics']['avg_similarity_compliance'] == current_result['aggregate_metrics']['avg_similarity_compliance'] and
                prev_result['total_queries'] == current_result['total_queries']
            )

            if not metrics_match:
                logger.warning(f"Idempotency check: Metrics differ between run {run_num-1} and run {run_num}")
            else:
                logger.info(f"Idempotency check: Run {run_num} matches run {run_num-1}")

    # Compile idempotency report
    idempotency_result = {
        'runs_executed': runs,
        'results': results,
        'consistent': all(
            r['aggregate_metrics']['avg_metadata_accuracy'] == results[0]['aggregate_metrics']['avg_metadata_accuracy']
            for r in results
        ),
        'final_status': results[-1]['status'] if results else 'UNKNOWN'
    }

    logger.info(f"Idempotency check completed: {'PASSED' if idempotency_result['consistent'] else 'FAILED'}")
    return idempotency_result


def main():
    """
    Main entry point for the RAG retrieval verification system.
    """
    parser = argparse.ArgumentParser(description='RAG Retrieval Verification System')
    parser.add_argument('--query', '-q', type=str, help='Single query to test')
    parser.add_argument('--queries-file', type=str, help='File containing queries to test (one per line)')
    parser.add_argument('--collection', '-c', type=str, default='rag_embedding', help='Qdrant collection name (default: rag_embedding)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of results to return per query (default: 5)')
    parser.add_argument('--min-similarity', type=float, default=0.7, help='Minimum similarity threshold (default: 0.7)')
    parser.add_argument('--idempotency-check', action='store_true', help='Run idempotency check with multiple runs')
    parser.add_argument('--runs', type=int, default=2, help='Number of runs for idempotency check (default: 2)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output', '-o', type=str, help='Output file for detailed results (JSON format)')

    args = parser.parse_args()

    # Setup logging based on verbosity
    log_level = 'DEBUG' if args.verbose else 'INFO'
    setup_logging(log_level)

    logger.info("Starting RAG Retrieval Verification System")

    try:
        # Determine queries to use
        queries = []

        if args.queries_file:
            with open(args.queries_file, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
        elif args.query:
            queries = [args.query]
        else:
            # Use sample validation queries
            validation_queries_data = get_validation_queries(args.min_similarity)
            queries = [vq['query'] for vq in validation_queries_data]
            logger.info(f"Using {len(queries)} validation sample queries")

        logger.info(f"Running verification with {len(queries)} queries on collection '{args.collection}'")

        # Run verification pipeline
        if args.idempotency_check:
            result = run_idempotency_check(
                queries=queries,
                collection_name=args.collection,
                runs=args.runs
            )
            # For idempotency check, we'll use the final result for reporting
            final_result = result['results'][-1] if result['results'] else {}
        else:
            result = run_verification_pipeline(
                queries=queries,
                collection_name=args.collection,
                top_k=args.top_k,
                min_similarity=args.min_similarity
            )
            final_result = result

        # Print summary
        print(f"\n{'='*60}")
        print("RAG RETRIEVAL VERIFICATION RESULTS")
        print(f"{'='*60}")
        print(f"Status: {final_result.get('status', 'UNKNOWN')}")
        print(f"Total Queries: {final_result.get('total_queries', 0)}")
        print(f"Successful: {final_result.get('successful_queries', 0)}")
        print(f"Failed: {final_result.get('failed_queries', 0)}")
        print(f"Success Rate: {final_result.get('summary', {}).get('success_rate', 0)}%")
        print(f"Execution Time: {final_result.get('execution_time_seconds', 0)}s")
        print(f"Average Metadata Accuracy: {final_result.get('aggregate_metrics', {}).get('avg_metadata_accuracy', 0)}%")
        print(f"Average Similarity Compliance: {final_result.get('aggregate_metrics', {}).get('avg_similarity_compliance', 0)}%")

        if final_result.get('errors'):
            print(f"\nErrors encountered: {len(final_result['errors'])}")
            for error in final_result['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(final_result['errors']) > 5:
                print(f"  ... and {len(final_result['errors']) - 5} more errors")

        print(f"{'='*60}")

        # Save detailed results if output file specified
        if args.output:
            import json
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Detailed results saved to: {args.output}")

        # Exit with appropriate code based on status
        if final_result.get('status') == 'FAILURE':
            sys.exit(1)
        elif final_result.get('status') == 'PARTIAL_SUCCESS':
            sys.exit(2)  # Different exit code for partial success
        else:
            sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Verification interrupted by user")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()