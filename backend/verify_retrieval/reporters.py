"""
Reporting module for the RAG Retrieval Verification system.

This module provides functions to generate verification reports and logs.
"""
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import VerificationResult, QueryRequest
from .validators import validate_metadata_consistency, validate_similarity_scores, validate_metadata_accuracy

logger = logging.getLogger(__name__)


class VerificationReporter:
    """
    A reporter class for generating verification reports and logs.
    """

    def __init__(self):
        """Initialize the verification reporter."""
        self.reports = []

    def generate_verification_report(self,
                                   query: str,
                                   results: List[Dict[str, Any]],
                                   config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive verification report based on query results.

        Args:
            query: The original query string
            results: List of retrieved results
            config: Configuration parameters used for the verification

        Returns:
            Dictionary containing the verification report
        """
        timestamp = datetime.now()

        # Perform validations
        metadata_validation = validate_metadata_consistency(results)
        similarity_validation = validate_similarity_scores(results, min_threshold=config.get('min_similarity', 0.7) if config else 0.7)
        metadata_accuracy = validate_metadata_accuracy(results)

        # Calculate overall metrics
        total_results = len(results)
        avg_similarity = sum(r.get('similarity_score', 0) for r in results) / total_results if total_results > 0 else 0

        report = {
            'timestamp': timestamp.isoformat(),
            'query': query,
            'config': config or {},
            'metrics': {
                'total_results': total_results,
                'average_similarity': round(avg_similarity, 3),
                'metadata_accuracy': metadata_accuracy,
                'threshold_compliance': similarity_validation['threshold_compliance'],
                'metadata_validation_score': metadata_validation['accuracy_percentage']
            },
            'validations': {
                'metadata': metadata_validation,
                'similarity': similarity_validation
            },
            'results_summary': [
                {
                    'id': r.get('id', 'unknown'),
                    'similarity_score': r.get('similarity_score', 0),
                    'content_preview': r.get('content', '')[:100] + '...' if len(r.get('content', '')) > 100 else r.get('content', '')
                } for r in results[:5]  # Just first 5 for summary
            ],
            'status': self._determine_status(metadata_accuracy, similarity_validation['threshold_compliance']),
            'recommendations': self._generate_recommendations(similarity_validation, metadata_validation)
        }

        self.reports.append(report)
        logger.info(f"Verification report generated for query: '{query[:50]}{'...' if len(query) > 50 else ''}'")

        return report

    def _determine_status(self, metadata_accuracy: float, threshold_compliance: float) -> str:
        """
        Determine the overall status based on validation scores.

        Args:
            metadata_accuracy: Metadata accuracy percentage
            threshold_compliance: Similarity threshold compliance percentage

        Returns:
            Status string (SUCCESS, PARTIAL_SUCCESS, WARNING, or FAILURE)
        """
        if metadata_accuracy >= 95.0 and threshold_compliance >= 90.0:
            return "SUCCESS"
        elif metadata_accuracy >= 80.0 and threshold_compliance >= 70.0:
            return "PARTIAL_SUCCESS"
        elif metadata_accuracy >= 50.0 or threshold_compliance >= 50.0:
            return "WARNING"
        else:
            return "FAILURE"

    def _generate_recommendations(self, similarity_validation: Dict[str, Any],
                                  metadata_validation: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on validation results.

        Args:
            similarity_validation: Results from similarity validation
            metadata_validation: Results from metadata validation

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if similarity_validation['below_threshold'] > 0:
            recommendations.append(f"Consider lowering the minimum similarity threshold or improving query formulation. {similarity_validation['below_threshold']} results were below threshold.")

        if metadata_validation['total_errors'] > 0:
            recommendations.append(f"Review metadata consistency. {metadata_validation['total_errors']} metadata inconsistencies were found.")

        if similarity_validation['average_score'] < 0.7:
            recommendations.append("Average similarity score is low. Consider reviewing embedding quality or query formulation.")

        return recommendations

    def save_report_to_file(self, report: Dict[str, Any], filepath: str) -> None:
        """
        Save a verification report to a JSON file.

        Args:
            report: The report dictionary to save
            filepath: Path to save the report file
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Report saved to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save report to {filepath}: {e}")
            raise

    def print_report_summary(self, report: Dict[str, Any]) -> None:
        """
        Print a formatted summary of the verification report.

        Args:
            report: The report dictionary to print
        """
        print("\n" + "="*60)
        print("RAG Retrieval Verification Results")
        print("="*60)
        print(f"Query: {report['query']}")
        print(f"Timestamp: {report['timestamp']}")
        print(f"Status: {report['status']}")
        print("-"*60)
        print(f"Total Results: {report['metrics']['total_results']}")
        print(f"Average Similarity: {report['metrics']['average_similarity']}")
        print(f"Metadata Accuracy: {report['metrics']['metadata_accuracy']}%")
        print(f"Threshold Compliance: {report['metrics']['threshold_compliance']}%")
        print(f"Metadata Validation Score: {report['metrics']['metadata_validation_score']}%")
        print("-"*60)

        if report['recommendations']:
            print("Recommendations:")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        print("="*60 + "\n")


def generate_verification_report(query: str,
                               results: List[Dict[str, Any]],
                               config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Generate a comprehensive verification report based on query results.

    Args:
        query: The original query string
        results: List of retrieved results
        config: Configuration parameters used for the verification

    Returns:
        Dictionary containing the verification report
    """
    reporter = VerificationReporter()
    return reporter.generate_verification_report(query, results, config)


def print_verification_summary(report: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the verification report.

    Args:
        report: The report dictionary to print
    """
    reporter = VerificationReporter()
    reporter.print_report_summary(report)


def save_verification_report(report: Dict[str, Any], filepath: str) -> None:
    """
    Save a verification report to a JSON file.

    Args:
        report: The report dictionary to save
        filepath: Path to save the report file
    """
    reporter = VerificationReporter()
    reporter.save_report_to_file(report, filepath)