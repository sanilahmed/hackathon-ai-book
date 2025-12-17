"""
Configuration module for the RAG Retrieval Verification system.

This module handles environment variable loading and configuration validation.
"""
import os
from typing import Optional
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class that manages all environment variables and settings
    for the RAG retrieval verification system.
    """

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'rag_embedding')
        self.default_top_k = int(os.getenv('DEFAULT_TOP_K', '5'))
        self.min_similarity = float(os.getenv('MIN_SIMILARITY', '0.5'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def validate(self) -> bool:
        """
        Validate that all required configuration values are present and valid.

        Returns:
            True if all required configuration is valid, False otherwise
        """
        errors = []

        if not self.qdrant_url:
            errors.append("QDRANT_URL environment variable not set")
        else:
            # Validate URL format
            if not self.qdrant_url.startswith(('http://', 'https://')):
                errors.append("QDRANT_URL must start with http:// or https://")

        if not self.qdrant_api_key:
            errors.append("QDRANT_API_KEY environment variable not set")
        elif len(self.qdrant_api_key) < 10:  # Basic check for API key length
            errors.append("QDRANT_API_KEY appears to be invalid (too short)")

        if not self.collection_name:
            errors.append("QDRANT_COLLECTION_NAME environment variable not set")
        elif not self._is_valid_collection_name(self.collection_name):
            errors.append("QDRANT_COLLECTION_NAME contains invalid characters")

        if self.default_top_k <= 0:
            errors.append("DEFAULT_TOP_K must be a positive integer")
        elif self.default_top_k > 100:  # Reasonable upper limit
            errors.append("DEFAULT_TOP_K should not exceed 100 for performance")

        if self.min_similarity < 0.0 or self.min_similarity > 1.0:
            errors.append("MIN_SIMILARITY must be between 0.0 and 1.0")

        if errors:
            for error in errors:
                logging.error(error)
            return False

        return True

    def _is_valid_collection_name(self, name: str) -> bool:
        """
        Validate collection name for allowed characters.

        Args:
            name: Collection name to validate

        Returns:
            True if name is valid, False otherwise
        """
        import re
        # Allow alphanumeric characters, hyphens, underscores, and dots
        pattern = r'^[a-zA-Z0-9._-]+$'
        return bool(re.match(pattern, name))

    def get_qdrant_config(self) -> dict:
        """
        Get Qdrant-specific configuration as a dictionary.

        Returns:
            Dictionary containing Qdrant configuration
        """
        return {
            'url': self.qdrant_url,
            'api_key': self.qdrant_api_key,
            'collection_name': self.collection_name
        }

    def get_verification_config(self) -> dict:
        """
        Get verification-specific configuration as a dictionary.

        Returns:
            Dictionary containing verification configuration
        """
        return {
            'top_k': self.default_top_k,
            'min_similarity': self.min_similarity
        }


# Global configuration instance
config = Config()


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance with loaded environment variables
    """
    return config


def validate_config() -> bool:
    """
    Validate the global configuration.

    Returns:
        True if configuration is valid, False otherwise
    """
    return config.validate()


def get_qdrant_config() -> dict:
    """
    Get Qdrant-specific configuration.

    Returns:
        Dictionary containing Qdrant configuration
    """
    return config.get_qdrant_config()


def get_verification_config() -> dict:
    """
    Get verification-specific configuration.

    Returns:
        Dictionary containing verification configuration
    """
    return config.get_verification_config()