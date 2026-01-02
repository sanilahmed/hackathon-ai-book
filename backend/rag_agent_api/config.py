"""
Configuration module for the RAG Agent and API Layer system.

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
    for the RAG agent and API system.
    """

    def __init__(self):
        """Initialize configuration by loading environment variables."""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.cohere_api_key = os.getenv('COHERE_API_KEY')
        self.openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        self.qdrant_url = os.getenv('QDRANT_URL')
        self.qdrant_api_key = os.getenv('QDRANT_API_KEY')
        self.qdrant_collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'rag_embedding')
        self.default_context_window = int(os.getenv('DEFAULT_CONTEXT_WINDOW', '5'))
        self.default_temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.1'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')

    def validate(self) -> bool:
        """
        Validate that all required configuration values are present and valid.

        Returns:
            True if all required configuration is valid, False otherwise
        """
        errors = []

        if not self.openrouter_api_key:
            errors.append("OPENROUTER_API_KEY environment variable not set")

        if not self.cohere_api_key:
            errors.append("COHERE_API_KEY environment variable not set")

        if not self.qdrant_url:
            errors.append("QDRANT_URL environment variable not set")

        if not self.qdrant_api_key:
            errors.append("QDRANT_API_KEY environment variable not set")

        if not self.qdrant_collection_name:
            errors.append("QDRANT_COLLECTION_NAME environment variable not set")

        if self.default_context_window <= 0 or self.default_context_window > 20:
            errors.append("DEFAULT_CONTEXT_WINDOW must be between 1 and 20")

        if self.default_temperature < 0.0 or self.default_temperature > 1.0:
            errors.append("DEFAULT_TEMPERATURE must be between 0.0 and 1.0")

        if errors:
            for error in errors:
                logging.error(error)
            return False

        return True

    def get_openai_config(self) -> dict:
        """
        Get OpenAI-specific configuration as a dictionary.

        Returns:
            Dictionary containing OpenAI configuration
        """
        return {
            'api_key': self.openai_api_key
        }

    def get_qdrant_config(self) -> dict:
        """
        Get Qdrant-specific configuration as a dictionary.

        Returns:
            Dictionary containing Qdrant configuration
        """
        return {
            'url': self.qdrant_url,
            'api_key': self.qdrant_api_key,
            'collection_name': self.qdrant_collection_name
        }

    def get_agent_config(self) -> dict:
        """
        Get agent-specific configuration as a dictionary.

        Returns:
            Dictionary containing agent configuration
        """
        return {
            'context_window': self.default_context_window,
            'temperature': self.default_temperature
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


def get_openai_config() -> dict:
    """
    Get OpenAI-specific configuration.

    Returns:
        Dictionary containing OpenAI configuration
    """
    return config.get_openai_config()


def get_qdrant_config() -> dict:
    """
    Get Qdrant-specific configuration.

    Returns:
        Dictionary containing Qdrant configuration
    """
    return config.get_qdrant_config()


def get_agent_config() -> dict:
    """
    Get agent-specific configuration.

    Returns:
        Dictionary containing agent configuration
    """
    return config.get_agent_config()