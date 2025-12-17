"""
RAG Agent and API Layer - __init__.py

Initialization module for the RAG Agent and API Layer system.
"""
__version__ = "1.0.0"
__author__ = "AI Engineer"
__license__ = "MIT"

# Import main components for easy access
from .main import app
from .config import Config, get_config, validate_config
from .agent import GeminiAgent
from .retrieval import QdrantRetriever

# Define what gets imported with "from rag_agent_api import *"
__all__ = [
    "app",
    "Config",
    "get_config",
    "validate_config",
    "GeminiAgent",
    "QdrantRetriever"
]