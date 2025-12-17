"""
Book Content Ingestion System
Extracts content from Docusaurus-based book websites, chunks and embeds it using Cohere,
and stores embeddings in Qdrant Cloud for RAG applications.
"""
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import cohere
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_urls(base_url: str) -> List[str]:
    """
    Collect and validate all URLs from a Docusaurus-based book website.

    Args:
        base_url: The base URL of the Docusaurus book

    Returns:
        List of valid URLs found on the site
    """
    urls = set()
    visited = set()

    def extract_urls_from_page(url: str) -> List[str]:
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            found_urls = []

            # Find all links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(url, href)

                # Only include URLs from the same domain
                if urlparse(full_url).netloc == urlparse(base_url).netloc:
                    found_urls.append(full_url)

            return found_urls
        except Exception as e:
            logger.error(f"Error extracting URLs from {url}: {e}")
            return []

    # Start with the base URL
    to_visit = [base_url]

    while to_visit:
        current_url = to_visit.pop(0)

        if current_url in visited:
            continue

        visited.add(current_url)
        urls.add(current_url)

        # Extract additional URLs from the current page
        new_urls = extract_urls_from_page(current_url)

        # Filter for new URLs that haven't been visited
        for url in new_urls:
            if url not in visited and len(urls) < 100:  # Limit to prevent infinite crawling
                to_visit.append(url)

    return list(urls)

def extract_text_from_url(url: str) -> Dict[str, Any]:
    """
    Extract and clean textual content from a single URL, optimized for Docusaurus sites.

    Args:
        url: The URL to extract content from

    Returns:
        Dictionary containing the extracted text and metadata
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove unwanted elements
        for element in soup(['nav', 'footer', 'aside', 'header', 'script', 'style']):
            element.decompose()

        # Try to find main content area (Docusaurus specific selectors)
        content_selectors = [
            'main',  # Most common for Docusaurus
            '.main-wrapper',  # Common Docusaurus wrapper
            '.container',  # General container
            '.doc-content',  # Docusaurus documentation content
            '.theme-doc-markdown',  # Docusaurus theme content
            '.markdown',  # Markdown content
            'article',  # Article tag as fallback
        ]

        content_element = None
        for selector in content_selectors:
            content_element = soup.select_one(selector)
            if content_element:
                break

        # If no specific content area found, try to get content differently
        if not content_element:
            # Try to find content by common Docusaurus classes
            content_selectors_alt = [
                '[class*="docItem"]',
                '[class*="doc-content"]',
                '[class*="content"]',
                '[role="main"]',
                '.content',
            ]
            for selector in content_selectors_alt:
                content_element = soup.select_one(selector)
                if content_element:
                    break

        if content_element:
            # Extract text and clean it
            text = content_element.get_text(separator=' ')

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)

            # Get page title
            title = soup.find('title')
            title = title.get_text().strip() if title else urlparse(url).path.split('/')[-1] or 'Untitled'

            # Check if the extracted text is meaningful
            if len(text) < 200:
                logger.warning(f"Insufficient content extracted from {url} (only {len(text)} characters)")
                return {
                    'url': url,
                    'title': title,
                    'text': '',
                    'length': 0
                }

            return {
                'url': url,
                'title': title,
                'text': text,
                'length': len(text)
            }
        else:
            logger.warning(f"No content area found at {url}")
            return {
                'url': url,
                'title': urlparse(url).path.split('/')[-1] or 'Untitled',
                'text': '',
                'length': 0
            }

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error extracting text from {url}: {e}")
        return {
            'url': url,
            'title': urlparse(url).path.split('/')[-1] or 'Error',
            'text': '',
            'length': 0
        }
    except Exception as e:
        logger.error(f"Error extracting text from {url}: {e}")
        return {
            'url': url,
            'title': urlparse(url).path.split('/')[-1] or 'Error',
            'text': '',
            'length': 0
        }

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Chunk text into fixed-size segments suitable for embedding generation (~1000 tokens).

    Args:
        text: The text to chunk
        chunk_size: The maximum size of each chunk (in characters, approximating tokens)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        # Find the best split point (try to break at sentence or word boundary)
        end = start + chunk_size

        if end >= len(text):
            # Last chunk
            chunk = text[start:].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            break

        # Try to find a good breaking point
        # First, try to break at sentence boundary
        sentence_break = text.rfind('.', start, end)
        if sentence_break > start and sentence_break < end:
            chunks.append(text[start:sentence_break + 1].strip())
            start = sentence_break + 1
            continue

        # Next, try to break at paragraph boundary
        para_break = text.rfind('\n\n', start, end)
        if para_break > start and para_break < end:
            chunks.append(text[start:para_break].strip())
            start = para_break + 2  # Skip the \n\n
            continue

        # Otherwise, try to break at word boundary
        word_break = text.rfind(' ', start, end)
        if word_break > start and word_break < end:
            chunks.append(text[start:word_break].strip())
            start = word_break + 1
            continue

        # If no good breaking point found, just break at the limit
        chunks.append(text[start:end].strip())
        start = end

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk]

    return chunks

def embed(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Cohere with retry logic.

    Args:
        texts: List of text chunks to embed

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    import time
    import random

    cohere_api_key = os.getenv('COHERE_API_KEY')
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")

    co = cohere.Client(cohere_api_key)

    # Retry up to 3 times with exponential backoff
    for attempt in range(3):
        try:
            response = co.embed(
                texts=texts,
                model='embed-english-v3.0',  # Using a valid Cohere embedding model
                input_type='search_document'  # Specify the input type
            )

            return response.embeddings
        except Exception as e:
            if attempt == 2:  # Last attempt
                logger.error(f"Error generating embeddings after 3 attempts: {e}")
                raise
            else:
                wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                time.sleep(wait_time)

def create_collection(client: QdrantClient, collection_name: str = "rag_embedding"):
    """
    Create a collection in Qdrant for storing embeddings.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create
    """
    try:
        # Check if collection already exists
        collections = client.get_collections()
        existing_collections = [c.name for c in collections.collections]

        if collection_name in existing_collections:
            logger.info(f"Collection '{collection_name}' already exists")
            return

        # Create the collection
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1024,  # Cohere embeddings are typically 1024 dimensions
                distance=models.Distance.COSINE
            )
        )

        logger.info(f"Collection '{collection_name}' created successfully")
    except Exception as e:
        logger.error(f"Error creating collection: {e}")
        raise

def save_chunk_to_qdrant(client: QdrantClient, collection_name: str, chunk: str,
                        embedding: List[float], metadata: Dict[str, Any]):
    """
    Save a text chunk with its embedding to Qdrant.

    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to save to
        chunk: The text chunk
        embedding: The embedding vector
        metadata: Additional metadata to store with the chunk
    """
    try:
        # Generate a unique ID for this chunk
        import hashlib
        chunk_id = hashlib.md5(f"{metadata['url']}_{chunk[:50]}".encode()).hexdigest()

        # Upsert the point to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=chunk_id,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "url": metadata.get('url', ''),
                        "title": metadata.get('title', ''),
                        "source": metadata.get('source', 'unknown'),
                        "timestamp": time.time()
                    }
                )
            ]
        )

        logger.info(f"Saved chunk to Qdrant: {chunk_id[:8]}...")
    except Exception as e:
        logger.error(f"Error saving chunk to Qdrant: {e}")
        raise

def main():
    """
    Main function to execute the complete book content ingestion pipeline.
    """
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()

    # Get environment variables
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    cohere_api_key = os.getenv('COHERE_API_KEY')

    if not all([qdrant_url, qdrant_api_key, cohere_api_key]):
        raise ValueError("Missing required environment variables: QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY")

    # Target book URL
    book_url = "https://sanilahmed.github.io/hackathon-ai-book/"

    logger.info(f"Starting ingestion pipeline for: {book_url}")

    try:
        # Initialize clients
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=30  # 30 second timeout for requests
        )

        # Step 1: Get all URLs from the book
        logger.info("Step 1: Collecting all URLs from the book...")
        urls = get_all_urls(book_url)
        logger.info(f"Found {len(urls)} URLs")

        if not urls:
            logger.warning("No URLs found, exiting.")
            return

        # Step 2: Create Qdrant collection
        logger.info("Step 2: Creating Qdrant collection...")
        create_collection(qdrant_client, "rag_embedding")

        # Step 3: Process each URL
        logger.info("Step 3: Processing URLs...")
        processed_count = 0

        for i, url in enumerate(urls):
            logger.info(f"Processing URL {i+1}/{len(urls)}: {url}")

            # Extract text from URL
            content_data = extract_text_from_url(url)

            if not content_data['text']:
                logger.warning(f"No content extracted from {url}, skipping...")
                continue

            # Chunk the text into ~1000 token segments (approx 1000 chars)
            chunks = chunk_text(content_data['text'], chunk_size=1000)

            if not chunks:
                logger.warning(f"No chunks created from {url}, skipping...")
                continue

            # Process each chunk
            for j, chunk in enumerate(chunks):
                try:
                    # Embed the chunk
                    embeddings = embed([chunk])
                    embedding = embeddings[0]  # Get the first (and only) embedding

                    # Prepare metadata
                    metadata = {
                        'url': url,
                        'title': content_data['title'],
                        'source': 'book_ingestion',
                        'chunk_index': j,
                        'total_chunks': len(chunks)
                    }

                    # Save to Qdrant
                    save_chunk_to_qdrant(
                        qdrant_client,
                        "rag_embedding",
                        chunk,
                        embedding,
                        metadata
                    )

                    processed_count += 1

                    # Add a small delay to respect rate limits
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing chunk {j} from {url}: {e}")
                    continue

        logger.info(f"Ingestion pipeline completed! Processed {processed_count} chunks from {len(urls)} URLs.")

    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise

if __name__ == "__main__":
    main()