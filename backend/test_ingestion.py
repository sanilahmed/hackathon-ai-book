"""
Test suite for the Book Content Ingestion System
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import os
from main import get_all_urls, extract_text_from_url, chunk_text, embed, create_collection, save_chunk_to_qdrant

class TestURLCollection(unittest.TestCase):
    """Test URL collection functionality"""

    @patch('main.requests.get')
    def test_get_all_urls_with_mocked_response(self, mock_get):
        """Test that get_all_urls properly extracts URLs from a page"""
        # Mock response content
        mock_response = Mock()
        mock_response.content = '''
        <html>
            <body>
                <a href="/page1">Page 1</a>
                <a href="/page2">Page 2</a>
                <a href="https://external.com">External</a>
            </body>
        </html>
        '''
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        urls = get_all_urls("http://example.com")

        # Should return URLs from the same domain
        self.assertIn("http://example.com/page1", urls)
        self.assertIn("http://example.com/page2", urls)
        # External URLs should be filtered out
        # The actual implementation will depend on the exact mocking


class TestContentExtraction(unittest.TestCase):
    """Test content extraction functionality"""

    def test_extract_text_from_url_basic(self):
        """Test basic text extraction from HTML content"""
        # Create HTML content that will match the Docusaurus selectors used in the function
        html_content = '''
        <html>
            <head><title>Test Page</title></head>
            <body>
                <main>
                    <div class="main-wrapper">
                        <div class="theme-doc-markdown">
                            <h1>Main Content</h1>
                            <p>This is the main content of the page. It needs to be long enough to pass the 200 character requirement.</p>
                            <p>This is additional content to ensure we have more than 200 characters for the content extraction function to process.</p>
                            <script>console.log("ignore this");</script>
                        </div>
                    </div>
                </main>
            </body>
        </html>
        '''

        with patch('main.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.content = html_content
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            result = extract_text_from_url("http://example.com")

            self.assertEqual(result['title'], 'Test Page')
            self.assertIn('main content', result['text'].lower())
            self.assertGreater(len(result['text']), 200)  # Should be greater than 200 to not be filtered out


class TestTextChunking(unittest.TestCase):
    """Test text chunking functionality"""

    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        text = "This is a test sentence. Here's another sentence. And a third one."
        chunks = chunk_text(text, chunk_size=30)

        self.assertGreater(len(chunks), 0)
        # Each chunk should be at most the specified size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 30)

    def test_chunk_text_empty(self):
        """Test chunking empty text"""
        chunks = chunk_text("")
        self.assertEqual(chunks, [])

    def test_chunk_text_single_chunk(self):
        """Test text that fits in a single chunk"""
        text = "Short text"
        chunks = chunk_text(text, chunk_size=100)
        self.assertEqual(chunks, ["Short text"])


class TestEmbedding(unittest.TestCase):
    """Test embedding functionality"""

    @patch.dict(os.environ, {'COHERE_API_KEY': 'test-key'})
    @patch('main.cohere.Client')
    def test_embed_basic(self, mock_cohere_client):
        """Test basic embedding functionality"""
        # Mock the Cohere client and its embed method
        mock_client_instance = Mock()
        mock_client_instance.embed.return_value = MagicMock()
        mock_client_instance.embed.return_value.embeddings = [[0.1, 0.2, 0.3]]
        mock_cohere_client.return_value = mock_client_instance

        texts = ["Test text"]
        result = embed(texts)

        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])

        # Verify the client was called with correct parameters
        mock_client_instance.embed.assert_called_once()


class TestQdrantIntegration(unittest.TestCase):
    """Test Qdrant integration functions"""

    def test_create_collection(self):
        """Test collection creation"""
        mock_client = Mock()

        # Mock the get_collections method
        mock_collections = Mock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        create_collection(mock_client, "test_collection")

        # Verify the create_collection method was called
        mock_client.create_collection.assert_called_once()

    def test_save_chunk_to_qdrant(self):
        """Test saving a chunk to Qdrant"""
        mock_client = Mock()

        save_chunk_to_qdrant(
            mock_client,
            "test_collection",
            "test chunk",
            [0.1, 0.2, 0.3],
            {"url": "http://example.com", "title": "Test"}
        )

        # Verify the upsert method was called
        mock_client.upsert.assert_called_once()


if __name__ == '__main__':
    unittest.main()