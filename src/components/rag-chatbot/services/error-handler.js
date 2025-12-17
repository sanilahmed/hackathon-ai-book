/**
 * Error handling service for the RAG Chatbot
 * Handles and formats errors for display
 */

/**
 * Handle API errors
 * @param {Error} error - The error to handle
 * @param {string} operation - The operation that failed
 * @returns {Error} Formatted error
 */
export const handleApiError = (error, operation) => {
  console.error(`API error in ${operation}:`, error);

  if (error.name === 'TypeError' && error.message.includes('fetch')) {
    return new Error('Network error: Unable to connect to the server. Please check your connection.');
  }

  if (error.message.includes('404')) {
    return new Error('Service not found: The requested service is not available.');
  }

  if (error.message.includes('500')) {
    return new Error('Server error: The server encountered an error. Please try again later.');
  }

  return new Error(`Request failed: ${error.message}`);
};

/**
 * Handle rate limit errors
 * @returns {Error} Rate limit error
 */
export const handleRateLimitError = () => {
  return new Error('Rate limit exceeded: Please wait before submitting another question.');
};

/**
 * Format error for display to user
 * @param {Error} error - The error to format
 * @returns {string} Formatted error message
 */
export const formatErrorForDisplay = (error) => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};