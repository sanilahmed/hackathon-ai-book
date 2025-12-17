/**
 * Rate limiting service for the RAG Chatbot
 * Prevents excessive requests to the backend
 */

class RateLimiter {
  constructor() {
    this.requests = new Map();
  }

  /**
   * Check if a request is allowed based on rate limits
   * @param {string} key - The rate limit key
   * @returns {boolean} Whether the request is allowed
   */
  isAllowed(key) {
    const now = Date.now();
    const limit = this.getLimitInfo(key);

    // Clean up old requests
    if (limit.requests) {
      limit.requests = limit.requests.filter(time => now - time < limit.window);
    } else {
      limit.requests = [];
    }

    if (limit.requests.length >= limit.maxRequests) {
      return false;
    }

    limit.requests.push(now);
    return true;
  }

  /**
   * Get rate limit information for a key
   * @param {string} key - The rate limit key
   * @returns {Object} Rate limit information
   */
  getLimitInfo(key) {
    // Default values
    const limits = {
      'chatbot-question': {
        maxRequests: 5,    // 5 requests
        window: 60000,     // per 1 minute (60000 ms)
        requests: []
      }
    };

    return limits[key] || limits['chatbot-question'];
  }
}

export const defaultRateLimiter = new RateLimiter();