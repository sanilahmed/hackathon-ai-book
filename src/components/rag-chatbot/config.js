/**
 * Configuration for the RAG Chatbot frontend
 * Manages environment variables and default settings
 */

// Default configuration values (browser-safe approach)
const DEFAULT_CONFIG = {
  // Backend API configuration
  BACKEND_URL: 'http://localhost:8000', // Default backend URL
  API_TIMEOUT: 30000, // 30 seconds
  MAX_QUESTION_LENGTH: 1000,
  MAX_RESPONSE_LENGTH: 5000,

  // Chatbot UI configuration
  DEFAULT_POSITION: 'bottom-right',
  DEFAULT_THEME: 'light',
  MAX_HISTORY_ITEMS: 10,
  ENABLE_HISTORY: true,
  ENABLE_RATE_LIMITING: true,

  // Rate limiting configuration
  RATE_LIMIT_REQUESTS: 5,
  RATE_LIMIT_WINDOW: 60000, // 1 minute in ms

  // Response handling
  RESPONSE_TIMEOUT: 10000, // 10 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second

  // Security
  ENABLE_INPUT_SANITIZATION: true,
  ENABLE_OUTPUT_SANITIZATION: true,

  // Feature flags
  ENABLE_CHATBOT: true,
  ENABLE_CONVERSATION_HISTORY: true,
  ENABLE_SOURCE_ATTRIBUTION: true,

  // Accessibility
  ENABLE_KEYBOARD_NAVIGATION: true,
  ENABLE_SCREEN_READER_SUPPORT: true,
};

// Validation rules for configuration
const CONFIG_VALIDATION = {
  BACKEND_URL: (value) => {
    try {
      new URL(value);
      return true;
    } catch {
      return false;
    }
  },
  API_TIMEOUT: (value) => typeof value === 'number' && value > 0,
  MAX_QUESTION_LENGTH: (value) => typeof value === 'number' && value > 0,
  MAX_HISTORY_ITEMS: (value) => typeof value === 'number' && value > 0,
  RATE_LIMIT_REQUESTS: (value) => typeof value === 'number' && value > 0,
  RATE_LIMIT_WINDOW: (value) => typeof value === 'number' && value > 0,
  RESPONSE_TIMEOUT: (value) => typeof value === 'number' && value > 0,
  RETRY_ATTEMPTS: (value) => typeof value === 'number' && value >= 0,
  RETRY_DELAY: (value) => typeof value === 'number' && value >= 0,
};

/**
 * Get configuration value with validation
 * @param {string} key - Configuration key
 * @param {*} defaultValue - Default value if not set
 * @returns {*} Configuration value
 */
const getConfigValue = (key, defaultValue = null) => {
  const value = DEFAULT_CONFIG[key];

  if (value === undefined) {
    return defaultValue;
  }

  // Validate the configuration value if validation rule exists
  if (CONFIG_VALIDATION[key] && !CONFIG_VALIDATION[key](value)) {
    console.warn(`Invalid configuration value for ${key}: ${value}`);
    return defaultValue;
  }

  return value;
};

/**
 * Get the complete configuration object
 * @returns {Object} Configuration object
 */
const getFullConfig = () => {
  const config = {};
  for (const [key] of Object.entries(DEFAULT_CONFIG)) {
    config[key] = getConfigValue(key);
  }
  return config;
};

/**
 * Check if the chatbot is enabled
 * @returns {boolean} Whether the chatbot is enabled
 */
const isChatbotEnabled = () => {
  return getConfigValue('ENABLE_CHATBOT', true); // Default to true for testing
};

/**
 * Get the backend API URL
 * @returns {string|null} Backend API URL or null if not available
 */
const getBackendUrl = () => {
  // Check if RAG_API_URL is available from Docusaurus config
  if (typeof window !== 'undefined' && window.RAG_API_URL !== undefined && window.RAG_API_URL !== 'null' &&
      window.RAG_API_URL !== 'https://your-backend-api-url.com' &&
      window.RAG_API_URL !== 'https://sanilahmed.github.io/hackathon-ai-book/') {
    // If it's the localhost default but we're on GitHub Pages, return null
    if (window.RAG_API_URL === 'http://localhost:8000' && window.location.hostname.includes('github.io')) {
      return null; // Disable chatbot backend calls on GitHub Pages with localhost URL
    }
    return window.RAG_API_URL;
  }

  // For GitHub Pages deployment without backend, return null
  if (typeof window !== 'undefined' && window.location.hostname.includes('github.io')) {
    return null; // Disable chatbot backend calls on GitHub Pages without backend
  }

  return getConfigValue('BACKEND_URL', 'http://localhost:8000');
};

/**
 * Check if backend API is available
 * @returns {boolean} Whether the backend API is accessible
 */
const isBackendAvailable = () => {
  const backendUrl = getBackendUrl();
  // Backend is available if URL is not null, not localhost, and not placeholder
  return backendUrl !== null &&
         backendUrl !== 'http://localhost:8000' &&
         !backendUrl.includes('your-backend-api-url.com') &&
         backendUrl !== 'null';
};

export {
  DEFAULT_CONFIG,
  CONFIG_VALIDATION,
  getConfigValue,
  getFullConfig,
  isChatbotEnabled,
  getBackendUrl,
  isBackendAvailable
};