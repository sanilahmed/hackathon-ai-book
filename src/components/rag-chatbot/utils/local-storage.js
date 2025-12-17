/**
 * Local storage utilities for the RAG Chatbot frontend
 * Manages conversation history persistence and other local data
 */

const STORAGE_PREFIX = 'rag-chatbot-';
const MAX_HISTORY_ITEMS = 50; // Maximum number of conversations to store

/**
 * Save conversation history to local storage
 * @param {Array} history - The conversation history to save
 * @param {string} userId - Optional user identifier
 * @returns {boolean} Whether the save was successful
 */
export const saveConversationHistory = (history, userId = null) => {
  try {
    if (!Array.isArray(history)) {
      console.warn('Invalid history provided to saveConversationHistory');
      return false;
    }

    // Limit history to prevent storage overflow
    const limitedHistory = history.slice(-MAX_HISTORY_ITEMS);

    const key = userId
      ? `${STORAGE_PREFIX}history-${userId}`
      : `${STORAGE_PREFIX}history`;

    const data = {
      history: limitedHistory,
      timestamp: new Date().toISOString(),
      version: '1.0'
    };

    localStorage.setItem(key, JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('Error saving conversation history:', error);
    return false;
  }
};

/**
 * Load conversation history from local storage
 * @param {string} userId - Optional user identifier
 * @returns {Array} The loaded conversation history, or empty array if none found
 */
export const loadConversationHistory = (userId = null) => {
  try {
    const key = userId
      ? `${STORAGE_PREFIX}history-${userId}`
      : `${STORAGE_PREFIX}history`;

    const stored = localStorage.getItem(key);
    if (!stored) {
      return [];
    }

    const data = JSON.parse(stored);

    // Validate data structure
    if (!data || !Array.isArray(data.history)) {
      console.warn('Invalid conversation history data found in storage');
      return [];
    }

    // Check if data is too old (older than 30 days)
    if (data.timestamp) {
      const storedDate = new Date(data.timestamp);
      const daysDiff = (Date.now() - storedDate.getTime()) / (1000 * 60 * 60 * 24);

      if (daysDiff > 30) {
        console.log('Conversation history is too old, clearing');
        localStorage.removeItem(key);
        return [];
      }
    }

    return data.history;
  } catch (error) {
    console.error('Error loading conversation history:', error);
    return [];
  }
};

/**
 * Clear conversation history from local storage
 * @param {string} userId - Optional user identifier
 * @returns {boolean} Whether the clear was successful
 */
export const clearConversationHistory = (userId = null) => {
  try {
    const key = userId
      ? `${STORAGE_PREFIX}history-${userId}`
      : `${STORAGE_PREFIX}history`;

    localStorage.removeItem(key);
    return true;
  } catch (error) {
    console.error('Error clearing conversation history:', error);
    return false;
  }
};

/**
 * Save chatbot settings to local storage
 * @param {Object} settings - The settings to save
 * @returns {boolean} Whether the save was successful
 */
export const saveSettings = (settings) => {
  try {
    if (!settings || typeof settings !== 'object') {
      console.warn('Invalid settings provided to saveSettings');
      return false;
    }

    const data = {
      settings,
      timestamp: new Date().toISOString(),
      version: '1.0'
    };

    localStorage.setItem(`${STORAGE_PREFIX}settings`, JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('Error saving settings:', error);
    return false;
  }
};

/**
 * Load chatbot settings from local storage
 * @returns {Object} The loaded settings, or default settings if none found
 */
export const loadSettings = () => {
  try {
    const stored = localStorage.getItem(`${STORAGE_PREFIX}settings`);
    if (!stored) {
      return {};
    }

    const data = JSON.parse(stored);

    // Validate data structure
    if (!data || typeof data.settings !== 'object') {
      console.warn('Invalid settings data found in storage');
      return {};
    }

    return data.settings;
  } catch (error) {
    console.error('Error loading settings:', error);
    return {};
  }
};

/**
 * Save backend status to local storage
 * @param {string} status - The backend status
 * @returns {boolean} Whether the save was successful
 */
export const saveBackendStatus = (status) => {
  try {
    const data = {
      status,
      timestamp: new Date().toISOString(),
      version: '1.0'
    };

    localStorage.setItem(`${STORAGE_PREFIX}backend-status`, JSON.stringify(data));
    return true;
  } catch (error) {
    console.error('Error saving backend status:', error);
    return false;
  }
};

/**
 * Load backend status from local storage
 * @returns {Object} The loaded status data with timestamp
 */
export const loadBackendStatus = () => {
  try {
    const stored = localStorage.getItem(`${STORAGE_PREFIX}backend-status`);
    if (!stored) {
      return { status: 'unknown', timestamp: null };
    }

    const data = JSON.parse(stored);

    // Validate data structure
    if (!data || typeof data.status !== 'string') {
      console.warn('Invalid backend status data found in storage');
      return { status: 'unknown', timestamp: null };
    }

    return {
      status: data.status,
      timestamp: data.timestamp
    };
  } catch (error) {
    console.error('Error loading backend status:', error);
    return { status: 'unknown', timestamp: null };
  }
};

/**
 * Clear all chatbot data from local storage
 * @returns {boolean} Whether the clear was successful
 */
export const clearAllData = () => {
  try {
    // Get all keys that start with our prefix
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(STORAGE_PREFIX)) {
        keysToRemove.push(key);
      }
    }

    // Remove all matching keys
    keysToRemove.forEach(key => localStorage.removeItem(key));

    return true;
  } catch (error) {
    console.error('Error clearing all chatbot data:', error);
    return false;
  }
};

/**
 * Get storage usage information
 * @returns {Object} Storage usage information
 */
export const getStorageUsage = () => {
  try {
    let totalSize = 0;
    const items = [];

    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(STORAGE_PREFIX)) {
        const value = localStorage.getItem(key);
        const size = new Blob([value]).size;

        items.push({
          key,
          size,
          sizeKB: Math.round(size / 1024 * 100) / 100
        });

        totalSize += size;
      }
    }

    return {
      items,
      totalSize,
      totalSizeKB: Math.round(totalSize / 1024 * 100) / 100,
      limit: navigator.storage && navigator.storage.estimate ? null : 'Limited by browser'
    };
  } catch (error) {
    console.error('Error getting storage usage:', error);
    return {
      items: [],
      totalSize: 0,
      totalSizeKB: 0
    };
  }
};

/**
 * Check if local storage is available and accessible
 * @returns {boolean} Whether local storage is available
 */
export const isLocalStorageAvailable = () => {
  try {
    const testKey = `${STORAGE_PREFIX}test`;
    localStorage.setItem(testKey, 'test');
    localStorage.removeItem(testKey);
    return true;
  } catch (error) {
    console.warn('Local storage not available:', error);
    return false;
  }
};