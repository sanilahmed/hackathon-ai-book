/**
 * API service for the RAG Chatbot
 * Handles communication with the backend API
 */
import { getBackendUrl } from '../config';

/**
 * Submit a question to the RAG backend
 * @param {string} question - The question to submit
 * @returns {Promise<Object>} The response from the backend
 */
export const submitQuestion = async (question) => {
  try {
    const backendUrl = getBackendUrl();
    const response = await fetch(`${backendUrl}/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: question,
        context_window: 5,
        include_sources: true,
        temperature: 0.1
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API request error:', error);
    throw error;
  }
};