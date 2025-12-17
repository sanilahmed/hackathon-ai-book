/**
 * Health check service for the RAG Chatbot backend
 * Checks the status of the backend API
 */
import { getBackendUrl } from '../config';

/**
 * Check the health status of the backend
 * @returns {Promise<string>} Health status ('healthy' or 'unhealthy')
 */
export const getBackendStatus = async () => {
  try {
    const backendUrl = getBackendUrl();
    const response = await fetch(`${backendUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 5000, // 5 seconds timeout
    });

    if (response.ok) {
      const data = await response.json();
      return data.status === 'healthy' ? 'healthy' : 'unhealthy';
    } else {
      return 'unhealthy';
    }
  } catch (error) {
    console.error('Health check error:', error);
    return 'unhealthy';
  }
};