/**
 * Validation utilities for the RAG Chatbot
 * Validates user input and other data
 */

/**
 * Validate a question before submitting
 * @param {string} question - The question to validate
 * @returns {Object} Validation result with isValid boolean and message
 */
export const validateQuestion = (question) => {
  if (!question || typeof question !== 'string') {
    return {
      isValid: false,
      message: 'Question is required and must be a string'
    };
  }

  if (question.trim().length === 0) {
    return {
      isValid: false,
      message: 'Question cannot be empty'
    };
  }

  if (question.length > 1000) { // This matches the config default
    return {
      isValid: false,
      message: 'Question is too long (maximum 1000 characters)'
    };
  }

  // Basic check for potentially harmful content
  if (/[<>{}]/.test(question)) {
    return {
      isValid: false,
      message: 'Question contains invalid characters'
    };
  }

  return {
    isValid: true,
    message: 'Question is valid'
  };
};