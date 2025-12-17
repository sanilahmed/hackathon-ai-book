import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatbotComponent from '../ChatbotComponent';

// Mock all the services to test integration
jest.mock('../services/api', () => ({
  submitQuestion: jest.fn()
}));

jest.mock('../services/health-check', () => ({
  getBackendStatus: jest.fn()
}));

jest.mock('../utils/local-storage', () => ({
  saveConversationHistory: jest.fn(),
  loadConversationHistory: jest.fn(() => []),
  saveSettings: jest.fn()
}));

jest.mock('../utils/validation', () => ({
  validateQuestion: jest.fn(() => ({ isValid: true, message: '' }))
}));

jest.mock('../services/rate-limiter', () => ({
  defaultRateLimiter: {
    isAllowed: jest.fn(() => true),
    getLimitInfo: jest.fn(() => ({ resetTime: Date.now() + 60000 }))
  }
}));

jest.mock('../services/error-handler', () => ({
  handleApiError: jest.fn(),
  handleRateLimitError: jest.fn(() => new Error('Rate limit exceeded')),
  formatErrorForDisplay: jest.fn((err) => err.message || 'An error occurred')
}));

describe('Integration Test: Chatbot Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    require('../services/health-check').getBackendStatus.mockResolvedValue('healthy');
  });

  test('full flow: question input -> API call -> response display', async () => {
    const mockResponse = {
      answer: 'This is the AI-generated answer',
      sources: [{ title: 'Book Chapter 1', url: 'http://book.example.com/chapter1' }],
      confidence: 0.92,
      timestamp: new Date().toISOString()
    };

    const { submitQuestion } = require('../services/api');
    submitQuestion.mockResolvedValue(mockResponse);

    render(<ChatbotComponent />);

    // Step 1: Enter a question
    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'What is artificial intelligence?' } });

    // Step 2: Submit the question (assuming there's a submit button or enter key)
    // Since the actual component might submit on Enter or have a button, let's check
    fireEvent.keyPress(input, { key: 'Enter', code: 'Enter', char: 'Enter' });

    // Step 3: Wait for the API call to complete and response to be displayed
    await waitFor(() => {
      expect(submitQuestion).toHaveBeenCalledWith('What is artificial intelligence?');
    });

    await waitFor(() => {
      expect(screen.getByText('This is the AI-generated answer')).toBeInTheDocument();
    });

    // Step 4: Verify source attribution is displayed
    await waitFor(() => {
      expect(screen.getByText('Sources:')).toBeInTheDocument();
      expect(screen.getByText('Book Chapter 1')).toBeInTheDocument();
    });

    // Step 5: Verify confidence score is displayed
    await waitFor(() => {
      expect(screen.getByText('Confidence: 92%')).toBeInTheDocument();
    });
  });

  test('error handling flow: API error -> error display', async () => {
    const error = new Error('API Error');
    require('../services/error-handler').handleApiError.mockReturnValue(error);
    require('../services/error-handler').formatErrorForDisplay.mockReturnValue('API Error occurred');

    const { submitQuestion } = require('../services/api');
    submitQuestion.mockRejectedValue(error);

    render(<ChatbotComponent />);

    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'This will fail' } });

    fireEvent.keyPress(input, { key: 'Enter', code: 'Enter', char: 'Enter' });

    await waitFor(() => {
      expect(screen.getByText('Error: API Error occurred')).toBeInTheDocument();
    });
  });
});