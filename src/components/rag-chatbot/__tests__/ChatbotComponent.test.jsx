import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatbotComponent from '../ChatbotComponent';

// Mock the API service
jest.mock('../services/api', () => ({
  submitQuestion: jest.fn()
}));

// Mock the health check service
jest.mock('../services/health-check', () => ({
  getBackendStatus: jest.fn()
}));

// Mock the local storage utilities
jest.mock('../utils/local-storage', () => ({
  saveConversationHistory: jest.fn(),
  loadConversationHistory: jest.fn(() => []),
  saveSettings: jest.fn()
}));

// Mock the validation utility
jest.mock('../utils/validation', () => ({
  validateQuestion: jest.fn(() => ({ isValid: true, message: '' }))
}));

// Mock the rate limiter
jest.mock('../services/rate-limiter', () => ({
  defaultRateLimiter: {
    isAllowed: jest.fn(() => true),
    getLimitInfo: jest.fn(() => ({ resetTime: Date.now() + 60000 }))
  }
}));

// Mock the error handler
jest.mock('../services/error-handler', () => ({
  handleApiError: jest.fn(),
  handleRateLimitError: jest.fn(() => new Error('Rate limit exceeded')),
  formatErrorForDisplay: jest.fn((err) => err.message || 'An error occurred')
}));

describe('ChatbotComponent', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock successful backend status
    require('./services/health-check').getBackendStatus.mockResolvedValue('healthy');
  });

  test('renders without crashing', () => {
    render(<ChatbotComponent />);
    expect(screen.getByText('Book Assistant')).toBeInTheDocument();
  });

  test('shows question input field', () => {
    render(<ChatbotComponent />);
    const input = screen.getByPlaceholderText('Ask about the book content...');
    expect(input).toBeInTheDocument();
  });

  test('handles question submission', async () => {
    const mockResponse = {
      answer: 'This is a test response',
      sources: [{ title: 'Test Source', url: 'http://example.com' }],
      confidence: 0.9
    };

    const { submitQuestion } = require('./services/api');
    submitQuestion.mockResolvedValue(mockResponse);

    render(<ChatbotComponent />);

    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'Test question?' } });

    const submitButton = screen.getByRole('button', { name: /submit/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(submitQuestion).toHaveBeenCalledWith('Test question?');
    });
  });

  test('displays error when question is empty', async () => {
    const { validateQuestion } = require('./utils/validation');
    validateQuestion.mockReturnValue({
      isValid: false,
      message: 'Question must be at least 3 characters long'
    });

    render(<ChatbotComponent />);

    const input = screen.getByPlaceholderText('Ask about the book content...');
    fireEvent.change(input, { target: { value: 'Hi' } }); // Too short

    const submitButton = screen.getByRole('button', { name: /submit/i });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText('Question must be at least 3 characters long')).toBeInTheDocument();
    });
  });

  test('toggles chat visibility', () => {
    render(<ChatbotComponent />);

    const toggleButton = screen.getByLabelText('Close chat'); // Initially open
    fireEvent.click(toggleButton);

    // The button text should change to indicate it's now to open the chat
    expect(screen.getByLabelText('Open chat')).toBeInTheDocument();
  });
});