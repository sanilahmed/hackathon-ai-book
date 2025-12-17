import { submitQuestion } from '../services/api';

// Mock fetch
global.fetch = jest.fn();

describe('API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('submitQuestion makes POST request with correct parameters', async () => {
    const mockResponse = {
      answer: 'Test answer',
      sources: [{ title: 'Test Source', url: 'http://example.com' }]
    };

    fetch.mockResolvedValue({
      ok: true,
      json: async () => mockResponse
    });

    const question = 'Test question';
    const result = await submitQuestion(question);

    expect(fetch).toHaveBeenCalledWith('http://localhost:8000/api/ask', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question })
    });

    expect(result).toEqual(mockResponse);
  });

  test('submitQuestion throws error when response is not ok', async () => {
    fetch.mockResolvedValue({
      ok: false,
      status: 500
    });

    await expect(submitQuestion('Test question')).rejects.toThrow('API request failed with status 500');
  });
});