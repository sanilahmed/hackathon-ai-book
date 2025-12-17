import { validateQuestion } from '../utils/validation';

describe('Validation Utility', () => {
  test('validates questions with proper length', () => {
    expect(validateQuestion('What is AI?')).toEqual({
      isValid: true,
      message: ''
    });

    expect(validateQuestion('Hi')).toEqual({
      isValid: false,
      message: 'Question must be at least 3 characters long'
    });

    expect(validateQuestion('')).toEqual({
      isValid: false,
      message: 'Question must be at least 3 characters long'
    });

    const longQuestion = 'A'.repeat(1001);
    expect(validateQuestion(longQuestion)).toEqual({
      isValid: false,
      message: 'Question must be no more than 1000 characters long'
    });
  });

  test('trims whitespace before validation', () => {
    expect(validateQuestion('  What is AI?  ')).toEqual({
      isValid: true,
      message: ''
    });

    expect(validateQuestion('   ')).toEqual({
      isValid: false,
      message: 'Question must be at least 3 characters long'
    });
  });
});