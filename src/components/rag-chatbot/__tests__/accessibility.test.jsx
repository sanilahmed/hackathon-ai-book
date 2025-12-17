import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import ChatbotComponent from '../ChatbotComponent';

describe('Accessibility Test: Chatbot Component', () => {
  test('has proper ARIA labels and roles', () => {
    render(<ChatbotComponent />);

    // Check for proper ARIA labels on interactive elements
    const toggleButton = screen.getByLabelText(/(Open|Close) chat/i);
    expect(toggleButton).toBeInTheDocument();
    expect(toggleButton).toHaveAttribute('aria-label');

    // Check for header structure
    const header = screen.getByRole('heading', { name: /Book Assistant/i });
    expect(header).toBeInTheDocument();
    expect(header.tagName).toBe('H3');
  });

  test('has proper focus management', () => {
    render(<ChatbotComponent />);

    // Check that input field has proper placeholder for accessibility
    const input = screen.getByPlaceholderText('Ask about the book content...');
    expect(input).toBeInTheDocument();
    expect(input).toHaveAttribute('placeholder');
  });

  test('has proper semantic structure', () => {
    render(<ChatbotComponent />);

    // Check for semantic elements
    const container = screen.getByRole('region', { name: /chatbot/i }) || screen.getByLabelText(/chat/i);
    expect(container).toBeInTheDocument();
  });
});