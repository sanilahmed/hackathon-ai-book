import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import ResponseRenderer from '../ResponseRenderer';

describe('ResponseRenderer', () => {
  test('renders answer correctly', () => {
    const response = {
      answer: 'This is the answer',
      sources: [],
      confidence: 0.85
    };

    render(<ResponseRenderer response={response} />);

    expect(screen.getByText('This is the answer')).toBeInTheDocument();
  });

  test('renders sources when available', () => {
    const response = {
      answer: 'This is the answer',
      sources: [
        { title: 'Source 1', url: 'http://example.com/1' },
        { title: 'Source 2', url: 'http://example.com/2' }
      ],
      confidence: 0.85
    };

    render(<ResponseRenderer response={response} />);

    expect(screen.getByText('Sources:')).toBeInTheDocument();
    expect(screen.getByText('Source 1')).toBeInTheDocument();
    expect(screen.getByText('Source 2')).toBeInTheDocument();
  });

  test('renders confidence score', () => {
    const response = {
      answer: 'This is the answer',
      sources: [],
      confidence: 0.85
    };

    render(<ResponseRenderer response={response} />);

    expect(screen.getByText('Confidence: 85%')).toBeInTheDocument();
  });

  test('does not render confidence when not provided', () => {
    const response = {
      answer: 'This is the answer',
      sources: []
    };

    render(<ResponseRenderer response={response} />);

    expect(screen.queryByText('Confidence:')).not.toBeInTheDocument();
  });

  test('does not render sources section when no sources', () => {
    const response = {
      answer: 'This is the answer',
      sources: [],
      confidence: 0.85
    };

    render(<ResponseRenderer response={response} />);

    expect(screen.queryByText('Sources:')).not.toBeInTheDocument();
  });
});