import React from 'react';
import SourceAttribution from './SourceAttribution';
import styles from './styles/response.module.css';

/**
 * Response renderer component for the RAG Chatbot
 * Renders the response from the backend with proper formatting
 */
const ResponseRenderer = ({ response }) => {
  // Handle both direct text responses and full API responses
  const answer = response.answer || response.text || response;
  const sources = response.sources || [];
  const confidence = response.confidence;

  return (
    <div className={styles.responseRenderer}>
      <div className={styles.responseRenderer__answer}>
        {answer}
      </div>

      {sources && sources.length > 0 && (
        <div className={styles.responseRenderer__sources}>
          <SourceAttribution sources={sources} />
        </div>
      )}

      {confidence !== undefined && confidence !== null && (
        <div className={styles.responseRenderer__confidence}>
          <small>Confidence: {Math.round(confidence * 100)}%</small>
        </div>
      )}
    </div>
  );
};

export default ResponseRenderer;