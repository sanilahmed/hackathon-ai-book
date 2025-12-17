import React from 'react';
import styles from './styles/loading-indicator.module.css';

/**
 * Loading indicator component for the RAG Chatbot
 * Shows a loading animation while waiting for responses
 */
const LoadingIndicator = () => {
  return (
    <div className={styles.loadingIndicator}>
      <div className={styles.loadingIndicator__dots}>
        <div className={styles.loadingIndicator__dot}></div>
        <div className={styles.loadingIndicator__dot}></div>
        <div className={styles.loadingIndicator__dot}></div>
      </div>
    </div>
  );
};

export default LoadingIndicator;