import React from 'react';
import styles from './styles/source-attribution.module.css';

/**
 * Source attribution component for the RAG Chatbot
 * Displays the sources used to generate the response
 */
const SourceAttribution = ({ sources }) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className={styles.sourceAttribution}>
      <h4 className={styles.sourceAttribution__title}>Sources:</h4>
      <ul className={styles.sourceAttribution__list}>
        {sources.map((source, index) => (
          <li key={index} className={styles.sourceAttribution__item}>
            <a
              href={source.url || '#'}
              className={styles.sourceAttribution__link}
              target={source.url ? "_blank" : undefined}
              rel={source.url ? "noopener noreferrer" : undefined}
            >
              {source.title || source.content?.substring(0, 50) + '...'}
            </a>
            {source.chunk_index && (
              <span className={styles.sourceAttribution__page}>
                (Chunk {source.chunk_index})
              </span>
            )}
            {source.similarity_score && (
              <span className={styles.sourceAttribution__similarity}>
                (Score: {Math.round(source.similarity_score * 100)}%)
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SourceAttribution;