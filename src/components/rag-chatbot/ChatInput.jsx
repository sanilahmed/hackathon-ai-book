import React, { useState } from 'react';
import styles from './styles/chat-input.module.css';

/**
 * Chat input component with fully functional send button
 */
const ChatInput = ({ onQuestionSubmit, isLoading }) => {
  const [inputValue, setInputValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onQuestionSubmit(inputValue.trim());
      setInputValue(''); // Clear input after sending
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (inputValue.trim() && !isLoading) {
        onQuestionSubmit(inputValue.trim());
        setInputValue(''); // Clear input after sending
      }
    }
  };

  return (
    <div className={styles.chatInput}>
      <form onSubmit={handleSubmit} className={styles.chatInput__form}>
        <textarea
          className={styles.chatInput__textarea}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          disabled={isLoading}
          rows={1}
          aria-label="Type your question"
        />
        <button
          type="submit"
          className={`${styles.chatInput__submit} ${isLoading ? styles['chatInput__submit--disabled'] : ''}`}
          disabled={!inputValue.trim() || isLoading}
          aria-label="Send message"
          style={{ pointerEvents: 'auto' }}
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  );
};

export default ChatInput;