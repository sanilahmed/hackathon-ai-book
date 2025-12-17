import React from 'react';
import ResponseRenderer from './ResponseRenderer';
import styles from './styles/chat-display.module.css';

/**
 * Chat display component to show messages
 */
const ChatDisplay = ({ messages, isLoading }) => {
  const messagesEndRef = React.useRef(null);

  // Scroll to bottom when messages change
  React.useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <div className={styles.chatDisplay}>
      <div className={styles.chatDisplay__messages}>
        {messages.length === 0 ? (
          <div className={styles.chatDisplay__welcome}>
            <p>Hello! How can I help you today?</p>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={`${styles.chatDisplay__message} ${
                message.sender === 'user'
                  ? styles['chatDisplay__message--user']
                  : styles['chatDisplay__message--bot']
              }`}
            >
              <div className={styles.chatDisplay__messageText}>
                {message.sender === 'user' ? (
                  message.text
                ) : (
                  <ResponseRenderer response={message} />
                )}
              </div>
              <div className={styles.chatDisplay__timestamp}>
                {message.timestamp ? message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className={`${styles.chatDisplay__message} ${styles['chatDisplay__message--bot']}`}>
            <div className={styles.chatDisplay__messageText}>
              <span className={styles.chatDisplay__typingIndicator}>
                <span className={styles.chatDisplay__typingDot}>•</span>
                <span className={styles.chatDisplay__typingDot}>•</span>
                <span className={styles.chatDisplay__typingDot}>•</span>
              </span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>
    </div>
  );
};

export default ChatDisplay;