import React, { useState } from 'react';
import ChatInput from './ChatInput';
import ChatDisplay from './ChatDisplay';
import styles from './styles/chatbot.module.css';
import { submitQuestion } from './services/api';

/**
 * Main Chatbot component with fully functional open/close and send functionality
 */
const ChatbotComponent = ({ position = 'bottom-right', theme = 'light' }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // Function to call the real backend API
  const callBackendAPI = async (question) => {
    try {
      const response = await submitQuestion(question);
      return response.answer || "I could not find this information in the book.";
    } catch (error) {
      console.error('Error calling backend API:', error);
      throw error;
    }
  };

  // Function to submit question to backend
  const handleQuestionSubmit = async (question) => {
    if (!question.trim()) return;

    setIsLoading(true);

    // Add user message to chat
    const userMessage = { id: Date.now(), text: question, sender: 'user', timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);

    try {
      // Call real backend API
      const response = await callBackendAPI(question);

      // Add bot response to chat
      const botMessage = {
        id: Date.now() + 1,
        text: response.answer || response,
        sender: 'bot',
        timestamp: new Date(),
        sources: response.sources || [],
        confidence: response.confidence || null
      };
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Toggle chat open/close
  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  // Position and theme classes
  const positionClass = styles[`chatbot--${position}`];
  const themeClass = styles[`chatbot--${theme}`];

  return (
    <div className={`${styles.chatbot} ${positionClass} ${themeClass}`}>
      {isChatOpen ? (
        <div className={styles.chatbot__container}>
          <div className={styles.chatbot__header}>
            <h3 className={styles.chatbot__title}>Chat Assistant</h3>
            <button
              className={styles.chatbot__closeButton}
              onClick={toggleChat}
              aria-label="Close chat"
              style={{ pointerEvents: 'auto' }}
            >
              ×
            </button>
          </div>

          <ChatDisplay
            messages={messages}
            isLoading={isLoading}
          />

          <ChatInput
            onQuestionSubmit={handleQuestionSubmit}
            isLoading={isLoading}
          />
        </div>
      ) : (
        <button
          className={`${styles.chatbot__toggle} ${styles.chatbot__toggleClosed}`}
          onClick={toggleChat}
          aria-label="Open chat"
          style={{ pointerEvents: 'auto', zIndex: 10001 }}
        >
          ?
        </button>
      )}
    </div>
  );
};

export default ChatbotComponent;