import React, { useEffect, useState } from 'react';
import { ChatbotComponent } from '../components/rag-chatbot';
import { isLocalStorageAvailable, loadSettings } from '../components/rag-chatbot/utils/local-storage';
import { isChatbotEnabled } from '../components/rag-chatbot/config';

/**
 * Docusaurus theme wrapper for the RAG Chatbot
 * Integrates the chatbot component into the Docusaurus layout
 */
const ChatbotWrapper = ({ children }) => {
  const [showChatbot, setShowChatbot] = useState(true); // Changed from false to true for default rendering
  const [chatbotSettings, setChatbotSettings] = useState({});

  useEffect(() => {
    // Check if chatbot is enabled
    const enabled = isChatbotEnabled();

    if (enabled) {
      // Load saved settings from local storage if available, with defaults
      if (isLocalStorageAvailable()) {
        const savedSettings = loadSettings();
        // Apply default values for any missing settings
        const settingsWithDefaults = {
          position: savedSettings.position || 'bottom-right',
          theme: savedSettings.theme || 'light',
          maxHistory: savedSettings.maxHistory || 10,
          enableHistory: typeof savedSettings.enableHistory === 'boolean' ? savedSettings.enableHistory : true,
          placeholder: savedSettings.placeholder || "Ask about the book content...",
        };
        setChatbotSettings(settingsWithDefaults);
      } else {
        // Use defaults if local storage is not available
        setChatbotSettings({
          position: 'bottom-right',
          theme: 'light',
          maxHistory: 10,
          enableHistory: true,
          placeholder: "Ask about the book content...",
        });
      }

      setShowChatbot(true);
    } else {
      // Even if disabled, show with defaults to allow user to enable
      setChatbotSettings({
        position: 'bottom-right',
        theme: 'light',
        maxHistory: 10,
        enableHistory: true,
        placeholder: "Ask about the book content...",
      });
      setShowChatbot(true); // Force showing for testing
    }
  }, []);

  return (
    <>
      {children}
      {showChatbot && (
        <ChatbotComponent
          position={chatbotSettings.position}
          theme={chatbotSettings.theme}
          maxHistory={chatbotSettings.maxHistory}
          enableHistory={chatbotSettings.enableHistory}
          placeholder={chatbotSettings.placeholder}
        />
      )}
    </>
  );
};

export default ChatbotWrapper;