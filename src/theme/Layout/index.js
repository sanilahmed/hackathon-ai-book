import React from 'react';
import OriginalLayout from '@theme-original/Layout';
import ChatbotWrapper from '@site/src/theme/ChatbotWrapper';

/**
 * Custom Layout component that wraps the original Docusaurus Layout
 * and integrates the RAG chatbot
 */
export default function Layout(props) {
  return (
    <OriginalLayout {...props}>
      <ChatbotWrapper>
        {props.children}
      </ChatbotWrapper>
    </OriginalLayout>
  );
}