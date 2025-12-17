/**
 * Docusaurus theme component to inject custom fields into the browser
 * This component renders a script tag that exposes custom fields to window
 */

import React from 'react';

// This component injects custom fields from Docusaurus config to browser window
const InjectCustomFields = () => {
  // This component will be server-side rendered to inject the script
  return (
    <script
      dangerouslySetInnerHTML={{
        __html: `
          (function() {
            if (typeof window !== 'undefined') {
              // Access the Docusaurus site config that's available in the browser
              var siteConfig = window.__PRELOADED_STATE__ ? window.__PRELOADED_STATE__.siteConfig : null;
              if (!siteConfig && window.__DOCUSAURUS__) {
                siteConfig = window.__DOCUSAURUS__.siteConfig;
              }

              // Initialize RAG_API_URL to null by default
              window.RAG_API_URL = null;

              if (siteConfig && siteConfig.customFields) {
                var customFields = siteConfig.customFields;

                // Expose each custom field as a top-level window property
                Object.keys(customFields).forEach(function(key) {
                  window[key] = customFields[key];
                });

                // Also make them available under a namespace
                window.docusaurusCustomFields = customFields;

                // Specifically ensure RAG_API_URL is set from custom fields
                if (customFields.RAG_API_URL !== undefined) {
                  window.RAG_API_URL = customFields.RAG_API_URL;
                }
              }

              // Log for debugging in development
              if (window.location.hostname !== 'localhost') {
                console.log('RAG_API_URL on GitHub Pages:', window.RAG_API_URL);
              }
            }
          })();
        `,
      }}
    />
  );
};

export default InjectCustomFields;