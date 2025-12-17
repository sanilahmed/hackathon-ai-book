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
              // Define the RAG_API_URL based on the deployment environment
              // For GitHub Pages, if RAG_API_URL is null in config, we still want to expose it
              window.RAG_API_URL = window.RAG_API_URL || null;

              // Access the Docusaurus site config that's available in the browser
              var siteConfig = window.__PRELOADED_STATE__ ? window.__PRELOADED_STATE__.siteConfig : null;
              if (!siteConfig && window.__DOCUSAURUS__) {
                siteConfig = window.__DOCUSAURUS__.siteConfig;
              }

              if (siteConfig && siteConfig.customFields) {
                var customFields = siteConfig.customFields;

                // Expose each custom field as a top-level window property
                Object.keys(customFields).forEach(function(key) {
                  window[key] = customFields[key];
                });

                // Also make them available under a namespace
                window.docusaurusCustomFields = customFields;
              }

              // Ensure RAG_API_URL is properly set even if not in custom fields
              if (typeof window.RAG_API_URL === 'undefined') {
                window.RAG_API_URL = null;
              }
            }
          })();
        `,
      }}
    />
  );
};

export default InjectCustomFields;