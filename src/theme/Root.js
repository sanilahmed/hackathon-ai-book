/**
 * Docusaurus Root component to inject custom fields into the browser
 * This component wraps the entire app and exposes custom fields to window
 */

import React from 'react';
import InjectCustomFields from './InjectCustomFields';

// Root component that wraps the entire Docusaurus app
const Root = ({ children }) => {
  return (
    <>
      {/* Inject custom fields script */}
      <InjectCustomFields />
      {/* Render the rest of the app */}
      {children}
    </>
  );
};

export default Root;