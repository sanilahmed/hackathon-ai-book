import DOMPurify from 'dompurify';

/**
 * Sanitizes HTML content to prevent XSS attacks
 * @param {string} htmlContent - The HTML content to sanitize
 * @returns {string} - Sanitized HTML content
 */
export const sanitizeHTML = (htmlContent) => {
  if (!htmlContent || typeof htmlContent !== 'string') {
    return '';
  }

  return DOMPurify.sanitize(htmlContent, {
    ALLOWED_TAGS: [
      'p', 'div', 'span', 'br',
      'strong', 'em', 'b', 'i', 'u',
      'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
      'ul', 'ol', 'li',
      'a', 'code', 'pre', 'blockquote',
      'table', 'thead', 'tbody', 'tr', 'th', 'td'
    ],
    ALLOWED_ATTR: ['href', 'target', 'rel', 'class', 'id']
  });
};

/**
 * Sanitizes text content for safe display
 * @param {string} textContent - The text content to sanitize
 * @returns {string} - Sanitized text content
 */
export const sanitizeText = (textContent) => {
  if (!textContent || typeof textContent !== 'string') {
    return '';
  }

  // Remove potentially dangerous characters/sequences
  return textContent
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    .replace(/javascript:/gi, '')
    .replace(/vbscript:/gi, '')
    .replace(/on\w+=/gi, '');
};

export default {
  sanitizeHTML,
  sanitizeText
};