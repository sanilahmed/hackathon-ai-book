// Test script to check if the Docusaurus server is responding
const http = require('http');

const options = {
  host: 'localhost',
  port: 3000,
  path: '/hackathon-ai-book/',
  timeout: 5000
};

console.log('Testing connection to Docusaurus server...');
console.log('URL: http://localhost:3000/hackathon-ai-book/');

const request = http.request(options, (res) => {
  console.log(`Status Code: ${res.statusCode}`);
  console.log(`Headers: ${JSON.stringify(res.headers, null, 2)}`);

  let data = '';
  res.on('data', (chunk) => {
    data += chunk;
  });

  res.on('end', () => {
    console.log('Response received (first 500 chars):');
    console.log(data.substring(0, 500) + (data.length > 500 ? '...' : ''));
  });
});

request.on('error', (err) => {
  console.error('Error connecting to server:', err.message);
  console.log('This could be due to:');
  console.log('1. The server is not running');
  console.log('2. Network access restrictions in your environment');
  console.log('3. Port forwarding not configured properly');
});

request.on('timeout', () => {
  console.error('Request timed out - server might be slow to respond');
});

request.end();