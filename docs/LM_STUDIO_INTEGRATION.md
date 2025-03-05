# LM Studio Integration Guide

This document outlines the complete integration setup for LM Studio, based on our working implementation.

## Overview

LM Studio provides a local API server that's compatible with the OpenAI API format. Our integration includes:
- A full-featured client implementation
- CORS configuration for local development
- Streaming and non-streaming support
- Model management
- Error handling

## Client Implementation

```javascript
// LM Studio Client Class
export class LMStudioClient {
    constructor(baseUrl = 'http://127.0.0.1:1234') {
      this.baseUrl = baseUrl;
      this.currentModel = null;
    }
  
    async fetchModels() {
      const response = await fetch(`${this.baseUrl}/v1/models`);
      if (!response.ok) {
        throw new Error(`Failed to fetch models: ${response.statusText}`);
      }
      const data = await response.json();
      // The API returns { data: [ {id: "modelid1", ...}, {...} ], object: "list" }
      return data.data;
    }
  
    async setCurrentModel(modelId) {
      this.currentModel = modelId;
      return true;
    }
  
    async sendMessage(prompt, options = {}) {
      if (!this.currentModel) {
        throw new Error('No LM Studio model selected');
      }
      const {
        conversationHistory = [],
        onToken = null,
        temperature = 0.7,
        maxTokens = 2048,
        stream = Boolean(onToken),
      } = options;
  
      // Merge conversation
      const messages = [...conversationHistory, { role: 'user', content: prompt }];
  
      // Prepare request
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.currentModel,
          messages,
          temperature,
          max_tokens: maxTokens,
          stream
        })
      });
  
      if (!response.ok) {
        throw new Error(`LM Studio request failed: ${response.statusText}`);
      }
  
      if (stream) {
        // Handle streaming
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullContent = '';
  
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
  
          const lines = buffer.split('\n');
          buffer = lines.pop() || ''; // leftover
  
          for (const line of lines) {
            if (!line.trim() || line.trim() === 'data: [DONE]') continue;
            try {
              const jsonLine = JSON.parse(line.replace(/^data:\s?/, ''));
              const token = jsonLine.choices[0]?.delta?.content || '';
              if (token) {
                fullContent += token;
                onToken(token, fullContent);
              }
            } catch (err) {
              console.warn('Stream parse error:', err);
            }
          }
        }
  
        return { content: fullContent, streaming: true };
      }
  
      // Non-streaming
      const data = await response.json();
      return {
        content: data.choices[0].message.content,
        streaming: false
      };
    }
}
```

## CORS Configuration

For local development, you'll need to configure CORS to allow communication between your frontend and the LM Studio server. Here's the Express.js middleware configuration:

```javascript
const express = require('express');
const cors = require('cors');

// Configure CORS
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Accept'],
    credentials: true,
    preflightContinue: true
}));

// Handle OPTIONS preflight requests
app.options('*', cors());
```

## Usage Example

```javascript
// Initialize client
const client = new LMStudioClient();

// List available models
const models = await client.fetchModels();
console.log('Available models:', models);

// Select a model
await client.setCurrentModel(models[0].id);

// Non-streaming example
const response = await client.sendMessage('Hello, how are you?');
console.log('Response:', response.content);

// Streaming example
const streamingResponse = await client.sendMessage('Tell me a story', {
    stream: true,
    onToken: (token, fullContent) => {
        console.log('New token:', token);
        console.log('Full content so far:', fullContent);
    }
});
```

## Critical Points

1. **Server Configuration**
   - LM Studio runs on `localhost:1234` by default
   - Uses OpenAI-compatible API endpoints
   - Supports both streaming and non-streaming responses

2. **API Endpoints**
   - `/v1/models` - GET request to list available models
   - `/v1/chat/completions` - POST request for chat interactions

3. **Request Format**
   ```javascript
   {
     "model": "model-id",
     "messages": [
       {"role": "user", "content": "Hello"},
       {"role": "assistant", "content": "Hi there"},
       {"role": "user", "content": "How are you?"}
     ],
     "temperature": 0.7,
     "max_tokens": 2048,
     "stream": true
   }
   ```

4. **Error Handling**
   - Check response.ok for request success
   - Handle stream parsing errors gracefully
   - Implement proper error boundaries in React components

5. **Performance Considerations**
   - Use streaming for better user experience with long responses
   - Implement proper cleanup for streaming connections
   - Handle connection timeouts

## Model Management

Our implementation supports various GGUF models, including:
- DeepSeek Coder series
- Qwen series
- Other compatible GGUF models

Model files should be placed in a directory accessible to LM Studio (e.g., `G:/Models/lmstudio-community/`).

## Security Notes

1. CORS is configured to accept all origins ('*') for development
2. In production, restrict CORS to specific origins
3. Implement rate limiting if needed
4. Consider adding authentication for production use

## Testing

Test the integration using the provided mock server:
```javascript
const MockLMStudioServer = require('./mock-lmstudio-server');
const server = new MockLMStudioServer(1235); // Use different port for testing
```

This allows testing the integration without running the actual LM Studio server.