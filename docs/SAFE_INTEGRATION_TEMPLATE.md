# Safe Integration Template for Lucid Recall

## Current Infrastructure (DO NOT MODIFY)

### Docker Container
- Name: nemo_sig_v2
- Status: Running
- Base Image: nemo-backup:2025-02-09
- Ports: 6006, 8888
- Memory System: Active and preserved

### LM Studio
- Endpoint: http://192.168.0.203:1234
- Model: qwen2.5-7b-instruct-1m
- Status: Active and responding

## Safe Integration Steps

1. Memory System Connection
   - Use existing WebSocket connection to nemo_sig_v2
   - Default ports: 6006, 8888
   - DO NOT modify container configuration
   - DO NOT restart services

2. LM Studio Integration
   ```javascript
   const LM_STUDIO_CONFIG = {
     baseUrl: 'http://192.168.0.203:1234',
     model: 'qwen2.5-7b-instruct-1m',
     headers: {
       'Content-Type': 'application/json'
     }
   };
   ```

3. API Endpoints
   - Chat Completion: POST /v1/chat/completions
   - Model Info: GET /v1/models
   - Health Check: GET /v1/health

4. Safety Guidelines
   - NO Docker commands
   - NO container modifications
   - NO service restarts
   - NO port changes

5. Testing Protocol
   - Use curl for API health checks
   - Monitor memory system responses
   - Verify WebSocket connections
   - Test LM Studio responses

## Implementation Notes

1. Memory System Access
   ```javascript
   // Use existing WebSocket connection
   const ws = new WebSocket('ws://localhost:6006');
   
   // Handle memory operations
   ws.onmessage = (event) => {
     const data = JSON.parse(event.data);
     // Process memory updates
   };
   ```

2. LM Studio Integration
   ```javascript
   // Example chat completion request
   async function sendChatRequest(messages) {
     const response = await fetch('http://192.168.0.203:1234/v1/chat/completions', {
       method: 'POST',
       headers: {
         'Content-Type': 'application/json'
       },
       body: JSON.stringify({
         model: 'qwen2.5-7b-instruct-1m',
         messages,
         stream: false
       })
     });
     return response.json();
   }
   ```

3. Health Monitoring
   ```javascript
   // Regular health checks
   async function checkHealth() {
     try {
       // Check LM Studio
       const lmStudioHealth = await fetch('http://192.168.0.203:1234/v1/health');
       
       // Check Memory System
       const memoryHealth = await fetch('http://localhost:6006/health');
       
       return {
         lmStudio: lmStudioHealth.ok,
         memorySystem: memoryHealth.ok
       };
     } catch (error) {
       console.error('Health check failed:', error);
       return {
         lmStudio: false,
         memorySystem: false
       };
     }
   }
   ```

## Emergency Procedures

If any issues occur:
1. DO NOT modify Docker container
2. DO NOT restart services
3. Log the error and contact system administrator
4. Keep existing connections alive
5. Use fallback to local memory if needed

Remember: This template prioritizes system stability and data preservation. Any modifications to the existing infrastructure must be approved by the system administrator.