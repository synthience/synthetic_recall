# Lucid Recall Quick Start Guide

## Prerequisites

1. Python Requirements:
```bash
pip install sentence-transformers websockets numpy torch aiohttp
```

2. System Requirements:
- CUDA-capable GPU (recommended)
- Python 3.8+
- Node.js 14+ (for development)

## Starting the System

1. **Set Environment Variable**:
```powershell
$env:PYTHONPATH = "h:/Development FAST/Lucid Recall FAST v1.1"
```

2. **Start Memory Server**:
```powershell
cd h:/Development FAST/Lucid Recall FAST v1.1/managers/python/nemo_integration
python tensor_server.py
```
Expected output:
```
INFO:__main__:GPU initialized: NVIDIA GeForce RTX XXXX
INFO:__main__:Available GPU memory: XX.XX GB
INFO:__main__:Initialized TensorServer with Memory System on device: cuda
INFO:websockets.server:server listening on 0.0.0.0:5000
```

3. **Start Inference Server**:
```powershell
cd h:/Development FAST/Lucid Recall FAST v1.1/managers/python/inference
python inference_server.py
```
Expected output:
```
INFO:embedding_engine:Initializing EmbeddingEngine with device: cuda
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: all-MiniLM-L6-v2
INFO:__main__:WebSocket server listening on 0.0.0.0:5005
```

4. **Start Web Server**:
```powershell
cd h:/Development FAST/Lucid Recall FAST v1.1/managers/python/nemo_integration
python serve_web.py
```
Expected output:
```
Serving files from: .../web
Server running at http://localhost:8081
```

## Verifying the Setup

1. Open http://localhost:8081 in your browser

2. Check the console for:
```
DOM Content Loaded - Starting initialization...
Detecting server configuration...
Memory server detected
Inference server detected
Server configuration detected
All components initialization completed
Memory Interface Status: connected
Connected to sentence model server
LoRA Interface Status: connected
```

3. Check the UI header for:
- Memory usage indicator
- Response time indicator
- Connection status showing "Connected"

## Testing the System

1. **Basic Chat Test**:
- Type a message in the input box
- Press Enter or click Send
- Verify the message appears in the chat
- Check console for "Storing memory" confirmation

2. **Memory Test**:
- Send a message about a specific topic
- Send a related message later
- Verify that suggestions appear based on the previous context
- Check the significance and surprise metrics

3. **Offline Mode Test**:
- Stop one of the servers
- Verify the UI shows "Offline Mode" indicator
- Verify graceful degradation of features

## Troubleshooting

1. **Server Connection Issues**:
- Verify all three ports are available (5000, 5005, 8081)
- Check PYTHONPATH is set correctly
- Ensure no other instances are running

2. **GPU Issues**:
- Verify CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory usage
- Consider falling back to CPU if needed

3. **Memory Issues**:
- Check memory usage in task manager
- Verify memory cleanup is working
- Adjust max_size in EphemeralMemory config if needed

## Development Mode Features

1. **Enhanced Logging**:
- Check browser console for detailed logs
- Server logs show additional debug information
- Performance metrics are displayed in real-time

2. **Model Compatibility**:
- Development mode accepts any model version
- Warnings instead of errors for version mismatches
- Detailed error messages in console

## Next Steps

1. **Customization**:
- Adjust memory parameters in tensor_server.py
- Modify significance weights in SignificanceCalculator
- Customize UI appearance in main.css

2. **Integration**:
- Review LUCID_RECALL_ARCHITECTURE.md for system overview
- Check SERVER_IMPLEMENTATIONS.md for detailed server code
- Explore frontend code in web/js directory

3. **Monitoring**:
- Watch memory usage and cleanup
- Monitor embedding generation performance
- Check response times and optimization opportunities

## Common Operations

1. **Memory Management**:
```python
# Store memory
await memory_interface.storeMemory("text", {
    type: "user",
    timestamp: Date.now()
})

# Retrieve memories
memories = await memory_interface.retrieveMemories(5, 0.7)
```

2. **Embeddings Generation**:
```python
# Generate embeddings
embeddings = await lora_interface.getEmbeddings("text")

# Get prediction
prediction = await lora_interface.getPrediction("text", {
    context: recentMessages
})
```

3. **UI Updates**:
```javascript
// Add message
ui_controller.addMessage('user', 'message text')

// Update metrics
ui_controller.updateMetrics()
```

For more detailed information:
- Architecture: LUCID_RECALL_ARCHITECTURE.md
- Server Implementation: SERVER_IMPLEMENTATIONS.md
- API Documentation: API_DOCUMENTATION.md