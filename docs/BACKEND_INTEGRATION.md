# Backend Integration Documentation

## Overview

This document describes the integration between the Next.js frontend chat application and the HPC-enabled backend system. The integration provides real-time metrics and memory management capabilities while maintaining high performance.

## Architecture

### Components

1. **HPCService (src/services/HPCService.ts)**
   - Singleton service managing HPC operations
   - Handles TensorFlow initialization and memory management
   - Provides metrics caching and real-time updates
   - Bridges frontend metrics display with backend HPC systems

2. **HPC API Route (src/pages/api/hpc-metrics.js)**
   - REST endpoint for retrieving HPC metrics
   - Provides real-time system status
   - Handles error cases and backend failures

3. **Chat Store Integration (src/store/chatStore.jsx)**
   - Extended with HPC metrics state
   - Implements polling mechanism for real-time updates
   - Manages metric persistence and updates

4. **Metrics Display (src/components/MetricsDisplay.jsx)**
   - Real-time visualization of HPC metrics
   - Memory usage and processing statistics
   - Backend status information

## Setup and Configuration

### Environment Requirements

```bash
# Required environment variables
NEXT_PUBLIC_ENABLE_HPC=true
NEXT_PUBLIC_HPC_POLL_INTERVAL=2000  # milliseconds
```

### TensorFlow Backend

The system automatically selects the optimal TensorFlow backend:
- WebGL for browser environments with GPU support
- TensorFlow Node.js for server-side operations
- CPU fallback when GPU is unavailable

## Usage

### Initializing HPC Services

```typescript
// In your Next.js API route
import HPCService from '@/services/HPCService';

const hpcService = HPCService.getInstance();
```

### Processing Embeddings

```typescript
// Example: Processing embeddings in an API route
const result = await hpcService.processEmbeddings(embeddings, chatId);
```

### Accessing Metrics

```typescript
// Get current metrics for a chat session
const metrics = hpcService.getMetrics(chatId);
```

## Memory Management

The system implements several strategies for efficient memory management:

1. **Chunked Processing**
   - Large embedding arrays are processed in chunks
   - Prevents memory overflow
   - Configurable chunk size

2. **Tensor Cleanup**
   - Automatic tensor disposal
   - Scope-based memory management
   - Regular garbage collection

3. **Metrics Caching**
   - Time-based cache invalidation
   - Chat-specific metric storage
   - Memory-efficient storage format

## Error Handling

The system implements comprehensive error handling:

```typescript
try {
  await hpcService.processEmbeddings(embeddings, chatId);
} catch (error) {
  logger.error('HPC Processing Error', { error, chatId });
  // Fallback to CPU processing or show user feedback
}
```

## Performance Considerations

1. **Polling Frequency**
   - Default: 2 seconds
   - Configurable through environment variables
   - Balance between responsiveness and performance

2. **Batch Processing**
   - Embeddings are processed in optimal batch sizes
   - Automatic batch size adjustment based on available memory
   - GPU utilization optimization

3. **Memory Limits**
   - Configurable memory thresholds
   - Automatic cleanup of old metrics
   - Resource monitoring and warnings

## Integration with Chat Flow

1. **Message Processing**
   ```typescript
   // In chat message handler
   const processMessage = async (message) => {
     const embeddings = await generateEmbeddings(message);
     const processed = await hpcService.processEmbeddings(embeddings, chatId);
     // Continue with chat flow
   };
   ```

2. **Metrics Display**
   ```jsx
   // In React component
   const ChatMetrics = () => {
     const metrics = useChatStore(state => state.hpcMetrics);
     return <MetricsDisplay metrics={metrics} />;
   };
   ```

## Troubleshooting

Common issues and solutions:

1. **High Memory Usage**
   - Check chunk size configuration
   - Verify tensor cleanup
   - Monitor GPU memory allocation

2. **Slow Performance**
   - Verify GPU availability
   - Check batch size settings
   - Monitor polling frequency

3. **Metric Updates**
   - Check network connectivity
   - Verify polling configuration
   - Monitor browser console for errors

## Future Improvements

Planned enhancements:

1. **WebSocket Integration**
   - Real-time metric updates
   - Reduced polling overhead
   - Better scalability

2. **Advanced Caching**
   - Redis integration
   - Distributed caching
   - Better memory efficiency

3. **Monitoring Dashboard**
   - Extended metrics visualization
   - System health monitoring
   - Performance analytics

## Contributing

When contributing to the backend integration:

1. Follow TypeScript best practices
2. Maintain comprehensive error handling
3. Update documentation for new features
4. Add appropriate tests
5. Consider performance implications

## Testing

Run the test suite:

```bash
npm run test:hpc
```

Key test areas:
- Memory management
- Metric accuracy
- Error handling
- Performance benchmarks