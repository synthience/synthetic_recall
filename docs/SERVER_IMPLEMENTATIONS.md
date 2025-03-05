# Lucid Recall Server Implementations

## Memory Server (tensor_server.py)

### EphemeralMemory Class
```python
class EphemeralMemory:
    def __init__(self, config):
        self.config = {
            'max_size': 1000,           # Maximum memories to store
            'decay_rate': 0.1,          # Memory decay rate
            'retention_threshold': 0.7,  # Minimum significance to retain
            'device': 'cuda',           # GPU acceleration
            'context_size': 10          # Context window size
        }
        self.memories = []
        self.embeddings = None          # Tensor of all embeddings
        self.device = torch.device(config['device'])

    async def store_memory(self, text, embeddings, metadata):
        # Convert embeddings to tensor
        embedding_tensor = torch.tensor(embeddings, device=self.device)
        
        # Calculate significance
        significance = self.calculate_significance(metadata)
        
        # Store memory with metadata
        memory = {
            'text': text,
            'embeddings': embedding_tensor,
            'metadata': {
                **metadata,
                'significance': significance,
                'timestamp': time.time()
            }
        }
        
        self.memories.append(memory)
        self.update_embeddings_tensor()
        self.cleanup_old_memories()

    async def retrieve_similar(self, query_embedding, count=5, threshold=0.7):
        if not self.memories:
            return []
            
        # Convert query to tensor
        query_tensor = torch.tensor(query_embedding, device=self.device)
        
        # Calculate similarities
        similarities = torch.cosine_similarity(
            query_tensor.unsqueeze(0),
            self.embeddings
        )
        
        # Get top matches
        top_indices = similarities.argsort(descending=True)[:count]
        
        return [
            self.memories[i] for i in top_indices 
            if similarities[i] > threshold
        ]

    def cleanup_old_memories(self):
        # Remove memories below retention threshold
        current_time = time.time()
        self.memories = [
            mem for mem in self.memories
            if self.calculate_retention(mem, current_time) > self.config['retention_threshold']
        ]
```

### SignificanceCalculator Class
```python
class SignificanceCalculator:
    def __init__(self, config):
        self.config = {
            'time_weight': 0.4,     # Recency importance
            'info_weight': 0.3,     # Information content importance
            'pred_weight': 0.3,     # Prediction accuracy importance
            'decay_rate': 0.1,      # Time decay rate
            'history_window': 1000,  # Historical context size
            'batch_size': 32        # Processing batch size
        }

    def calculate_significance(self, memory, context):
        # Time-based significance
        time_score = self.calculate_time_significance(memory)
        
        # Information-based significance
        info_score = self.calculate_information_significance(memory)
        
        # Prediction-based significance
        pred_score = self.calculate_prediction_significance(memory, context)
        
        # Weighted combination
        return (
            self.config['time_weight'] * time_score +
            self.config['info_weight'] * info_score +
            self.config['pred_weight'] * pred_score
        )
```

## Inference Server (inference_server.py)

### EmbeddingEngine Class
```python
class EmbeddingEngine:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name).to(self.device)
        self.tokenizer = self.model.tokenizer
        self.max_seq_length = self.model.max_seq_length

    async def generate_embeddings(self, texts):
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device
        )
        
        return embeddings.cpu().numpy()

    async def get_similarity(self, text1, text2):
        # Get embeddings
        emb1, emb2 = await self.generate_embeddings([text1, text2])
        
        # Calculate cosine similarity
        return cosine_similarity([emb1], [emb2])[0][0]
```

### StreamProcessor Class
```python
class StreamProcessor:
    def __init__(self, config=None):
        self.config = {
            'batch_size': 32,
            'max_length': 512,
            'stride': 256,
            'temperature': 0.7
        }
        self.embedding_engine = EmbeddingEngine()

    async def process_stream(self, text_stream, context=None):
        # Initialize processing state
        state = {
            'buffer': '',
            'embeddings': [],
            'predictions': []
        }
        
        async for chunk in text_stream:
            # Update buffer
            state['buffer'] += chunk
            
            # Process complete sentences
            if self.is_complete_sentence(state['buffer']):
                # Generate embeddings
                emb = await self.embedding_engine.generate_embeddings(state['buffer'])
                state['embeddings'].append(emb)
                
                # Generate predictions
                if context:
                    pred = await self.generate_prediction(state['buffer'], context)
                    state['predictions'].append(pred)
                
                # Clear buffer
                state['buffer'] = ''
            
            # Yield processed results
            yield {
                'embeddings': state['embeddings'],
                'predictions': state['predictions']
            }
```

## Web Server (serve_web.py)

```python
class WebServer:
    def __init__(self, directory, port=8081):
        self.directory = directory
        self.port = port
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        # Serve static files
        self.app.router.add_static('/', self.directory)
        
        # Serve index.html for root
        async def serve_index(request):
            return web.FileResponse(
                path=os.path.join(self.directory, 'index.html'),
                headers={'Content-Type': 'text/html'}
            )
        self.app.router.add_get('/', serve_index)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        print(f'Server running at http://localhost:{self.port}')
```

## Integration Points

1. **Memory-Inference Integration**:
   - Inference server generates embeddings
   - Memory server stores and retrieves based on embeddings
   - Significance calculator uses both for scoring

2. **WebSocket Communication**:
   - All servers use WebSocket for real-time communication
   - Binary data transfer for embeddings
   - JSON for metadata and control messages

3. **Error Handling**:
   - Graceful degradation
   - Automatic reconnection
   - Error propagation to clients

4. **Performance Optimization**:
   - GPU acceleration for tensor operations
   - Batch processing where possible
   - Efficient memory management

This implementation provides:
- Real-time text processing
- Semantic memory storage and retrieval
- GPU-accelerated embeddings generation
- Efficient WebSocket communication
- Robust error handling