# API Documentation

## Docker Server API

### WebSocket Endpoints

**Base URL**: `ws://localhost:8080`

| Endpoint | Description | Parameters | Response |
|----------|-------------|------------|----------|
| `/interact` | Send user interaction | `{"message": string, "context": object}` | `{"response": string, "thoughts": object, "memories": array}` |
| `/system/status` | Get system status | N/A | `{"status": string, "uptime": number, "current_model": string, "state": string}` |
| `/system/model` | Change active model | `{"model": string}` | `{"success": boolean, "model": string, "error": string}` |

### HTTP Endpoints

**Base URL**: `http://localhost:8081`

| Endpoint | Method | Description | Parameters | Response |
|----------|--------|-------------|------------|----------|
| `/api/memory/recent` | GET | Get recent memories | `?limit=10&type=interaction` | `{"memories": array}` |
| `/api/knowledge/search` | GET | Search knowledge graph | `?query=string&limit=10` | `{"results": array}` |
| `/api/model/status` | GET | Get model status | N/A | `{"current": string, "available": array}` |
| `/api/dream/insights` | GET | Get dream insights | `?limit=5&since=timestamp` | `{"insights": array}` |

## Dream API Test Endpoints

**Base URL**: `http://localhost:8081`

| Endpoint | Method | Description | Parameters | Response |
|----------|--------|-------------|------------|----------|
| `/api/dream/test/batch_embedding` | POST | Process batch of embeddings | `{"texts": array, "use_hypersphere": boolean}` | `{"status": string, "count": number, "successful": number, "embeddings": array}` |
| `/api/dream/test/similarity_search` | POST | Search for similar memories | `{"query": string, "top_k": number}` | `{"status": string, "results": array, "total_matches": number, "query": string}` |
| `/api/dream/test/create_test_report` | POST | Create a test dream report | `{"title": string, "fragments": array}` | `{"status": string, "report_id": string, "fragment_count": number}` |
| `/api/dream/test/refine_report` | POST | Refine an existing report | `{"report_id": string}` | `{"status": string, "report_id": string, "refinement_count": number, "confidence": number, "reason": string}` |
| `/api/dream/test/tensor_connection` | GET | Test tensor server connection | N/A | `{"status": string, "connected": boolean}` |
| `/api/dream/test/hpc_connection` | GET | Test HPC server connection | N/A | `{"status": string, "connected": boolean}` |
| `/api/dream/test/process_embedding` | GET | Test embedding processing | `?text=string` | `{"status": string, "embedding": array}` |
| `/api/dream/health` | GET | API health check | N/A | `{"status": string, "timestamp": string}` |

## TensorServer API

**Base URL**: `ws://localhost:5001`

| Command | Description | Parameters | Response |
|---------|-------------|------------|----------|
| `embed` | Generate embeddings | `{"text": string, "id": string}` | `{"embedding": array, "id": string}` |
| `search` | Search for similar memories | `{"embedding": array, "limit": number}` | `{"results": array}` |
| `stats` | Get server statistics | N/A | `{"embeddings_count": number, "gpu_utilization": number}` |

## HPCServer API

**Base URL**: `ws://localhost:5005`

| Command | Description | Parameters | Response |
|---------|-------------|------------|----------|
| `process` | Process embeddings | `{"embedding": array, "operation": string}` | `{"result": object, "operation": string}` |
| `stats` | Get HPC statistics | N/A | `{"cpu_utilization": number, "memory_utilization": number}` |

## LM Studio Server API

**Base URL**: `http://127.0.0.1:1234`

Standard OpenAI-compatible API:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Generate chat completions |
| `/v1/embeddings` | POST | Generate embeddings |

## API Usage Examples

### Testing Dream API Endpoints

```bash
# Test batch embedding processing
curl -X POST http://localhost:8081/api/dream/test/batch_embedding \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is a test", "Another test"], "use_hypersphere": false}'

# Test similarity search
curl -X POST http://localhost:8081/api/dream/test/similarity_search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 3}'

# Test dream report creation
curl -X POST http://localhost:8081/api/dream/test/create_test_report \
  -H "Content-Type: application/json" \
  -d '{"title": "Test Report", "fragments": [{"content": "Test insight", "type": "insight", "confidence": 0.8}]}'

# Test report refinement
curl -X POST http://localhost:8081/api/dream/test/refine_report \
  -H "Content-Type: application/json" \
  -d '{"report_id": "report:12345"}'

# Test health check
curl http://localhost:8081/api/dream/health
```

### Using the LM Studio API

```python
import requests
import json

# List available models
response = requests.get("http://127.0.0.1:1234/v1/models")
models = response.json()
print(json.dumps(models, indent=2))

# Generate chat completion
response = requests.post(
    "http://127.0.0.1:1234/v1/chat/completions",
    json={
        "model": "qwen2.5-7b-instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "temperature": 0.7
    }
)
result = response.json()
print(json.dumps(result, indent=2))
```