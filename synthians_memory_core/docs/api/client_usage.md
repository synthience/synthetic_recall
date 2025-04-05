# Memory Core Python Client Usage Guide

This document provides guidelines and code examples for using the asynchronous Python client to interact with the Synthians Memory Core API.

## 1. Installation & Setup

```python
# Import necessary modules
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp

# Define the client class
class SynthiansClient:
    def __init__(self, base_url: str = "http://localhost:5010"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    # Client methods will be defined below
```

## 2. Basic Operations

### Process Memory

```python
async def process_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Process and store a memory."""
    if metadata is None:
        metadata = {}
    
    async with self.session.post(
        f"{self.base_url}/process_memory",
        json={"content": content, "metadata": metadata}
    ) as response:
        return await response.json()

# Example Usage
async def example_process_memory(client: SynthiansClient):
    response = await client.process_memory(
        content="Paris is the capital of France",
        metadata={"source": "wikipedia", "type": "fact"}
    )
    print(f"Memory ID: {response.get('memory_id')}")
    print(f"QuickRecal Score: {response.get('quick_recal_score')}")
```

### Retrieve Memories

```python
async def retrieve_memories(
    self, 
    query: str, 
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    offset: int = 0,
    include_embedding: bool = False
) -> Dict[str, Any]:
    """Retrieve memories based on a query."""
    if filters is None:
        filters = {}
    
    async with self.session.post(
        f"{self.base_url}/retrieve",
        json={
            "query": query,
            "filters": filters,
            "limit": limit,
            "offset": offset,
            "include_embedding": include_embedding
        }
    ) as response:
        return await response.json()

# Example Usage
async def example_retrieve_memories(client: SynthiansClient):
    response = await client.retrieve_memories(
        query="capital of France",
        filters={"metadata": {"type": "fact"}},
        limit=5
    )
    
    if response.get("success"):
        memories = response.get("memories", [])
        print(f"Found {len(memories)} memories:")
        for memory in memories:
            print(f"- {memory.get('content')} (ID: {memory.get('memory_id')})")
    else:
        print(f"Error: {response.get('error')}")
```

### Generate Embedding

```python
async def generate_embedding(self, text: str) -> Dict[str, Any]:
    """Generate an embedding for the provided text."""
    async with self.session.post(
        f"{self.base_url}/generate_embedding",
        json={"text": text}
    ) as response:
        return await response.json()

# Example Usage
async def example_generate_embedding(client: SynthiansClient):
    response = await client.generate_embedding("Paris is beautiful")
    
    if response.get("success"):
        embedding = response.get("embedding")
        dimensions = response.get("dimensions")
        print(f"Generated embedding with {dimensions} dimensions")
        print(f"First 5 values: {embedding[:5]}")
    else:
        print(f"Error: {response.get('error')}")
```

## 3. Assembly Operations

### List Assemblies

```python
async def list_assemblies(
    self,
    limit: int = 100,
    offset: int = 0,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    """List all memory assemblies."""
    params = {
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by,
        "sort_order": sort_order
    }
    
    async with self.session.get(
        f"{self.base_url}/assemblies", params=params
    ) as response:
        return await response.json()

# Example Usage
async def example_list_assemblies(client: SynthiansClient):
    response = await client.list_assemblies(limit=5)
    
    if response.get("success"):
        assemblies = response.get("assemblies", [])
        print(f"Found {len(assemblies)} assemblies:")
        for assembly in assemblies:
            print(f"- {assembly.get('name')} (ID: {assembly.get('assembly_id')})")
    else:
        print(f"Error: {response.get('error')}")
```

### Get Assembly Details

```python
async def get_assembly(self, assembly_id: str) -> Dict[str, Any]:
    """Get details of a specific assembly."""
    async with self.session.get(
        f"{self.base_url}/assemblies/{assembly_id}"
    ) as response:
        return await response.json()

# Example Usage
async def example_get_assembly(client: SynthiansClient, assembly_id: str):
    response = await client.get_assembly(assembly_id)
    
    if response.get("success"):
        assembly = response.get("assembly", {})
        print(f"Assembly: {assembly.get('name')} (ID: {assembly.get('assembly_id')})")
        print(f"Memory Count: {len(assembly.get('memory_ids', []))}")
        print(f"Created At: {assembly.get('created_at')}")
        print(f"Updated At: {assembly.get('updated_at')}")
        
        # Display merged_from ancestry (Phase 5.9)
        if "merged_from" in assembly and assembly["merged_from"]:
            print(f"Merged From: {', '.join(assembly['merged_from'])}")
    else:
        print(f"Error: {response.get('error')}")
```

## 4. Advanced Features (Updated for 5.9)

### Feedback and Contradiction Detection

```python
async def provide_feedback(
    self, 
    memory_id: str, 
    feedback_type: str, 
    value: float
) -> Dict[str, Any]:
    """Provide feedback on a memory."""
    async with self.session.post(
        f"{self.base_url}/feedback",
        json={"memory_id": memory_id, "feedback_type": feedback_type, "value": value}
    ) as response:
        return await response.json()

async def detect_contradictions(self, content: str) -> Dict[str, Any]:
    """Detect contradictions between content and existing memories."""
    async with self.session.post(
        f"{self.base_url}/detect_contradictions",
        json={"content": content}
    ) as response:
        return await response.json()

# Example Usage
async def example_feedback(client: SynthiansClient, memory_id: str):
    response = await client.provide_feedback(
        memory_id=memory_id,
        feedback_type="surprise",
        value=0.9
    )
    print(f"Updated QuickRecal: {response.get('updated_quick_recal_score')}")

async def example_contradictions(client: SynthiansClient):
    response = await client.detect_contradictions(
        content="Paris is the capital of Germany"
    )
    
    if response.get("success"):
        contradictions = response.get("contradictions", [])
        if contradictions:
            print(f"Found {len(contradictions)} contradictions:")
            for contradiction in contradictions:
                print(f"- {contradiction.get('content')}")
                print(f"  Score: {contradiction.get('contradiction_score')}")
                print(f"  Explanation: {contradiction.get('explanation')}")
        else:
            print("No contradictions found")
    else:
        print(f"Error: {response.get('error')}")
```

### Transcription Processing

```python
async def process_transcription(
    self, 
    transcription: str, 
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a transcription, breaking it into chunks and storing as memories."""
    if metadata is None:
        metadata = {}
    
    async with self.session.post(
        f"{self.base_url}/process_transcription",
        json={"transcription": transcription, "metadata": metadata}
    ) as response:
        return await response.json()

# Example Usage
async def example_transcription(client: SynthiansClient):
    transcription = """
    This is a meeting transcription.
    We discussed several topics including the new product launch.
    The marketing team will prepare materials by next week.
    """
    
    response = await client.process_transcription(
        transcription=transcription,
        metadata={"source": "meeting", "participants": ["Alice", "Bob"]}
    )
    
    if response.get("success"):
        print(f"Created {response.get('chunk_count')} memory chunks")
        print(f"Memory IDs: {response.get('memory_ids')}")
    else:
        print(f"Error: {response.get('error')}")
```

### **(NEW)** Explainability Endpoints (Requires `ENABLE_EXPLAINABILITY=true`)

```python
async def explain_activation(self, assembly_id: str, memory_id: Optional[str] = None) -> Dict[str, Any]:
    """Explain assembly activation."""
    params = {"memory_id": memory_id} if memory_id else {}
    async with self.session.get(
        f"{self.base_url}/assemblies/{assembly_id}/explain_activation", params=params
    ) as response:
        return await response.json()

async def explain_merge(self, assembly_id: str) -> Dict[str, Any]:
    """Explain assembly merge."""
    async with self.session.get(
        f"{self.base_url}/assemblies/{assembly_id}/explain_merge"
    ) as response:
        return await response.json()

async def get_lineage(self, assembly_id: str) -> Dict[str, Any]:
    """Get assembly lineage."""
    async with self.session.get(
        f"{self.base_url}/assemblies/{assembly_id}/lineage"
    ) as response:
        return await response.json()

# Example Usage
async def explainability_example(client: SynthiansClient, assembly_id: str, memory_id: Optional[str] = None):
    # --- Explain Activation ---
    try:
        activation_explain = await client.explain_activation(assembly_id, memory_id=memory_id)
        
        if activation_explain.get("success"):
            explanation = activation_explain.get("explanation", {})
            print(f"Activation Explanation:")
            print(f"  Assembly: {explanation.get('assembly_id')}")
            if explanation.get('memory_id'):
                print(f"  Memory: {explanation.get('memory_id')}")
            print(f"  Timestamp: {explanation.get('check_timestamp')}")
            print(f"  Similarity: {explanation.get('calculated_similarity')}")
            print(f"  Threshold: {explanation.get('activation_threshold')}")
            print(f"  Passed: {explanation.get('passed_threshold')}")
            print(f"  Notes: {explanation.get('notes')}")
        else:
            print(f"Error: {activation_explain.get('error')}")
    except Exception as e:
        print(f"Error accessing activation endpoint: {e}")
        print("Ensure ENABLE_EXPLAINABILITY=true is set on the server")
    
    # --- Explain Merge ---
    try:
        merge_explain = await client.explain_merge(assembly_id)
        
        if merge_explain.get("success"):
            explanation = merge_explain.get("explanation", {})
            print(f"\nMerge Explanation:")
            if "notes" in explanation and explanation["notes"] == "Assembly was not formed by a merge.":
                print(f"  {explanation['notes']}")
            else:
                print(f"  Target Assembly: {explanation.get('target_assembly_id')}")
                print(f"  Merge Event ID: {explanation.get('merge_event_id')}")
                print(f"  Timestamp: {explanation.get('merge_timestamp')}")
                print(f"  Source Assemblies: {', '.join(explanation.get('source_assembly_ids', []))}")
                print(f"  Similarity: {explanation.get('similarity_at_merge')}")
                print(f"  Threshold: {explanation.get('threshold_at_merge')}")
        else:
            print(f"Error: {merge_explain.get('error')}")
    except Exception as e:
        print(f"Error accessing merge explanation endpoint: {e}")
    
    # --- Get Lineage ---
    try:
        lineage_response = await client.get_lineage(assembly_id)
        
        if lineage_response.get("success"):
            lineage = lineage_response.get("lineage", [])
            print(f"\nAssembly Lineage:")
            for entry in lineage:
                indent = "  " * entry.get("depth", 0)
                print(f"{indent}- {entry.get('name')} (ID: {entry.get('assembly_id')})")
            
            if lineage_response.get("max_depth_reached"):
                print("  (Maximum depth reached, lineage may be truncated)")
        else:
            print(f"Error: {lineage_response.get('error')}")
    except Exception as e:
        print(f"Error accessing lineage endpoint: {e}")
```

### **(NEW)** Diagnostics Endpoints (Requires `ENABLE_EXPLAINABILITY=true`)

```python
async def get_merge_log(self, limit: int = 100) -> Dict[str, Any]:
    """Get recent merge log entries."""
    params = {"limit": limit}
    async with self.session.get(
        f"{self.base_url}/diagnostics/merge_log", params=params
    ) as response:
        return await response.json()

async def get_runtime_config(self, service_name: str = "memory-core") -> Dict[str, Any]:
    """Get runtime configuration for a service."""
    async with self.session.get(
        f"{self.base_url}/config/runtime/{service_name}"
    ) as response:
        return await response.json()

# Example Usage
async def diagnostics_example(client: SynthiansClient):
    # --- Get Merge Log ---
    try:
        merge_log = await client.get_merge_log(limit=5)
        
        if merge_log.get("success"):
            entries = merge_log.get("log_entries", [])
            print(f"Recent Merge Events ({len(entries)}):")
            for entry in entries:
                print(f"  Event: {entry.get('merge_event_id')}")
                print(f"  Timestamp: {entry.get('timestamp')}")
                print(f"  Target: {entry.get('target_assembly_id')}")
                print(f"  Source: {', '.join(entry.get('source_assembly_ids', []))}")
                print(f"  Similarity: {entry.get('similarity_at_merge')}")
                print(f"  Threshold: {entry.get('merge_threshold')}")
                print(f"  Outcome: {entry.get('outcome')}")
                print("  ---")
        else:
            print(f"Error: {merge_log.get('error')}")
    except Exception as e:
        print(f"Error accessing merge log endpoint: {e}")
        print("Ensure ENABLE_EXPLAINABILITY=true is set on the server")
    
    # --- Get Runtime Config ---
    try:
        for service in ["memory-core", "neural-memory", "cce"]:
            config = await client.get_runtime_config(service)
            
            if config.get("success"):
                print(f"\nRuntime Configuration for {service}:")
                for key, value in config.get("config", {}).items():
                    print(f"  {key}: {value}")
            else:
                print(f"Error retrieving {service} config: {config.get('error')}")
    except Exception as e:
        print(f"Error accessing runtime config endpoint: {e}")
```

## 5. Error Handling

```python
async def example_with_error_handling(client: SynthiansClient):
    try:
        response = await client.process_memory(
            content="Paris is the capital of France",
            metadata={"source": "wikipedia"}
        )
        
        if response.get("success"):
            print(f"Memory ID: {response.get('memory_id')}")
        else:
            error = response.get("error", "Unknown error")
            details = response.get("details", {})
            print(f"API Error: {error}")
            if details:
                print(f"Details: {details}")
    except aiohttp.ClientError as e:
        print(f"HTTP Error: {e}")
    except asyncio.TimeoutError:
        print("Request timed out")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

## 6. Best Practices

- **Connection Management:** Use the client as an async context manager to ensure proper session cleanup.
  ```python
  async with SynthiansClient() as client:
      result = await client.process_memory(...)
  ```

- **Error Handling:** Always check the `success` field in responses and handle errors appropriately.

- **Rate Limiting:** Implement backoff logic for rate-limited requests if making many calls.

- **Feature Flags:** When using explainability endpoints, check if they're enabled on the server by testing a request and handling any `403 Forbidden` responses gracefully.

- **Resource Cleanup:** Ensure all sessions are properly closed, especially when handling exceptions.

## 7. Complete Example

```python
import asyncio
from typing import Dict, List, Any, Optional
import aiohttp

class SynthiansClient:
    # ... (include all methods defined above)

async def main():
    async with SynthiansClient() as client:
        # Basic operations
        print("\n=== Process Memory ===")
        process_result = await client.process_memory(
            content="The sky appears blue due to Rayleigh scattering of sunlight.",
            metadata={"source": "science", "type": "fact", "tags": ["physics", "optics"]}
        )
        
        if process_result.get("success"):
            memory_id = process_result.get("memory_id")
            print(f"Created memory: {memory_id}")
            
            # Get assemblies
            print("\n=== List Assemblies ===")
            assemblies_result = await client.list_assemblies(limit=3)
            
            if assemblies_result.get("success") and assemblies_result.get("assemblies"):
                assembly_id = assemblies_result.get("assemblies")[0].get("assembly_id")
                print(f"First assembly ID: {assembly_id}")
                
                # Try explainability features if available
                print("\n=== Explainability Features ===")
                await explainability_example(client, assembly_id, memory_id)
                
                # Try diagnostics features if available  
                print("\n=== Diagnostics Features ===")
                await diagnostics_example(client)
            
            # Retrieve similar memories
            print("\n=== Retrieve Memories ===")
            retrieve_result = await client.retrieve_memories(
                query="blue sky optics",
                limit=3
            )
            
            if retrieve_result.get("success"):
                memories = retrieve_result.get("memories", [])
                print(f"Found {len(memories)} related memories")
                for memory in memories:
                    print(f"- {memory.get('content')}")
        else:
            print(f"Error: {process_result.get('error')}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 8. Using with Alternative HTTP Clients

While the examples use `aiohttp`, the patterns can be adapted to other HTTP client libraries:

### httpx (Async)

```python
import httpx

class SynthiansHttpxClient:
    def __init__(self, base_url: str = "http://localhost:5010"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def close(self):
        await self.client.aclose()
    
    async def process_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
        
        response = await self.client.post(
            f"{self.base_url}/process_memory",
            json={"content": content, "metadata": metadata}
        )
        return response.json()

# Usage
async def httpx_example():
    client = SynthiansHttpxClient()
    try:
        result = await client.process_memory("Example memory", {"source": "test"})
        print(result)
    finally:
        await client.close()
```

### requests (Sync)

```python
import requests

class SynthiansSyncClient:
    def __init__(self, base_url: str = "http://localhost:5010"):
        self.base_url = base_url
    
    def process_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if metadata is None:
            metadata = {}
        
        response = requests.post(
            f"{self.base_url}/process_memory",
            json={"content": content, "metadata": metadata}
        )
        return response.json()

# Usage
def sync_example():
    client = SynthiansSyncClient()
    result = client.process_memory("Example memory", {"source": "test"})
    print(result)
