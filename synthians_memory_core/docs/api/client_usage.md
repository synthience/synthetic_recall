# Memory Core Python Client Usage Guide

*This document provides examples and best practices for using the asynchronous Python client (`SynthiansClient`) to interact with the Synthians Memory Core API.*

## 1. Installation & Setup

Ensure the client class is accessible within your project.

```python
import asyncio
import numpy as np
import json # Added for pretty printing
from synthians_memory_core.api.client.client import SynthiansClient

# Initialize the client within an async context
async def main():
    # Use default localhost and port unless configured otherwise
    client_instance = SynthiansClient(base_url="http://localhost:5010")
    async with client_instance as client:
        # --- Use client methods here ---
        print("Client initialized.")
        # Example calls (uncomment to run)
        # await store_example(client)
        # await retrieve_example(client)
        # await get_by_id_example(client, "some_memory_id") # Replace with a real ID
        # await embedding_example(client)
        # await emotion_example(client)
        # await quickrecal_example(client)
        # await feedback_example(client, "some_memory_id") # Replace with a real ID
        # await contradict_example(client)
        # await transcription_example(client)
        # await safe_call(client)
        print("Client operations finished.")

# To run the examples:
# if __name__ == "__main__":
#     asyncio.run(main())
```

## 2. Basic Operations

### Storing a Memory

The server handles embedding generation if not provided. Metadata is enriched server-side.

```python
async def store_example(client: SynthiansClient):
    # Store with content only (embedding generated server-side)
    response1 = await client.process_memory(
        content="This is an important memory about project Alpha."
    )
    if response1.get("success"):
        print(f"Stored memory 1 with ID: {response1['memory_id']}")
        return response1['memory_id'] # Return ID for potential use later
    else:
        print(f"Failed to store memory 1: {response1.get('error')}")
        return None

    # Store with custom metadata
    response2 = await client.process_memory(
        content="Meeting notes regarding the Q3 roadmap.",
        metadata={
            "source": "meeting_notes",
            "project": "RoadmapQ3",
            "attendees": ["Alice", "Bob"]
        }
    )
    if response2.get("success"):
        print(f"Stored memory 2 with ID: {response2['memory_id']}")
        print(f"  -> Returned metadata: {response2.get('metadata')}") # Note enriched metadata
        return response2['memory_id']
    else:
        print(f"Failed to store memory 2: {response2.get('error')}")
        return None
```

### Retrieving Memories

Retrieve memories based on a text query. Emotional gating and adaptive thresholding are applied server-side if configured.

```python
async def retrieve_example(client: SynthiansClient):
    # Basic retrieval by query
    response1 = await client.retrieve_memories(
        query="project Alpha roadmap",
        top_k=3
    )
    if response1.get("success"):
        print(f"
Retrieved {len(response1['memories'])} memories for 'project Alpha roadmap':")
        for i, memory in enumerate(response1['memories']):
            print(f"  {i+1}. ID: {memory.get('id')}, Score: {memory.get('similarity'):.4f}, Content: {memory.get('content', '')[:60]}...")
    else:
        print(f"Retrieval failed: {response1.get('error')}")

    # Retrieve with metadata filtering
    response2 = await client.retrieve_memories(
        query="meeting notes",
        top_k=5,
        metadata_filter={
            "source": "meeting_notes",
            "project": "RoadmapQ3"
        }
    )
    if response2.get("success"):
        print(f"
Retrieved {len(response2['memories'])} memories matching metadata filter:")
        for memory in response2['memories']:
             print(f"  - ID: {memory.get('id')}, Source: {memory.get('metadata', {}).get('source')}")
    else:
        print(f"Metadata retrieval failed: {response2.get('error')}")


    # Retrieve with emotional context (provide dominant emotion)
    response3 = await client.retrieve_memories(
        query="important decision",
        user_emotion={"dominant_emotion": "focused"}, # Or just "focused"
        cognitive_load=0.3, # Lower value allows more results through gating
        top_k=3
    )
    if response3.get("success"):
        print(f"
Retrieved {len(response3['memories'])} memories with 'focused' emotion:")
        # Check 'emotional_resonance' or 'final_score' if available
        for memory in response3['memories']:
            print(f"  - ID: {memory.get('id')}, Resonance: {memory.get('emotional_resonance', 'N/A')}")
    else:
        print(f"Emotional retrieval failed: {response3.get('error')}")

    # Retrieve with explicit threshold override
    response4 = await client.retrieve_memories(
        query="roadmap",
        threshold=0.1, # Very low threshold for broad recall
        top_k=10
    )
    if response4.get("success"):
         print(f"
Retrieved {len(response4['memories'])} memories with low threshold (0.1):")
    else:
         print(f"Low threshold retrieval failed: {response4.get('error')}")

```

### Retrieving a Specific Memory by ID

```python
async def get_by_id_example(client: SynthiansClient, memory_id: str):
    if not memory_id:
        print("Cannot retrieve by ID: memory_id is missing.")
        return

    response = await client.get_memory_by_id(memory_id) # Corrected method name
    if response.get("success") and response.get("memory"):
        print(f"
Successfully retrieved memory by ID {memory_id}:")
        # Use json.dumps for readable output of the memory dict
        print(json.dumps(response["memory"], indent=2, default=str)) # Use default=str for non-serializable types like datetime
    elif not response.get("success") and "not found" in response.get("error", "").lower(): # Check error message for 404
         print(f"
Memory with ID {memory_id} not found.")
    else:
        print(f"
Failed to retrieve memory by ID {memory_id}: {response.get('error')}")

```

## 3. Utility Endpoints

### Generating Embeddings

```python
async def embedding_example(client: SynthiansClient):
    response = await client.generate_embedding("Generate an embedding for this sentence.")
    if response.get("success"):
        embedding = response.get("embedding")
        dimension = response.get("dimension")
        print(f"
Generated embedding (Dimension: {dimension}): {str(embedding)[:100]}...") # Print snippet
    else:
        print(f"
Failed to generate embedding: {response.get('error')}")
```

### Analyzing Emotion

```python
async def emotion_example(client: SynthiansClient):
    response = await client.analyze_emotion("This is a surprisingly complex and intriguing challenge!")
    if response.get("success"):
        print(f"
Emotion Analysis Result:")
        print(f"  Dominant: {response.get('dominant_emotion')}")
        print(f"  Scores: {response.get('emotions')}")
    else:
        print(f"
Failed to analyze emotion: {response.get('error')}")
```

### Calculating QuickRecal Score

```python
async def quickrecal_example(client: SynthiansClient):
    # Calculate score for text (embedding generated server-side)
    response1 = await client.calculate_quickrecal(
        text="Calculate the relevance score for this piece of text.",
        context={"importance": 0.7, "source": "user_query"}
    )
    if response1.get("success"):
        print(f"
QuickRecal Score (from text): {response1.get('quickrecal_score'):.4f}")
        print(f"  Factors: {response1.get('factors')}")
    else:
        print(f"
Failed QuickRecal calculation (text): {response1.get('error')}")

    # Calculate score for pre-computed embedding
    embedding_resp = await client.generate_embedding("Some other text")
    if embedding_resp.get("success"):
        embedding = embedding_resp.get("embedding")
        response2 = await client.calculate_quickrecal(embedding=embedding)
        if response2.get("success"):
             print(f"
QuickRecal Score (from embedding): {response2.get('quickrecal_score'):.4f}")
        else:
             print(f"
Failed QuickRecal calculation (embedding): {response2.get('error')}")

```

## 4. Advanced Features

### Providing Feedback

Used to tune the adaptive retrieval threshold.

```python
async def feedback_example(client: SynthiansClient, memory_id: str):
    if not memory_id:
        print("Cannot provide feedback: memory_id is missing.")
        return

    # Example: Assume a memory was retrieved with score 0.82 and user found it relevant
    response = await client.provide_feedback(
        memory_id=memory_id,
        similarity_score=0.82,
        was_relevant=True
    )
    if response.get("success"):
        print(f"
Feedback provided successfully. New threshold: {response.get('new_threshold'):.4f}")
    else:
        print(f"
Failed to provide feedback: {response.get('error')}")

```

### Detecting Contradictions

```python
async def contradict_example(client: SynthiansClient):
    # Add potentially contradictory memories first
    await client.process_memory("Statement A implies outcome X.")
    await client.process_memory("Statement B prevents outcome X.")
    await asyncio.sleep(1.0) # Allow indexing time

    response = await client.detect_contradictions(threshold=0.7)
    if response.get("success"):
        print(f"
Contradiction Detection Found: {response.get('count')} potential contradictions.")
        # Pretty print the list of contradictions
        print(json.dumps(response.get("contradictions", []), indent=2))
    else:
         print(f"
Contradiction detection failed: {response.get('error')}")
```

### Processing Transcriptions

Enriches transcriptions with audio features before storing.

```python
async def transcription_example(client: SynthiansClient):
    # Assuming client has process_transcription method
    if not hasattr(client, 'process_transcription'):
        print("
Skipping transcription example: client.process_transcription not implemented.")
        return

    response = await client.process_transcription(
        text="Okay, let me think... The first point is about integration, and the second... involves the API.",
        audio_metadata={
            "duration_sec": 6.8,
            "speaking_rate": 2.5, # Words per second, example
            "pause_count": 3,
            "longest_pause_ms": 800,
            "was_interrupted": False
        },
        importance=0.8 # Optional importance score
    )
    if response.get("success"):
        print("
Transcription processed successfully:")
        # Pretty print the response
        print(json.dumps(response, indent=2))
    else:
        print(f"
Failed to process transcription: {response.get('error')}")

```

## 5. Error Handling

The client methods return dictionaries. Check the `"success"` key (usually boolean) or look for an `"error"` key. Handle potential `aiohttp` exceptions.

```python
import aiohttp # Import for exception handling

async def safe_call(client: SynthiansClient):
    try:
        response = await client.health_check()
        if response.get("status") == "healthy":
            print("Server is healthy.")
        # Handle structured error response from health check
        elif "error" in response:
            print(f"Health check failed: {response['error']} (Status: {response.get('status')})")
        else:
            print(f"Health check returned unexpected response: {response}")

        # Example of handling potential failure during retrieval
        retrieve_response = await client.retrieve_memories("nonexistent query for testing")
        if not retrieve_response.get("success"):
             print(f"Retrieval Error: {retrieve_response.get('error')}")
        else:
             print("Retrieval successful (may return 0 memories).")

    except aiohttp.ClientConnectorError as e:
        print(f"Connection Error: Could not connect to the server at {client.base_url}. Is it running? ({e})")
    except aiohttp.ClientResponseError as e:
        print(f"HTTP Error: Received status {e.status} from server. Message: {e.message}")
    except asyncio.TimeoutError:
        print(f"Request Timeout: The request to {client.base_url} timed out.")
    except Exception as e:
        # Catch other unexpected client-side or parsing errors
        print(f"An unexpected error occurred: {e}")

```

## 6. Best Practices

1.  **Use Async Context Manager:** Always use `async with SynthiansClient(...) as client:` to ensure the `aiohttp.ClientSession` is properly managed and closed.
2.  **Check Responses:** Robustly check the `"success"` or `"error"` keys in the returned dictionary before assuming an operation succeeded. Handle potential `None` returns or missing keys.
3.  **Rate Limiting:** Be mindful of server load. Avoid sending extremely high volumes of requests without appropriate delays or batching strategies (client doesn't implement batching itself). Use `asyncio.sleep()` if needed.
4.  **Metadata:** Use meaningful and structured metadata when storing memories to improve filtering, context, and retrieval relevance.
5.  **Thresholds:** Understand the impact of the `threshold` parameter in `retrieve_memories`. Lower values increase recall but may decrease precision. Use the feedback mechanism if adaptive thresholding is enabled on the server.
6.  **Error Logging:** Implement robust logging on the client-side to capture API errors, unexpected responses, and connection issues. Use the specific `aiohttp` exceptions for better diagnostics.
7.  **Embedding Handling:** Be aware that the server handles embedding generation and dimension alignment. Provide pre-computed embeddings only if necessary and ensure they are valid lists of floats.
