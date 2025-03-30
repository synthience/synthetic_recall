# synthians_memory_core/api/client/client.py

import sys
import json
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
import aiohttp
import argparse
from datetime import datetime

class SynthiansClient:
    """A simple client for testing the Synthians Memory Core API."""
    
    def __init__(self, base_url: str = "http://localhost:5010"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check if the server is healthy."""
        async with self.session.get(f"{self.base_url}/health") as response:
            return await response.json()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        async with self.session.get(f"{self.base_url}/stats") as response:
            return await response.json()
    
    async def process_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process and store a new memory."""
        payload = {
            "content": content,
            "metadata": metadata or {}
        }
        async with self.session.post(
            f"{self.base_url}/process_memory", json=payload
        ) as response:
            return await response.json()
    
    async def retrieve_memories(self, query: str, top_k: int = 5, 
                               user_emotion: Optional[Dict[str, Any]] = None,
                               cognitive_load: float = 0.5,
                               threshold: Optional[float] = None,
                               metadata_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieve relevant memories."""
        payload = {
            "query": query,
            "top_k": top_k,
            "user_emotion": user_emotion,
            "cognitive_load": cognitive_load,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if metadata_filter is not None:
            payload["metadata_filter"] = metadata_filter
        async with self.session.post(
            f"{self.base_url}/retrieve_memories", json=payload
        ) as response:
            return await response.json()
    
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        """Generate embedding for text."""
        payload = {"text": text}
        async with self.session.post(
            f"{self.base_url}/generate_embedding", json=payload
        ) as response:
            return await response.json()
    
    async def calculate_quickrecal(self, text: str = None, embedding: List[float] = None, 
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate QuickRecal score."""
        payload = {
            "text": text,
            "embedding": embedding,
            "context": context or {}
        }
        async with self.session.post(
            f"{self.base_url}/calculate_quickrecal", json=payload
        ) as response:
            return await response.json()
    
    async def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """Analyze emotional content of text."""
        payload = {"text": text}
        async with self.session.post(
            f"{self.base_url}/analyze_emotion", json=payload
        ) as response:
            return await response.json()
    
    async def provide_feedback(self, memory_id: str, similarity_score: float, 
                             was_relevant: bool) -> Dict[str, Any]:
        """Provide feedback on memory retrieval."""
        payload = {
            "memory_id": memory_id,
            "similarity_score": similarity_score,
            "was_relevant": was_relevant
        }
        async with self.session.post(
            f"{self.base_url}/provide_feedback", json=payload
        ) as response:
            return await response.json()
    
    async def detect_contradictions(self, threshold: float = 0.75) -> Dict[str, Any]:
        """Detect potential contradictions in memories."""
        async with self.session.post(
            f"{self.base_url}/detect_contradictions?threshold={threshold}"
        ) as response:
            return await response.json()
    
    async def get_memory_by_id(self, memory_id: str) -> Dict[str, Any]:
        """Retrieve a specific memory by its ID."""
        async with self.session.get(
            f"{self.base_url}/api/memories/{memory_id}"
        ) as response:
            return await response.json()
    
    async def process_transcription(self, text: str, audio_metadata: Dict[str, Any] = None, 
                                  importance: float = None) -> Dict[str, Any]:
        """Process a transcription with audio features."""
        payload = {
            "text": text,
            "audio_metadata": audio_metadata or {},
        }
        if importance is not None:
            payload["importance"] = importance
            
        async with self.session.post(
            f"{self.base_url}/process_transcription", json=payload
        ) as response:
            return await response.json()


async def run_tests(client: SynthiansClient):
    """Run a series of tests to verify API functionality."""
    print("Running API tests...\n")
    
    try:
        print("1. Health Check Test")
        health = await client.health_check()
        print(f"Health check result: {json.dumps(health, indent=2)}\n")
        
        print("2. Stats Test")
        stats = await client.get_stats()
        print(f"Stats result: {json.dumps(stats, indent=2)}\n")
        
        print("3. Embedding Generation Test")
        embed_resp = await client.generate_embedding("Testing the embedding generation API")
        if embed_resp["success"]:
            embed_dim = len(embed_resp["embedding"])
            print(f"Successfully generated embedding with dimension {embed_dim}\n")
        else:
            print(f"Failed to generate embedding: {embed_resp.get('error')}\n")
        
        print("4. QuickRecal Calculation Test")
        qr_resp = await client.calculate_quickrecal(text="Testing the QuickRecal API")
        print(f"QuickRecal result: {json.dumps(qr_resp, indent=2)}\n")
        
        print("5. Emotion Analysis Test")
        emotion_resp = await client.analyze_emotion("I am feeling very happy today")
        print(f"Emotion analysis result: {json.dumps(emotion_resp, indent=2)}\n")
        
        print("6. Memory Processing Test")
        mem_resp = await client.process_memory(
            content="This is a test memory created at " + datetime.now().isoformat(),
            metadata={"source": "test_client", "importance": 0.8}
        )
        print(f"Memory processing result: {json.dumps(mem_resp, indent=2)}\n")
        
        if mem_resp.get("success"):
            memory_id = mem_resp.get("memory_id")
            
            print("7. Memory Retrieval Test")
            retrieve_resp = await client.retrieve_memories("test memory", top_k=3)
            print(f"Memory retrieval result: {json.dumps(retrieve_resp, indent=2)}\n")
            
            print("8. Feedback Test")
            feedback_resp = await client.provide_feedback(
                memory_id=memory_id,
                similarity_score=0.85,
                was_relevant=True
            )
            print(f"Feedback result: {json.dumps(feedback_resp, indent=2)}\n")
        
        print("9. Contradiction Detection Test")
        contradict_resp = await client.detect_contradictions(threshold=0.7)
        print(f"Contradiction detection result: {json.dumps(contradict_resp, indent=2)}\n")
        
        print("All tests completed.")
    except Exception as e:
        print(f"Test failed with error: {str(e)}")


async def main():
    parser = argparse.ArgumentParser(description="Synthians Memory Core API Client")
    parser.add_argument("--url", default="http://localhost:5010", help="API server URL")
    parser.add_argument("--action", choices=["test", "health", "stats", "add", "retrieve", "embedding", "quickrecal", "emotion"], 
                       default="test", help="Action to perform")
    parser.add_argument("--query", help="Query for memory retrieval")
    parser.add_argument("--content", help="Content for memory processing or analysis")
    parser.add_argument("--metadata", help="JSON metadata string for memory processing")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to return for memory retrieval")
    parser.add_argument("--cognitive_load", type=float, default=0.5, help="Cognitive load for memory retrieval")
    parser.add_argument("--threshold", type=float, help="Threshold for memory retrieval")
    
    args = parser.parse_args()
    
    async with SynthiansClient(base_url=args.url) as client:
        if args.action == "test":
            await run_tests(client)
        
        elif args.action == "health":
            result = await client.health_check()
            print(json.dumps(result, indent=2))
        
        elif args.action == "stats":
            result = await client.get_stats()
            print(json.dumps(result, indent=2))
        
        elif args.action == "add" and args.content:
            metadata = {}
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    print("Error: metadata must be valid JSON")
                    return
            
            result = await client.process_memory(content=args.content, metadata=metadata)
            print(json.dumps(result, indent=2))
        
        elif args.action == "retrieve" and args.query:
            result = await client.retrieve_memories(
                query=args.query, 
                top_k=args.top_k,
                cognitive_load=args.cognitive_load,
                threshold=args.threshold if hasattr(args, 'threshold') and args.threshold is not None else None
            )
            print(json.dumps(result, indent=2))
        
        elif args.action == "embedding" and args.content:
            result = await client.generate_embedding(text=args.content)
            print(json.dumps(result, indent=2))
        
        elif args.action == "quickrecal" and args.content:
            result = await client.calculate_quickrecal(text=args.content)
            print(json.dumps(result, indent=2))
        
        elif args.action == "emotion" and args.content:
            result = await client.analyze_emotion(text=args.content)
            print(json.dumps(result, indent=2))
        
        else:
            print("Invalid action or missing required arguments")
            parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
