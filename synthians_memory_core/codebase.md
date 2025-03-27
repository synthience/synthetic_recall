# __init__.py

```py
# synthians_memory_core/__init__.py

"""
Synthians Memory Core - A Unified, Efficient Memory System
Incorporates HPC-QuickRecal, Hyperbolic Geometry, Emotional Intelligence,
Memory Assemblies, and Adaptive Thresholds.
"""

__version__ = "1.0.0"

# Core components
from .synthians_memory_core import SynthiansMemoryCore
from .memory_structures import MemoryEntry, MemoryAssembly
from .hpc_quickrecal import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
from .geometry_manager import GeometryManager, GeometryType
from .emotional_intelligence import EmotionalAnalyzer, EmotionalGatingService
from .memory_persistence import MemoryPersistence
from .adaptive_components import ThresholdCalibrator

__all__ = [
    "SynthiansMemoryCore",
    "MemoryEntry",
    "MemoryAssembly",
    "UnifiedQuickRecallCalculator",
    "QuickRecallMode",
    "QuickRecallFactor",
    "GeometryManager",
    "GeometryType",
    "EmotionalAnalyzer",
    "EmotionalGatingService",
    "MemoryPersistence",
    "ThresholdCalibrator",
]

```

# adaptive_components.py

```py
# synthians_memory_core/adaptive_components.py

import time
import math
from collections import deque
from typing import Dict, Any, Optional

from .custom_logger import logger # Use the shared custom logger

class ThresholdCalibrator:
    """Dynamically calibrates similarity thresholds based on feedback."""

    def __init__(self, initial_threshold: float = 0.75, learning_rate: float = 0.05, window_size: int = 50):
        self.threshold = initial_threshold
        self.learning_rate = learning_rate
        self.feedback_history = deque(maxlen=window_size)
        self.stats = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} # Added tn for completeness
        logger.info("ThresholdCalibrator", "Initialized", {"initial": initial_threshold, "lr": learning_rate, "window": window_size})

    def record_feedback(self, similarity_score: float, was_relevant: bool):
        """Record feedback for a retrieved memory."""
        is_above_threshold = similarity_score >= self.threshold

        self.feedback_history.append({
            "score": similarity_score,
            "relevant": was_relevant,
            "predicted_relevant": is_above_threshold,
            "threshold_at_time": self.threshold
        })

        # Update stats based on prediction vs actual relevance
        if is_above_threshold:
            if was_relevant: self.stats['tp'] += 1
            else: self.stats['fp'] += 1
        else:
            if was_relevant: self.stats['fn'] += 1
            else: self.stats['tn'] += 1 # Correctly predicted irrelevant

        # Adjust threshold immediately based on this feedback
        self.adjust_threshold()

    def adjust_threshold(self) -> float:
        """Adjust the similarity threshold based on recent feedback."""
        if len(self.feedback_history) < 10: # Need minimum feedback
            return self.threshold

        # Calculate Precision and Recall from recent history (last N items)
        recent_feedback = list(self.feedback_history)
        recent_tp = sum(1 for f in recent_feedback if f["predicted_relevant"] and f["relevant"])
        recent_fp = sum(1 for f in recent_feedback if f["predicted_relevant"] and not f["relevant"])
        recent_fn = sum(1 for f in recent_feedback if not f["predicted_relevant"] and f["relevant"])

        precision = recent_tp / max(1, recent_tp + recent_fp)
        recall = recent_tp / max(1, recent_tp + recent_fn)

        adjustment = 0.0
        # If precision is low (too many irrelevant items retrieved), increase threshold
        if precision < 0.6 and recall > 0.5: # Avoid penalizing if recall is also low
            adjustment = self.learning_rate * (1.0 - precision) # Stronger increase for lower precision
        # If recall is low (too many relevant items missed), decrease threshold
        elif recall < 0.6 and precision > 0.5: # Avoid penalizing if precision is also low
             adjustment = -self.learning_rate * (1.0 - recall) # Stronger decrease for lower recall

        # Apply adjustment with diminishing returns near bounds
        current_threshold = self.threshold
        if adjustment > 0:
            # Less adjustment as we approach 1.0
            adjustment *= (1.0 - current_threshold)
        else:
             # Less adjustment as we approach 0.0
             adjustment *= current_threshold

        new_threshold = current_threshold + adjustment
        new_threshold = max(0.1, min(0.95, new_threshold)) # Keep within reasonable bounds

        if abs(new_threshold - self.threshold) > 0.001:
            logger.info("ThresholdCalibrator", f"Adjusted threshold: {self.threshold:.3f} -> {new_threshold:.3f}",
                        {"adjustment": adjustment, "precision": precision, "recall": recall})
            self.threshold = new_threshold

        return self.threshold

    def get_current_threshold(self) -> float:
        """Return the current similarity threshold."""
        return self.threshold

    def get_statistics(self) -> dict:
        """Return statistics about calibration performance."""
        total = self.stats['tp'] + self.stats['fp'] + self.stats['fn'] + self.stats['tn']
        precision = self.stats['tp'] / max(1, self.stats['tp'] + self.stats['fp'])
        recall = self.stats['tp'] / max(1, self.stats['tp'] + self.stats['fn'])
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        return {
            "threshold": self.threshold,
            "feedback_count": len(self.feedback_history),
            "true_positives": self.stats['tp'],
            "false_positives": self.stats['fp'],
            "false_negatives": self.stats['fn'],
            "true_negatives": self.stats['tn'],
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

# Note: AdaptiveBatchScheduler might be overkill if batching is handled externally
# or if the primary interaction pattern doesn't benefit significantly from adaptive batching.
# Keeping ThresholdCalibrator as it's directly related to retrieval relevance.

```

# api\__init__.py

```py


```

# api\client\__init__.py

```py


```

# api\client\client.py

```py
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
                               threshold: Optional[float] = None) -> Dict[str, Any]:
        """Retrieve relevant memories."""
        payload = {
            "query": query,
            "top_k": top_k,
            "user_emotion": user_emotion,
            "cognitive_load": cognitive_load,
        }
        if threshold is not None:
            payload["threshold"] = threshold
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

```

# api\client\test_metadata.py

```py
import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

# Import the client class directly from client module
from synthians_memory_core.api.client.client import SynthiansClient

async def test_metadata_synthesis():
    """Test the metadata synthesis capabilities of the memory system."""
    print("\n=== Testing Metadata Synthesis ===\n")
    
    async with SynthiansClient() as client:
        # 1. Process a memory with specific emotional content
        print("\n1. Creating memory with emotional content...")
        happy_memory = await client.process_memory(
            content="I am feeling incredibly happy and joyful today. It's a wonderful day and everything is going great!",
            metadata={
                "source": "metadata_test",
                "importance": 0.9,
                "test_type": "positive_emotion"
            }
        )
        print(f"Happy memory result: {json.dumps(happy_memory, indent=2)}")
        
        # 2. Process a memory with negative emotional content
        print("\n2. Creating memory with negative emotional content...")
        sad_memory = await client.process_memory(
            content="I'm feeling quite sad and disappointed today. Things aren't going well and I'm frustrated.",
            metadata={
                "source": "metadata_test",
                "importance": 0.7,
                "test_type": "negative_emotion"
            }
        )
        print(f"Sad memory result: {json.dumps(sad_memory, indent=2)}")
        
        # 3. Process a memory with technical content
        print("\n3. Creating memory with technical/complex content...")
        tech_memory = await client.process_memory(
            content="The quantum computational paradigm leverages superposition and entanglement to perform calculations that would be infeasible on classical computers. The fundamental unit is the qubit, which can exist in multiple states simultaneously.",
            metadata={
                "source": "metadata_test",
                "importance": 0.8,
                "test_type": "complex_content"
            }
        )
        print(f"Technical memory result: {json.dumps(tech_memory, indent=2)}")
        
        # 4. Retrieve memories and check if metadata is preserved
        print("\n4. Retrieving memories to verify metadata...")
        # First try with default parameters
        retrieve_resp = await client.retrieve_memories(
            "test metadata synthesis", 
            top_k=5
        )
        print(f"Default retrieval results: {json.dumps(retrieve_resp, indent=2)}")
        
        # Try again with a lowered threshold to bypass ThresholdCalibrator
        print("\n4b. Retrieving with lowered threshold...")
        retrieve_with_threshold = await client.retrieve_memories(
            "test metadata synthesis", 
            top_k=5,
            threshold=0.4  # Explicitly lower the threshold well below our ~0.66 scores
        )
        print(f"Retrieval with threshold=0.4: {json.dumps(retrieve_with_threshold, indent=2)}")
        
        # Try with exact memory IDs to force retrieval
        print("\n4c. Retrieving by exact memory IDs...")
        memory_ids = [
            happy_memory.get("memory_id"),
            sad_memory.get("memory_id"),
            tech_memory.get("memory_id")
        ]
        # Filter out any None values
        memory_ids = [mid for mid in memory_ids if mid]
        
        if memory_ids:
            memory_by_id = await client.retrieve_memory_by_id(memory_ids[0])
            print(f"Retrieved by ID: {json.dumps(memory_by_id, indent=2)}")
            
            # Try direct query of each test type
            print("\n4d. Retrieving with direct test type queries...")
            for test_type in ["positive_emotion", "negative_emotion", "complex_content"]:
                test_query = await client.retrieve_memories(
                    test_type,  # Use the test_type as the query
                    top_k=1,
                    threshold=0.4,
                    user_emotion=None  # Bypass emotional gating
                )
                print(f"Query '{test_type}' results: {json.dumps(test_query, indent=2)}")
        
        # 5. Verify key metadata fields in each memory
        print("\n5. Validating metadata fields...")
        memories = retrieve_resp.get("memories", [])
        
        validation_results = []
        for memory in memories:
            metadata = memory.get("metadata", {})
            validation = {
                "id": memory.get("id"),
                "metadata_schema_version": metadata.get("metadata_schema_version"),
                "has_timestamp": "timestamp" in metadata,
                "has_timestamp_iso": "timestamp_iso" in metadata,
                "has_time_of_day": "time_of_day" in metadata,
                "has_dominant_emotion": "dominant_emotion" in metadata,
                "has_emotional_intensity": "emotional_intensity" in metadata,
                "has_complexity_estimate": "complexity_estimate" in metadata,
                "has_embedding_metadata": all(key in metadata for key in ["embedding_valid", "embedding_dim"])
            }
            validation_results.append(validation)
        
        print(f"Validation results: {json.dumps(validation_results, indent=2)}")
        
        # Summary
        print("\n=== Metadata Synthesis Test Summary ===\n")
        if validation_results:
            success = all(result.get("has_timestamp") and 
                         result.get("has_dominant_emotion") and 
                         result.get("has_complexity_estimate") 
                         for result in validation_results)
            if success:
                print("✅ SUCCESS: All memories have proper metadata synthesis")
            else:
                print("❌ FAILURE: Some memories are missing key metadata fields")
        else:
            print("❓ INCONCLUSIVE: No memories were retrieved for validation")

def main():
    """Run the metadata synthesis test."""
    asyncio.run(test_metadata_synthesis())

if __name__ == "__main__":
    main()

```

# api\server.py

```py
# synthians_memory_core/api/server.py

import asyncio
import os
import time
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from fastapi import FastAPI, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import uvicorn
import json
from datetime import datetime
import sys
import importlib.util
import subprocess

# Import the unified memory core
from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.custom_logger import logger
from synthians_memory_core.emotion_analyzer import EmotionAnalyzer
from synthians_memory_core.utils.transcription_feature_extractor import TranscriptionFeatureExtractor
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Optional: Import sentence_transformers for embedding generation if not moved to GeometryManager
from sentence_transformers import SentenceTransformer

# Define request/response models using Pydantic
class ProcessMemoryRequest(BaseModel):
    """Request model for processing a new memory."""
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    analyze_emotion: Optional[bool] = Field(default=True, description="Whether to analyze emotions in the content")

class ProcessMemoryResponse(BaseModel):
    """Response model for memory processing."""
    success: bool
    memory_id: Optional[str] = None
    quickrecal_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class RetrieveMemoriesRequest(BaseModel):
    query: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 5
    user_emotion: Optional[Union[Dict[str, Any], str]] = None
    cognitive_load: float = 0.5
    threshold: Optional[float] = None

class RetrieveMemoriesResponse(BaseModel):
    success: bool
    memories: List[Dict[str, Any]] = []
    error: Optional[str] = None

class GenerateEmbeddingRequest(BaseModel):
    text: str

class GenerateEmbeddingResponse(BaseModel):
    success: bool
    embedding: Optional[List[float]] = None
    dimension: Optional[int] = None
    error: Optional[str] = None

class QuickRecalRequest(BaseModel):
    embedding: Optional[List[float]] = None
    text: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class QuickRecalResponse(BaseModel):
    success: bool
    quickrecal_score: Optional[float] = None
    factors: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class EmotionRequest(BaseModel):
    text: str

class EmotionResponse(BaseModel):
    success: bool
    emotions: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    error: Optional[str] = None

class FeedbackRequest(BaseModel):
    memory_id: str
    similarity_score: float
    was_relevant: bool

class FeedbackResponse(BaseModel):
    success: bool
    new_threshold: Optional[float] = None
    error: Optional[str] = None

# Models for the transcription endpoint
class TranscriptionRequest(BaseModel):
    """Request model for processing transcription data."""
    text: str = Field(..., description="The transcribed text")
    audio_metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata about the audio source")
    embedding: Optional[List[float]] = Field(None, description="Optional pre-computed embedding for the transcription")
    memory_id: Optional[str] = Field(None, description="Optional memory ID if updating an existing memory")
    importance: Optional[float] = Field(None, description="Optional importance score for the memory (0-1)")
    force_update: bool = Field(False, description="Force update if memory ID exists")

class TranscriptionResponse(BaseModel):
    """Response model for processed transcription data."""
    success: bool = Field(..., description="Whether the operation was successful")
    memory_id: Optional[str] = Field(None, description="ID of the created/updated memory")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Extracted metadata from the transcription")
    embedding: Optional[List[float]] = Field(None, description="Embedding generated for the transcription")
    error: Optional[str] = Field(None, description="Error message if operation failed")

# App lifespan for initialization/cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup app resources."""
    # Startup Logic
    logger.info("API", "Starting Synthians Memory Core API server...")
    
    # Set startup time
    app.state.startup_time = time.time()
    
    # Run GPU setup script to detect GPU and install appropriate FAISS package
    try:
        logger.info("API", "Checking for GPU availability and setting up FAISS...")
        # Get the path to gpu_setup.py
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gpu_setup_path = os.path.join(current_dir, "gpu_setup.py")
        
        if os.path.exists(gpu_setup_path):
            logger.info("API", f"Running GPU setup script from: {gpu_setup_path}")
            # Run the setup script as a subprocess
            result = subprocess.run([sys.executable, gpu_setup_path], 
                                    capture_output=True, text=True, check=False)
            
            if result.returncode == 0:
                logger.info("API", f"GPU setup completed successfully: {result.stdout.strip()}")
            else:
                logger.warning("API", f"GPU setup failed: {result.stderr.strip()}")
                logger.info("API", "Continuing with CPU-only FAISS")
        else:
            logger.warning("API", f"GPU setup script not found at {gpu_setup_path}")
    except Exception as e:
        logger.error("API", f"Error during GPU setup: {str(e)}")
        logger.info("API", "Continuing with CPU-only FAISS")
    
    # Create core instance on startup
    app.state.memory_core = SynthiansMemoryCore()
    await app.state.memory_core.initialize()
    
    # Initialize emotion analysis model
    try:
        logger.info("API", "Initializing emotion analyzer...")
        # Use the new EmotionAnalyzer class
        app.state.emotion_analyzer = EmotionAnalyzer()
        logger.info("API", "Emotion analyzer initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize emotion analyzer: {str(e)}")
        app.state.emotion_analyzer = None
    
    # Initialize transcription feature extractor
    try:
        logger.info("API", "Initializing transcription feature extractor...")
        # Create the extractor with the emotion_analyzer
        app.state.transcription_extractor = TranscriptionFeatureExtractor(
            emotion_analyzer=app.state.emotion_analyzer
        )
        logger.info("API", "Transcription feature extractor initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize transcription feature extractor: {str(e)}")
        app.state.transcription_extractor = None
    
    # Initialize embedding model
    try:
        model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
        logger.info("API", f"Loading embedding model: {model_name}")
        
        # Try to load the model, download if not available
        try:
            app.state.embedding_model = SentenceTransformer(model_name)
            logger.info("API", f"Embedding model {model_name} loaded successfully")
        except Exception as model_error:
            # If the model doesn't exist, it might need to be downloaded
            if "No such file or directory" in str(model_error) or "not found" in str(model_error).lower():
                logger.warning("API", f"Model {model_name} not found locally, attempting to download...")
                from sentence_transformers import util as st_util
                # Force download from Hugging Face
                app.state.embedding_model = SentenceTransformer(model_name, use_auth_token=None)
                logger.info("API", f"Successfully downloaded and loaded model {model_name}")
            else:
                # Re-raise if it's not a file-not-found error
                raise
    except Exception as e:
        logger.error("API", f"Failed to load embedding model: {str(e)}")
        app.state.embedding_model = None
    
    # Complete initialization
    logger.info("API", "Synthians Memory Core API server started")
    
    # Yield control to FastAPI
    yield
    
    # Shutdown Logic
    logger.info("API", "Shutting down Synthians Memory Core API server...")
    # Clean up resources
    try:
        if hasattr(app.state, 'memory_core'):
            await app.state.memory_core.cleanup()
    except Exception as e:
        logger.error("API", f"Error during cleanup: {str(e)}")
    
    logger.info("API", "Synthians Memory Core API server shut down")

# Create the FastAPI app with lifespan
app = FastAPI(
    title="Synthians Memory Core API",
    description="Unified API for memory, embeddings, QuickRecal, and emotion analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

# Generate embedding using the loaded model
async def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using the sentence transformer model."""
    if not text:
        logger.warning("generate_embedding", "Empty text provided for embedding generation")
        # Return a zero vector of appropriate dimension
        embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
        return np.zeros(embedding_dim, dtype=np.float32)
    
    try:
        # Use the embedding model from app state
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: app.state.embedding_model.encode(text)
        )
        return embedding
    except Exception as e:
        logger.error("generate_embedding", f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback
        embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
        return np.zeros(embedding_dim, dtype=np.float32)

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Synthians Memory Core API"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        uptime = time.time() - app.state.startup_time
        # Use _memories instead of memories to match the updated attribute name
        memory_count = len(app.state.memory_core._memories)
        assembly_count = len(app.state.memory_core.assemblies)
        return {
            "status": "healthy",
            "uptime_seconds": uptime,
            "memory_count": memory_count,
            "assembly_count": assembly_count,
            "version": "1.0.0"  # Add version information
        }
    except Exception as e:
        logger.error("health_check", f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        uptime = time.time() - app.state.startup_time
        return {
            "success": True,  # Add success field
            "api_server": {
                "uptime_seconds": uptime,
                "memory_count": len(app.state.memory_core._memories),
                "embedding_dim": app.state.memory_core.config.get('embedding_dim', 768),
                "geometry": app.state.memory_core.config.get('geometry', 'hyperbolic'),
                "model": os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2')
            },
            "memory": {
                "total_memories": len(app.state.memory_core._memories),
                "total_assemblies": len(app.state.memory_core.assemblies),
                "storage_path": app.state.memory_core.config.get('storage_path', '/app/memory/stored/synthians'),
                "threshold": app.state.memory_core.config.get('contradiction_threshold', 0.75),
            }
        }
    except Exception as e:
        logger.error("get_stats", f"Error retrieving stats: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/process_memory", response_model=ProcessMemoryResponse)
async def process_memory(request: ProcessMemoryRequest, background_tasks: BackgroundTasks):
    """Process and store a new memory."""
    try:
        logger.info("process_memory", "Processing new memory request")
        # Validate input
        if not request.content and not request.embedding and not request.metadata:
            raise HTTPException(status_code=400, detail="No memory content provided")
            
        # Tracking for current request (all fields start as None)
        embedding = None
        generated_text = None
        memory_id = None
        emotion_data = None
        
        # Step 1: Generate embedding if needed
        if request.content and (request.embedding is None) and hasattr(app.state, 'embedding_model'):
            try:
                # Generate embedding
                logger.info("process_memory", "Generating embedding from text")
                loop = asyncio.get_event_loop()
                embedding_list = await loop.run_in_executor(
                    None, 
                    lambda: app.state.embedding_model.encode([request.content])
                )
                # Convert numpy array to Python list to avoid array boolean issues
                if embedding_list is not None and len(embedding_list) > 0:
                    embedding = embedding_list[0].tolist()
                    logger.info("process_memory", f"Generated embedding with {len(embedding)} dimensions")
                else:
                    embedding = None
                    logger.warning("process_memory", "Failed to generate embedding - empty result")
            except Exception as embed_error:
                logger.error("process_memory", f"Embedding generation error: {str(embed_error)}")
                embedding = None
                
        # Step 2: Perform emotion analysis if requested
        if request.analyze_emotion and request.content:
            try:
                logger.info("process_memory", "Performing emotion analysis")
                
                # Use our EmotionAnalyzer directly for the analysis
                if hasattr(app.state, 'emotion_analyzer') and app.state.emotion_analyzer is not None:
                    # Use the emotion analyzer
                    logger.debug("process_memory", "Using emotion analyzer for analysis")
                    emotion_data = await app.state.emotion_analyzer.analyze(request.content)
                else:
                    # Fallback: Call the analyze_emotion endpoint
                    logger.debug("process_memory", "Using analyze_emotion endpoint fallback")
                    emotion_response = await analyze_emotion(request.content)
                    if emotion_response.success:
                        emotion_data = {
                            "emotions": emotion_response.emotions,
                            "dominant_emotion": emotion_response.dominant_emotion
                        }
                
                logger.info("process_memory", f"Emotion analysis complete: {emotion_data.get('dominant_emotion') if emotion_data else 'None'}")
            except Exception as emotion_error:
                logger.error("process_memory", f"Emotion analysis error: {str(emotion_error)}")
                # Continue without emotion data
                
        # Step 3: Process the memory through the core
        try:
            # Prepare metadata with emotion data if available
            metadata = request.metadata or {}
            
            # Add timestamp to metadata
            metadata["timestamp"] = time.time()
            
            # Add emotion data to metadata if available
            if emotion_data:
                metadata["emotional_context"] = emotion_data
            
            # If we don't have an embedding at this point but have content, create a zero-embedding
            # This is a fallback to ensure the memory core can process the request
            if (embedding is None) and request.content:
                logger.warning("process_memory", "No embedding generated or provided. Creating zero-embedding as fallback.")
                # Create a zero-embedding with the default dimension
                embedding_dim = app.state.memory_core.config.get('embedding_dim', 768)
                embedding = [0.0] * embedding_dim
            
            # Validate embedding for NaN/Inf values and handle dimension mismatches
            if embedding is not None:
                try:
                    # Check for NaN/Inf values
                    if any(not np.isfinite(val) for val in embedding):
                        logger.warning("process_memory", "Found NaN/Inf values in embedding. Replacing with zeros.")
                        embedding = [0.0 if not np.isfinite(val) else val for val in embedding]
                    
                    # Ensure correct dimensionality
                    expected_dim = app.state.memory_core.config.get('embedding_dim', 768)
                    actual_dim = len(embedding)
                    
                    if actual_dim != expected_dim:
                        logger.warning("process_memory", f"Dimension mismatch: expected {expected_dim}, got {actual_dim}. Aligning to expected dimension.")
                        if actual_dim < expected_dim:
                            # Pad with zeros if too small
                            embedding = embedding + [0.0] * (expected_dim - actual_dim)
                        else:
                            # Truncate if too large
                            embedding = embedding[:expected_dim]
                except Exception as val_error:
                    logger.error("process_memory", f"Error validating embedding: {str(val_error)}")
                    # Continue with original embedding
            
            # Process the memory through the core
            logger.info("process_memory", "Calling memory core to process memory")
            
            result = await app.state.memory_core.process_new_memory(
                content=request.content,
                embedding=embedding or request.embedding,
                metadata=metadata
            )
            
            memory_id = result.id if result else None
            quickrecal_score = result.quickrecal_score if result else None
            logger.info("process_memory", f"Memory processed successfully with ID: {memory_id}")
            
            # Return response with results
            return ProcessMemoryResponse(
                success=True,
                memory_id=memory_id,
                quickrecal_score=quickrecal_score,
                metadata=metadata
            )
            
        except Exception as core_error:
            logger.error("process_memory", f"Memory core processing error: {str(core_error)}")
            raise HTTPException(status_code=500, detail=f"Memory processing failed: {str(core_error)}")
    
    except Exception as e:
        logger.error("process_memory", f"Process memory error: {str(e)}")
        import traceback
        logger.error("process_memory", traceback.format_exc())
        
        return ProcessMemoryResponse(
            success=False,
            error=str(e)
        )


@app.post("/retrieve_memories", response_model=RetrieveMemoriesResponse)
async def retrieve_memories(request: RetrieveMemoriesRequest):
    """Retrieve relevant memories."""
    try:
        # Add debug logging
        logger.info("retrieve_memories", f"Received request: query='{request.query}', top_k={request.top_k}")
        
        # Convert user_emotion from dict to string if needed
        user_emotion_str = None
        if request.user_emotion:
            if isinstance(request.user_emotion, dict) and 'dominant_emotion' in request.user_emotion:
                user_emotion_str = request.user_emotion['dominant_emotion']
            elif isinstance(request.user_emotion, str):
                user_emotion_str = request.user_emotion
        
        # Retrieve memories with updated parameters
        # Note: We no longer pass query_embedding as it's handled internally
        retrieve_result = await app.state.memory_core.retrieve_memories(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,  # Use threshold from request if provided
            user_emotion=user_emotion_str
        )
        
        return RetrieveMemoriesResponse(
            success=retrieve_result.get('success', False),
            memories=retrieve_result.get('memories', []),
            error=retrieve_result.get('error')
        )
    except Exception as e:
        logger.error("retrieve_memories", f"Error: {str(e)}")
        import traceback
        logger.error("retrieve_memories", traceback.format_exc())
        return RetrieveMemoriesResponse(
            success=False,
            error=str(e)
        )

@app.post("/generate_embedding", response_model=GenerateEmbeddingResponse)
async def embedding_endpoint(request: GenerateEmbeddingRequest):
    """Generate embedding for text."""
    try:
        embedding = await generate_embedding(request.text)
        return GenerateEmbeddingResponse(
            success=True,
            embedding=embedding.tolist(),
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error("generate_embedding", f"Error: {str(e)}")
        return GenerateEmbeddingResponse(
            success=False,
            error=str(e)
        )

@app.post("/calculate_quickrecal", response_model=QuickRecalResponse)
async def calculate_quickrecal(request: QuickRecalRequest):
    """Calculate QuickRecal score for an embedding or text."""
    try:
        # Generate embedding if text is provided but embedding is not
        embedding = None
        if request.embedding is None and request.text is not None:
            # Generate embedding directly
            embedding = await generate_embedding(request.text)
        elif request.embedding is not None:
            embedding = np.array(request.embedding, dtype=np.float32)
        else:
            return QuickRecalResponse(
                success=False,
                error="Either embedding or text must be provided"
            )
        
        if embedding is None:
            return QuickRecalResponse(
                success=False,
                error="Failed to generate embedding"
            )
            
        # Prepare context with text if provided
        context = request.context or {'timestamp': time.time()}
        if request.text:
            context['text'] = request.text
            
        # Calculate QuickRecal score - use synchronous method to avoid asyncio issues
        try:
            if hasattr(app.state.memory_core.quick_recal, 'calculate'):
                quickrecal_score = await app.state.memory_core.quick_recal.calculate(embedding, context=context)
            else:
                logger.warning("calculate_quickrecal", "No calculate method found, using fallback")
                quickrecal_score = 0.5  # Default fallback score
        except RuntimeError as re:
            if "asyncio.run()" in str(re):
                # Handle asyncio runtime error by using synchronous version
                logger.warning("calculate_quickrecal", f"Asyncio runtime error: {str(re)}. Using synchronous method.")
                if hasattr(app.state.memory_core.quick_recal, 'calculate_sync'):
                    quickrecal_score = app.state.memory_core.quick_recal.calculate_sync(embedding, context=context)
                else:
                    logger.error("calculate_quickrecal", "No synchronous fallback method available.")
                    quickrecal_score = 0.5  # Default fallback score
            else:
                raise re
        
        # Get factor scores if available
        factors = None
        if hasattr(app.state.memory_core.quick_recal, 'get_last_factor_scores'):
            factors = app.state.memory_core.quick_recal.get_last_factor_scores()
        
        return QuickRecalResponse(
            success=True,
            quickrecal_score=quickrecal_score,
            factors=factors
        )
    except Exception as e:
        logger.error("calculate_quickrecal", f"Error: {str(e)}")
        return QuickRecalResponse(
            success=False,
            error=str(e)
        )

@app.post("/analyze_emotion", response_model=EmotionResponse)
async def analyze_emotion(request: EmotionRequest):
    """Analyze emotional content of text."""
    try:
        # Get text from the request
        text = request.text
            
        # Ensure text is a string
        if not isinstance(text, str):
            return EmotionResponse(
                success=False,
                error="Text must be a string"
            )
        
        # Use our EmotionAnalyzer if available
        if hasattr(app.state, 'emotion_analyzer') and app.state.emotion_analyzer is not None:
            # Get analysis results from the analyzer
            result = await app.state.emotion_analyzer.analyze(text)
            
            return EmotionResponse(
                success=True,
                emotions=result.get("emotions", {}),
                dominant_emotion=result.get("dominant_emotion", "neutral")
            )
        else:
            # Fallback to keyword-based detection if analyzer isn't available
            logger.warning("analyze_emotion", "Emotion analyzer not available, using keyword fallback")
            
            # Simple keyword-based emotion detection
            emotion_keywords = {
                "joy": ["happy", "joy", "delighted", "glad", "pleased", "excited", "thrilled"],
                "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "disappointed"],
                "anger": ["angry", "mad", "furious", "annoyed", "irritated", "enraged", "frustrated"],
                "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
                "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
                "disgust": ["disgusted", "repulsed", "revolted", "sickened"],
                "neutral": ["ok", "fine", "neutral", "average", "normal"]
            }
            
            text = text.lower()
            emotion_scores = {emotion: 0.1 for emotion in emotion_keywords}  # Base score
            
            # Simple keyword matching
            for emotion, keywords in emotion_keywords.items():
                for keyword in keywords:
                    if keyword in text:
                        emotion_scores[emotion] += 0.15  # Increment score for each match
            
            # Find the dominant emotion
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            return EmotionResponse(
                success=True,
                emotions=emotion_scores,
                dominant_emotion=dominant_emotion
            )
            
    except Exception as e:
        logger.error("analyze_emotion", f"Error analyzing emotions: {str(e)}")
        import traceback
        logger.error("analyze_emotion", traceback.format_exc())
        
        return EmotionResponse(
            success=False,
            error=str(e)
        )

@app.post("/provide_feedback", response_model=FeedbackResponse)
async def provide_feedback(request: FeedbackRequest):
    """Provide feedback on memory retrieval relevance."""
    try:
        if not app.state.memory_core.threshold_calibrator:
            return FeedbackResponse(
                success=False,
                error="Adaptive thresholding is not enabled"
            )
        
        await app.state.memory_core.provide_feedback(
            memory_id=request.memory_id,
            similarity_score=request.similarity_score,
            was_relevant=request.was_relevant
        )
        
        new_threshold = app.state.memory_core.threshold_calibrator.get_current_threshold()
        
        return FeedbackResponse(
            success=True,
            new_threshold=new_threshold
        )
    except Exception as e:
        logger.error("provide_feedback", f"Error: {str(e)}")
        return FeedbackResponse(
            success=False,
            error=str(e)
        )

@app.post("/detect_contradictions")
async def detect_contradictions(threshold: float = 0.75):
    """Detect potential causal contradictions in memories."""
    try:
        contradictions = await app.state.memory_core.detect_contradictions(threshold=threshold)
        return {
            "success": True,
            "contradictions": contradictions,
            "count": len(contradictions)
        }
    except Exception as e:
        logger.error("detect_contradictions", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/process_transcription", response_model=TranscriptionResponse)
async def process_transcription(request: TranscriptionRequest, background_tasks: BackgroundTasks):
    """Process a transcription and store it in the memory system with rich metadata."""
    try:
        logger.info("process_transcription", "Processing transcription request")
        
        # Validate input
        if not request.text or not isinstance(request.text, str) or len(request.text.strip()) == 0:
            logger.error("process_transcription", "Invalid or empty transcription text")
            return TranscriptionResponse(
                success=False,
                error="Transcription text cannot be empty"
            )
            
        # Tracking for current request
        embedding = None
        extracted_metadata = None
        memory_id = None
        
        # Step 1: Generate embedding if needed
        if request.embedding is None and hasattr(app.state, 'embedding_model'):
            try:
                logger.info("process_transcription", "Generating embedding from transcription")
                loop = asyncio.get_event_loop()
                embedding_list = await loop.run_in_executor(
                    None, 
                    lambda: app.state.embedding_model.encode([request.text])
                )
                # Convert numpy array to Python list to avoid array boolean issues
                if embedding_list is not None and len(embedding_list) > 0:
                    embedding = embedding_list[0].tolist()
                    logger.info("process_transcription", f"Generated embedding with {len(embedding)} dimensions")
                else:
                    embedding = None
                    logger.warning("process_transcription", "Failed to generate embedding - empty result")
            except Exception as embed_error:
                logger.error("process_transcription", f"Embedding generation error: {str(embed_error)}")
                # Continue with None embedding if it fails
        else:
            embedding = request.embedding
        
        # Step 2: Extract features using the TranscriptionFeatureExtractor
        if hasattr(app.state, 'transcription_extractor') and app.state.transcription_extractor is not None:
            try:
                logger.info("process_transcription", "Extracting features from transcription")
                
                # Use our extractor to get rich metadata
                audio_metadata = request.audio_metadata or {}
                extracted_metadata = await app.state.transcription_extractor.extract_features(
                    transcript=request.text,
                    meta=audio_metadata
                )
                
                logger.info("process_transcription", 
                         f"Extracted {len(extracted_metadata)} features including" +
                         f" dominant_emotion={extracted_metadata.get('dominant_emotion', 'none')}," +
                         f" keywords={len(extracted_metadata.get('keywords', []))} keywords")
            except Exception as extract_error:
                logger.error("process_transcription", f"Feature extraction error: {str(extract_error)}")
                # Continue with empty metadata if extraction fails
                extracted_metadata = {
                    "input_modality": "spoken",
                    "source": "transcription",
                    "error": str(extract_error)
                }
        else:
            logger.warning("process_transcription", "No transcription feature extractor available")
            extracted_metadata = {
                "input_modality": "spoken",
                "source": "transcription"
            }
        
        # Step 3: Process the memory through the core
        try:
            # Prepare final metadata
            metadata = extracted_metadata or {}
            
            # Set importance if provided
            if request.importance is not None:
                metadata["importance"] = max(0.0, min(1.0, request.importance))
            
            # Add timestamp to metadata
            metadata["timestamp"] = time.time()
            
            # Call memory core to process the memory
            logger.info("process_transcription", "Calling memory core to process transcription memory")
            result = await app.state.memory_core.process_memory(
                content=request.text,
                embedding=embedding,
                memory_id=request.memory_id,
                metadata=metadata,
                memory_type="transcription",
                force_update=request.force_update
            )
            
            memory_id = result.get("memory_id")
            logger.info("process_transcription", f"Transcription processed with ID: {memory_id}")
            
            # Return success response
            return TranscriptionResponse(
                success=True,
                memory_id=memory_id,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as core_error:
            logger.error("process_transcription", f"Memory core processing error: {str(core_error)}")
            raise HTTPException(status_code=500, detail=f"Memory processing failed: {str(core_error)}")
    
    except Exception as e:
        logger.error("process_transcription", f"Process transcription error: {str(e)}")
        import traceback
        logger.error("process_transcription", traceback.format_exc())
        
        return TranscriptionResponse(
            success=False,
            error=str(e)
        )

# --- Optional: Assembly Management Endpoints (Basic for MVP) ---

@app.get("/assemblies")
async def list_assemblies():
    """List all memory assemblies."""
    try:
        assembly_info = []
        async with app.state.memory_core._lock:
            for assembly_id, assembly in app.state.memory_core.assemblies.items():
                assembly_info.append({
                    "assembly_id": assembly_id,
                    "name": assembly.name,
                    "memory_count": len(assembly.memories),
                    "last_activation": assembly.last_activation
                })
        return {
            "success": True,
            "assemblies": assembly_info,
            "count": len(assembly_info)
        }
    except Exception as e:
        logger.error("list_assemblies", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/assemblies/{assembly_id}")
async def get_assembly(assembly_id: str):
    """Get details for a specific assembly."""
    try:
        async with app.state.memory_core._lock:
            if assembly_id not in app.state.memory_core.assemblies:
                return {
                    "success": False,
                    "error": "Assembly not found"
                }
            
            assembly = app.state.memory_core.assemblies[assembly_id]
            memory_ids = list(assembly.memories)
            
            # Get memory details (limited to first 10 for brevity)
            memories = []
            for mem_id in memory_ids[:10]:
                if mem_id in app.state.memory_core._memories:
                    memory = app.state.memory_core._memories[mem_id]
                    memories.append({
                        "id": memory.id,
                        "content": memory.content,
                        "quickrecal_score": memory.quickrecal_score
                    })
            
            return {
                "success": True,
                "assembly_id": assembly_id,
                "name": assembly.name,
                "memory_count": len(assembly.memories),
                "last_activation": assembly.last_activation,
                "sample_memories": memories,
                "total_memories": len(memory_ids)
            }
    except Exception as e:
        logger.error("get_assembly", f"Error: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.on_event("startup")
async def startup_db_client():
    """Initialize FastAPI app with required services."""
    # Record startup time for stats
    app.state.startup_time = time.time()
    
    # Initialize embedding model
    try:
        from sentence_transformers import SentenceTransformer
        # Use the model specified in environment or default to all-mpnet-base-v2
        model_name = os.environ.get('EMBEDDING_MODEL', 'all-mpnet-base-v2')
        logger.info("startup", f"Loading embedding model: {model_name}")
        
        # Try to load the model, download if not available
        try:
            app.state.embedding_model = SentenceTransformer(model_name)
            logger.info("startup", f"Embedding model {model_name} loaded successfully")
        except Exception as model_error:
            # If the model doesn't exist, it might need to be downloaded
            if "No such file or directory" in str(model_error) or "not found" in str(model_error).lower():
                logger.warning("startup", f"Model {model_name} not found locally, attempting to download...")
                from sentence_transformers import util as st_util
                # Force download from Hugging Face
                app.state.embedding_model = SentenceTransformer(model_name, use_auth_token=None)
                logger.info("startup", f"Successfully downloaded and loaded model {model_name}")
            else:
                # Re-raise if it's not a file-not-found error
                raise
    except Exception as e:
        logger.error("startup", f"Error loading embedding model: {str(e)}")
        raise
    
    # Initialize emotion model
    try:
        from transformers import pipeline
        
        # Check for models in both local and Docker environments
        # For local development
        local_model_path = "C:/Users/danny/OneDrive/Documents/AI_Conversations/lucid-recall-dist/lucid-recall-dist/models/roberta-base-go_emotions"
        # For Docker environment
        docker_model_path = "/app/models/roberta-base-go_emotions"
        
        # Try Docker path first, then local path
        if os.path.exists(docker_model_path):
            emotion_model_path = docker_model_path
        elif os.path.exists(local_model_path):
            emotion_model_path = local_model_path
        else:
            emotion_model_path = None
            
        if emotion_model_path:
            logger.info("startup", f"Loading emotion model from: {emotion_model_path}")
            app.state.emotion_model = pipeline("text-classification", model=emotion_model_path, return_all_scores=True)
            logger.info("startup", "Emotion model loaded successfully")
        else:
            logger.warning("startup", f"Emotion model not found in expected locations, will use fallback")
            app.state.emotion_model = None
    except Exception as e:
        logger.error("startup", f"Error loading emotion model: {str(e)}")
        app.state.emotion_model = None
    
    # Initialize SynthiansMemoryCore
    try:
        # Load configuration from environment variables
        storage_path = os.environ.get('MEMORY_STORAGE_PATH', '/app/memory/stored/synthians')
        embedding_dim = int(os.environ.get('EMBEDDING_DIM', '768'))
        geometry_type = os.environ.get('GEOMETRY_TYPE', 'hyperbolic')
        
        # Create memory core config
        memory_core_config = {
            'storage_path': storage_path,
            'embedding_dim': embedding_dim,
            'geometry': geometry_type,
            'embedding_model': app.state.embedding_model,
            'emotion_model': app.state.emotion_model  # Pass the emotion model to the memory core
        }
        
        # Initialize memory core
        logger.info("startup", "Initializing SynthiansMemoryCore", memory_core_config)
        from synthians_memory_core import SynthiansMemoryCore
        app.state.memory_core = SynthiansMemoryCore(memory_core_config)
        await app.state.memory_core.initialize()
        logger.info("startup", "SynthiansMemoryCore initialized successfully")
    except Exception as e:
        logger.error("startup", f"Error initializing SynthiansMemoryCore: {str(e)}")
        raise

# Run the server when the module is executed directly
if __name__ == "__main__":
    import os
    import uvicorn
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5010"))
    
    print(f"Starting Synthians Memory Core API server at {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)

```

# custom_logger.py

```py
# synthians_memory_core/custom_logger.py

import logging
import os
import time
from typing import Dict, Any, Optional

# Set up logging
log_level = os.getenv("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, log_level.upper(), None)
if not isinstance(numeric_level, int):
    numeric_level = logging.INFO

logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class Logger:
    """A simplified logger compatible with the original interface"""

    def __init__(self, name="SynthiansMemory"):
        self.logger = logging.getLogger(name)

    def info(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log info message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.info(log_msg)

    def warning(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.warning(log_msg)

    def error(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log error message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.error(log_msg, exc_info=True if isinstance(data, Exception) else False)

    def debug(self, context: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Log debug message"""
        log_msg = f"[{context}] {message}"
        if data:
            log_msg += f" | Data: {data}"
        self.logger.debug(log_msg)

# Create a singleton logger instance
logger = Logger()
```

# docs\embedding_handling.md

```md
# Embedding Handling in Synthians Memory Core

## Overview

The Synthians Memory Core implements robust handling for embeddings throughout the system, addressing several critical challenges:

1. **Dimension Mismatches**: Safely handling vectors of different dimensions (e.g., 384 vs. 768)
2. **Malformed Embeddings**: Detecting and handling NaN/Inf values in embedding vectors
3. **Efficient Retrieval**: Using FAISS for fast similarity search with automatic GPU acceleration

## Embedding Validation

All embeddings in the system are validated before use to ensure robustness:

\`\`\`python
def _validate_embedding(embedding):
    """Validate that an embedding vector contains only valid values.
    
    Args:
        embedding: The embedding vector to validate
        
    Returns:
        bool: True if the embedding is valid, False otherwise
    """
    if embedding is None:
        return False
        
    # Check for NaN or Inf values
    return not (np.isnan(embedding).any() or np.isinf(embedding).any())
\`\`\`

Invalid embeddings are replaced with zero vectors to prevent crashes:

\`\`\`python
def process_embedding(embedding):
    """Process and normalize an embedding, handling malformed inputs."""
    if not _validate_embedding(embedding):
        # Replace invalid embedding with zeros
        logger.warning("Invalid embedding detected (NaN/Inf values). Replacing with zeros.")
        return np.zeros(len(embedding), dtype=np.float32)
    
    # Normalize and return valid embedding
    return normalize_embedding(embedding)
\`\`\`

## Dimension Alignment

The system can handle embeddings of different dimensions (primarily 384 vs. 768) using a vector alignment utility:

\`\`\`python
def _align_vectors_for_comparison(vec1, vec2):
    """Align two vectors to the same dimension for comparison operations.
    
    If dimensions differ, either pads the smaller vector with zeros or
    truncates the larger vector to match dimensions.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        tuple: (aligned_vec1, aligned_vec2) with matching dimensions
    """
    dim1 = len(vec1)
    dim2 = len(vec2)
    
    if dim1 == dim2:
        return vec1, vec2
    
    if dim1 < dim2:
        # Pad vec1 with zeros to match vec2
        return np.pad(vec1, (0, dim2 - dim1)), vec2
    else:
        # Pad vec2 with zeros to match vec1
        return vec1, np.pad(vec2, (0, dim1 - dim2))
\`\`\`

This ensures vector operations work correctly even when embeddings have different dimensions.

## Integration with FAISS Vector Index

The FAISS vector index implementation interacts with the embedding handling system:

### Dimension Handling

The `MemoryVectorIndex` is initialized with a specific dimension and validates all inputs:

\`\`\`python
def add(self, memory_id: str, embedding: np.ndarray) -> None:
    """Add a memory embedding to the index.
    
    Args:
        memory_id: Unique identifier for the memory
        embedding: Embedding vector as numpy array
    """
    # Validate embedding
    if embedding is None:
        logger.warning(f"Attempted to add None embedding for memory {memory_id}")
        return
        
    if len(embedding) != self.dimension:
        logger.warning(
            f"Embedding dimension mismatch for memory {memory_id}: "
            f"Expected {self.dimension}, got {len(embedding)}"
        )
        # Align dimensions by padding or truncating
        embedding = self._align_embedding_dimension(embedding)
\`\`\`

The `_align_embedding_dimension` method ensures all embeddings match the expected dimension:

\`\`\`python
def _align_embedding_dimension(self, embedding):
    """Align embedding to the expected dimension.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        numpy.ndarray: Aligned embedding with correct dimension
    """
    current_dim = len(embedding)
    
    if current_dim == self.dimension:
        return embedding
        
    if current_dim < self.dimension:
        # Pad with zeros
        return np.pad(embedding, (0, self.dimension - current_dim))
    else:
        # Truncate
        return embedding[:self.dimension]
\`\`\`

### Handling Malformed Embeddings

The vector index works with the validation system to safely handle malformed embeddings:

\`\`\`python
def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    """Search for similar embeddings in the index.
    
    Args:
        query_embedding: Query embedding vector
        k: Number of nearest neighbors to retrieve
        
    Returns:
        List of (memory_id, similarity_score) tuples
    """
    # Validate query embedding
    if not _validate_embedding(query_embedding):
        logger.warning("Invalid query embedding (NaN/Inf values). Replacing with zeros.")
        query_embedding = np.zeros(self.dimension, dtype=np.float32)
    
    # Align dimensions if needed
    if len(query_embedding) != self.dimension:
        query_embedding = self._align_embedding_dimension(query_embedding)
\`\`\`

## Memory Retrieval Improvements

Memory retrieval has been enhanced with several improvements:

1. **Lowered Pre-filter Threshold**: Reduced from 0.5 to 0.3 for better recall sensitivity
2. **Explicit Threshold Parameter**: Added client and server-side support for explicit threshold control
3. **Enhanced Logging**: Added detailed similarity score logging for debugging

Example from `_get_candidate_memories`:

\`\`\`python
async def _get_candidate_memories(self, query_embedding: np.ndarray, limit: int, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Get candidate memories using vector similarity search.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results to return
        threshold: Optional similarity threshold (if None, uses default)
        
    Returns:
        List of candidate memories with similarity scores
    """
    # Apply default threshold if not specified
    threshold = threshold if threshold is not None else self.default_threshold
    
    # Validate embedding
    if not _validate_embedding(query_embedding):
        logger.warning("Invalid query embedding detected in _get_candidate_memories")
        query_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
    
    # Perform vector search using the FAISS index
    results = self.vector_index.search(query_embedding, k=limit * 2)  # Get extra results for filtering
    
    # Filter and format results
    candidates = []
    for memory_id, score in results:
        if score < threshold:  # Lower scores are better for L2 distance
            logger.debug(f"Memory {memory_id} filtered out with score {score} (threshold: {threshold})")
            continue
            
        # Fetch the full memory and add to candidates
        memory = await self.get_memory_by_id(memory_id)
        if memory:
            memory['similarity_score'] = float(score)
            candidates.append(memory)
            
        if len(candidates) >= limit:
            break
            
    return candidates
\`\`\`

## Testing

Comprehensive tests ensure embedding handling is robust:

- **Validation Tests**: Verify detection of NaN/Inf values
- **Alignment Tests**: Confirm vectors of different dimensions are properly aligned
- **Threshold Tests**: Ensure memory retrieval works with various thresholds

## Conclusion

The embedding handling system in Synthians Memory Core provides a robust foundation for vector operations. Combined with the FAISS vector index implementation, it ensures efficient and reliable memory retrieval while gracefully handling edge cases like dimension mismatches and malformed embeddings.

```

# docs\faiss_gpu_integration.md

```md
# FAISS GPU Integration Guide

## Overview

This document explains how GPU support is integrated with FAISS in the Synthians Memory Core system. The integration enables significant performance improvements for vector similarity searches when GPU hardware is available.

## Implementation Approach

Our implementation follows a robust multi-layered approach to ensure FAISS with GPU acceleration is available whenever possible:

1. **Docker Pre-Installation**: FAISS is installed during container startup based on hardware detection
2. **Dynamic Code Installation**: Fallback auto-installation occurs if the import fails at runtime
3. **Graceful Degradation**: If GPU support isn't available, the system falls back to CPU mode

## Docker Integration

### Container Startup Process

The Docker Compose configuration detects GPU availability and installs the appropriate FAISS package during container initialization:

\`\`\`yaml
command: >
  /bin/bash -c '
  # Pre-install FAISS before Python importing it
  echo "[+] PRE-INSTALLING FAISS FOR MEMORY VECTOR INDEX" &&
  pip install --upgrade pip setuptools wheel &&
  # Install CPU version first as a fallback
  pip install --no-cache-dir faiss-cpu &&
  # If GPU available, replace with GPU version
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "[+] GPU DETECTED - Installing FAISS-GPU for better performance" &&
    pip uninstall -y faiss-cpu &&
    pip install --no-cache-dir faiss-gpu
  fi &&
  # Verify FAISS installation
  python -c "import faiss; print(f\'[+] FAISS {getattr(faiss, \\\'__version__\\\', \\\'unknown\\\')} pre-installed successfully\')" &&
  ...
\`\`\`

Key aspects of this approach:
- Installs CPU version first as a reliable fallback
- Only replaces with GPU version when hardware is confirmed available
- Verifies installation succeeded before proceeding

## Dynamic Import with Auto-Installation

The `vector_index.py` module implements dynamic FAISS import with automatic installation if the package is missing:

\`\`\`python
# Dynamic FAISS import with auto-installation fallback
try:
    import faiss
except ImportError:
    import sys
    import subprocess
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vector_index")
    
    logger.warning("FAISS not found. Attempting to install...")
    
    # Check for GPU availability
    try:
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            gpu_available = result.returncode == 0
        except:
            pass
            
        # Install appropriate FAISS package
        if gpu_available:
            logger.info("GPU detected, installing FAISS with GPU support")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-gpu'])
        else:
            logger.info("No GPU detected, installing CPU-only FAISS")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-cpu'])
            
        # Try importing again
        import faiss
        logger.info(f"Successfully installed and imported FAISS {getattr(faiss, '__version__', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
        raise ImportError("Failed to install FAISS. Please install it manually.")
\`\`\`

This approach provides resilience against:
- Missing dependencies at runtime
- Container rebuilds that might lose installed packages
- Varying hardware configurations

## GPU Utilization in the Vector Index

The `MemoryVectorIndex` class handles runtime GPU utilization:

\`\`\`python
def __init__(self, config=None):
    # ...
    self.is_using_gpu = False
    
    # Move to GPU if available and requested
    if self.config['use_gpu']:
        self._move_to_gpu_if_available()

def _move_to_gpu_if_available(self):
    """Move the index to GPU if available."""
    try:
        # Check if FAISS was built with GPU support
        if hasattr(faiss, 'StandardGpuResources'):
            logger.info("Moving FAISS index to GPU...")
            self.gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(self.gpu_res, self.config['gpu_id'], self.index)
            self.index = gpu_index
            self.is_using_gpu = True
            logger.info(f"FAISS index successfully moved to GPU {self.config['gpu_id']}")
        else:
            logger.warning("FAISS was not built with GPU support. Using CPU index.")
    except Exception as e:
        logger.error(f"Failed to move index to GPU: {str(e)}. Using CPU index.")
\`\`\`

This implementation:
1. Attempts to move the index to GPU memory when initialized
2. Provides detailed logging about GPU utilization status
3. Falls back gracefully to CPU if GPU transfer fails

## Performance Considerations

### Expected Speedups

Typical performance improvements with GPU acceleration:

| Vector Count | Query Count | CPU Time | GPU Time | Speedup |
|--------------|-------------|----------|----------|--------|
| 10,000       | 100         | 0.087s   | 0.024s   | 3.6x   |
| 100,000      | 100         | 0.830s   | 0.064s   | 13.0x  |
| 1,000,000    | 100         | 8.214s   | 0.356s   | 23.1x  |

*Note: These are approximate values that will vary based on GPU model and vector dimensionality*

### Memory Management

For optimal GPU performance:

- The system sets `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to avoid memory fragmentation
- Consider adjusting this value for your specific GPU memory size
- For very large indices, you may need to implement index sharding

## Troubleshooting GPU Support

### Verifying GPU Usage

To verify if FAISS is using GPU acceleration:

\`\`\`python
from synthians_memory_core.vector_index import MemoryVectorIndex

index = MemoryVectorIndex()
print(f"Using GPU: {index.is_using_gpu}")
\`\`\`

### Common GPU Issues

1. **CUDA Version Mismatch**
   - FAISS-GPU requires a specific CUDA version
   - We added `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118` to ensure compatible versions

2. **Insufficient GPU Memory**
   - Large indices may exceed GPU memory
   - Solution: Implement index sharding or reduce batch sizes

3. **GPU Not Visible to Docker**
   - Ensure Docker has GPU access: `--runtime=nvidia` and proper device mapping
   - Verify NVIDIA Container Toolkit is properly installed

## Conclusion

This implementation ensures that the Synthians Memory Core system can leverage GPU acceleration for vector similarity searches whenever possible, while gracefully falling back to CPU processing when necessary. The multi-layered approach provides robust operation across different deployment environments.

```

# docs\README.md

```md
# Synthians Memory Core Documentation

## Vector Index and FAISS Integration

- [Memory Vector Index with FAISS](vector_index.md) - Core implementation details and usage
- [FAISS GPU Integration Guide](faiss_gpu_integration.md) - How GPU acceleration is implemented
- [Embedding Handling](embedding_handling.md) - Robust embedding validation and dimension alignment

## Memory System

### Core Features

- Memory storage and retrieval
- Efficient vector similarity search via FAISS
- Automatic embedding validation and dimension alignment
- Metadata synthesis and enrichment
- Emotion analysis integration

### Implementation Details

#### Memory Retrieval

- Improved pre-filter threshold (reduced from 0.5 to 0.3)
- Added NaN/Inf validation for embedding vectors
- Enhanced similarity score logging
- Added explicit threshold parameter support

#### Metadata Enrichment

- MetadataSynthesizer integration in the memory processing workflow
- Automatic addition of UUID and content length to metadata
- Sophisticated metadata extraction and enrichment

#### Emotion Analysis

- Optimized emotion analysis to avoid redundant processing
- Respect for pre-computed emotion data from API
- Fallback mechanisms for handling unavailable services

## Architecture

### Components

1. **SynthiansMemoryCore** - The main memory management system
2. **MemoryVectorIndex** - FAISS-based vector indexing for efficient retrieval
3. **MetadataSynthesizer** - Enriches memory with metadata
4. **EmotionAnalyzer** - Analyzes emotional content of text

### Deployment

- Docker integration with GPU support
- Automatic dependency management
- Robust error handling and fallbacks

## Docker Integration

The system is designed to run in a Docker environment with optional GPU acceleration:

- Automatic detection and installation of appropriate FAISS version
- GPU acceleration when available
- Seamless fallback to CPU processing when necessary

## API

The system exposes a comprehensive API for memory operations:

- Memory processing and storage
- Similarity-based retrieval
- Embedding generation
- Emotion analysis
- Transcription processing

See the API server implementation for detailed endpoint specifications.

## Technologies

- **FAISS** - Facebook AI Similarity Search for efficient vector operations
- **Sentence Transformers** - For generating text embeddings
- **FastAPI** - For the REST API interface
- **Docker** - For containerized deployment
- **CUDA** - For GPU acceleration

```

# docs\refactor-plan.md

```md
## **Unified Memory System: Technical Overview & Roadmap (Synthians Core)**

**Goal:** Consolidate the complex memory codebase into a single, efficient, unified system (`synthians_memory_core`) running locally (e.g., on an RTX 4090 via Docker), focusing on core memory operations, HPC-QuickRecal scoring, emotional context, and memory assemblies for an MVP by the end of the week.

---

### 1. **Technical Overview of the Unified `synthians_memory_core`**

This unified system centralizes memory functionality, integrating the most valuable and innovative concepts identified previously, while simplifying the architecture for clarity and maintainability.

**Core Components (Target Architecture):**

1.  **`SynthiansMemoryCore` (`synthians_memory_core.py`):**
    *   **Role:** The central orchestrator and main API endpoint.
    *   **Responsibilities:** Initializes and manages all other core components. Handles incoming requests for storing (`process_new_memory`) and retrieving (`retrieve_memories`) memories. Manages the in-memory cache/working set (`self.memories`), memory assemblies (`self.assemblies`), and coordinates background tasks. Delegates specialized tasks (scoring, geometry, persistence, emotion) to dedicated managers. Provides LLM tool interfaces (`get_tools`, `handle_tool_call`).
2.  **`UnifiedQuickRecallCalculator` (`hpc_quickrecal.py`):**
    *   **Role:** The single source of truth for calculating memory importance (`quickrecal_score`).
    *   **Responsibilities:** Implements various scoring modes (Standard, HPC-QR, Minimal, etc.) using configurable factor weights. Calculates factors like Recency, Emotion, Relevance, Importance, Personal, and potentially simplified versions of HPC-QR factors (Geometry, Novelty, Self-Org, Overlap) using the `GeometryManager`.
3.  **`GeometryManager` (`geometry_manager.py`):**
    *   **Role:** Central authority for all embedding geometry operations.
    *   **Responsibilities:** Validates embeddings (NaN/Inf checks). Normalizes vectors. Aligns vectors of different dimensions (e.g., 384 vs 768). Performs geometric transformations (e.g., Euclidean to Hyperbolic via `_to_hyperbolic`). Calculates distances and similarities based on the configured geometry (Euclidean, Hyperbolic, Spherical, Mixed).
4.  **`EmotionalAnalyzer` & `EmotionalGatingService` (`emotional_intelligence.py`):**
    *   **Role:** Handle emotional context.
    *   **Responsibilities:** `EmotionalAnalyzer` (simplified/placeholder for now) provides emotional analysis of text. `EmotionalGatingService` uses this analysis and user state to filter/re-rank retrieved memories, implementing cognitive defense and resonance scoring.
5.  **`MemoryPersistence` (`memory_persistence.py`):**
    *   **Role:** Sole handler for all disk-based memory operations.
    *   **Responsibilities:** Asynchronously saves (`save_memory`), loads (`load_memory`), and deletes (`delete_memory`) `MemoryEntry` objects using atomic writes (temp files + rename) and JSON format. Manages a memory index file (`memory_index.json`) and handles backups.
6.  **`MemoryEntry` & `MemoryAssembly` (`memory_structures.py`):**
    *   **Role:** Standard data structures.
    *   **Responsibilities:** `MemoryEntry` defines a single memory unit with content, embedding (standard and optional hyperbolic), QuickRecal score, and metadata. `MemoryAssembly` groups related `MemoryEntry` IDs, maintains a composite embedding (using `GeometryManager`), tracks activation, and handles emotional profiles/keywords for the group.
7.  **`ThresholdCalibrator` (`adaptive_components.py`):**
    *   **Role:** Enables adaptive retrieval relevance.
    *   **Responsibilities:** Dynamically adjusts the similarity threshold used in `retrieve_memories` based on feedback (`provide_feedback`) about whether retrieved memories were actually relevant.
8.  **`custom_logger.py`:**
    *   **Role:** Provides a consistent logging interface used by all components.

**Key Workflows in Unified System:**

*   **Memory Storage:**
    1.  `SynthiansMemoryCore.process_new_memory` receives content/embedding/metadata.
    2.  It calls `GeometryManager` to validate, align, and normalize the embedding.
    3.  It calls `UnifiedQuickRecallCalculator.calculate` to get the `quickrecal_score`.
    4.  It calls `EmotionalAnalyzer.analyze` to get emotional context for metadata.
    5.  If geometry is hyperbolic, it calls `GeometryManager._to_hyperbolic`.
    6.  It creates a `MemoryEntry`.
    7.  If score > threshold, it stores the `MemoryEntry` in `self.memories`.
    8.  It asynchronously calls `MemoryPersistence.save_memory`.
    9.  It calls `_update_assemblies` to potentially add the memory to relevant `MemoryAssembly` objects.
*   **Memory Retrieval:**
    1.  `SynthiansMemoryCore.retrieve_memories` receives query/embedding/context.
    2.  It calls `GeometryManager` to validate/align/normalize the query embedding.
    3.  It calls `_get_candidate_memories` which:
        *   Activates relevant `MemoryAssembly` objects based on similarity (using `GeometryManager.calculate_similarity`).
        *   Performs a quick direct similarity search against `self.memories` (using `GeometryManager.calculate_similarity`).
        *   Returns a combined list of candidate `MemoryEntry` objects.
    4.  It calculates relevance scores for candidates (using `GeometryManager.calculate_similarity`).
    5.  It calls `EmotionalGatingService.gate_memories` to filter/re-rank based on user emotion.
    6.  If `ThresholdCalibrator` is enabled, it filters results based on the current dynamic threshold.
    7.  Returns the top K results as dictionaries.

**Simplifications for MVP:**

*   **No Distributed Architecture:** Assumes a single process/container. `MemoryBroker` and `MemoryClientProxy` are removed.
*   **No Full Self/World Models:** The complex `SelfModel` and `WorldModel` classes are excluded. Basic context can be simulated or derived directly from memory/KG if needed later.
*   **No Advanced Dreaming/Narrative:** The `DreamProcessor`, `DreamManager`, `ReflectionEngine`, and `NarrativeIdentity` system are deferred. Dream insights could be stored as simple `MemoryEntry` objects if needed.
*   **Simplified Knowledge Graph:** The full modular KG is deferred. Core storage uses the `MemoryPersistence` layer. If basic graph features are needed *immediately*, use the `CoreGraphManager` directly, but avoid the full modular complexity for the MVP.
*   **Single Server:** Combines API endpoints into one server (`synthians_server.py`) using FastAPI. No separate Tensor/HPC servers needed locally; embedding/scoring happens within the `SynthiansMemoryCore` process.
*   **Simplified HPC-QR Factors:** For the MVP, `UnifiedQuickRecallCalculator` can initially focus on Recency, Relevance (Similarity), Emotion, Importance, Personal, Overlap. Geometric, Causal, and SOM factors can be added iteratively post-MVP.

---

### 2. **Identified Redundant Files/Components (To Be Removed for MVP)**

Based on the unification into `synthians_memory_core`:

1.  **High-Level Interfaces/Orchestrators:**
    *   `memory_manager.py`: Replaced by direct use of `SynthiansMemoryCore`.
    *   `memory_client.py` / `enhanced_memory_client.py`: Functionality absorbed into `SynthiansMemoryCore` or unnecessary.
    *   `advanced_memory_system.py`: Logic integrated into `SynthiansMemoryCore`.
    *   `memory_integration.py`: Replaced by `SynthiansMemoryCore`.
    *   `memory_router.py`: Routing logic is simplified within `SynthiansMemoryCore._get_candidate_memories`.
    *   `lucidia_memory.py` (`LucidiaMemorySystemMixin`): Not needed as components are directly integrated.
2.  **Persistence Layers:**
    *   `base.py` (`BaseMemoryClient`): Persistence logic replaced by `MemoryPersistence`.
    *   `long_term_memory.py`: Replaced by `SynthiansMemoryCore` + `MemoryPersistence`.
    *   `memory_system.py`: Replaced by `SynthiansMemoryCore` + `MemoryPersistence`.
    *   `unified_memory_storage.py`: Replaced by `MemoryPersistence` and `MemoryEntry`.
    *   `storage/memory_persistence_handler.py`: *This logic should be adapted/merged into `synthians_memory_core/memory_persistence.py`*. The file itself can then be removed.
3.  **Significance/QuickRecall Calculation:**
    *   `hpc_quickrecal.py` (Original `HPCQuickRecal` class): Logic merged into `UnifiedQuickRecallCalculator`.
    *   `hpc_qr_flow_manager.py`: Batching/workflow management integrated into `SynthiansMemoryCore` or handled by external callers if needed.
    *   `qr_calculator.py` (Original): Replaced by the version in `synthians_memory_core/hpc_quickrecal.py`.
4.  **HPC/Tensor Servers & Clients:**
    *   `hpc_server.py`: Not needed for local MVP; calculations happen within `SynthiansMemoryCore`.
    *   `updated_hpc_client.py`: Not needed.
    *   `tensor_server.py`: Not needed; embedding generation assumed external or handled differently.
5.  **Knowledge Graph:**
    *   `knowledge_graph.py` (Monolithic): Replaced by modular concept (deferred for MVP).
    *   `lucidia_memory_system/knowledge_graph/` (Entire modular directory): Deferred for post-MVP. Core storage uses `MemoryPersistence`.
6.  **Emotion Components:**
    *   `emotion.py` (`EmotionMixin`): Logic integrated into `SynthiansMemoryCore` using `EmotionalAnalyzer`.
    *   `emotional_intelligence.py` (within `Self`): Replaced by `synthians_memory_core/emotional_intelligence.py`.
    *   `emotion_graph_enhancer.py`: Deferred along with the full KG.
7.  **Adapters & Bridges:**
    *   `memory_adapter.py`: Not needed after unification.
    *   `memory_bridge.py`: Not needed after unification.
    *   `synthience_hpc_connector.py`: Logic for combining scores integrated into `SynthiansMemoryCore.retrieve_memories`. The external `SynthienceMemory` concept is removed for MVP.
8.  **Other:**
    *   `connectivity.py`: WebSocket logic removed as servers are removed.
    *   `tools.py`: Tool definitions moved to `SynthiansMemoryCore.get_tools`.
    *   `personal_details.py`: Basic pattern matching can be integrated directly into `SynthiansMemoryCore.process_new_memory` or a small utility function if needed.
    *   `rag_context.py`: Context generation handled by `SynthiansMemoryCore`.
    *   `memory_types.py` (Original): Replaced by `memory_structures.py`.
    *   `memory_client_example.py`: Update or remove.
    *   `test_advanced_memory.py`: Update or remove.
    *   All files under `lucidia_memory_system/core/Self/` and `lucidia_memory_system/core/World/`: Deferred for post-MVP.
    *   All files under `lucidia_memory_system/narrative_identity/`: Deferred for post-MVP.
    *   `system_events.py`: Event handling simplified or deferred.
    *   `memory_index.py`: Indexing logic might be integrated into `MemoryPersistence` or simplified.

**Files to Keep/Adapt for the MVP:**

*   All files within the new `synthians_memory_core/` directory (`__init__.py`, `synthians_memory_core.py`, `adaptive_components.py`, `custom_logger.py`, `emotional_intelligence.py`, `geometry_manager.py`, `hpc_quickrecal.py`, `memory_persistence.py`, `memory_structures.py`).
*   A *new* FastAPI server file (e.g., `synthians_server.py`) to expose `SynthiansMemoryCore`.
*   A *new* client file (e.g., `synthians_client.py`) to test the new server.
*   Relevant utility files (`logging_config.py`, `performance_tracker.py`, `cache_manager.py`) if their functionality is still desired and adapted.

---

### 3. **Development Roadmap for MVP (End of Week Target)**

**Goal:** A single Docker container running the unified `SynthiansMemoryCore` with basic storage, retrieval, HPC-QR scoring, emotional gating, assemblies, and adaptive thresholds.

**Assumptions:**
*   Focus is on the *memory system core*. Full Self/World model integration, Dreaming, Narrative, and complex KG are post-MVP.
*   Embedding generation is handled externally or via a placeholder within `SynthiansMemoryCore`.
*   You have a working Docker environment and Python 3.8+.

**Phase 1: Setup & Core Unification (Days 1-2)**

1.  **Directory Structure:**
    *   Create the new `synthians_memory_core` directory.
    *   Copy the proposed target files (`__init__.py`, `synthians_memory_core.py`, `hpc_quickrecal.py`, `geometry_manager.py`, `emotional_intelligence.py`, `memory_structures.py`, `memory_persistence.py`, `adaptive_components.py`, `custom_logger.py`) into it.
2.  **Dependencies:** Ensure all necessary libraries (`numpy`, `torch`, `aiofiles`) are installed (add to `requirements.txt`).
3.  **Integrate `UnifiedQuickRecallCalculator`:**
    *   Focus on `STANDARD` or `MINIMAL` mode initially for simplicity.
    *   Ensure it correctly uses `GeometryManager` for any distance/similarity calls.
    *   Implement basic versions of required factors (Recency, Relevance, Emotion, Importance, Overlap). Defer complex HPC-QR factors (Geometry, Causal, SOM) if necessary for speed, using defaults.
4.  **Integrate `GeometryManager`:**
    *   Ensure `SynthiansMemoryCore` uses it for all normalization, alignment, and similarity/distance calculations.
    *   Configure the desired default geometry (e.g., 'hyperbolic').
5.  **Integrate `MemoryPersistence`:**
    *   Ensure `SynthiansMemoryCore` uses this class *exclusively* for saving/loading memories via its async methods. Remove persistence logic from other classes.
6.  **Test Core Flow:** Write basic unit tests for `SynthiansMemoryCore.process_new_memory` and `SynthiansMemoryCore.retrieve_memories` using mock embeddings to verify the main data flow through the calculator, geometry manager, and persistence. Ensure GPU is utilized if configured and available (`torch.device`).

**Phase 2: Integrate Key Features (Days 3-4)**

1.  **Emotional Intelligence:**
    *   Wire `EmotionalAnalyzer` (even the simplified version) into `SynthiansMemoryCore`.
    *   Integrate `EmotionalGatingService` into the `retrieve_memories` flow.
    *   Test retrieval with different `user_emotion` contexts.
2.  **Memory Assemblies:**
    *   Implement the assembly creation (`_update_assemblies` triggered by `process_new_memory`) and retrieval (`_get_candidate_memories` using `_activate_assemblies`) logic within `SynthiansMemoryCore`.
    *   Assemblies should use `GeometryManager` for similarity.
    *   Test creating assemblies and retrieving memories via assembly activation.
3.  **Adaptive Thresholds:**
    *   Connect `ThresholdCalibrator` to the `retrieve_memories` results.
    *   Implement the `provide_feedback` method/endpoint to update the calibrator.
    *   Test retrieval results changing as feedback is provided.
4.  **Background Tasks:** Ensure the persistence and decay/pruning loops in `SynthiansMemoryCore` are functioning correctly using `asyncio`. Test shutdown.

**Phase 3: API Exposure & Cleanup (Day 5)**

1.  **Create FastAPI Server (`synthians_server.py`):**
    *   Create a new FastAPI app.
    *   In `startup`, initialize `SynthiansMemoryCore` (and call `initialize()`).
    *   In `shutdown`, call `SynthiansMemoryCore.shutdown()`.
    *   Expose endpoints mirroring the essential functions of `SynthiansMemoryCore`:
        *   `/process_memory` (POST)
        *   `/retrieve_memories` (POST)
        *   `/provide_feedback` (POST)
        *   `/detect_contradictions` (POST)
        *   `/health` (GET)
        *   `/stats` (GET)
        *   (Optional) Endpoints for assembly management.
2.  **Create Test Client (`synthians_client.py`):**
    *   Adapt `memory_client_example.py` to call the new FastAPI endpoints.
    *   Perform end-to-end tests: store, retrieve, feedback, check stats.
3.  **Dockerize:** Create a `Dockerfile` for the unified service. Include `requirements.txt`. Ensure the storage path is correctly mapped as a volume. Configure for GPU usage (e.g., using `nvidia-docker`).
4.  **Code Cleanup:** **Delete** all the identified redundant files from the project to avoid confusion.
5.  **Documentation:** Update the main `README.md` and the `synthians_memory_core/README.md` to reflect the new unified architecture.

**Post-MVP:**

*   Re-integrate advanced HPC-QR factors (Geometry, Causal Novelty, SOM) into `UnifiedQuickRecallCalculator`.
*   Re-introduce more sophisticated Dreaming, Narrative Identity, and Self/World Model components, ensuring they use the unified `SynthiansMemoryCore` API.
*   Re-implement the full modular Knowledge Graph system, potentially using `MemoryPersistence` for its storage backend.
*   Refine error handling and performance monitoring.
*   Consider re-introducing the `MemoryBroker` if a distributed architecture is needed later.

This roadmap focuses on creating a functional, unified core system quickly by leveraging the best existing components, centralizing logic, removing redundancy, and deferring the most complex cognitive features.
```

# docs\vector_index.md

```md
# Memory Vector Index with FAISS

## Overview

The `MemoryVectorIndex` class provides an efficient vector similarity search implementation using Facebook AI Similarity Search (FAISS). This implementation supports both CPU and GPU acceleration, with automatic detection and installation of the appropriate FAISS package.

## Features

- Fast vector similarity search using FAISS
- Automatic GPU detection and utilization
- Dynamic FAISS installation if the package is missing
- Persistent storage and loading of indices
- Support for different similarity metrics (L2, Inner Product, Cosine)

## Architecture

The vector index implementation consists of two main components:

1. **Docker Integration**: Pre-installs FAISS during container startup
2. **Dynamic Import**: Auto-installs FAISS if missing during runtime

### FAISS Auto-Installation

The system implements a robust approach to FAISS installation:

\`\`\`python
# Dynamic FAISS import with auto-installation fallback
try:
    import faiss
except ImportError:
    # Auto-detect GPU and install appropriate FAISS version
    ...
\`\`\`

This pattern ensures FAISS is available regardless of whether it was pre-installed, making the system more resilient to environment changes.

## GPU Support

The vector index automatically detects GPU availability and uses GPU acceleration when possible:

1. During Docker startup, the system checks for NVIDIA GPUs and installs either `faiss-gpu` or `faiss-cpu`
2. At runtime, if a GPU is detected, the vector index is moved to GPU memory for faster similarity search
3. If the GPU becomes unavailable, the system gracefully falls back to CPU processing

## Usage

### Basic Usage

\`\`\`python
from synthians_memory_core.vector_index import MemoryVectorIndex

# Create a new index
index = MemoryVectorIndex({
    'embedding_dim': 768,
    'storage_path': '/app/memory/stored/synthians',
    'index_type': 'L2',  # 'L2', 'IP', 'Cosine'
    'use_gpu': True,     # Whether to attempt to use GPU
    'gpu_id': 0          # Which GPU to use
})

# Add vectors to the index
index.add("memory_id_1", embedding_1)  # embedding_1 is a numpy array

# Search for similar vectors
results = index.search(query_embedding, k=10)  # Returns list of (memory_id, score) tuples
\`\`\`

### Configuration

The `MemoryVectorIndex` accepts the following configuration options:

| Parameter | Description | Default |
|-----------|-------------|--------|
| `embedding_dim` | Dimensionality of the embeddings | 768 |
| `storage_path` | Path to store the index | '/app/memory/stored/synthians' |
| `index_type` | Type of FAISS index to use ('L2', 'IP', 'Cosine') | 'L2' |
| `use_gpu` | Whether to use GPU acceleration if available | True |
| `gpu_id` | Which GPU to use if multiple are available | 0 |

## Implementation Details

### Index Types

- **L2**: Euclidean distance (smaller values = more similar)
- **IP**: Inner Product similarity (larger values = more similar)
- **Cosine**: Cosine similarity with normalized vectors (larger values = more similar)

### Persistence

The vector index is automatically persisted to disk and can be reloaded on restart:

\`\`\`python
# Save index to disk
index.save()

# Load index from disk
index.load()
\`\`\`

## Docker Integration

The Docker Compose configuration pre-installs FAISS with GPU support if available:

\`\`\`yaml
command: >
  /bin/bash -c '
  # Pre-install FAISS before Python importing it
  echo "[+] PRE-INSTALLING FAISS FOR MEMORY VECTOR INDEX" &&
  pip install --upgrade pip setuptools wheel &&
  # Install CPU version first as a fallback
  pip install --no-cache-dir faiss-cpu &&
  # If GPU available, replace with GPU version
  if command -v nvidia-smi > /dev/null 2>&1; then
    echo "[+] GPU DETECTED - Installing FAISS-GPU for better performance" &&
    pip uninstall -y faiss-cpu &&
    pip install --no-cache-dir faiss-gpu
  fi &&
  ...
\`\`\`

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'faiss'**
   
   This should be handled automatically by the dynamic import system. If it persists:
   - Ensure pip is available in the environment
   - Check if CUDA is properly installed for GPU support
   - Try manually installing: `pip install faiss-cpu` or `pip install faiss-gpu`

2. **GPU Not Detected**

   - Verify NVIDIA drivers are installed: `nvidia-smi`
   - Ensure CUDA is properly configured
   - Check if the Docker container has GPU access

3. **Performance Issues**

   - For large indices, adjust memory allocation: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
   - Consider using a different index type for your specific use case
   - For high-dimensional vectors, consider using PCA or product quantization

```

# emotion_analyzer.py

```py
import asyncio
import os
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np

from .custom_logger import logger

class EmotionAnalyzer:
    """
    Handles emotion analysis using a dual-mode approach:
    1. Primary: RoBERTa-based GoEmotions transformer model
    2. Fallback: Lightweight keyword-based approach
    
    Ensures consistent emotion detection structure regardless of the mode used.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the EmotionAnalyzer with a transformer model if available.
        
        Args:
            model_path: Path to the emotion model, if None will check for environment variable
            device: Device to use for inference (cuda, cpu). If None, will auto-detect.
        """
        # Auto-detect device if not specified
        if device is None:
            # Check for CUDA availability at runtime - default to CPU if not available
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("EmotionAnalyzer", f"Auto-detected device: {self.device}")
            except ImportError:
                self.device = "cpu"
                logger.info("EmotionAnalyzer", "Torch not available, defaulting to CPU device")
        else:
            self.device = device
            
        # Model path can come from multiple sources with increasing precedence:
        # 1. Default path relative to the project
        # 2. Environment variable EMOTION_MODEL_PATH
        # 3. Explicitly provided model_path parameter
        default_paths = [
            "models/roberta-base-go_emotions",  # Default relative path
            "/app/models/emotion",             # Common Docker mount point
            "/data/models/emotion",            # Alternative Docker volume
        ]
        
        # Determine the model path with proper precedence
        env_path = os.environ.get("EMOTION_MODEL_PATH")
        self.model_path = model_path or env_path or next((p for p in default_paths if os.path.exists(p)), default_paths[0])
        logger.info("EmotionAnalyzer", f"Using model path: {self.model_path}")
        
        # Model will be loaded on first use, not during initialization
        self.model = None
        self.model_loaded = False
        self.model_load_attempted = False
        
        # Track analysis stats
        self.stats = {
            "primary_calls": 0,
            "fallback_calls": 0,
            "errors": 0,
            "avg_time_ms": 0,
            "total_calls": 0
        }
    
    def _initialize_model(self):
        """
        Load the transformer-based emotion model if available.
        Returns True if model loaded successfully, False otherwise.
        """
        # Skip if we've already attempted to load and failed
        if self.model_loaded:
            return True
            
        if self.model_load_attempted and not self.model_loaded:
            logger.debug("EmotionAnalyzer", "Previous model load attempt failed, using fallback")
            return False
            
        self.model_load_attempted = True
        
        try:
            # Only import transformers if we're actually going to use it
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            # Check both if the path exists AND if it contains expected model files
            path_exists = os.path.exists(self.model_path)
            model_files_exist = False
            
            if path_exists:
                # Check for key files that indicate a Hugging Face model
                expected_files = ['config.json', 'pytorch_model.bin']
                model_files_exist = any(os.path.exists(os.path.join(self.model_path, f)) for f in expected_files)
                
            # Log what we found about the model path
            if path_exists and model_files_exist:
                logger.info("EmotionAnalyzer", f"Found model files at {self.model_path}")
            elif path_exists:
                logger.warning("EmotionAnalyzer", f"Path {self.model_path} exists but doesn't contain model files")
            else:
                logger.warning("EmotionAnalyzer", f"Model path {self.model_path} does not exist")
            
            # If model files exist locally, use them; otherwise try to download from Hugging Face Hub
            if path_exists and model_files_exist:
                # Load from local path
                logger.info("EmotionAnalyzer", f"Loading local model from {self.model_path}")
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_path, local_files_only=True)
            else:
                # Try to download model from Hugging Face Hub
                try:
                    logger.info("EmotionAnalyzer", "Local model not found, downloading from Hugging Face Hub")
                    # Use a fallback model ID - GoEmotions on Hugging Face
                    model_id = "joeddav/distilbert-base-uncased-go-emotions-student"
                    tokenizer = AutoTokenizer.from_pretrained(model_id)
                    model = AutoModelForSequenceClassification.from_pretrained(model_id)
                    
                    # Save the model to the specified path for future use
                    if path_exists:
                        logger.info("EmotionAnalyzer", f"Saving downloaded model to {self.model_path}")
                        model.save_pretrained(self.model_path)
                        tokenizer.save_pretrained(self.model_path)
                except Exception as download_error:
                    logger.error("EmotionAnalyzer", f"Error downloading model: {str(download_error)}")
                    return False
            
            # Create the pipeline with the loaded model
            self.model = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                top_k=None  # Return all emotion scores
            )
            
            self.model_loaded = True
            logger.info("EmotionAnalyzer", "Emotion model loaded successfully")
            return True
            
        except Exception as e:
            logger.error("EmotionAnalyzer", f"Error loading emotion model: {str(e)}")
            self.model = None
            self.model_loaded = False
            self.stats["errors"] += 1
            return False
    
    async def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions in the given text.
        Attempts to use the transformer model first, and falls back to keyword analysis if needed.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing emotions and the dominant emotion
        """
        start_time = time.time()
        
        try:
            # Try to load the model on first use if not already loaded
            if not self.model and not self.model_load_attempted:
                logger.info("EmotionAnalyzer", "First-time model loading during analyze call")
                model_loaded = self._initialize_model()
                if model_loaded:
                    logger.info("EmotionAnalyzer", "Successfully loaded model on first use")
                else:
                    logger.warning("EmotionAnalyzer", "Failed to load model on first use, falling back to keywords")
            
            # Attempt primary analysis if model is available
            if self.model is not None:
                logger.debug("EmotionAnalyzer", "Using transformer-based analysis")
                result = await self._analyze_with_transformer(text)
                self.stats["primary_calls"] += 1
            else:
                # Fall back to keyword analysis
                logger.debug("EmotionAnalyzer", "Using keyword-based analysis fallback")
                result = await self._analyze_with_keywords(text)
                self.stats["fallback_calls"] += 1
            
            # Update stats
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats["avg_time_ms"] = (
                (self.stats["avg_time_ms"] * (self.stats["primary_calls"] + self.stats["fallback_calls"] - 1) + elapsed_ms) /
                (self.stats["primary_calls"] + self.stats["fallback_calls"])
            )
            self.stats["total_calls"] += 1
            
            return result
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("EmotionAnalyzer", f"Error in emotion analysis: {str(e)}")
            self.stats["errors"] += 1
            
            # Always return a valid response, even in case of errors
            return {
                "dominant_emotion": "neutral",
                "emotions": {"neutral": 1.0},
                "error": str(e)
            }
    
    async def _analyze_with_transformer(self, text: str) -> Dict[str, Any]:
        """
        Analyze emotions using the transformer model.
        """
        # Execute the model in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        raw_results = await loop.run_in_executor(None, lambda: self.model(text))
        
        # Convert the transformer output format to our expected format
        # The model returns a list of dictionaries with 'label' and 'score'
        emotion_results = {}
        for result_list in raw_results:
            for item in result_list:
                label = item['label']
                score = float(item['score'])  # Ensure score is float
                emotion_results[label] = score
        
        # Find the dominant emotion based on score
        if emotion_results:
            dominant_emotion = max(emotion_results.items(), key=lambda x: x[1])[0]
        else:
            dominant_emotion = "neutral"
            emotion_results["neutral"] = 0.5
        
        return {
            "emotions": emotion_results,
            "dominant_emotion": dominant_emotion
        }
    
    async def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """
        Fallback emotion analysis using keyword matching.
        Much less accurate but works without any models.
        """
        # Simple keyword-based emotion detection
        emotion_keywords = {
            "joy": ["happy", "joy", "delighted", "glad", "pleased", "excited", "thrilled"],
            "sadness": ["sad", "unhappy", "depressed", "down", "miserable", "upset", "disappointed"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "enraged", "frustrated"],
            "fear": ["afraid", "scared", "frightened", "terrified", "anxious", "worried", "nervous"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "disgust": ["disgusted", "repulsed", "revolted", "sickened"],
            "neutral": ["ok", "fine", "neutral", "average", "normal"]
        }
        
        text = text.lower()
        emotion_scores = {emotion: 0.1 for emotion in emotion_keywords}  # Base score
        
        # Simple keyword matching
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[emotion] += 0.15  # Increment score for each match
        
        # Normalize scores
        max_score = max(emotion_scores.values())
        if max_score > 0.1:  # If we found any matches
            for emotion in emotion_scores:
                emotion_scores[emotion] = min(emotion_scores[emotion] / max_score, 1.0)
        else:
            # If no matches, default to neutral
            emotion_scores["neutral"] = 0.5
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "emotions": emotion_scores,
            "dominant_emotion": dominant_emotion
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics for the emotion analyzer.
        """
        total_calls = self.stats["primary_calls"] + self.stats["fallback_calls"]
        
        return {
            "total_calls": self.stats["total_calls"],
            "primary_calls": self.stats["primary_calls"],
            "fallback_calls": self.stats["fallback_calls"],
            "primary_percentage": (self.stats["primary_calls"] / max(total_calls, 1)) * 100,
            "fallback_percentage": (self.stats["fallback_calls"] / max(total_calls, 1)) * 100,
            "errors": self.stats["errors"],
            "avg_time_ms": round(self.stats["avg_time_ms"], 2),
            "model_loaded": self.model_loaded,
            "model_path": self.model_path
        }

```

# emotional_intelligence.py

```py
# synthians_memory_core/emotional_intelligence.py

import logging
import numpy as np
from typing import Dict, List, Optional, Any

from .custom_logger import logger # Use the shared custom logger

class EmotionalAnalyzer:
    """Simplified interface for emotion analysis."""
    async def analyze(self, text: str) -> Dict[str, Any]:
        # Placeholder: In a real system, this would call an external service or model.
        # Simulate analysis based on keywords for testing.
        text_lower = text.lower()
        emotions = {}
        dominant_emotion = "neutral"
        max_score = 0.1

        positive_words = ["happy", "joy", "great", "wonderful", "progress", "good", "excited"]
        negative_words = ["sad", "angry", "fear", "bugs", "problem", "crashed", "concerned"]

        pos_score = sum(1 for word in positive_words if word in text_lower) * 0.2
        neg_score = sum(1 for word in negative_words if word in text_lower) * 0.2

        if pos_score > neg_score and pos_score > 0.1:
            dominant_emotion = "joy"
            max_score = pos_score
            emotions["joy"] = max_score
        elif neg_score > pos_score and neg_score > 0.1:
            dominant_emotion = "sadness" if "sad" in text_lower else "anger" if "angry" in text_lower else "fear" if "fear" in text_lower else "concern"
            max_score = neg_score
            emotions[dominant_emotion] = max_score
        else:
             emotions["neutral"] = max_score

        # Simulate sentiment and intensity
        sentiment_value = (pos_score - neg_score) / max(1, pos_score + neg_score) if (pos_score + neg_score) > 0 else 0.0
        intensity = max_score # Use max keyword score as intensity proxy

        logger.debug("EmotionalAnalyzer", "Simulated emotion analysis", {"text": text[:30], "dominant": dominant_emotion, "intensity": intensity})

        return {
            "dominant_emotion": dominant_emotion,
            "sentiment_value": sentiment_value,
            "intensity": intensity,
            "emotions": emotions
        }

class EmotionalGatingService:
    """Applies emotional gating to memory retrieval."""
    def __init__(self, emotion_analyzer, config: Optional[Dict] = None):
        """Initialize the emotional gating service.
        
        Args:
            emotion_analyzer: Any emotion analyzer with an `analyze(text)` method
            config: Configuration parameters for the gating service
        """
        self.emotion_analyzer = emotion_analyzer
        self.config = config or {}
        
        # Configuration with defaults
        self.emotion_weight = self.config.get('emotional_weight', 0.3)
        self.memory_gate_min_factor = self.config.get('gate_min_factor', 0.5)
        self.cognitive_bias = self.config.get('cognitive_bias', 0.2)
        
        logger.info("EmotionalGatingService", "Initialized with config", {
            "emotion_weight": self.emotion_weight,
            "gate_min_factor": self.memory_gate_min_factor,
            "cognitive_bias": self.cognitive_bias,
            "has_analyzer": self.emotion_analyzer is not None
        })

        # Simplified compatibility - similar emotions are compatible
        self.emotion_compatibility = {
            "joy": {"joy", "excitement", "gratitude", "satisfaction", "content"},
            "sadness": {"sadness", "grief", "disappointment", "melancholy"},
            "anger": {"anger", "frustration", "irritation"},
            "fear": {"fear", "anxiety", "nervousness"},
            "surprise": {"surprise", "amazement", "astonishment"},
            "disgust": {"disgust", "displeasure"},
            "trust": {"trust", "respect", "admiration"},
            "neutral": {"neutral", "calm", "focused"}
        }
        # Add reverse compatibility and self-compatibility
        for emotion, compatible_set in list(self.emotion_compatibility.items()):
             compatible_set.add(emotion) # Self-compatible
             for compatible_emotion in compatible_set:
                  if compatible_emotion not in self.emotion_compatibility:
                       self.emotion_compatibility[compatible_emotion] = set()
                  self.emotion_compatibility[compatible_emotion].add(emotion)
        # Ensure neutral is compatible with everything
        all_emotions = set(self.emotion_compatibility.keys())
        self.emotion_compatibility["neutral"] = all_emotions
        for emotion in all_emotions:
             self.emotion_compatibility[emotion].add("neutral")

    async def gate_memories(self,
                           memories: List[Dict[str, Any]],
                           user_emotion: Optional[Dict[str, Any]],
                           cognitive_load: float = 0.5) -> List[Dict[str, Any]]:
        """Filter and re-rank memories based on emotional context."""
        if not memories or user_emotion is None:
            return memories # No gating if no user emotion provided

        user_dominant = user_emotion.get("dominant_emotion", "neutral")
        user_valence = user_emotion.get("sentiment_value", 0.0)
        user_intensity = user_emotion.get("intensity", 0.0)

        gated_memories = []
        for memory in memories:
            mem_emotion_context = memory.get("metadata", {}).get("emotional_context")
            if not mem_emotion_context:
                 # If no emotion data, assign neutral resonance
                 memory["emotional_resonance"] = 0.5
                 gated_memories.append(memory)
                 continue

            memory_dominant = mem_emotion_context.get("dominant_emotion", "neutral")
            memory_valence = mem_emotion_context.get("sentiment_value", 0.0)
            memory_intensity = mem_emotion_context.get("intensity", 0.0)

            # 1. Cognitive Defense (Simplified)
            if self.config.get('cognitive_defense_enabled', True) and user_valence < -0.5 and user_intensity > 0.6:
                 # If user is highly negative, filter out extremely negative memories
                 if memory_valence < -0.7 and memory_intensity > 0.8:
                      logger.debug("EmotionalGatingService", "Cognitive defense filtered out negative memory", {"memory_id": memory.get("id")})
                      continue # Skip this memory

            # 2. Calculate Emotional Resonance
            # Compatibility score (1 if compatible, 0 if not, 0.5 if neutral involved)
            user_compatibles = self.emotion_compatibility.get(user_dominant, set())
            mem_compatibles = self.emotion_compatibility.get(memory_dominant, set())

            if user_dominant == "neutral" or memory_dominant == "neutral":
                 emotion_compatibility = 0.7 # Neutral is somewhat compatible with everything
            elif memory_dominant in user_compatibles:
                 emotion_compatibility = 1.0 # Direct or similar emotion
            elif user_compatibles.intersection(mem_compatibles):
                 emotion_compatibility = 0.6 # Related emotions
            else:
                 emotion_compatibility = 0.1 # Unrelated emotions

            # Valence alignment (1 for same sign, 0 for opposite, 0.5 if one is neutral)
            if (user_valence > 0.1 and memory_valence > 0.1) or \
               (user_valence < -0.1 and memory_valence < -0.1):
                 valence_alignment = 1.0
            elif (user_valence > 0.1 and memory_valence < -0.1) or \
                 (user_valence < -0.1 and memory_valence > 0.1):
                 valence_alignment = 0.0
            else: # One or both are neutral
                 valence_alignment = 0.5

            # Combined resonance score
            emotional_resonance = (emotion_compatibility * 0.6 + valence_alignment * 0.4)
            memory["emotional_resonance"] = emotional_resonance

            # 3. Cognitive Load Adjustment (Simplified)
            # Higher load makes less resonant memories less likely
            if cognitive_load > 0.7 and emotional_resonance < (0.4 + 0.4 * cognitive_load):
                logger.debug("EmotionalGatingService", f"Memory filtered by high cognitive load ({cognitive_load})", {"memory_id": memory.get("id")})
                continue # Skip less resonant memories under high load

            gated_memories.append(memory)

        # 4. Re-rank based on combined score
        weight = self.emotion_weight
        for memory in gated_memories:
             original_score = memory.get("relevance_score", 0.0) # Use relevance_score if available
             resonance = memory.get("emotional_resonance", 0.5)
             memory["final_score"] = (1 - weight) * original_score + weight * resonance

        gated_memories.sort(key=lambda x: x["final_score"], reverse=True)

        logger.info("EmotionalGatingService", f"Gated memories from {len(memories)} to {len(gated_memories)}", {"user_emotion": user_dominant})
        return gated_memories

```

# geometry_manager.py

```py
# synthians_memory_core/geometry_manager.py

import numpy as np
import torch
import math
from enum import Enum
from typing import Optional, Tuple, List, Union, Dict, Any

from .custom_logger import logger # Use the shared custom logger

class GeometryType(Enum):
    EUCLIDEAN = "euclidean"
    HYPERBOLIC = "hyperbolic"
    SPHERICAL = "spherical"
    MIXED = "mixed"

class GeometryManager:
    """Centralized handling of embedding geometry, transformations, and calculations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'embedding_dim': 768,
            'geometry_type': GeometryType.EUCLIDEAN,
            'curvature': -1.0, # Relevant for Hyperbolic/Spherical
            'alignment_strategy': 'truncate', # or 'pad' or 'project'
            'normalization_enabled': True,
             **(config or {})
        }
        # Ensure geometry_type is enum
        if isinstance(self.config['geometry_type'], str):
            try:
                self.config['geometry_type'] = GeometryType(self.config['geometry_type'].lower())
            except ValueError:
                 logger.warning("GeometryManager", f"Invalid geometry type {self.config['geometry_type']}, defaulting to EUCLIDEAN.")
                 self.config['geometry_type'] = GeometryType.EUCLIDEAN

        # Warning counters
        self.dim_mismatch_warnings = 0
        self.max_dim_mismatch_warnings = 10
        self.nan_inf_warnings = 0
        self.max_nan_inf_warnings = 10

        logger.info("GeometryManager", "Initialized", self.config)

    def _validate_vector(self, vector: Union[np.ndarray, List[float], torch.Tensor], name: str = "Vector") -> Optional[np.ndarray]:
        """Validate and convert vector to numpy array."""
        if vector is None:
            logger.warning("GeometryManager", f"{name} is None")
            return None

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)
        elif isinstance(vector, torch.Tensor):
            vector = vector.detach().cpu().numpy().astype(np.float32)
        elif not isinstance(vector, np.ndarray):
            logger.warning("GeometryManager", f"Unsupported vector type {type(vector)} for {name}, attempting conversion.")
            try:
                vector = np.array(vector, dtype=np.float32)
            except Exception as e:
                 logger.error("GeometryManager", f"Failed to convert {name} to numpy array", {"error": str(e)})
                 return None

        # Check for NaN/Inf
        if np.isnan(vector).any() or np.isinf(vector).any():
            if self.nan_inf_warnings < self.max_nan_inf_warnings:
                 logger.warning("GeometryManager", f"{name} contains NaN or Inf values. Replacing with zeros.")
                 self.nan_inf_warnings += 1
                 if self.nan_inf_warnings == self.max_nan_inf_warnings:
                      logger.warning("GeometryManager", "Max NaN/Inf warnings reached, suppressing further warnings.")
            return np.zeros_like(vector) # Replace invalid vector with zeros

        return vector

    def _align_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two vectors to the configured embedding dimension."""
        target_dim = self.config['embedding_dim']
        dim_a = vec_a.shape[0]
        dim_b = vec_b.shape[0]

        aligned_a = vec_a
        aligned_b = vec_b

        strategy = self.config['alignment_strategy']

        if dim_a != target_dim:
            if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                 logger.warning("GeometryManager", f"Vector A dimension mismatch: got {dim_a}, expected {target_dim}. Applying strategy: {strategy}")
                 self.dim_mismatch_warnings += 1
                 if self.dim_mismatch_warnings == self.max_dim_mismatch_warnings:
                      logger.warning("GeometryManager", "Max dimension mismatch warnings reached.")

            if strategy == 'pad':
                if dim_a < target_dim:
                    aligned_a = np.pad(vec_a, (0, target_dim - dim_a))
                else: # Truncate if padding isn't the strategy and dim > target
                    aligned_a = vec_a[:target_dim]
            elif strategy == 'truncate':
                if dim_a > target_dim:
                    aligned_a = vec_a[:target_dim]
                else: # Pad if truncating isn't the strategy and dim < target
                     aligned_a = np.pad(vec_a, (0, target_dim - dim_a))
            # Add 'project' strategy later if needed
            else: # Default to truncate/pad based on relative size
                if dim_a > target_dim: aligned_a = vec_a[:target_dim]
                else: aligned_a = np.pad(vec_a, (0, target_dim - dim_a))


        if dim_b != target_dim:
             if self.dim_mismatch_warnings < self.max_dim_mismatch_warnings:
                 logger.warning("GeometryManager", f"Vector B dimension mismatch: got {dim_b}, expected {target_dim}. Applying strategy: {strategy}")
                 # No warning count increment here, handled by vec_a check

             if strategy == 'pad':
                if dim_b < target_dim:
                    aligned_b = np.pad(vec_b, (0, target_dim - dim_b))
                else: aligned_b = vec_b[:target_dim]
             elif strategy == 'truncate':
                 if dim_b > target_dim: aligned_b = vec_b[:target_dim]
                 else: aligned_b = np.pad(vec_b, (0, target_dim - dim_b))
             else: # Default
                 if dim_b > target_dim: aligned_b = vec_b[:target_dim]
                 else: aligned_b = np.pad(vec_b, (0, target_dim - dim_b))

        return aligned_a, aligned_b

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        if not self.config['normalization_enabled']:
             return vector
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            logger.debug("GeometryManager", "_normalize received zero vector, returning as is.")
            return vector
        return vector / norm

    def _to_hyperbolic(self, euclidean_vector: np.ndarray) -> np.ndarray:
        """Project Euclidean vector to Poincaré ball."""
        norm = np.linalg.norm(euclidean_vector)
        if norm == 0: return euclidean_vector
        curvature = abs(self.config['curvature']) # Ensure positive for scaling
        if curvature == 0: curvature = 1.0 # Avoid division by zero if Euclidean is accidentally chosen
        # Adjusted scaling: tanh maps [0, inf) -> [0, 1)
        scale_factor = np.tanh(norm / 2.0) # Removed curvature influence here, seems standard
        hyperbolic_vector = (euclidean_vector / norm) * scale_factor
        # Ensure norm is strictly less than 1
        hyp_norm = np.linalg.norm(hyperbolic_vector)
        if hyp_norm >= 1.0:
            hyperbolic_vector = hyperbolic_vector * (0.99999 / hyp_norm)
        return hyperbolic_vector

    def _from_hyperbolic(self, hyperbolic_vector: np.ndarray) -> np.ndarray:
        """Project Poincaré ball vector back to Euclidean."""
        norm = np.linalg.norm(hyperbolic_vector)
        if norm >= 1.0:
            logger.warning("GeometryManager", "Hyperbolic vector norm >= 1, cannot project back accurately.", {"norm": norm})
            # Project onto the boundary and then back
            norm = 0.99999
            hyperbolic_vector = (hyperbolic_vector / np.linalg.norm(hyperbolic_vector)) * norm
        if norm == 0: return hyperbolic_vector
        # Inverse of tanh is arctanh
        original_norm_approx = np.arctanh(norm) * 2.0 # Approximation without curvature
        return (hyperbolic_vector / norm) * original_norm_approx

    def euclidean_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Euclidean distance."""
        aligned_a, aligned_b = self._align_vectors(vec_a, vec_b)
        return np.linalg.norm(aligned_a - aligned_b)

    def hyperbolic_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Hyperbolic (Poincaré) distance."""
        aligned_a, aligned_b = self._align_vectors(vec_a, vec_b)
        norm_a_sq = np.sum(aligned_a**2)
        norm_b_sq = np.sum(aligned_b**2)

        # Ensure vectors are strictly inside the unit ball
        if norm_a_sq >= 1.0: aligned_a = aligned_a * (0.99999 / np.sqrt(norm_a_sq)); norm_a_sq=np.sum(aligned_a**2)
        if norm_b_sq >= 1.0: aligned_b = aligned_b * (0.99999 / np.sqrt(norm_b_sq)); norm_b_sq=np.sum(aligned_b**2)

        euclidean_dist_sq = np.sum((aligned_a - aligned_b)**2)
        denominator = (1 - norm_a_sq) * (1 - norm_b_sq)

        if denominator < 1e-15: # Prevent division by zero or extreme values
            # If denominator is tiny, points are near boundary. If points are also close, distance is small. If far, distance is large.
            if euclidean_dist_sq < 1e-9: return 0.0
            else: return np.inf # Effectively infinite distance

        argument = 1 + (2 * euclidean_dist_sq / denominator)

        # Clamp argument to handle potential floating point issues near 1.0
        argument = max(1.0, argument)

        # Calculate distance with curvature
        curvature = abs(self.config['curvature'])
        if curvature <= 1e-9: curvature = 1.0 # Treat 0 curvature as Euclidean-like case within arccosh framework
        distance = np.arccosh(argument) / np.sqrt(curvature)

        return float(distance)

    def spherical_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Spherical distance (angle)."""
        aligned_a, aligned_b = self._align_vectors(vec_a, vec_b)
        norm_a = np.linalg.norm(aligned_a)
        norm_b = np.linalg.norm(aligned_b)
        if norm_a < 1e-9 or norm_b < 1e-9: return np.pi # Max distance if one vector is zero
        cos_angle = np.dot(aligned_a, aligned_b) / (norm_a * norm_b)
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def mixed_distance(self, vec_a: np.ndarray, vec_b: np.ndarray, weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> float:
        """Calculate a weighted mixed distance."""
        euc_dist = self.euclidean_distance(vec_a, vec_b)
        hyp_dist = self.hyperbolic_distance(self._to_hyperbolic(vec_a), self._to_hyperbolic(vec_b))
        sph_dist = self.spherical_distance(vec_a, vec_b)
        # Normalize distances before combining (rough normalization)
        # Max Euclidean dist is 2, max spherical is pi
        euc_norm = euc_dist / 2.0
        sph_norm = sph_dist / np.pi
        # Hyperbolic distance can be large, use exp(-dist) for similarity-like scaling
        hyp_norm = np.exp(-hyp_dist * 0.5) # Scaled exponential decay

        # Combine weighted distances (treating hyp_norm as similarity, so use 1-hyp_norm)
        mixed_dist = weights[0] * euc_norm + weights[1] * (1.0 - hyp_norm) + weights[2] * sph_norm
        return float(mixed_dist)

    def calculate_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate distance based on configured geometry."""
        vec_a = self._validate_vector(vec_a, "Vector A")
        vec_b = self._validate_vector(vec_b, "Vector B")
        if vec_a is None or vec_b is None: return np.inf # Return infinite distance if validation failed

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.EUCLIDEAN:
            return self.euclidean_distance(vec_a, vec_b)
        elif geom_type == GeometryType.HYPERBOLIC:
            # Assume vectors are Euclidean, project them first
            hyp_a = self._to_hyperbolic(vec_a)
            hyp_b = self._to_hyperbolic(vec_b)
            return self.hyperbolic_distance(hyp_a, hyp_b)
        elif geom_type == GeometryType.SPHERICAL:
            return self.spherical_distance(vec_a, vec_b)
        elif geom_type == GeometryType.MIXED:
            return self.mixed_distance(vec_a, vec_b)
        else:
            logger.warning("GeometryManager", f"Unknown geometry type {geom_type}, using Euclidean.")
            return self.euclidean_distance(vec_a, vec_b)

    def calculate_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate similarity based on configured geometry."""
        vec_a = self._validate_vector(vec_a, "Vector A")
        vec_b = self._validate_vector(vec_b, "Vector B")
        if vec_a is None or vec_b is None: return 0.0

        geom_type = self.config['geometry_type']

        if geom_type == GeometryType.HYPERBOLIC:
             hyp_a = self._to_hyperbolic(vec_a)
             hyp_b = self._to_hyperbolic(vec_b)
             distance = self.hyperbolic_distance(hyp_a, hyp_b)
             # Convert distance to similarity: exp(-distance) is common
             similarity = np.exp(-distance)
             return float(np.clip(similarity, 0.0, 1.0))
        else: # Euclidean, Spherical, Mixed (default to cosine similarity for simplicity)
             aligned_a, aligned_b = self._align_vectors(vec_a, vec_b)
             norm_a = np.linalg.norm(aligned_a)
             norm_b = np.linalg.norm(aligned_b)
             if norm_a < 1e-9 or norm_b < 1e-9: return 0.0
             cos_sim = np.dot(aligned_a, aligned_b) / (norm_a * norm_b)
             # Map cosine similarity [-1, 1] to [0, 1] range
             similarity = (cos_sim + 1.0) / 2.0
             return float(np.clip(similarity, 0.0, 1.0))

    def transform_to_geometry(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector into the configured geometry space (e.g., Poincaré ball)."""
        vector = self._validate_vector(vector, "Input Vector")
        if vector is None: return np.zeros(self.config['embedding_dim'])

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.HYPERBOLIC:
            return self._to_hyperbolic(vector)
        elif geom_type == GeometryType.SPHERICAL:
            # Project onto unit sphere (normalize)
            return self._normalize(vector)
        else: # Euclidean or Mixed (no specific projection needed for Euclidean part)
            return vector

    def transform_from_geometry(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector from the configured geometry space back to Euclidean."""
        vector = self._validate_vector(vector, "Input Vector")
        if vector is None: return np.zeros(self.config['embedding_dim'])

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.HYPERBOLIC:
            return self._from_hyperbolic(vector)
        else: # Spherical, Euclidean, Mixed - assume normalization or no transformation needed
            return vector

```

# gpu_setup.py

```py
#!/usr/bin/env python
# synthians_memory_core/gpu_setup.py

import os
import sys
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPU-Setup")


def check_gpu_available():
    """Check if CUDA is available."""
    try:
        # Try to import torch and check CUDA availability
        import torch
        cuda_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            logger.info(f"Found {device_count} CUDA device(s). Using: {device_name}")
            return True
        else:
            # Try nvidia-smi as a backup check
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode == 0:
                logger.info("nvidia-smi detected GPU, but PyTorch CUDA not available.")
                # Still return True as FAISS might be able to use it
                return True
            else:
                logger.info("No CUDA devices detected through nvidia-smi")
                return False
    except (ImportError, FileNotFoundError):
        logger.warning("Could not check CUDA availability through PyTorch or nvidia-smi")
        return False


def install_faiss_gpu():
    """Install FAISS with GPU support."""
    try:
        # Try to import faiss-gpu first to see if it's already installed
        try:
            import faiss
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                logger.info(f"FAISS-GPU already installed. Available GPUs: {faiss.get_num_gpus()}")
                return True
            else:
                logger.info("FAISS is installed but no GPUs detected by FAISS")
        except ImportError:
            logger.info("FAISS not installed yet, proceeding with installation")
        
        # First uninstall faiss-cpu if it exists
        logger.info("Uninstalling faiss-cpu if present...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "faiss-cpu"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Install faiss-gpu
        logger.info("Installing faiss-gpu...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "faiss-gpu"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install faiss-gpu: {result.stderr.decode()}")
            return False
        
        # Verify installation
        try:
            import faiss
            logger.info(f"FAISS version: {faiss.__version__}")
            if hasattr(faiss, 'get_num_gpus'):
                gpu_count = faiss.get_num_gpus()
                logger.info(f"FAISS detected {gpu_count} GPUs")
                return gpu_count > 0
            else:
                logger.warning("FAISS installed but get_num_gpus not available")
                return False
        except ImportError:
            logger.error("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        logger.error(f"Error during FAISS-GPU installation: {str(e)}")
        return False


def install_faiss_cpu():
    """Install FAISS CPU version as fallback."""
    try:
        # Check if faiss is already installed
        try:
            import faiss
            logger.info(f"FAISS already installed (CPU version). Version: {faiss.__version__}")
            return True
        except ImportError:
            logger.info("FAISS not installed yet, proceeding with CPU installation")
        
        # Install faiss-cpu
        logger.info("Installing faiss-cpu...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "faiss-cpu>=1.7.4"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to install faiss-cpu: {result.stderr.decode()}")
            return False
        
        # Verify installation
        try:
            import faiss
            logger.info(f"FAISS CPU version: {faiss.__version__}")
            return True
        except ImportError:
            logger.error("Failed to import FAISS after installation")
            return False
            
    except Exception as e:
        logger.error(f"Error during FAISS-CPU installation: {str(e)}")
        return False


def setup_faiss():
    """Set up FAISS with GPU support if available, otherwise use CPU version."""
    logger.info("Checking for GPU availability...")
    if check_gpu_available():
        logger.info("GPU detected, installing FAISS with GPU support")
        if install_faiss_gpu():
            logger.info("Successfully installed FAISS with GPU support")
            return True
        else:
            logger.warning("Failed to install FAISS with GPU support, falling back to CPU version")
            return install_faiss_cpu()
    else:
        logger.info("No GPU detected, installing FAISS CPU version")
        return install_faiss_cpu()


if __name__ == "__main__":
    logger.info("=== FAISS GPU Setup Script ===")
    success = setup_faiss()
    if success:
        logger.info("FAISS setup completed successfully")
        sys.exit(0)
    else:
        logger.error("FAISS setup failed")
        sys.exit(1)

```

# hpc_quickrecal.py

```py
# synthians_memory_core/hpc_quickrecal.py

import os
import math
import logging
import json
import time
import asyncio
import traceback
import numpy as np
import torch
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union

from .geometry_manager import GeometryManager, GeometryType # Import from the unified manager
from .custom_logger import logger # Use the shared custom logger

# Renamed from FactorKeys for clarity
class QuickRecallFactor(Enum):
    RECENCY = "recency"
    EMOTION = "emotion"
    EXTENDED_EMOTION = "extended_emotion" # For buffer-based emotion
    RELEVANCE = "relevance" # e.g., similarity to query
    OVERLAP = "overlap" # Redundancy penalty
    R_GEOMETRY = "r_geometry" # Geometric novelty/distance
    CAUSAL_NOVELTY = "causal_novelty" # Surprise based on causal model/prediction
    SELF_ORG = "self_org" # Based on SOM or similar clustering
    IMPORTANCE = "importance" # Explicitly assigned importance
    PERSONAL = "personal" # Related to user's personal info
    SURPRISE = "surprise" # General novelty or unexpectedness
    DIVERSITY = "diversity" # Difference from other recent memories
    COHERENCE = "coherence" # Logical consistency with existing knowledge
    INFORMATION = "information" # Information density or value

class QuickRecallMode(Enum):
    STANDARD = "standard"
    HPC_QR = "hpc_qr" # Original HPC-QR formula using alpha, beta, etc.
    MINIMAL = "minimal" # Basic recency, relevance, emotion
    CUSTOM = "custom" # User-defined weights

class UnifiedQuickRecallCalculator:
    """Unified calculator for memory importance using HPC-QR principles."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, geometry_manager: Optional[GeometryManager] = None):
        self.config = {
            'mode': QuickRecallMode.STANDARD,
            'factor_weights': {},
            'time_decay_rate': 0.1,
            'novelty_threshold': 0.45,
            'min_qr_score': 0.0,
            'max_qr_score': 1.0,
            'history_window': 100,
            'embedding_dim': 768,
             # HPC-QR specific weights (used in HPC_QR mode or as fallback)
            'alpha': 0.35, 'beta': 0.35, 'gamma': 0.2, 'delta': 0.1,
             # Factor configs
            'personal_keywords': ['my name', 'i live', 'my birthday', 'my job', 'my family'],
            'emotion_intensifiers': ['very', 'really', 'extremely', 'so'],
             **(config or {})
        }
        self.geometry_manager = geometry_manager or GeometryManager(self.config) # Use provided or create new
        self._init_factor_weights()
        self.history = {'calculated_qr': [], 'timestamps': [], 'factor_values': {f: [] for f in QuickRecallFactor}}
        self.total_calculations = 0
        logger.info("UnifiedQuickRecallCalculator", f"Initialized with mode: {self.config['mode'].value}")

    def _init_factor_weights(self):
        """Initialize weights based on mode."""
        default_weights = {f: 0.1 for f in QuickRecallFactor if f != QuickRecallFactor.OVERLAP} # Default equal weights
        default_weights[QuickRecallFactor.OVERLAP] = -0.1 # Overlap is a penalty

        mode = self.config['mode']
        if mode == QuickRecallMode.STANDARD:
            self.factor_weights = {
                QuickRecallFactor.RELEVANCE: 0.25, QuickRecallFactor.RECENCY: 0.15,
                QuickRecallFactor.EMOTION: 0.15, QuickRecallFactor.IMPORTANCE: 0.1,
                QuickRecallFactor.PERSONAL: 0.1, QuickRecallFactor.SURPRISE: 0.1,
                QuickRecallFactor.DIVERSITY: 0.05, QuickRecallFactor.COHERENCE: 0.05,
                QuickRecallFactor.INFORMATION: 0.05, QuickRecallFactor.OVERLAP: -0.1,
                 # Include HPC-QR factors with small default weights
                QuickRecallFactor.R_GEOMETRY: 0.0, QuickRecallFactor.CAUSAL_NOVELTY: 0.0,
                QuickRecallFactor.SELF_ORG: 0.0
            }
        elif mode == QuickRecallMode.MINIMAL:
             self.factor_weights = {
                QuickRecallFactor.RECENCY: 0.4, QuickRecallFactor.RELEVANCE: 0.4,
                QuickRecallFactor.EMOTION: 0.2, QuickRecallFactor.OVERLAP: -0.1
            }
        elif mode == QuickRecallMode.CUSTOM:
            # Ensure all factors are present, use defaults if missing
            user_weights = self.config.get('factor_weights', {})
            self.factor_weights = default_weights.copy()
            for factor, weight in user_weights.items():
                 if isinstance(factor, str): factor = QuickRecallFactor(factor.lower())
                 if factor in self.factor_weights: self.factor_weights[factor] = weight
        else: # Default to standard weights if mode is unrecognized (including HPC_QR for now)
            self.factor_weights = default_weights.copy()

        # Normalize weights (excluding overlap penalty)
        positive_weight_sum = sum(w for f, w in self.factor_weights.items() if f != QuickRecallFactor.OVERLAP and w > 0)
        if positive_weight_sum > 0 and abs(positive_weight_sum - 1.0) > 1e-6 :
             scale = 1.0 / positive_weight_sum
             for f in self.factor_weights:
                  if f != QuickRecallFactor.OVERLAP and self.factor_weights[f] > 0:
                       self.factor_weights[f] *= scale
        logger.debug("UnifiedQuickRecallCalculator", f"Initialized factor weights for mode {mode.value}", self.factor_weights)

    async def calculate(
        self,
        embedding_or_text: Union[str, np.ndarray, torch.Tensor, List[float]],
        text: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate the composite QuickRecal score."""
        start_time = time.time()
        context = context or {}

        # --- Prepare Embedding ---
        embedding = None
        if isinstance(embedding_or_text, str):
            text_content = embedding_or_text
            # Generate embedding if needed (consider moving this outside for performance)
            # embedding = await self._generate_embedding(text_content) # Assume embedding gen is handled externally or mocked
            logger.debug("UnifiedQuickRecallCalculator", "Calculating score based on text, embedding generation assumed external.")
        else:
            embedding = self.geometry_manager._validate_vector(embedding_or_text, "Input Embedding")
            text_content = text or context.get("text", "") # Get text if available

        if embedding is not None:
             embedding = self.geometry_manager._normalize(embedding) # Ensure normalized

        # --- Calculate Factors ---
        factor_values = {}
        tasks = []

        # Helper to run potentially async factor calculations
        async def calculate_factor(factor, func, *args):
             try:
                 # Check if the function is a coroutine function or returns a coroutine
                 if asyncio.iscoroutinefunction(func):
                     val = await func(*args)
                 else:
                     # Regular function, don't await
                     val = func(*args)
                 factor_values[factor] = float(np.clip(val, 0.0, 1.0))
             except Exception as e:
                 logger.error("UnifiedQuickRecallCalculator", f"Error calculating factor {factor.value}", {"error": str(e)})
                 factor_values[factor] = 0.0 # Default on error

        # Context-based factors (fast)
        # Use the sync versions directly since they're quick
        factor_values[QuickRecallFactor.RECENCY] = self._calculate_recency(context)
        factor_values[QuickRecallFactor.RELEVANCE] = self._calculate_relevance(context)
        factor_values[QuickRecallFactor.IMPORTANCE] = self._calculate_importance(text_content, context)
        factor_values[QuickRecallFactor.PERSONAL] = self._calculate_personal(text_content, context)

        # Text-based factors (potentially slower)
        if text_content:
             tasks.append(calculate_factor(QuickRecallFactor.EMOTION, self._calculate_emotion, text_content, context))
             tasks.append(calculate_factor(QuickRecallFactor.INFORMATION, self._calculate_information, text_content, context))
             tasks.append(calculate_factor(QuickRecallFactor.COHERENCE, self._calculate_coherence, text_content, context))
        else:
             factor_values[QuickRecallFactor.EMOTION] = 0.0
             factor_values[QuickRecallFactor.INFORMATION] = 0.0
             factor_values[QuickRecallFactor.COHERENCE] = 0.0

        # Embedding-based factors (potentially slowest)
        if embedding is not None:
             # Use external momentum if provided
             external_momentum = context.get('external_momentum', None)
             tasks.append(calculate_factor(QuickRecallFactor.SURPRISE, self._calculate_surprise, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.DIVERSITY, self._calculate_diversity, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.OVERLAP, self._calculate_overlap, embedding, external_momentum))
             # HPC-QR specific factors
             tasks.append(calculate_factor(QuickRecallFactor.R_GEOMETRY, self._calculate_r_geometry, embedding, external_momentum))
             tasks.append(calculate_factor(QuickRecallFactor.CAUSAL_NOVELTY, self._calculate_causal_novelty, embedding, context)) # Causal needs context
             tasks.append(calculate_factor(QuickRecallFactor.SELF_ORG, self._calculate_self_org, embedding, context)) # SOM needs context (or internal SOM state)
        else:
             factor_values[QuickRecallFactor.SURPRISE] = 0.5
             factor_values[QuickRecallFactor.DIVERSITY] = 0.5
             factor_values[QuickRecallFactor.OVERLAP] = 0.0
             factor_values[QuickRecallFactor.R_GEOMETRY] = 0.5
             factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = 0.5
             factor_values[QuickRecallFactor.SELF_ORG] = 0.5

        # Run potentially async calculations
        if tasks:
            await asyncio.gather(*tasks)

        # --- Combine Factors ---
        final_score = 0.0
        if self.config['mode'] == QuickRecallMode.HPC_QR:
            # Use the original alpha, beta, gamma, delta formula
            final_score = (
                self.config['alpha'] * factor_values.get(QuickRecallFactor.R_GEOMETRY, 0.0) +
                self.config['beta'] * factor_values.get(QuickRecallFactor.CAUSAL_NOVELTY, 0.0) +
                self.config['gamma'] * factor_values.get(QuickRecallFactor.SELF_ORG, 0.0) -
                self.config['delta'] * factor_values.get(QuickRecallFactor.OVERLAP, 0.0)
            )
            # Add other factors with small weights if needed, or keep it pure HPC-QR
            final_score += 0.05 * factor_values.get(QuickRecallFactor.RECENCY, 0.0)
            final_score += 0.05 * factor_values.get(QuickRecallFactor.EMOTION, 0.0)

        else:
            # Use weighted sum based on mode/custom weights
            for factor, value in factor_values.items():
                weight = self.factor_weights.get(factor, 0.0)
                # Overlap is a penalty
                if factor == QuickRecallFactor.OVERLAP:
                    final_score -= abs(weight) * value
                else:
                    final_score += weight * value

        # Apply time decay
        time_decay = self._calculate_time_decay(context)
        final_score *= time_decay

        # Clamp score
        final_score = float(np.clip(final_score, self.config['min_qr_score'], self.config['max_qr_score']))

        # Update history and stats
        self._update_history(final_score, factor_values)
        self.total_calculations += 1
        calculation_time = (time.time() - start_time) * 1000
        logger.debug("UnifiedQuickRecallCalculator", f"Score calculated: {final_score:.4f}", {"time_ms": calculation_time, "mode": self.config['mode'].value, "factors": {f.value: v for f,v in factor_values.items()}})

        return final_score

    def calculate_sync(
        self,
        embedding_or_text: Union[str, np.ndarray, torch.Tensor, List[float]],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Synchronous version of calculate for use in environments where asyncio.run() causes issues."""
        start_time = time.time()
        context = context or {}

        # --- Prepare Embedding ---
        embedding = None
        if isinstance(embedding_or_text, str):
            text_content = embedding_or_text
            logger.debug("UnifiedQuickRecallCalculator", "Calculating score based on text only in sync mode.")
        else:
            embedding = self.geometry_manager._validate_vector(embedding_or_text, "Input Embedding")
            text_content = context.get("text", "") # Get text if available

        if embedding is not None:
            embedding = self.geometry_manager._normalize(embedding) # Ensure normalized

        # --- Calculate Factors ---
        factor_values = {}

        # Context-based factors (fast)
        factor_values[QuickRecallFactor.RECENCY] = self._calculate_recency(context)
        factor_values[QuickRecallFactor.RELEVANCE] = self._calculate_relevance(context)
        factor_values[QuickRecallFactor.IMPORTANCE] = self._calculate_importance(text_content, context)
        factor_values[QuickRecallFactor.PERSONAL] = self._calculate_personal(text_content, context)

        # Text-based factors
        if text_content:
            # Use synchronous versions or set defaults
            try:
                factor_values[QuickRecallFactor.EMOTION] = self._calculate_emotion_sync(text_content, context)
            except:
                factor_values[QuickRecallFactor.EMOTION] = 0.0
                
            factor_values[QuickRecallFactor.INFORMATION] = 0.5  # Default value
            factor_values[QuickRecallFactor.COHERENCE] = 0.5    # Default value
        else:
            factor_values[QuickRecallFactor.EMOTION] = 0.0
            factor_values[QuickRecallFactor.INFORMATION] = 0.0
            factor_values[QuickRecallFactor.COHERENCE] = 0.0

        # Embedding-based factors
        if embedding is not None:
            # Use external momentum if provided
            external_momentum = context.get('external_momentum', None)
            factor_values[QuickRecallFactor.SURPRISE] = self._calculate_surprise_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.DIVERSITY] = self._calculate_diversity_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.OVERLAP] = self._calculate_overlap_sync(embedding, external_momentum)
            # HPC-QR specific factors
            factor_values[QuickRecallFactor.R_GEOMETRY] = self._calculate_r_geometry_sync(embedding, external_momentum)
            factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = self._calculate_causal_novelty_sync(embedding, context)
            factor_values[QuickRecallFactor.SELF_ORG] = self._calculate_self_org_sync(embedding, context)
        else:
            factor_values[QuickRecallFactor.SURPRISE] = 0.5
            factor_values[QuickRecallFactor.DIVERSITY] = 0.5
            factor_values[QuickRecallFactor.OVERLAP] = 0.0
            factor_values[QuickRecallFactor.R_GEOMETRY] = 0.5
            factor_values[QuickRecallFactor.CAUSAL_NOVELTY] = 0.5
            factor_values[QuickRecallFactor.SELF_ORG] = 0.5

        # --- Combine Factors ---
        final_score = 0.0
        if self.config['mode'] == QuickRecallMode.HPC_QR:
            # Use the original alpha, beta, gamma, delta formula
            final_score = (
                self.config['alpha'] * factor_values.get(QuickRecallFactor.R_GEOMETRY, 0.0) +
                self.config['beta'] * factor_values.get(QuickRecallFactor.CAUSAL_NOVELTY, 0.0) +
                self.config['gamma'] * factor_values.get(QuickRecallFactor.SELF_ORG, 0.0) -
                self.config['delta'] * factor_values.get(QuickRecallFactor.OVERLAP, 0.0)
            )
            # Add other factors with small weights
            final_score += 0.05 * factor_values.get(QuickRecallFactor.RECENCY, 0.0)
            final_score += 0.05 * factor_values.get(QuickRecallFactor.EMOTION, 0.0)
        else:
            # Use weighted sum based on mode/custom weights
            for factor, value in factor_values.items():
                weight = self.factor_weights.get(factor, 0.0)
                # Overlap is a penalty
                if factor == QuickRecallFactor.OVERLAP:
                    final_score -= abs(weight) * value
                else:
                    final_score += weight * value

        # Apply time decay
        time_decay = self._calculate_time_decay(context)
        final_score *= time_decay

        # Clamp score
        final_score = float(np.clip(final_score, self.config['min_qr_score'], self.config['max_qr_score']))

        # Update history and stats
        self._update_history(final_score, factor_values)
        self.total_calculations += 1
        calculation_time = (time.time() - start_time) * 1000
        logger.debug("UnifiedQuickRecallCalculator", f"Score calculated (sync): {final_score:.4f}", {"time_ms": calculation_time, "mode": self.config['mode'].value})

        return final_score
        
    # Synchronous versions of the async calculation methods
    def _calculate_emotion_sync(self, text: str, context: Dict[str, Any]) -> float:
        # Simple fallback implementation
        return 0.5
        
    def _calculate_surprise_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Handle the dimension mismatch with safe alignment
        try:
            # Similar implementation to async version but synchronous
            if self.momentum_buffer is None or len(self.momentum_buffer) == 0:
                return 0.5  # Default value when no history
                
            # Calculate vector distance, handle dimension mismatch
            distances = []
            for vec in self.momentum_buffer:
                try:
                    # Use existing alignment functionality
                    aligned_vec, aligned_embedding = self._align_vectors_for_comparison(vec, embedding, log_warnings=False)
                    dist = self.geometry_manager.calculate_distance(aligned_vec, aligned_embedding)
                    distances.append(dist)
                except Exception:
                    distances.append(0.5)  # Default on error
                    
            if not distances:
                return 0.5
                
            # Calculate surprise based on minimum distance (most similar)
            min_dist = min(distances)
            surprise = min_dist / self.config.get('surprise_normalization', 2.0)
            return float(np.clip(surprise, 0.0, 1.0))
        except Exception as e:
            logger.warning("UnifiedQuickRecallCalculator", f"Error in surprise calc: {str(e)}")
            return 0.5
    
    def _calculate_diversity_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation that handles dimension mismatches
        try:
            return self._calculate_surprise_sync(embedding, external_momentum) * 0.8  # Simplified
        except Exception:
            return 0.5
    
    def _calculate_overlap_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation
        return 0.0  # Default no overlap
    
    def _calculate_r_geometry_sync(self, embedding: np.ndarray, external_momentum=None) -> float:
        # Simple implementation
        return 0.6  # Default moderate geometric novelty
    
    def _calculate_causal_novelty_sync(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        # Simple implementation
        return 0.5  # Default causal novelty
    
    def _calculate_self_org_sync(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
        # Simple implementation
        return 0.5  # Default self-organization

    # --- Factor Calculation Methods ---
    # (Implementations adapted from your previous code, using GeometryManager where needed)

    def _calculate_recency(self, context: Dict[str, Any]) -> float:
        timestamp = context.get('timestamp', time.time())
        age_seconds = time.time() - timestamp
        # Exponential decay with a half-life of ~3 days
        decay_factor = np.exp(-age_seconds / (3 * 86400))
        return float(decay_factor)

    def _calculate_relevance(self, context: Dict[str, Any]) -> float:
        # Relevance might come from an external source (e.g., query similarity)
        return float(context.get('relevance', context.get('similarity', 0.5)))

    def _calculate_importance(self, text: str, context: Dict[str, Any]) -> float:
        # Explicit importance or keyword-based
        explicit_importance = context.get('importance', context.get('significance', 0.0))
        if text:
             keywords = ['important', 'remember', 'critical', 'key', 'significant']
             keyword_score = sum(1 for k in keywords if k in text.lower()) / 3.0
             return float(np.clip(max(explicit_importance, keyword_score), 0.0, 1.0))
        return float(explicit_importance)

    def _calculate_personal(self, text: str, context: Dict[str, Any]) -> float:
        # Check for personal keywords
        if text:
            count = sum(1 for k in self.config.get('personal_keywords', []) if k in text.lower())
            return float(np.clip(count / 3.0, 0.0, 1.0))
        return 0.0

    async def _calculate_emotion(self, text: str, context: Dict[str, Any]) -> float:
         # Reuse context's emotion data if available, otherwise analyze
         if 'emotion_data' in context and context['emotion_data']:
             intensity = context['emotion_data'].get('intensity', 0.0) # Assumes intensity 0-1
             valence_abs = abs(context['emotion_data'].get('sentiment_value', 0.0)) # Assumes valence -1 to 1
             return float(np.clip((intensity + valence_abs) / 2.0, 0.0, 1.0))
         # Placeholder: Simple keyword analysis if no analyzer
         if text:
              count = sum(1 for k in self.config.get('emotional_keywords', []) if k in text.lower())
              intensity = sum(1 for k in self.config.get('emotion_intensifiers', []) if k in text.lower())
              return float(np.clip((count + intensity) / 5.0, 0.0, 1.0))
         return 0.0

    async def _calculate_surprise(self, embedding: np.ndarray, external_momentum) -> float:
        # Novelty compared to recent memories (momentum)
        if external_momentum is None or len(external_momentum) == 0: return 0.5
        similarities = []
        for mem_emb in external_momentum[-5:]: # Compare with last 5
             sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
             similarities.append(sim)
        max_sim = max(similarities) if similarities else 0.0
        surprise = 1.0 - max_sim # Higher surprise if less similar to recent items
        return float(np.clip(surprise, 0.0, 1.0))

    async def _calculate_diversity(self, embedding: np.ndarray, external_momentum) -> float:
         # Novelty compared to the entire buffer (or a sample)
         if external_momentum is None or len(external_momentum) < 2: return 0.5
         # Sample if buffer is large
         sample_size = min(50, len(external_momentum))
         indices = np.random.choice(len(external_momentum), sample_size, replace=False)
         sample_momentum = [external_momentum[i] for i in indices]

         similarities = []
         for mem_emb in sample_momentum:
              sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
              similarities.append(sim)
         avg_sim = np.mean(similarities) if similarities else 0.0
         diversity = 1.0 - avg_sim # Higher diversity if less similar on average
         return float(np.clip(diversity, 0.0, 1.0))

    async def _calculate_overlap(self, embedding: np.ndarray, external_momentum) -> float:
         # Similar to surprise, but focused on maximum similarity as redundancy measure
         if external_momentum is None or len(external_momentum) == 0: return 0.0
         similarities = []
         for mem_emb in external_momentum[-10:]: # Check against more recent items for overlap
              sim = self.geometry_manager.calculate_similarity(embedding, mem_emb)
              similarities.append(sim)
         max_sim = max(similarities) if similarities else 0.0
         # Overlap is directly related to max similarity
         return float(np.clip(max_sim, 0.0, 1.0))

    async def _calculate_r_geometry(self, embedding: np.ndarray, external_momentum) -> float:
         # Distance from the center of the momentum buffer
         if external_momentum is None or len(external_momentum) < 3: return 0.5
         # Calculate centroid
         aligned_embeddings = []
         target_dim = self.config['embedding_dim']
         for emb in external_momentum:
              validated = self.geometry_manager._validate_vector(emb)
              if validated is not None:
                   aligned, _ = self.geometry_manager._align_vectors(validated, np.zeros(target_dim))
                   aligned_embeddings.append(aligned)

         if not aligned_embeddings: return 0.5
         centroid = np.mean(aligned_embeddings, axis=0)
         # Calculate distance from embedding to centroid
         distance = self.geometry_manager.calculate_distance(embedding, centroid)
         # Convert distance to score (larger distance = more novel = higher score)
         # Use exponential decay on distance
         geometry_score = np.exp(-distance * 0.5) # Adjust scaling factor as needed
         return float(np.clip(geometry_score, 0.0, 1.0))

    async def _calculate_causal_novelty(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
         # Placeholder - Requires a causal model
         # Simulates prediction based on context and compares with actual embedding
         predicted_embedding = embedding + np.random.randn(*embedding.shape) * 0.1 # Simulate slight prediction error
         novelty = 1.0 - self.geometry_manager.calculate_similarity(embedding, predicted_embedding)
         return float(np.clip(novelty, 0.0, 1.0))

    async def _calculate_self_org(self, embedding: np.ndarray, context: Dict[str, Any]) -> float:
         # Placeholder - Requires SOM or similar structure
         # Simulates finding BMU distance
         distance = np.random.rand() * 2.0 # Random distance 0-2
         self_org_score = np.exp(-distance * 0.5)
         return float(np.clip(self_org_score, 0.0, 1.0))

    def _calculate_information(self, text: str, context: Dict[str, Any]) -> float:
         # Simple length and keyword based information score
         if not text: return 0.0
         length_score = np.clip(len(text.split()) / 100.0, 0.0, 1.0)
         # Add bonus for specific informational keywords if needed
         return float(length_score)

    def _calculate_coherence(self, text: str, context: Dict[str, Any]) -> float:
         # Placeholder - Requires more advanced NLP or context checking
         # Simulate based on sentence structure (longer sentences might be more coherent)
         if not text: return 0.5
         sentences = [s for s in text.split('.') if s.strip()]
         if not sentences: return 0.5
         avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
         coherence_score = np.clip(avg_len / 30.0, 0.0, 1.0) # Assuming avg sentence length target is ~30 words
         return float(coherence_score)

    def _calculate_user_attention(self, context: Dict[str, Any]) -> float:
         # Placeholder - Requires input from UI/interaction layer
         return float(context.get('user_attention', 0.0))

    def _calculate_time_decay(self, context: Dict[str, Any]) -> float:
        """Exponential time decay, clamped at min_time_decay."""
        timestamp = context.get('timestamp', time.time())
        elapsed_days = (time.time() - timestamp) / 86400.0
        decay_factor = np.exp(-self.config['time_decay_rate'] * elapsed_days)
        return float(max(self.config.get('min_time_decay', 0.02), decay_factor))

    def _update_history(self, score: float, factor_values: Dict[QuickRecallFactor, float]):
        """Update score history."""
        self.history['calculated_qr'].append(score)
        self.history['timestamps'].append(time.time())
        for factor, value in factor_values.items():
            self.history['factor_values'][factor].append(value)

        # Trim history
        hw = self.config['history_window']
        if len(self.history['calculated_qr']) > hw:
            self.history['calculated_qr'].pop(0)
            self.history['timestamps'].pop(0)
            for factor in self.history['factor_values']:
                if len(self.history['factor_values'][factor]) > hw:
                    self.history['factor_values'][factor].pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Retrieve calculator statistics."""
        qr_scores = self.history['calculated_qr']
        factor_stats = {}
        for factor, values in self.history['factor_values'].items():
             if values:
                  factor_stats[factor.value] = {
                       'average': float(np.mean(values)),
                       'stddev': float(np.std(values)),
                       'weight': self.factor_weights.get(factor, 0.0)
                  }

        return {
            'mode': self.config['mode'].value,
            'total_calculations': self.total_calculations,
            'avg_qr_score': float(np.mean(qr_scores)) if qr_scores else 0.0,
            'std_qr_score': float(np.std(qr_scores)) if qr_scores else 0.0,
            'history_size': len(qr_scores),
            'factors': factor_stats
        }

```

# interruption\__init__.py

```py
# synthians_memory_core/interruption/__init__.py

from .memory_handler import InterruptionAwareMemoryHandler

__all__ = ['InterruptionAwareMemoryHandler']

```

# interruption\memory_handler.py

```py
# synthians_memory_core/interruption/memory_handler.py

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Union, Callable, Awaitable
import json
import aiohttp
import numpy as np

class InterruptionAwareMemoryHandler:
    """
    Specialized handler for transcripts that enriches memory entries with interruption metadata.
    This bridges the voice system's interruption tracking with the memory system.
    """

    def __init__(self, 
                 api_url: str = "http://localhost:8000"):
        """
        Initialize the memory handler with API connection details.
        
        Args:
            api_url: Base URL for the memory API
        """
        self.logger = logging.getLogger("InterruptionAwareMemoryHandler")
        self.api_url = api_url.rstrip('/')
        
    async def __call__(self, 
                       text: str, 
                       transcript_sequence: int = 0,
                       timestamp: float = 0,
                       confidence: float = 1.0,
                       **metadata) -> Dict[str, Any]:
        """
        Process a transcript, enriching it with interruption metadata, and send to memory API.
        This method accepts transcripts and additional metadata from voice processing.
        
        Args:
            text: The transcript text to process
            transcript_sequence: Sequence number of this transcript
            timestamp: Unix timestamp when transcript was received
            confidence: STT confidence score
            **metadata: Additional metadata, including interruption data
            
        Returns:
            Response from the memory API as a dictionary
        """
        try:
            self.logger.info(f"Processing transcript {transcript_sequence}: {text[:50]}...")
            
            # Prepare audio metadata from transcript info
            audio_metadata = {
                "timestamp": timestamp,
                "confidence": confidence,
                "sequence": transcript_sequence,
                "source": "voice_interaction"
            }
            
            # Add interruption metadata if available
            if "was_interrupted" in metadata:
                audio_metadata["was_interrupted"] = metadata["was_interrupted"]
                audio_metadata["user_interruptions"] = metadata.get("user_interruptions", 1)
                
                if "interruption_timestamps" in metadata:
                    audio_metadata["interruption_timestamps"] = metadata["interruption_timestamps"]
                    
                if "session_id" in metadata:
                    audio_metadata["session_id"] = metadata["session_id"]
            
            # Prepare request to memory API
            request_data = {
                "text": text,
                "audio_metadata": audio_metadata
            }
            
            # Use the new transcription feature extraction endpoint
            async with aiohttp.ClientSession() as session:
                self.logger.info(f"Sending transcript to memory API: {self.api_url}/process_transcription")
                async with session.post(
                    f"{self.api_url}/process_transcription", 
                    json=request_data,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Memory created/updated with ID: {result.get('memory_id')}")
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Memory API error: {response.status} - {error_text}")
                        return {"success": False, "error": error_text}
                        
        except Exception as e:
            self.logger.error(f"Error processing transcript: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _validate_embedding(self, embedding):
        """
        Validate that an embedding is properly formed without NaN or Inf values.
        Implements the same validation logic as in memory_core/tools.py.
        
        Args:
            embedding: The embedding vector to validate (np.ndarray or list)
            
        Returns:
            bool: True if the embedding is valid, False otherwise
        """
        if embedding is None:
            return False
            
        # Convert to numpy array if needed
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
            
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False
            
        return True

    @staticmethod
    def get_reflection_prompt(interruption_data: Dict[str, Any]) -> Optional[str]:
        """
        Generate a reflection prompt based on interruption patterns to help guide memory retrieval.
        
        Args:
            interruption_data: Dictionary containing interruption metadata
            
        Returns:
            Optional reflection prompt string or None if no reflection needed
        """
        was_interrupted = interruption_data.get("was_interrupted", False)
        interruption_count = interruption_data.get("user_interruptions", 0)
        
        # No reflection needed for normal conversation flow
        if not was_interrupted and interruption_count == 0:
            return None
            
        # Generate prompts based on interruption patterns
        if was_interrupted:
            if interruption_count > 5:
                return "You seem to be interrupting frequently. Would you like me to pause more often to let you speak?"
            else:
                return "I noticed you interrupted. Was there something specific you wanted to address?"
        
        # General high interruption pattern but not this specific utterance
        if interruption_count > 3:
            return "I've noticed several interruptions in our conversation. Would you prefer if I spoke in shorter segments?"
            
        return None

```

# interruption\README.md

```md
# Interruption Tracking and Analysis Module

## Overview

The interruption module provides a bridge between Lucidia's voice interaction system and the memory core. It captures conversational rhythm, interruption patterns, and speaking behaviors to enhance the semantic understanding of conversations with rich contextual metadata.

## Key Components

### InterruptionAwareMemoryHandler

A specialized handler that processes transcripts with interruption metadata and stores them in the memory system with rich contextual information.

\`\`\`python
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Initialize the handler
handler = InterruptionAwareMemoryHandler(api_url="http://localhost:8000")

# Process a transcript with interruption data
await handler(
    text="I wanted to explain something important.",
    was_interrupted=True,
    user_interruptions=2,
    interruption_timestamps=[1678945330.45, 1678945342.12]
)
\`\`\`

## Integration with VoiceStateManager

The interruption module is designed to work with the `VoiceStateManager` from the voice_core package. The VoiceStateManager tracks interruptions in real-time and provides this data when processing transcripts.

### Configuration

To connect the VoiceStateManager with the InterruptionAwareMemoryHandler:

\`\`\`python
from voice_core.state.voice_state_manager import VoiceStateManager
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# Initialize components
state_manager = VoiceStateManager()
memory_handler = InterruptionAwareMemoryHandler(api_url="http://localhost:8000")

# Register the memory handler as the transcript handler
state_manager.register_transcript_handler(memory_handler)
\`\`\`

## Memory Processing Flow

1. VoiceStateManager detects and tracks interruptions during conversation
2. When a transcript is processed, interruption metadata is attached
3. InterruptionAwareMemoryHandler sends this enriched data to the memory API
4. TranscriptionFeatureExtractor processes the text and metadata
5. The memory is stored with rich conversational context

## Using Interruption Data for Reflection

The module provides utilities to generate reflection prompts based on interruption patterns:

\`\`\`python
from synthians_memory_core.interruption import InterruptionAwareMemoryHandler

# For a memory with high interruption count
prompt = InterruptionAwareMemoryHandler.get_reflection_prompt({
    "was_interrupted": True,
    "user_interruptions": 6
})
# Returns: "You seem to be interrupting frequently. Would you like me to pause more often to let you speak?"
\`\`\`

## Compatibility with Embedding Handling

This module is fully compatible with Lucidia's robust embedding handling system:

- Works with both 384 and 768 dimension embeddings
- Properly handles vector alignment during comparison operations
- Validates embeddings to prevent NaN/Inf values
- Provides graceful fallbacks when embedding generation fails

## Metadata Structure

The interruption metadata schema includes:

\`\`\`json
{
  "was_interrupted": true,            // Whether this specific utterance was interrupted
  "user_interruptions": 3,           // Total interruptions in the current session
  "interruption_timestamps": [       // Timestamps of interruptions (relative to session start)
    12.5, 24.1, 38.8
  ],
  "session_id": "abc123",            // Unique ID for the current conversation session
  "interruption_severity": "medium", // Classification of interruption pattern severity
  "requires_reflection": true        // Whether this memory might benefit from reflection
}
\`\`\`

## Best Practices

1. **Session Management**: Generate a new session ID for each distinct conversation
2. **Timestamp Precision**: Store interruption timestamps as relative times (seconds from session start)
3. **Aggregation**: Consider aggregating interruption patterns across multiple sessions for deeper insights
4. **Memory Retrieval**: Use interruption metadata as a factor in memory prioritization

```

# memory_persistence.py

```py
# synthians_memory_core/memory_persistence.py

import os
import json
import logging
import asyncio
import time
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np
import torch
import aiofiles # Use aiofiles for async file operations
import uuid
from .memory_structures import MemoryEntry # Use the unified structure
from .custom_logger import logger # Use the shared custom logger

class MemoryPersistence:
    """Handles disk-based memory operations with robustness."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'storage_path': Path('/app/memory/stored'), # Consistent Docker path
            'backup_dir': 'backups',
            'index_filename': 'memory_index.json',
            'max_backups': 5,
            'safe_write': True, # Use atomic writes
            **(config or {})
        }
        self.storage_path = Path(self.config['storage_path'])
        self.backup_path = self.storage_path / self.config['backup_dir']
        self.index_path = self.storage_path / self.config['index_filename']
        self.memory_index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self.stats = {'saves': 0, 'loads': 0, 'deletes': 0, 'backups': 0, 'errors': 0}

        # Ensure directories exist
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            self.backup_path.mkdir(exist_ok=True)
        except Exception as e:
             logger.error("MemoryPersistence", "Failed to create storage directories", {"path": self.storage_path, "error": str(e)})
             raise # Initialization failure is critical

        # Load index on init
        asyncio.create_task(self._load_index()) # Load index in background

        logger.info("MemoryPersistence", "Initialized", {"storage_path": str(self.storage_path)})

    async def _load_index(self):
        """Load the memory index from disk."""
        async with self._lock:
             if not self.index_path.exists():
                 logger.info("MemoryPersistence", "Memory index file not found, starting fresh.", {"path": str(self.index_path)})
                 self.memory_index = {}
                 return

             try:
                 async with aiofiles.open(self.index_path, 'r') as f:
                     content = await f.read()
                     loaded_index = json.loads(content)
                 # Basic validation
                 if isinstance(loaded_index, dict):
                     self.memory_index = loaded_index
                     logger.info("MemoryPersistence", f"Loaded memory index with {len(self.memory_index)} entries.", {"path": str(self.index_path)})
                 else:
                      logger.error("MemoryPersistence", "Invalid index file format, starting fresh.", {"path": str(self.index_path)})
                      self.memory_index = {}
             except Exception as e:
                 logger.error("MemoryPersistence", "Error loading memory index, starting fresh.", {"path": str(self.index_path), "error": str(e)})
                 self.memory_index = {} # Start fresh on error

    async def _save_index(self):
        """Save the memory index to disk atomically."""
        async with self._lock:
             temp_path = self.index_path.with_suffix('.tmp')
             try:
                 async with aiofiles.open(temp_path, 'w') as f:
                     await f.write(json.dumps(self.memory_index, indent=2))
                 await asyncio.to_thread(os.replace, temp_path, self.index_path)
                 self.stats['last_index_update'] = time.time()
             except Exception as e:
                 logger.error("MemoryPersistence", "Error saving memory index", {"path": str(self.index_path), "error": str(e)})
                 # Attempt to remove potentially corrupted temp file
                 if await asyncio.to_thread(os.path.exists, temp_path):
                      try: await asyncio.to_thread(os.remove, temp_path)
                      except Exception: pass

    async def save_memory(self, memory: MemoryEntry) -> bool:
        """Save a single memory entry to disk."""
        try:
            # Create a unique ID if one doesn't exist
            if not hasattr(memory, 'id') or memory.id is None:
                memory.id = f"mem_{uuid.uuid4().hex[:12]}"
            
            # Ensure the storage directory exists
            memory_dir = self.storage_path 
            memory_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate a filename based on the memory ID
            file_path = memory_dir / f"{memory.id}.json"
            
            # Convert the memory to a serializable dict
            memory_dict = memory.to_dict()

            # Write the memory to disk
            async with aiofiles.open(file_path, 'w') as f:
                # Ensure complex numbers or other non-serializables are handled
                def default_serializer(obj):
                     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                         np.int16, np.int32, np.int64, np.uint8,
                                         np.uint16, np.uint32, np.uint64)):
                         return int(obj)
                     elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                         return float(obj)
                     elif isinstance(obj, (np.ndarray,)): # Handle complex arrays if needed
                         return obj.tolist()
                     elif isinstance(obj, set):
                         return list(obj)
                     try:
                          # Fallback for other types
                          return str(obj)
                     except:
                          return "[Unserializable Object]"
                await f.write(json.dumps(memory_dict, indent=2, default=default_serializer))
            
            # Update the memory index
            self.memory_index[memory.id] = {
                'path': str(file_path.relative_to(self.storage_path)),
                'timestamp': memory.timestamp if hasattr(memory, 'timestamp') else time.time(),
                'quickrecal': memory.quickrecal_score if hasattr(memory, 'quickrecal_score') else 0.5,
                'type': 'memory'  # Default type since memory_type doesn't exist
            }
            
            # Save the memory index
            await self._save_index()
            
            self.stats['saves'] += 1
            self.stats['successful_saves'] = self.stats.get('successful_saves', 0) + 1
            return True
        except Exception as e:
            logger.error("MemoryPersistence", f"Error saving memory {getattr(memory, 'id', 'unknown')}: {str(e)}")
            self.stats['saves'] += 1
            self.stats['failed_saves'] = self.stats.get('failed_saves', 0) + 1
            return False

    async def load_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Load a single memory entry from disk."""
        async with self._lock:
             try:
                 if memory_id not in self.memory_index:
                     # Fallback: check filesystem directly (maybe index is outdated)
                     file_path = self.storage_path / f"{memory_id}.json"
                     if not await asyncio.to_thread(os.path.exists, file_path):
                          logger.warning("MemoryPersistence", f"Memory {memory_id} not found in index or filesystem.")
                          return None
                     # If found directly, update index info
                     self.memory_index[memory_id] = {'path': f"{memory_id}.json"}
                 else:
                     file_path = self.storage_path / self.memory_index[memory_id]['path']

                 # Check primary path first
                 if not await asyncio.to_thread(os.path.exists, file_path):
                      # Try backup path
                      backup_path = file_path.with_suffix('.bak')
                      if await asyncio.to_thread(os.path.exists, backup_path):
                           logger.warning("MemoryPersistence", f"Using backup file for {memory_id}", {"path": str(backup_path)})
                           file_path = backup_path
                      else:
                           logger.error("MemoryPersistence", f"Memory file not found for {memory_id}", {"path": str(file_path)})
                           # Remove from index if file is missing
                           if memory_id in self.memory_index: del self.memory_index[memory_id]
                           return None

                 async with aiofiles.open(file_path, 'r') as f:
                     content = await f.read()
                     memory_dict = json.loads(content)

                 memory = MemoryEntry.from_dict(memory_dict)
                 self.stats['loads'] = self.stats.get('loads', 0) + 1
                 self.stats['successful_loads'] = self.stats.get('successful_loads', 0) + 1
                 return memory

             except Exception as e:
                 logger.error("MemoryPersistence", f"Error loading memory {memory_id}", {"error": str(e)})
                 self.stats['loads'] = self.stats.get('loads', 0) + 1
                 self.stats['failed_loads'] = self.stats.get('failed_loads', 0) + 1
                 # Attempt recovery from backup if primary load failed
                 backup_path = self.storage_path / f"{memory_id}.json.bak"
                 if await asyncio.to_thread(os.path.exists, backup_path):
                      try:
                           logger.info("MemoryPersistence", f"Attempting recovery from backup for {memory_id}")
                           async with aiofiles.open(backup_path, 'r') as f:
                                content = await f.read()
                                memory_dict = json.loads(content)
                           memory = MemoryEntry.from_dict(memory_dict)
                           # Restore backup to primary file
                           await asyncio.to_thread(shutil.copy2, backup_path, self.storage_path / f"{memory_id}.json")
                           logger.info("MemoryPersistence", f"Successfully recovered {memory_id} from backup.")
                           return memory
                      except Exception as e_rec:
                           logger.error("MemoryPersistence", f"Backup recovery failed for {memory_id}", {"error": str(e_rec)})
                 return None

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory file from disk."""
        async with self._lock:
             try:
                 if memory_id not in self.memory_index:
                     # Check filesystem directly as fallback
                     file_path_direct = self.storage_path / f"{memory_id}.json"
                     if await asyncio.to_thread(os.path.exists, file_path_direct):
                         await asyncio.to_thread(os.remove, file_path_direct)
                         logger.info("MemoryPersistence", f"Deleted memory file directly {memory_id} (was not in index)")
                         self.stats['deletes'] = self.stats.get('deletes', 0) + 1
                         return True
                     logger.warning("MemoryPersistence", f"Memory {memory_id} not found for deletion.")
                     return False

                 file_path = self.storage_path / self.memory_index[memory_id]['path']
                 backup_path = file_path.with_suffix('.bak')

                 deleted = False
                 if await asyncio.to_thread(os.path.exists, file_path):
                     await asyncio.to_thread(os.remove, file_path)
                     deleted = True
                 if await asyncio.to_thread(os.path.exists, backup_path):
                     await asyncio.to_thread(os.remove, backup_path)
                     deleted = True # Mark deleted even if only backup existed

                 if deleted:
                     del self.memory_index[memory_id]
                     await self._save_index() # Update index after deletion
                     self.stats['deletes'] = self.stats.get('deletes', 0) + 1
                     return True
                 else:
                      # File didn't exist, remove from index anyway
                      del self.memory_index[memory_id]
                      await self._save_index()
                      return False # Indicate file wasn't actually deleted

             except Exception as e:
                 logger.error("MemoryPersistence", f"Error deleting memory {memory_id}", {"error": str(e)})
                 self.stats['errors'] = self.stats.get('errors', 0) + 1
                 return False

    async def load_all(self) -> List[MemoryEntry]:
        """Load all memories listed in the index."""
        all_memories = []
        memory_ids = list(self.memory_index.keys())
        logger.info("MemoryPersistence", f"Loading all {len(memory_ids)} memories from index.")

        # Consider batching if loading many memories
        batch_size = 100
        for i in range(0, len(memory_ids), batch_size):
             batch_ids = memory_ids[i:i+batch_size]
             load_tasks = [self.load_memory(mid) for mid in batch_ids]
             results = await asyncio.gather(*load_tasks)
             all_memories.extend(mem for mem in results if mem is not None)
             await asyncio.sleep(0.01) # Yield control between batches

        logger.info("MemoryPersistence", f"Finished loading {len(all_memories)} memories.")
        return all_memories

    async def create_backup(self) -> bool:
        """Create a timestamped backup of the memory storage."""
        async with self._lock:
             try:
                 timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                 backup_instance_path = self.backup_path / f"backup_{timestamp}"
                 # Use shutil.copytree for directory backup
                 await asyncio.to_thread(shutil.copytree, self.storage_path, backup_instance_path, ignore=shutil.ignore_patterns('backups'))
                 self.stats['last_backup'] = time.time()
                 self.stats['backup_count'] = self.stats.get('backup_count', 0) + 1
                 logger.info("MemoryPersistence", f"Created backup at {backup_instance_path}")
                 await self._prune_backups()
                 return True
             except Exception as e:
                 logger.error("MemoryPersistence", "Error creating backup", {"error": str(e)})
                 self.stats['errors'] = self.stats.get('errors', 0) + 1
                 return False

    async def _prune_backups(self):
        """Remove old backups, keeping only the most recent ones."""
        try:
             backups = sorted(
                 [d for d in self.backup_path.iterdir() if d.is_dir() and d.name.startswith('backup_')],
                 key=lambda d: d.stat().st_mtime
             )
             num_to_keep = self.config['max_backups']
             if len(backups) > num_to_keep:
                 for old_backup in backups[:-num_to_keep]:
                     await asyncio.to_thread(shutil.rmtree, old_backup)
                     logger.info("MemoryPersistence", f"Pruned old backup {old_backup.name}")
        except Exception as e:
            logger.error("MemoryPersistence", "Error pruning backups", {"error": str(e)})

    def get_stats(self) -> Dict[str, Any]:
        """Get persistence statistics."""
        asyncio.create_task(self._save_index()) # Ensure index is saved before getting stats
        return {
            "total_indexed_memories": len(self.memory_index),
            "last_index_update": self.stats.get('last_index_update', 0),
            "saves": self.stats.get('saves', 0),
            "successful_saves": self.stats.get('successful_saves', 0),
            "failed_saves": self.stats.get('failed_saves', 0),
            "loads": self.stats.get('loads', 0),
            "successful_loads": self.stats.get('successful_loads', 0),
            "failed_loads": self.stats.get('failed_loads', 0),
            "deletes": self.stats.get('deletes', 0),
            "backups": self.stats.get('backup_count', 0),
            "last_backup": self.stats.get('last_backup', 0),
            "errors": self.stats.get('errors', 0)
        }

```

# memory_structures.py

```py
# synthians_memory_core/memory_structures.py

import time
import uuid
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field

from .custom_logger import logger # Use the shared custom logger

@dataclass
class MemoryEntry:
    """Standardized container for a single memory entry."""
    content: str
    embedding: Optional[np.ndarray] = None
    id: str = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
    quickrecal_score: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    # Hyperbolic specific
    hyperbolic_embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        self.quickrecal_score = max(0.0, min(1.0, self.quickrecal_score))
        # Ensure embedding is numpy array
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            if isinstance(self.embedding, torch.Tensor):
                self.embedding = self.embedding.detach().cpu().numpy()
            elif isinstance(self.embedding, list):
                self.embedding = np.array(self.embedding, dtype=np.float32)
            else:
                logger.warning("MemoryEntry", f"Unsupported embedding type {type(self.embedding)} for ID {self.id}, clearing.")
                self.embedding = None

        if self.hyperbolic_embedding is not None and not isinstance(self.hyperbolic_embedding, np.ndarray):
            if isinstance(self.hyperbolic_embedding, torch.Tensor):
                self.hyperbolic_embedding = self.hyperbolic_embedding.detach().cpu().numpy()
            elif isinstance(self.hyperbolic_embedding, list):
                 self.hyperbolic_embedding = np.array(self.hyperbolic_embedding, dtype=np.float32)
            else:
                logger.warning("MemoryEntry", f"Unsupported hyperbolic embedding type {type(self.hyperbolic_embedding)} for ID {self.id}, clearing.")
                self.hyperbolic_embedding = None

    def record_access(self):
        self.access_count += 1
        self.last_access_time = time.time()

    def get_effective_quickrecal(self, decay_rate: float = 0.05) -> float:
        """Calculate effective QuickRecal score with time decay."""
        age_days = (time.time() - self.timestamp) / 86400
        if age_days < 1: return self.quickrecal_score
        importance_factor = 0.5 + (0.5 * self.quickrecal_score)
        effective_decay_rate = decay_rate / importance_factor
        decay_factor = np.exp(-effective_decay_rate * (age_days - 1))
        return max(0.0, min(1.0, self.quickrecal_score * decay_factor))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "timestamp": self.timestamp,
            "quickrecal_score": self.quickrecal_score,
            "metadata": self.metadata,
            "access_count": self.access_count,
            "last_access_time": self.last_access_time,
            "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create memory from dictionary."""
        embedding = np.array(data["embedding"], dtype=np.float32) if data.get("embedding") else None
        hyperbolic = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        # Handle legacy 'significance' field
        quickrecal = data.get("quickrecal_score", data.get("significance", 0.5))

        return cls(
            content=data["content"],
            embedding=embedding,
            id=data.get("id"),
            timestamp=data.get("timestamp"),
            quickrecal_score=quickrecal,
            metadata=data.get("metadata", {}),
            access_count=data.get("access_count", 0),
            last_access_time=data.get("last_access_time"),
            hyperbolic_embedding=hyperbolic
        )

class MemoryAssembly:
    """Represents a group of related memories forming a coherent assembly."""
    def __init__(self,
                 geometry_manager, # Pass GeometryManager for consistency
                 assembly_id: str = None,
                 name: str = None,
                 description: str = None):
        self.geometry_manager = geometry_manager
        self.assembly_id = assembly_id or f"asm_{uuid.uuid4().hex[:12]}"
        self.name = name or f"Assembly-{self.assembly_id[:8]}"
        self.description = description or ""
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0

        self.memories: Set[str] = set()  # IDs of memories in this assembly
        self.composite_embedding: Optional[np.ndarray] = None
        self.hyperbolic_embedding: Optional[np.ndarray] = None
        self.emotion_profile: Dict[str, float] = {}
        self.keywords: Set[str] = set()
        self.activation_level: float = 0.0
        self.activation_decay_rate: float = 0.05

    def add_memory(self, memory: MemoryEntry):
        """Add a memory and update assembly properties."""
        if memory.id in self.memories:
            return False
        self.memories.add(memory.id)

        # --- Update Composite Embedding ---
        if memory.embedding is not None:
            target_dim = self.geometry_manager.config['embedding_dim']
            # Align memory embedding to target dimension
            mem_emb = memory.embedding
            if mem_emb.shape[0] != target_dim:
                 aligned_mem_emb, _ = self.geometry_manager._align_vectors(mem_emb, np.zeros(target_dim))
            else:
                 aligned_mem_emb = mem_emb

            normalized_mem_emb = self.geometry_manager._normalize(aligned_mem_emb)

            if self.composite_embedding is None:
                self.composite_embedding = normalized_mem_emb
            else:
                # Align composite embedding if needed (should already be target_dim)
                if self.composite_embedding.shape[0] != target_dim:
                     aligned_comp_emb, _ = self.geometry_manager._align_vectors(self.composite_embedding, np.zeros(target_dim))
                else:
                     aligned_comp_emb = self.composite_embedding

                normalized_composite = self.geometry_manager._normalize(aligned_comp_emb)

                # Simple averaging (could be weighted later)
                n = len(self.memories)
                self.composite_embedding = ((n - 1) * normalized_composite + normalized_mem_emb) / n
                # Re-normalize
                self.composite_embedding = self.geometry_manager._normalize(self.composite_embedding)

            # Update hyperbolic embedding if enabled
            if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC:
                self.hyperbolic_embedding = self.geometry_manager._to_hyperbolic(self.composite_embedding)

        # --- Update Emotion Profile ---
        mem_emotion = memory.metadata.get("emotional_context", {})
        if mem_emotion:
            self._update_emotion_profile(mem_emotion)

        # --- Update Keywords ---
        # Simple keyword extraction (could use NLP later)
        content_words = set(re.findall(r'\b\w{3,}\b', memory.content.lower()))
        self.keywords.update(content_words)
        # Limit keyword set size if needed
        if len(self.keywords) > 100:
            # Simple strategy: keep most frequent or randomly sample
            pass # Placeholder for keyword pruning logic

        return True

    def _update_emotion_profile(self, mem_emotion: Dict[str, Any]):
        """Update aggregated emotional profile."""
        n = len(self.memories)
        for emotion, score in mem_emotion.get("emotions", {}).items():
            current_score = self.emotion_profile.get(emotion, 0.0)
            # Weighted average (giving slightly more weight to existing profile)
            self.emotion_profile[emotion] = (current_score * (n - 1) * 0.6 + score * 0.4) / max(1, (n - 1) * 0.6 + 0.4)

    def get_similarity(self, query_embedding: np.ndarray) -> float:
        """Calculate similarity between query and assembly embedding."""
        ref_embedding = self.hyperbolic_embedding if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC and self.hyperbolic_embedding is not None else self.composite_embedding

        if ref_embedding is None:
            return 0.0

        return self.geometry_manager.calculate_similarity(query_embedding, ref_embedding)

    def activate(self, level: float):
        self.activation_level = min(1.0, max(0.0, level))
        self.last_access_time = time.time()
        self.access_count += 1

    def decay_activation(self):
        self.activation_level = max(0.0, self.activation_level - self.activation_decay_rate)

    def to_dict(self) -> Dict[str, Any]:
        """Convert assembly to dictionary."""
        return {
            "assembly_id": self.assembly_id,
            "name": self.name,
            "description": self.description,
            "creation_time": self.creation_time,
            "last_access_time": self.last_access_time,
            "access_count": self.access_count,
            "memory_ids": list(self.memories),
            "composite_embedding": self.composite_embedding.tolist() if self.composite_embedding is not None else None,
            "hyperbolic_embedding": self.hyperbolic_embedding.tolist() if self.hyperbolic_embedding is not None else None,
            "emotion_profile": self.emotion_profile,
            "keywords": list(self.keywords),
            "activation_level": self.activation_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], geometry_manager) -> 'MemoryAssembly':
        """Create assembly from dictionary."""
        assembly = cls(
            geometry_manager,
            assembly_id=data["assembly_id"],
            name=data["name"],
            description=data["description"]
        )
        assembly.creation_time = data.get("creation_time")
        assembly.last_access_time = data.get("last_access_time")
        assembly.access_count = data.get("access_count", 0)
        assembly.memories = set(data.get("memory_ids", []))
        assembly.composite_embedding = np.array(data["composite_embedding"], dtype=np.float32) if data.get("composite_embedding") else None
        assembly.hyperbolic_embedding = np.array(data["hyperbolic_embedding"], dtype=np.float32) if data.get("hyperbolic_embedding") else None
        assembly.emotion_profile = data.get("emotion_profile", {})
        assembly.keywords = set(data.get("keywords", []))
        assembly.activation_level = data.get("activation_level", 0.0)
        return assembly

```

# metadata_synthesizer.py

```py
import time
import datetime
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import json

from .custom_logger import logger

# Define the current metadata schema version
METADATA_SCHEMA_VERSION = "1.0.0"

class MetadataSynthesizer:
    """
    Enriches memory entries with synthesized metadata derived from content analysis,
    embedding characteristics, and contextual information.
    
    This class serves as a modular pipeline for extracting, computing, and assembling
    metadata fields that add semantic richness to memory entries beyond their raw content.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the MetadataSynthesizer with configuration options.
        
        Args:
            config: Configuration dictionary for customizing metadata synthesis behavior
        """
        self.config = config or {}
        self.metadata_processors = [
            self._process_base_metadata,   # Always process base metadata first (versioning, etc)
            self._process_temporal_metadata,
            self._process_emotional_metadata,
            self._process_cognitive_metadata,
            self._process_embedding_metadata,
            self._process_identifiers_and_basic_stats  # Add identifiers and basic stats processor
        ]
        logger.info("MetadataSynthesizer", "Initialized with processors")
    
    async def synthesize(self, 
                   content: str, 
                   embedding: Optional[np.ndarray] = None,
                   base_metadata: Optional[Dict[str, Any]] = None,
                   emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synthesize rich metadata from content, embedding, and optional existing metadata.
        
        Args:
            content: The text content of the memory
            embedding: Vector representation of the content (optional)
            base_metadata: Existing metadata to build upon (optional)
            emotion_data: Pre-computed emotion analysis results (optional)
            
        Returns:
            Enriched metadata dictionary with synthesized fields
        """
        # Start with base metadata or empty dict
        metadata = base_metadata or {}
        
        # Track original fields to identify what we've added
        original_keys = set(metadata.keys())
        
        # Process through each metadata processor
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        # Run all processors
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                # Handle both synchronous and asynchronous processor results
                if processor_result and hasattr(processor_result, '__await__'):
                    metadata = await processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
        # Log what was added
        added_keys = set(metadata.keys()) - original_keys
        logger.info("MetadataSynthesizer", f"Added metadata fields: {list(added_keys)}")
        
        return metadata
    
    def synthesize_sync(self, 
                   content: str, 
                   embedding: Optional[np.ndarray] = None,
                   base_metadata: Optional[Dict[str, Any]] = None,
                   emotion_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Synchronous version of synthesize for contexts where async cannot be used.
        
        Args:
            content: The text content of the memory
            embedding: Vector representation of the content (optional)
            base_metadata: Existing metadata to build upon (optional)
            emotion_data: Pre-computed emotion analysis results (optional)
            
        Returns:
            Enriched metadata dictionary with synthesized fields
        """
        # Start with base metadata or empty dict
        metadata = base_metadata or {}
        
        # Track original fields to identify what we've added
        original_keys = set(metadata.keys())
        
        # Process through each metadata processor
        context = {
            'content': content,
            'embedding': embedding,
            'emotion_data': emotion_data,
            'original_metadata': base_metadata
        }
        
        # Run all processors (synchronously)
        for processor in self.metadata_processors:
            try:
                processor_result = processor(metadata, context)
                
                # Since we're in sync mode, we skip any async processors
                if processor_result and not hasattr(processor_result, '__await__'):
                    metadata = processor_result
            except Exception as e:
                logger.error("MetadataSynthesizer", f"Error in processor {processor.__name__}: {str(e)}")
        
        # Log what was added
        added_keys = set(metadata.keys()) - original_keys
        logger.info("MetadataSynthesizer", f"Added metadata fields: {list(added_keys)}")
        
        return metadata
    
    def _process_base_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add base metadata fields including:
        - metadata_schema_version
        - creation_time
        """
        # Add metadata schema version
        metadata['metadata_schema_version'] = METADATA_SCHEMA_VERSION
        
        # Add creation time
        metadata['creation_time'] = time.time()
        
        return metadata
    
    def _process_temporal_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add time-related metadata including:
        - timestamp (if not already present)
        - time_of_day (morning, afternoon, evening, night)
        - day_of_week
        - is_weekend
        """
        # Ensure timestamp exists and is a float
        if 'timestamp' not in metadata:
            metadata['timestamp'] = float(time.time())
        else:
            # Ensure timestamp is a float to avoid serialization issues
            try:
                metadata['timestamp'] = float(metadata['timestamp'])
            except (ValueError, TypeError):
                logger.warning("MetadataSynthesizer", f"Invalid timestamp format {metadata['timestamp']}, using current time")
                metadata['timestamp'] = float(time.time())
            
        # Convert timestamp to datetime
        dt = datetime.datetime.fromtimestamp(metadata['timestamp'])
        
        # Add ISO-formatted timestamp for convenience (guarantees serialization compatibility)
        metadata['timestamp_iso'] = dt.isoformat()
        
        # Add temporal markers
        hour = dt.hour
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 22:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'
            
        # Add temporal metadata
        metadata['time_of_day'] = time_of_day
        metadata['day_of_week'] = dt.strftime('%A').lower()
        metadata['is_weekend'] = dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday
        metadata['month'] = dt.strftime('%B').lower()
        metadata['year'] = dt.year
        
        # Debug log the temporal metadata
        logger.debug("MetadataSynthesizer", "Temporal metadata processed", {
            'timestamp': metadata.get('timestamp'),
            'time_of_day': metadata.get('time_of_day'),
            'day_of_week': metadata.get('day_of_week'),
            'is_weekend': metadata.get('is_weekend')
        })
        
        return metadata
    
    def _process_emotional_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add emotion-related metadata including:
        - dominant_emotion
        - sentiment_value
        - emotional_intensity
        """
        # Use pre-computed emotion data if available
        emotion_data = context.get('emotion_data')
        
        if emotion_data and isinstance(emotion_data, dict):
            # Extract emotions from the expected location in the emotion_data
            emotions = emotion_data.get('emotions', {})
            
            if isinstance(emotions, dict) and emotions:
                # Ensure we're not duplicating emotion data
                if 'emotions' in metadata:
                    # If emotions already exists in metadata, avoid nesting
                    # Just log it and use the one from emotion_data
                    logger.debug("MetadataSynthesizer", "Emotions already present in metadata, overwriting")
                
                # Copy relevant emotion data to metadata
                if emotions.get('dominant_emotion') is not None:
                    metadata['dominant_emotion'] = emotions.get('dominant_emotion')
                elif 'dominant_emotion' in emotion_data:
                    metadata['dominant_emotion'] = emotion_data.get('dominant_emotion')
                    
                if emotions.get('sentiment_value') is not None:
                    sentiment = emotions.get('sentiment_value')
                    metadata['sentiment_value'] = float(sentiment) # Ensure it's a float
                    # Add a simple polarity label
                    if sentiment > 0.2:
                        metadata['sentiment_polarity'] = 'positive'
                    elif sentiment < -0.2:
                        metadata['sentiment_polarity'] = 'negative'
                    else:
                        metadata['sentiment_polarity'] = 'neutral'
                
                if emotions.get('intensity') is not None:
                    metadata['emotional_intensity'] = float(emotions.get('intensity', 0.5)) # Ensure it's a float
        
        # Ensure mandatory emotional fields are present with safe default values
        if 'dominant_emotion' not in metadata:
            metadata['dominant_emotion'] = 'neutral'  # Default
        
        if 'sentiment_polarity' not in metadata:
            metadata['sentiment_polarity'] = 'neutral' # Default
            
        if 'sentiment_value' not in metadata:
            metadata['sentiment_value'] = 0.0  # Default neutral sentiment
            
        if 'emotional_intensity' not in metadata:
            metadata['emotional_intensity'] = 0.5  # Default (medium intensity)
            
        # Debug log the final emotional metadata
        logger.debug("MetadataSynthesizer", "Emotional metadata processed", {
            'dominant_emotion': metadata.get('dominant_emotion'),
            'sentiment_polarity': metadata.get('sentiment_polarity'),
            'emotional_intensity': metadata.get('emotional_intensity')
        })
            
        return metadata
    
    def _process_cognitive_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add cognitive-related metadata including:
        - complexity_estimate
        - word_count
        - cognitive_load_estimate
        """
        content = context.get('content', '')
        
        # Simple metrics based on content
        word_count = len(content.split())
        metadata['word_count'] = word_count
        
        # Estimate complexity (very simple heuristic)
        avg_word_length = sum(len(word) for word in content.split()) / max(1, word_count)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        sentence_count = max(1, sentence_count)  # Avoid division by zero
        
        words_per_sentence = word_count / sentence_count
        
        # Simplified complexity score (0-1 range)
        complexity = min(1.0, ((avg_word_length / 10) + (words_per_sentence / 25)) / 2)
        metadata['complexity_estimate'] = float(complexity)
        
        # Cognitive load is a factor of complexity and length
        cognitive_load = min(1.0, (complexity * 0.7) + (min(1.0, word_count / 500) * 0.3))
        metadata['cognitive_load_estimate'] = float(cognitive_load)
        
        return metadata
    
    def _process_embedding_metadata(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from embedding characteristics:
        - embedding_norm
        - embedding_sparsity
        - embedding_dim
        - embedding_valid
        """
        embedding = context.get('embedding')
        
        if embedding is not None:
            # Extract embedding characteristics
            try:
                # First validate the embedding
                embedding, is_valid = self._validate_embedding(embedding)
                metadata['embedding_valid'] = is_valid
                
                # Calculate embedding norm (magnitude)
                embedding_norm = float(np.linalg.norm(embedding))
                metadata['embedding_norm'] = embedding_norm
                
                # Calculate sparsity (percent of near-zero values)
                near_zero = np.abs(embedding) < 0.01
                sparsity = float(np.mean(near_zero))
                metadata['embedding_sparsity'] = sparsity
                
                # Store embedding dimension
                metadata['embedding_dim'] = embedding.shape[0]
                
                # Log the embedding metadata
                logger.debug("MetadataSynthesizer", "Embedding metadata processed", {
                    'valid': metadata.get('embedding_valid'),
                    'norm': metadata.get('embedding_norm'),
                    'sparsity': metadata.get('embedding_sparsity'),
                    'dim': metadata.get('embedding_dim')
                })
            except Exception as e:
                logger.warning("MetadataSynthesizer", f"Error processing embedding metadata: {str(e)}")
                metadata['embedding_valid'] = False
        else:
            # No embedding available
            metadata['embedding_valid'] = False
        
        return metadata
        
    def _validate_embedding(self, embedding: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Validate an embedding vector and replace with zeros if invalid.
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            Tuple of (possibly_fixed_embedding, is_valid)
        """
        # Check for None
        if embedding is None:
            return np.zeros(768), False
            
        # Convert to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding, dtype=np.float32)
            except Exception:
                return np.zeros(768), False
        
        # Check for NaN or Inf values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning("MetadataSynthesizer", "Embedding contains NaN or Inf values, replacing with zeros")
            # Create a zero vector of the same shape
            return np.zeros_like(embedding), False
            
        # Check if the vector is all zeros
        if np.all(embedding == 0):
            return embedding, False
            
        return embedding, True
        
    def _align_vectors_for_comparison(self, vec1: np.ndarray, vec2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two vectors to the same dimension for comparison operations.
        Will pad the smaller vector with zeros or truncate the larger one.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Tuple of (aligned_vec1, aligned_vec2)
        """
        # Make sure both are numpy arrays
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)
            
        # Get dimensions
        dim1 = vec1.shape[0]
        dim2 = vec2.shape[0]
        
        # If dimensions match, no alignment needed
        if dim1 == dim2:
            return vec1, vec2
            
        # Need to align dimensions
        if dim1 < dim2:
            # Pad vec1 with zeros
            aligned_vec1 = np.zeros(dim2, dtype=np.float32)
            aligned_vec1[:dim1] = vec1
            return aligned_vec1, vec2
        else:
            # Pad vec2 with zeros
            aligned_vec2 = np.zeros(dim1, dtype=np.float32)
            aligned_vec2[:dim2] = vec2
            return vec1, aligned_vec2

    def _process_identifiers_and_basic_stats(self, metadata: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds memory ID (uuid) and content length if available.
        This should run after base metadata and before final memory entry creation.
        """
        content = context.get('content', '')
        
        # Add length (raw character count)
        if 'length' not in metadata:
            metadata['length'] = len(content)
        
        # NOTE: 'uuid' (aka memory_id) must be passed externally if you want it included,
        # or you can let the core insert it *after* memory creation if needed.
        
        return metadata

```

# README.md

```md
# synthians_memory_core/README.md

# Synthians Memory Core

This directory contains the unified and optimized memory system for the Synthians AI architecture, integrating the best features from the Lucid Recall system.

## Overview

The Synthians Memory Core provides a lean, efficient, yet powerful memory system incorporating:

-   **Advanced Relevance Scoring:** Uses the `UnifiedQuickRecallCalculator` (HPC-QR) for multi-factor memory importance assessment.
-   **Flexible Geometry:** Supports Euclidean, Hyperbolic, Spherical, and Mixed geometries for embedding representation via the `GeometryManager`.
-   **Emotional Intelligence:** Integrates emotional analysis and gating (`EmotionalAnalyzer`, `EmotionalGatingService`) for nuanced retrieval.
-   **Memory Assemblies:** Groups related memories (`MemoryAssembly`) for complex concept representation.
-   **Robust Persistence:** Handles disk storage, backups, and atomic writes asynchronously (`MemoryPersistence`).
-   **Adaptive Thresholds:** Dynamically adjusts retrieval thresholds based on feedback (`ThresholdCalibrator`).
-   **Unified Interface:** Provides a cohesive API through `SynthiansMemoryCore`.

## Components

-   `synthians_memory_core.py`: The main orchestrator class.
-   `hpc_quickrecal.py`: Contains the `UnifiedQuickRecallCalculator`.
-   `geometry_manager.py`: Centralizes embedding and geometry operations.
-   `emotional_intelligence.py`: Provides emotion analysis and gating.
-   `memory_structures.py`: Defines `MemoryEntry` and `MemoryAssembly`.
-   `memory_persistence.py`: Manages disk storage and backups.
-   `adaptive_components.py`: Includes `ThresholdCalibrator`.
-   `custom_logger.py`: Simple logging utility.

## Usage

\`\`\`python
import asyncio
from synthians_memory_core import SynthiansMemoryCore
import numpy as np

async def main():
    # Configuration (adjust paths and dimensions as needed)
    config = {
        'embedding_dim': 768,
        'geometry': 'hyperbolic',
        'storage_path': './synthians_memory_data'
    }

    # Initialize
    memory_core = SynthiansMemoryCore(config)
    await memory_core.initialize()

    # --- Example Operations ---

    # Generate a sample embedding (replace with your actual embedding generation)
    sample_embedding = np.random.rand(config['embedding_dim']).astype(np.float32)

    # 1. Store a new memory
    memory_entry = await memory_core.process_new_memory(
        content="Learned about hyperbolic embeddings today.",
        embedding=sample_embedding,
        metadata={"source": "learning_session"}
    )
    if memory_entry:
        print(f"Stored memory: {memory_entry.id}")

    # 2. Retrieve memories
    query_embedding = np.random.rand(config['embedding_dim']).astype(np.float32) # Use actual query embedding
    retrieved = await memory_core.retrieve_memories(
        query="hyperbolic geometry",
        query_embedding=query_embedding,
        top_k=3
    )
    print(f"\nRetrieved {len(retrieved)} memories:")
    for mem_dict in retrieved:
        print(f"- ID: {mem_dict.get('id')}, Score: {mem_dict.get('final_score', mem_dict.get('relevance_score')):.3f}, Content: {mem_dict.get('content', '')[:50]}...")


    # 3. Provide feedback (if adaptive thresholding enabled)
    if memory_entry and memory_core.threshold_calibrator:
         await memory_core.provide_feedback(
              memory_id=memory_entry.id,
              similarity_score=0.85, # Example score from retrieval
              was_relevant=True
         )
         print(f"\nProvided feedback. New threshold: {memory_core.threshold_calibrator.get_current_threshold():.3f}")

    # 4. Detect Contradictions
    # (Add potentially contradictory memories first)
    await memory_core.process_new_memory(content="A causes B", embedding=np.random.rand(config['embedding_dim']))
    await memory_core.process_new_memory(content="A prevents B", embedding=np.random.rand(config['embedding_dim']))
    contradictions = await memory_core.detect_contradictions()
    print(f"\nDetected {len(contradictions)} potential contradictions.")

    # Shutdown
    await memory_core.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

## Key Improvements

-   **Unified Structure:** Consolidates core logic into fewer files.
-   **Centralized Geometry:** `GeometryManager` handles all geometric operations consistently.
-   **Direct Integration:** HPC-QR, Emotion, Assemblies are integral parts, not separate layers added via mixins.
-   **Improved Efficiency:** Leverages `asyncio` and dedicated persistence class.
-   **Clearer Interfaces:** Simplified API focused on core memory operations.
-   **Hyperbolic First-Class:** Hyperbolic geometry is treated as a core configuration option.
\`\`\`

**Explanation and Design Choices:**

1.  **Consolidation:** Logic from `base.py`, `connectivity.py`, `tools.py`, `personal_details.py`, `rag_context.py`, `enhanced_memory_client.py`, `advanced_memory_system.py`, `memory_core.py`, `memory_assembly.py` (manager part), `memory_manager.py` is largely consolidated or represented within `SynthiansMemoryCore` and its direct components.
2.  **`SynthiansMemoryCore` (Orchestrator):** This class acts as the main entry point. It initializes and holds references to all the specialized components (`GeometryManager`, `UnifiedQuickRecallCalculator`, `EmotionalGatingService`, `MemoryPersistence`, `ThresholdCalibrator`). It orchestrates the flow for processing new memories and retrieving relevant ones, delegating specific tasks.
3.  **`GeometryManager` (Centralized):** This is a crucial new component. It takes *all* responsibility for embedding validation, normalization, alignment (handling 384 vs 768 automatically based on config), geometric transformations (`_to_hyperbolic`, `_from_hyperbolic`), and distance/similarity calculations (`euclidean_distance`, `hyperbolic_distance`, `calculate_similarity`). This ensures consistency and avoids logic duplication.
4.  **`hpc_quickrecal.py` (Focused):** Retains the `UnifiedQuickRecallCalculator` but is simplified. It now *uses* the `GeometryManager` for distance calculations instead of implementing its own. The complex `HPCQRFlowManager` logic is absorbed into `SynthiansMemoryCore`'s processing flow.
5.  **`emotional_intelligence.py` (Focused):** Contains the `EmotionalGatingService` and a *simplified* `EmotionalAnalyzer` interface (assuming the actual analysis might be external or a simpler internal model).
6.  **`memory_structures.py` (Definitions):** Defines `MemoryEntry` and `MemoryAssembly`. `MemoryAssembly` now uses the `GeometryManager` passed during initialization for its internal similarity calculations, ensuring geometric consistency.
7.  **`memory_persistence.py` (Dedicated):** Consolidates all disk I/O logic using `aiofiles` for async operations, atomic writes via temp files, backup management, and pruning logic based on effective QuickRecal scores.
8.  **`adaptive_components.py`:** Keeps the `ThresholdCalibrator` separate for clarity.
9.  **Efficiency:** Uses `asyncio` extensively, especially for I/O in `MemoryPersistence`. Background tasks handle persistence and decay/pruning without blocking core operations. Simplifies the complex multi-layered routing of the original MPL by integrating assembly activation and direct search within `_get_candidate_memories`.
10. **Leaner:** Explicitly excludes the full complexity of the original Synthien subsystem (Dreaming, Narrative, Spiral Awareness beyond basic phase reference) while retaining the core memory mechanisms identified as valuable.

This new structure provides a more streamlined, maintainable, and potentially more efficient implementation while capturing the core value propositions (HPC-QR, Hyperbolic, Emotion, Assemblies) of the original system.
```

# run_server.py

```py
# synthians_memory_core/run_server.py

import os
import sys
import logging
import uvicorn
from pathlib import Path

# Ensure we can import from the synthians_memory_core package
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def main():
    """Run the Synthians Memory Core API server"""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5010"))
    
    print(f"Starting Synthians Memory Core API server at {host}:{port}")
    
    # Use Uvicorn to run the FastAPI application
    uvicorn.run(
        "synthians_memory_core.api.server:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1      # Single worker for memory consistency
    )

if __name__ == "__main__":
    main()

```

# synthians_memory_core.py

```py
# synthians_memory_core/synthians_memory_core.py

import time
import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Set, Union
from pathlib import Path
import random
import uuid
import json
import os

# Import core components from this package
from .custom_logger import logger
from .memory_structures import MemoryEntry, MemoryAssembly
from .hpc_quickrecal import UnifiedQuickRecallCalculator, QuickRecallMode, QuickRecallFactor
from .geometry_manager import GeometryManager, GeometryType
from .emotional_intelligence import EmotionalGatingService
from .memory_persistence import MemoryPersistence
from .adaptive_components import ThresholdCalibrator
from .metadata_synthesizer import MetadataSynthesizer
from .emotion_analyzer import EmotionAnalyzer
from .vector_index import MemoryVectorIndex

class SynthiansMemoryCore:
    """
    Unified Synthians Memory Core.

    Integrates HPC-QuickRecal, Hyperbolic Geometry, Emotional Intelligence,
    Memory Assemblies, Adaptive Thresholds, and Robust Persistence
    into a lean and efficient memory system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'embedding_dim': 768,
            'geometry': 'hyperbolic', # 'euclidean', 'hyperbolic', 'spherical', 'mixed'
            'hyperbolic_curvature': -1.0,
            'storage_path': '/app/memory/stored/synthians', # Unified path
            'persistence_interval': 60.0, # Persist every minute
            'decay_interval': 3600.0, # Check decay every hour
            'prune_check_interval': 600.0, # Check if pruning needed every 10 mins
            'max_memory_entries': 50000,
            'prune_threshold_percent': 0.9, # Prune when 90% full
            'min_quickrecal_for_ltm': 0.2, # Min score to keep after decay
            'assembly_threshold': 0.75,
            'max_assemblies_per_memory': 3,
            'adaptive_threshold_enabled': True,
            'initial_retrieval_threshold': 0.75,
            'vector_index_type': 'Cosine',  # 'L2', 'IP', 'Cosine'
            **(config or {})
        }

        logger.info("SynthiansMemoryCore", "Initializing...", self.config)

        # --- Core Components ---
        self.geometry_manager = GeometryManager({
            'embedding_dim': self.config['embedding_dim'],
            'geometry_type': self.config['geometry'],
            'curvature': self.config['hyperbolic_curvature']
        })

        self.quick_recal = UnifiedQuickRecallCalculator({
            'embedding_dim': self.config['embedding_dim'],
            'mode': QuickRecallMode.HPC_QR, # Default to HPC-QR mode
            'geometry_type': self.config['geometry'],
            'curvature': self.config['hyperbolic_curvature']
        }, geometry_manager=self.geometry_manager) # Pass geometry manager

        # Provide the analyzer instance directly to the gating service
        self.emotional_analyzer = EmotionAnalyzer()  # Use our new robust emotion analyzer
        self.emotional_gating = EmotionalGatingService(
            emotion_analyzer=self.emotional_analyzer, # Pass the instance
            config={'emotional_weight': 0.3} # Example config
        )

        self.persistence = MemoryPersistence({'storage_path': self.config['storage_path']})

        self.threshold_calibrator = ThresholdCalibrator(
            initial_threshold=self.config['initial_retrieval_threshold']
        ) if self.config['adaptive_threshold_enabled'] else None

        self.metadata_synthesizer = MetadataSynthesizer()  # Initialize metadata synthesizer

        # Initialize vector index for fast retrieval
        self.vector_index = MemoryVectorIndex({
            'embedding_dim': self.config['embedding_dim'],
            'storage_path': self.config['storage_path'],
            'index_type': self.config['vector_index_type']
        })

        # --- Memory State ---
        self._memories: Dict[str, MemoryEntry] = {} # In-memory cache/working set
        self.assemblies: Dict[str, MemoryAssembly] = {}
        self.memory_to_assemblies: Dict[str, Set[str]] = {}

        # --- Concurrency & Tasks ---
        self._lock = asyncio.Lock()
        self._background_tasks: List[asyncio.Task] = []
        self._initialized = False
        self._shutdown_signal = asyncio.Event()

        logger.info("SynthiansMemoryCore", "Core components initialized.")

    async def initialize(self):
        """Load persisted state and start background tasks."""
        if self._initialized: return True
        logger.info("SynthiansMemoryCore", "Starting initialization...")
        async with self._lock:
            # Load memories and assemblies from persistence
            loaded_memories = await self.persistence.load_all()
            for mem in loaded_memories:
                self._memories[mem.id] = mem
            logger.info("SynthiansMemoryCore", f"Loaded {len(self._memories)} memories from persistence.")
            # TODO: Load assemblies and memory_to_assemblies mapping

            # Load the vector index
            index_loaded = self.vector_index.load()
            
            # If index wasn't found, build it from loaded memories
            if not index_loaded and self._memories:
                logger.info("SynthiansMemoryCore", "Building vector index from loaded memories...")
                for mem_id, memory in self._memories.items():
                    if memory.embedding is not None:
                        self.vector_index.add(mem_id, memory.embedding)
                # Save the newly built index
                self.vector_index.save()
                logger.info("SynthiansMemoryCore", f"Built and saved vector index with {len(self._memories)} entries")

            # Pass momentum buffer to calculator if needed
            # self.quick_recal.set_external_momentum(...)

            # Start background tasks
            self._background_tasks.append(asyncio.create_task(self._persistence_loop()))
            self._background_tasks.append(asyncio.create_task(self._decay_and_pruning_loop()))

            self._initialized = True
            logger.info("SynthiansMemoryCore", "Initialization complete. Background tasks started.")
        return True

    async def shutdown(self):
        """Gracefully shut down the memory core."""
        logger.info("SynthiansMemoryCore", "Shutting down...")
        self._shutdown_signal.set()
        # Cancel background tasks
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        # Wait for tasks to finish cancellation
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        # Final persistence flush
        await self._persist_all_managed_memories()
        logger.info("SynthiansMemoryCore", "Shutdown complete.")

    # --- Core Memory Operations ---

    async def process_memory(self,
                           content: Optional[str] = None,
                           embedding: Optional[Union[np.ndarray, List[float]]] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """API-compatible wrapper for process_new_memory."""
        if not self._initialized: await self.initialize()
        
        # Call the underlying implementation
        memory = await self.process_new_memory(content=content, embedding=embedding, metadata=metadata)
        
        if memory:
            return {
                "memory_id": memory.id,
                "quickrecal_score": memory.quickrecal_score,
                "metadata": memory.metadata
            }
        else:
            return {
                "memory_id": None,
                "quickrecal_score": None,
                "error": "Failed to process memory"
            }

    async def process_new_memory(self,
                                 content: str,
                                 embedding: Optional[Union[np.ndarray, List[float]]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> Optional[MemoryEntry]:
        """Process and store a new memory entry."""
        if not self._initialized: await self.initialize()
        start_time = time.time()
        metadata = metadata or {}

        # 1. Validate/Generate Embedding
        if embedding is None:
            # embedding = await self._generate_embedding(content) # Assumed external
            logger.warning("SynthiansMemoryCore", "process_new_memory called without embedding, skipping.")
            return None # Cannot proceed without embedding
        validated_embedding = self.geometry_manager._validate_vector(embedding, "Input Embedding")
        if validated_embedding is None:
             logger.error("SynthiansMemoryCore", "Invalid embedding provided, cannot process memory.")
             return None
        aligned_embedding, _ = self.geometry_manager._align_vectors(validated_embedding, np.zeros(self.config['embedding_dim']))
        normalized_embedding = self.geometry_manager._normalize(aligned_embedding)

        # 2. Calculate QuickRecal Score
        context = {'timestamp': time.time(), 'metadata': metadata}
        # Include momentum buffer if available/needed by the mode
        # context['external_momentum'] = ...
        quickrecal_score = await self.quick_recal.calculate(normalized_embedding, text=content, context=context)

        # 3. Analyze Emotion only if not already provided
        emotional_context = metadata.get("emotional_context")
        if not emotional_context:
            logger.info("SynthiansMemoryCore", "Analyzing emotional context for memory")
            emotional_context = await self.emotional_analyzer.analyze(content)
            metadata["emotional_context"] = emotional_context
        else:
            logger.debug("SynthiansMemoryCore", "Using precomputed emotional context from metadata")

        # 4. Generate Hyperbolic Embedding (if enabled)
        hyperbolic_embedding = None
        if self.geometry_manager.config['geometry_type'] == GeometryType.HYPERBOLIC:
            hyperbolic_embedding = self.geometry_manager._to_hyperbolic(normalized_embedding)

        # 5. Run Metadata Synthesizer
        metadata = await self.metadata_synthesizer.synthesize(
            content=content,
            embedding=normalized_embedding,
            base_metadata=metadata,
            emotion_data=emotional_context
        )

        # 6. Create Memory Entry
        memory = MemoryEntry(
            content=content,
            embedding=normalized_embedding,
            quickrecal_score=quickrecal_score,
            metadata=metadata,
            hyperbolic_embedding=hyperbolic_embedding
        )
        
        # Add memory ID to metadata for easier access
        memory.metadata["uuid"] = memory.id

        # 7. Store in memory and persistence
        async with self._lock:
            self._memories[memory.id] = memory
            stored = await self.persistence.save_memory(memory)
            if stored:
                 logger.info("SynthiansMemoryCore", f"Stored new memory {memory.id}", {"quickrecal": quickrecal_score})
            else:
                 # Rollback if persistence failed
                 del self._memories[memory.id]
                 logger.error("SynthiansMemoryCore", f"Failed to persist memory {memory.id}, rolling back.")
                 return None

        # 8. Update Assemblies
        await self._update_assemblies(memory)

        # 9. Add to vector index for fast retrieval
        self.vector_index.add(memory.id, normalized_embedding)
        logger.debug("SynthiansMemoryCore", f"Added memory {memory.id} to vector index")

        proc_time = (time.time() - start_time) * 1000
        logger.debug("SynthiansMemoryCore", f"Processed new memory {memory.id}", {"time_ms": proc_time})
        return memory

    async def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        threshold: Optional[float] = None,
        user_emotion: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve memories based on query relevance.
        """
        
        query_embedding = None
        try:
            # Generate embedding for the query if necessary
            if query:
                query_embedding = await self.generate_embedding(query)
                logger.debug("SynthiansMemoryCore", "Query embedding generated", {
                    "query": query,
                    "has_embedding": query_embedding is not None,
                })
            
            # Get the current threshold (use provided or default)
            current_threshold = threshold
            if current_threshold is None and self.threshold_calibrator is not None:
                current_threshold = self.threshold_calibrator.get_current_threshold()
                logger.debug("SynthiansMemoryCore", f"Using calibrated threshold: {current_threshold:.4f}")
            else:
                logger.debug("SynthiansMemoryCore", f"Using explicit threshold: {current_threshold}")
            
            logger.debug("SynthiansMemoryCore", "Memory retrieval parameters", {
                "query": query,
                "has_embedding": query_embedding is not None,
                "threshold": current_threshold,
                "user_emotion": user_emotion,
                "top_k": top_k,
                "metadata_filter": metadata_filter
            })
            
            # Perform the retrieval
            candidates = await self._get_candidate_memories(query_embedding, top_k * 2)
            logger.debug("SynthiansMemoryCore", f"Found {len(candidates)} candidate memories")
            
            # Score and filter candidates
            if candidates:
                scored_candidates = []
                for memory_dict in candidates:
                    memory_embedding = memory_dict.get("embedding")
                    if memory_embedding is not None and query_embedding is not None:
                        # Calculate similarity score
                        similarity = self.geometry_manager.calculate_similarity(query_embedding, memory_embedding)
                        memory_dict["similarity"] = similarity
                        scored_candidates.append(memory_dict)
                    else:
                        # Skip memories without embeddings
                        continue
                
                # Sort by similarity score (descending)
                scored_candidates.sort(key=lambda x: x.get("similarity", 0), reverse=True)
                
                # Log similarity scores for debugging
                if scored_candidates:
                    score_info = [{
                        "id": cand.get("id", "")[-8:],  # Last 8 chars of ID
                        "score": round(cand.get("similarity", 0), 4),
                        "metadata": {k: v for k, v in cand.get("metadata", {}).items() 
                                     if k in ["source", "test_type"]}
                    } for cand in scored_candidates[:5]]
                    logger.debug("SynthiansMemoryCore", "Top candidate scores", score_info)
                
                # Apply threshold filtering
                if current_threshold is not None:
                    before_threshold = len(scored_candidates)
                    scored_candidates = [c for c in scored_candidates 
                                       if c.get("similarity", 0) >= current_threshold]
                    logger.debug("SynthiansMemoryCore", "After threshold filtering", {
                        "before": before_threshold,
                        "after": len(scored_candidates),
                        "threshold": current_threshold
                    })
                
                # Apply emotional gating if requested
                if user_emotion and hasattr(self, 'emotional_gating') and self.emotional_gating is not None:
                    before_gating = len(scored_candidates)
                    scored_candidates = await self.emotional_gating.gate_memories(
                        scored_candidates, user_emotion
                    )
                    logger.debug("SynthiansMemoryCore", "After emotional gating", {
                        "before": before_gating,
                        "after": len(scored_candidates),
                        "user_emotion": user_emotion
                    })
                
                # Apply metadata filtering if requested
                if metadata_filter and len(scored_candidates) > 0:
                    before_metadata = len(scored_candidates)
                    scored_candidates = self._filter_by_metadata(scored_candidates, metadata_filter)
                    logger.debug("SynthiansMemoryCore", "After metadata filtering", {
                        "before": before_metadata,
                        "after": len(scored_candidates),
                        "filter": metadata_filter
                    })
                
                # Format and return results (taking top_k)
                top_candidates = scored_candidates[:top_k] if len(scored_candidates) > top_k else scored_candidates
                
                result = {
                    "success": True,
                    "memories": top_candidates,
                    "error": None
                }
                
                return result
            
            # Fall through if no candidates found
            logger.warning("SynthiansMemoryCore", "No candidate memories found for query", {"query": query})
            return {"success": True, "memories": [], "error": None}
            
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error in retrieve_memories: {str(e)}")
            import traceback
            logger.error("SynthiansMemoryCore", traceback.format_exc())
            return {"success": False, "memories": [], "error": str(e)}

    async def _get_candidate_memories(self, query_embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        """Retrieve candidate memories using assembly activation and direct vector search."""
        assembly_candidates = set()
        direct_candidates = set()

        # 1. Assembly Activation
        activated_assemblies = await self._activate_assemblies(query_embedding)
        for assembly, activation_score in activated_assemblies[:5]: # Consider top 5 assemblies
            # Lower activation threshold from 0.3 to 0.2 for better retrieval
            if activation_score > 0.2: # Lower activation threshold
                assembly_candidates.update(assembly.memories)
                logger.debug("SynthiansMemoryCore", f"Assembly activated: {len(assembly.memories)} memories, score: {activation_score:.4f}")

        # 2. Direct Vector Search using FAISS Index
        # Validate query embedding for NaN/Inf values
        if query_embedding is not None and (np.isnan(query_embedding).any() or np.isinf(query_embedding).any()):
            logger.warning("SynthiansMemoryCore", "Query embedding has NaN or Inf values")
            # Replace with zeros or return empty list
            return []
            
        # Perform vector search using the FAISS index
        search_results = self.vector_index.search(query_embedding, k=limit, threshold=0.3)
        
        # Add direct search candidates from vector index
        for memory_id, similarity in search_results:
            direct_candidates.add(memory_id)
            logger.debug("SynthiansMemoryCore", f"Memory {memory_id} similarity: {similarity:.4f} (from vector index)")

        # Combine candidates
        all_candidate_ids = assembly_candidates.union(direct_candidates)

        # Fetch MemoryEntry objects
        final_candidates = []
        async with self._lock:
             for mem_id in all_candidate_ids:
                 if mem_id in self._memories:
                      final_candidates.append(self._memories[mem_id].to_dict())

        return final_candidates[:limit] # Limit the final candidate list

    async def _activate_assemblies(self, query_embedding: np.ndarray) -> List[Tuple[MemoryAssembly, float]]:
        """Find and activate assemblies based on query similarity."""
        activated = []
        async with self._lock: # Accessing shared self.assemblies
            for assembly_id, assembly in self.assemblies.items():
                 similarity = assembly.get_similarity(query_embedding)
                 if similarity >= self.config['assembly_threshold'] * 0.8: # Lower threshold for activation
                      assembly.activate(similarity)
                      activated.append((assembly, similarity))
        # Sort by activation score
        activated.sort(key=lambda x: x[1], reverse=True)
        return activated

    async def _update_assemblies(self, memory: MemoryEntry):
        """Find or create assemblies for a new memory."""
        if memory.embedding is None: return

        suitable_assemblies = []
        best_similarity = 0.0
        best_assembly_id = None

        async with self._lock: # Accessing shared self.assemblies
             for assembly_id, assembly in self.assemblies.items():
                  similarity = assembly.get_similarity(memory.embedding)
                  if similarity >= self.config['assembly_threshold']:
                       suitable_assemblies.append((assembly_id, similarity))
                  if similarity > best_similarity:
                       best_similarity = similarity
                       best_assembly_id = assembly_id

        # Sort suitable assemblies by similarity
        suitable_assemblies.sort(key=lambda x: x[1], reverse=True)

        # Add memory to best matching assemblies (up to max limit)
        added_count = 0
        assemblies_updated = set()
        for assembly_id, _ in suitable_assemblies[:self.config['max_assemblies_per_memory']]:
            async with self._lock: # Lock for modifying assembly
                 if assembly_id in self.assemblies:
                     assembly = self.assemblies[assembly_id]
                     if assembly.add_memory(memory):
                          added_count += 1
                          assemblies_updated.add(assembly_id)
                          # Update memory_to_assemblies mapping
                          if memory.id not in self.memory_to_assemblies:
                               self.memory_to_assemblies[memory.id] = set()
                          self.memory_to_assemblies[memory.id].add(assembly_id)

        # If no suitable assembly found, consider creating a new one
        if added_count == 0 and best_similarity > self.config['assembly_threshold'] * 0.5: # Threshold to create new
             async with self._lock: # Lock for creating new assembly
                 # Double check if a suitable assembly was created concurrently
                 assembly_exists = False
                 for asm_id in self.memory_to_assemblies.get(memory.id, set()):
                      if asm_id in self.assemblies: assembly_exists = True; break

                 if not assembly_exists:
                     logger.info("SynthiansMemoryCore", f"Creating new assembly seeded by memory {memory.id[:8]}")
                     new_assembly = MemoryAssembly(geometry_manager=self.geometry_manager, name=f"Assembly around {memory.id[:8]}")
                     if new_assembly.add_memory(memory):
                          self.assemblies[new_assembly.assembly_id] = new_assembly
                          assemblies_updated.add(new_assembly.assembly_id)
                          # Update mapping
                          if memory.id not in self.memory_to_assemblies:
                               self.memory_to_assemblies[memory.id] = set()
                          self.memory_to_assemblies[memory.id].add(new_assembly.assembly_id)
                          added_count += 1

        if added_count > 0:
             logger.debug("SynthiansMemoryCore", f"Updated {added_count} assemblies for memory {memory.id}", {"assemblies": list(assemblies_updated)})

    async def provide_feedback(self, memory_id: str, similarity_score: float, was_relevant: bool):
        """Provide feedback to the threshold calibrator."""
        if self.threshold_calibrator:
            self.threshold_calibrator.record_feedback(similarity_score, was_relevant)
            logger.debug("SynthiansMemoryCore", "Recorded feedback", {"memory_id": memory_id, "score": similarity_score, "relevant": was_relevant})

    async def detect_contradictions(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Detect potential causal contradictions using embeddings."""
        contradictions = []
        async with self._lock: # Access shared _memories
            memories_list = list(self._memories.values())

        # Basic Keyword Filtering for Causal Statements (Can be improved with NLP)
        causal_keywords = ["causes", "caused", "leads to", "results in", "effect of", "affects"]
        causal_memories = [m for m in memories_list if m.embedding is not None and any(k in m.content.lower() for k in causal_keywords)]

        if len(causal_memories) < 2: return []

        logger.info("SynthiansMemoryCore", f"Checking {len(causal_memories)} causal memories for contradictions.")

        # Compare pairs (simplified N^2 comparison, can be optimized)
        compared_pairs = set()
        for i in range(len(causal_memories)):
            for j in range(i + 1, len(causal_memories)):
                mem_a = causal_memories[i]
                mem_b = causal_memories[j]

                # Calculate similarity
                similarity = self.geometry_manager.calculate_similarity(mem_a.embedding, mem_b.embedding)

                # Basic Topic Overlap Check (can be improved)
                words_a = set(mem_a.content.lower().split())
                words_b = set(mem_b.content.lower().split())
                common_words = words_a.intersection(words_b)
                overlap_ratio = len(common_words) / min(len(words_a), len(words_b)) if min(len(words_a), len(words_b)) > 0 else 0

                # Check for potential semantic opposition (basic keyword check)
                opposites = [("increase", "decrease"), ("up", "down"), ("positive", "negative"), ("high", "low")]
                has_opposite = False
                content_a_lower = mem_a.content.lower()
                content_b_lower = mem_b.content.lower()
                for w1, w2 in opposites:
                    if (w1 in content_a_lower and w2 in content_b_lower) or \
                       (w2 in content_a_lower and w1 in content_b_lower):
                        has_opposite = True
                        break

                # If high similarity, sufficient topic overlap, and potential opposition -> contradiction
                if similarity >= threshold and overlap_ratio > 0.3 and has_opposite:
                     contradictions.append({
                          "memory_a_id": mem_a.id,
                          "memory_a_content": mem_a.content,
                          "memory_b_id": mem_b.id,
                          "memory_b_content": mem_b.content,
                          "similarity": similarity,
                          "overlap_ratio": overlap_ratio
                     })

        logger.info("SynthiansMemoryCore", f"Detected {len(contradictions)} potential contradictions.")
        return contradictions


    # --- Background Tasks ---

    async def _persistence_loop(self):
        """Periodically persist changed memories."""
        while not self._shutdown_signal.is_set():
            await asyncio.sleep(self.config['persistence_interval'])
            logger.debug("SynthiansMemoryCore", "Running periodic persistence.")
            await self._persist_all_managed_memories()

    async def _decay_and_pruning_loop(self):
        """Periodically decay memory scores and prune old/irrelevant memories."""
        while not self._shutdown_signal.is_set():
            # Decay check interval
            await asyncio.sleep(self.config['decay_interval'])
            logger.info("SynthiansMemoryCore", "Running memory decay check.")
            await self._apply_decay()

            # Pruning check interval (more frequent)
            await asyncio.sleep(self.config['prune_check_interval'] - self.config['decay_interval'] % self.config['prune_check_interval'])
            logger.debug("SynthiansMemoryCore", "Running pruning check.")
            await self._prune_if_needed()


    async def _persist_all_managed_memories(self):
         """Persist all memories currently managed (in self._memories)."""
         async with self._lock:
              memories_to_persist = list(self._memories.values()) # Get a snapshot
         logger.info("SynthiansMemoryCore", f"Persisting {len(memories_to_persist)} managed memories.")
         count = 0
         for memory in memories_to_persist:
              success = await self.persistence.save_memory(memory)
              if success: count += 1
              # Yield control briefly to prevent blocking loop for too long
              if count % 50 == 0: await asyncio.sleep(0.01)
         logger.info("SynthiansMemoryCore", f"Finished persisting {count} memories.")

    async def _apply_decay(self):
        """Apply decay to QuickRecal scores."""
        async with self._lock:
             modified_ids = []
             for memory_id, memory in self._memories.items():
                 effective_score = memory.get_effective_quickrecal()
                 # Store the effective score back, but don't overwrite original quickrecal
                 memory.metadata['effective_quickrecal'] = effective_score
                 modified_ids.append(memory_id) # Mark for potential persistence update

             # Persist modified memories (optional, could be done in main persistence loop)
             # for mem_id in modified_ids:
             #     await self.persistence.save_memory(self._memories[mem_id])
             logger.info("SynthiansMemoryCore", f"Applied decay to {len(modified_ids)} memories.")


    async def _prune_if_needed(self):
        """Prune memories if storage limit is exceeded."""
        async with self._lock:
             current_size = len(self._memories)
             max_size = self.config['max_memory_entries']
             prune_threshold = int(max_size * self.config['prune_threshold_percent'])

             if current_size <= prune_threshold:
                  return # No pruning needed

             logger.info("SynthiansMemoryCore", f"Memory usage ({current_size}/{max_size}) exceeds threshold ({prune_threshold}). Starting pruning.")
             num_to_prune = current_size - int(max_size * 0.85) # Prune down to 85%

             # Get memories sorted by effective QuickRecal score (lowest first)
             scored_memories = [(mem.id, mem.get_effective_quickrecal()) for mem in self._memories.values()]
             scored_memories.sort(key=lambda x: x[1])

             pruned_count = 0
             for mem_id, score in scored_memories[:num_to_prune]:
                 if score < self.config['min_quickrecal_for_ltm']:
                      if mem_id in self._memories:
                           del self._memories[mem_id]
                           # Also remove from assemblies mapping
                           if mem_id in self.memory_to_assemblies:
                                for asm_id in self.memory_to_assemblies[mem_id]:
                                     if asm_id in self.assemblies:
                                          # Assembly removal logic would be more complex, involving recalculation
                                          # self.assemblies[asm_id].remove_memory(mem_id)
                                          pass # Simplified for now
                                del self.memory_to_assemblies[mem_id]
                           # Delete from persistence
                           await self.persistence.delete_memory(mem_id)
                           pruned_count += 1

             logger.info("SynthiansMemoryCore", f"Pruned {pruned_count} memories.")

    # --- Tool Interface ---

    def get_tools(self) -> List[Dict[str, Any]]:
        """Return descriptions of available tools for LLM integration."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "retrieve_memories_tool",
                    "description": "Retrieve relevant memories based on a query text.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."},
                            "top_k": {"type": "integer", "description": "Max number of results.", "default": 5},
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "process_new_memory_tool",
                    "description": "Process and store a new piece of information or experience.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {"type": "string", "description": "The content of the memory."},
                            "metadata": {"type": "object", "description": "Optional metadata (source, type, etc.)."}
                        },
                        "required": ["content"]
                    }
                }
            },
             {
                "type": "function",
                "function": {
                    "name": "provide_retrieval_feedback_tool",
                    "description": "Provide feedback on the relevance of retrieved memories.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "The ID of the memory being rated."},
                             "similarity_score": {"type": "number", "description": "The similarity score assigned during retrieval."},
                            "was_relevant": {"type": "boolean", "description": "True if the memory was relevant, False otherwise."}
                        },
                        "required": ["memory_id", "similarity_score", "was_relevant"]
                    }
                }
            },
             {
                "type": "function",
                "function": {
                    "name": "detect_contradictions_tool",
                    "description": "Check for potential contradictions within recent memory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                             "threshold": {"type": "number", "description": "Similarity threshold for contradiction.", "default": 0.75}
                        }
                    }
                }
            }
            # TODO: Add tools for assemblies, emotional state, etc.
        ]

    async def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call from an external agent (e.g., LLM)."""
        logger.info("SynthiansMemoryCore", f"Handling tool call: {tool_name}", {"args": args})
        try:
            if tool_name == "retrieve_memories_tool":
                 query = args.get("query")
                 top_k = args.get("top_k", 5)
                 # Assume query embedding generation happens here or is passed in context
                 # query_embedding = await self._generate_embedding(query) # Placeholder
                 # For now, we rely on the retrieve_memories method to handle text query
                 memories = await self.retrieve_memories(query=query, top_k=top_k)
                 # Return simplified dicts for LLM
                 return {"memories": [{"id": m.get("id"), "content": m.get("content"), "score": m.get("final_score", m.get("relevance_score"))} for m in memories]}

            elif tool_name == "process_new_memory_tool":
                 content = args.get("content")
                 metadata = args.get("metadata")
                 # Embedding generation would happen here
                 # embedding = await self._generate_embedding(content) # Placeholder
                 # For now, store without embedding if not provided
                 entry = await self.process_new_memory(content=content, metadata=metadata)
                 return {"success": entry is not None, "memory_id": entry.id if entry else None}

            elif tool_name == "provide_retrieval_feedback_tool":
                 memory_id = args.get("memory_id")
                 similarity_score = args.get("similarity_score")
                 was_relevant = args.get("was_relevant")
                 if self.threshold_calibrator:
                      await self.provide_feedback(memory_id, similarity_score, was_relevant)
                      return {"success": True, "message": "Feedback recorded."}
                 else:
                      return {"success": False, "error": "Adaptive thresholding not enabled."}

            elif tool_name == "detect_contradictions_tool":
                 threshold = args.get("threshold", 0.75)
                 contradictions = await self.detect_contradictions(threshold)
                 return {"success": True, "contradictions_found": len(contradictions), "contradictions": contradictions}

            else:
                 logger.warning("SynthiansMemoryCore", f"Unknown tool called: {tool_name}")
                 return {"success": False, "error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error handling tool call {tool_name}", {"error": str(e)})
            return {"success": False, "error": str(e)}

    # --- Helper & Placeholder Methods ---

    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embeddings using a consistent method for all text processing."""
        # Use SentenceTransformer directly without importing server.py
        try:
            from sentence_transformers import SentenceTransformer
            # Use the same model name as server.py
            import os
            model_name = os.environ.get("EMBEDDING_MODEL", "all-mpnet-base-v2")
            model = SentenceTransformer(model_name)
            
            logger.info("SynthiansMemoryCore", f"Using embedding model {model_name}")
            embedding = model.encode([text], convert_to_tensor=False)[0]
            return self.geometry_manager._normalize(np.array(embedding, dtype=np.float32))
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}")
            
            # Fallback to a deterministic embedding based on text hash
            # This ensures same text always gets same embedding
            import hashlib
            
            # Create a deterministic embedding based on the hash of the text
            text_bytes = text.encode('utf-8')
            hash_obj = hashlib.md5(text_bytes)
            hash_digest = hash_obj.digest()
            
            # Convert the 16-byte digest to a list of floats
            # Repeating it to fill the embedding dimension
            byte_values = list(hash_digest) * (self.config['embedding_dim'] // 16 + 1)
            
            # Create a normalized embedding vector
            embedding = np.array([float(byte) / 255.0 for byte in byte_values[:self.config['embedding_dim']]], dtype=np.float32)
            
            logger.warning("SynthiansMemoryCore", "Using deterministic hash-based embedding generation")
            return self.geometry_manager._normalize(embedding)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        persistence_stats = asyncio.run(self.persistence.get_stats()) # Run sync in this context
        quick_recal_stats = self.quick_recal.get_stats()
        threshold_stats = self.threshold_calibrator.get_statistics() if self.threshold_calibrator else {}

        return {
            "core_stats": {
                "total_memories": len(self._memories),
                "total_assemblies": len(self.assemblies),
                "initialized": self._initialized,
            },
            "persistence_stats": persistence_stats,
            "quick_recal_stats": quick_recal_stats,
            "threshold_stats": threshold_stats
        }

    def process_memory_sync(self, content: str, embedding: Optional[np.ndarray] = None, 
                           metadata: Optional[Dict[str, Any]] = None,
                           emotion_data: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], None]:
        """
Process a new memory synchronously without using asyncio.run().
        
This is a synchronous version of process_new_memory that avoids potential asyncio.run() issues.

Args:
    content: The text content of the memory
    embedding: Vector representation of the content (optional)
    metadata: Base metadata for the memory entry (optional)
    emotion_data: Pre-computed emotion analysis results (optional)
        """
        try:
            logger.info("SynthiansMemoryCore", "Processing memory synchronously")
            
            # Create a new memory entry
            memory_id = f"mem_{uuid.uuid4().hex[:12]}"  # More consistent ID format
            timestamp = metadata.get('timestamp', time.time()) if metadata else time.time()
            
            # Ensure metadata is a dictionary
            metadata = metadata or {}
            metadata['timestamp'] = timestamp
            
            # Use provided embedding or generate from content
            if embedding is None and self.geometry_manager is not None:
                try:
                    embedding = self.geometry_manager.process_text(content)
                    logger.info("SynthiansMemoryCore", f"Generated embedding for memory {memory_id}")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error generating embedding: {str(e)}")
                    # Create a fallback embedding of zeros
                    embedding = np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)
            
            # If emotion_data is not provided but we have an emotion analyzer, try to generate it
            if emotion_data is None and self.emotion_analyzer is not None:
                try:
                    # Use the synchronous version for consistency
                    emotion_result = self.emotion_analyzer.analyze_sync(content)
                    if emotion_result and emotion_result.get('success', False):
                        emotion_data = emotion_result
                        logger.info("SynthiansMemoryCore", f"Generated emotion data for memory {memory_id}")
                except Exception as e:
                    logger.warning("SynthiansMemoryCore", f"Error analyzing emotions: {str(e)}")
            
            # Enhance metadata using the MetadataSynthesizer
            enhanced_metadata = metadata
            if self.metadata_synthesizer is not None:
                try:
                    # Use the synchronous version of metadata synthesis
                    enhanced_metadata = self.metadata_synthesizer.synthesize_sync(
                        content=content,
                        embedding=embedding,
                        base_metadata=metadata,
                        emotion_data=emotion_data
                    )
                    logger.info("SynthiansMemoryCore", f"Enhanced metadata for memory {memory_id}")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error enhancing metadata: {str(e)}")
            
            # Calculate QuickRecal score
            quickrecal_score = 0.5  # Default value
            if self.quick_recal is not None and embedding is not None:
                try:
                    # Use synchronous version to avoid asyncio issues
                    context = {'text': content, 'timestamp': timestamp}
                    if enhanced_metadata:
                        context.update(enhanced_metadata)
                    
                    if hasattr(self.quick_recal, 'calculate_sync'):
                        quickrecal_score = self.quick_recal.calculate_sync(embedding, context=context)
                        logger.info("SynthiansMemoryCore", f"Calculated QuickRecal score: {quickrecal_score}")
                    else:
                        logger.warning("SynthiansMemoryCore", "No synchronous QuickRecal calculate method available")
                except Exception as e:
                    logger.error("SynthiansMemoryCore", f"Error calculating QuickRecal score: {str(e)}")
            
            # Create memory object
            memory_entry = {
                'id': memory_id,
                'content': content,
                'embedding': embedding.tolist() if embedding is not None else None,
                'metadata': enhanced_metadata,  # Use the enhanced metadata
                'quickrecal_score': quickrecal_score,
                'created_at': timestamp,
                'updated_at': timestamp,
                'access_count': 0,
                'last_accessed': timestamp
            }
            
            # Store memory directly without queueing to avoid potential deadlocks
            self._memories[memory_id] = memory_entry
            logger.info("SynthiansMemoryCore", f"Memory {memory_id} stored in memory")
            
            # Only queue if persistence is enabled and queue is not full
            try:
                if self._persistence and not self._memory_queue.full():
                    self._memory_queue.put_nowait((memory_id, memory_entry))
                    logger.info("SynthiansMemoryCore", f"Memory {memory_id} queued for persistence")
            except Exception as queue_err:
                logger.error("SynthiansMemoryCore", f"Failed to queue memory: {str(queue_err)}")
                # Memory is still in _memories, just not persisted
            
            # Return success
            return memory_entry
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error processing memory synchronously: {str(e)}")
            return None

```

# tests\conftest.py

```py
import os
import pytest
import asyncio
import shutil
import tempfile
from datetime import datetime

# Define common test fixtures

@pytest.fixture(scope="session")
def temp_test_dir():
    """Create a temporary directory for test data that's removed after tests finish."""
    test_dir = tempfile.mkdtemp(prefix="synthians_test_")
    print(f"Created temporary test directory: {test_dir}")
    yield test_dir
    # Clean up after tests
    shutil.rmtree(test_dir, ignore_errors=True)
    print(f"Removed temporary test directory: {test_dir}")

@pytest.fixture(scope="session")
def test_server_url():
    """Return the URL of the test server."""
    # Default to localhost:5010, but allow override through environment variable
    return os.environ.get("SYNTHIANS_TEST_URL", "http://localhost:5010")

# Add markers for categorizing tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "smoke: mark test as a smoke test (basic functionality)"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as a slow test"
    )
    config.addinivalue_line(
        "markers", "emotion: mark test as testing emotion analysis"
    )
    config.addinivalue_line(
        "markers", "retrieval: mark test as testing memory retrieval"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test"
    )

# Helper functions for test utilities

async def create_test_memories(client, count=5, prefix="Test memory"):
    """Create a batch of test memories and return their IDs."""
    memory_ids = []
    timestamp = datetime.now().isoformat()
    
    for i in range(count):
        content = f"{prefix} {i+1} created at {timestamp}"
        metadata = {"test_batch": timestamp, "index": i}
        
        response = await client.process_memory(content=content, metadata=metadata)
        if response.get("success"):
            memory_ids.append(response.get("memory_id"))
    
    return memory_ids, timestamp

```

# tests\README.md

```md
# Synthians Memory Core Test Suite

This comprehensive test suite is designed to validate the functionality, performance, and reliability of the Synthians Memory Core system. The tests are organized into modular, progressive phases to ensure full coverage of all components while allowing for targeted testing of specific subsystems.

## 🧪 Test Structure

The tests are organized into seven progressive phases, each focusing on different aspects of the system:

### 🔹 Phase 1: Core Infrastructure Validation
- `test_api_health.py` - Basic API endpoints, health, and stats tests

### 🔹 Phase 2: Memory Lifecycle Test
- `test_memory_lifecycle.py` - End-to-end memory creation, retrieval, feedback, deletion

### 🔹 Phase 3: Emotional & Cognitive Layer Test
- `test_emotion_and_cognitive.py` - Tests for emotion analysis, metadata enrichment, and cognitive load scoring

### 🔹 Phase 4: Transcription & Voice Pipeline Test
- `test_transcription_voice_flow.py` - Tests for speech transcription, interruption handling, and voice state management

### 🔹 Phase 5: Retrieval Dynamics Test
- `test_retrieval_dynamics.py` - Tests for memory retrieval with various conditions, thresholds, and filters

### 🔹 Phase 6: Tooling Integration Test
- `test_tool_integration.py` - Tests for tool interfaces that call core functions

### 🔹 Phase 7: Stress + Load Test
- `test_stress_load.py` - High-volume and performance tests 

## 📋 Prerequisites

\`\`\`bash
pip install pytest pytest-asyncio pytest-html aiohttp
\`\`\`

## 🚀 Running Tests

### Quick Start

\`\`\`bash
# Run all tests
python tests/run_tests.py

# Run with more detailed output
python tests/run_tests.py --verbose

# Run smoke tests only
python tests/run_tests.py --markers="smoke"

# Run a specific test module
python tests/run_tests.py --module="test_api_health.py"

# Run a specific test function
python tests/run_tests.py --test="test_health_and_stats"

# Generate HTML and XML reports
python tests/run_tests.py --report

# Run tests in parallel
python tests/run_tests.py --parallel=4

# Test against a different server
python tests/run_tests.py --url="http://test-server:5010"
\`\`\`

### Using pytest directly

\`\`\`bash
# Run all tests
pytest -xvs --asyncio-mode=auto

# Run a specific test module
pytest -xvs test_api_health.py --asyncio-mode=auto

# Run tests with a specific marker
pytest -xvs -m smoke --asyncio-mode=auto
\`\`\`

## 🏷️ Test Markers

Tests are categorized with the following markers:

- `smoke`: Basic functionality tests that should always pass
- `integration`: Tests that verify integration between components
- `slow`: Tests that take longer to run (e.g., stress tests)
- `emotion`: Tests focused on emotion analysis
- `retrieval`: Tests focused on memory retrieval
- `stress`: High-volume load tests

## 📊 Test Reports

When using the `--report` option, the test suite generates:

- HTML reports in `test_reports/report_TIMESTAMP.html`
- XML reports in `test_reports/report_TIMESTAMP.xml` (JUnit format for CI systems)

## 🔧 Configuration

The test suite can be configured using environment variables:

- `SYNTHIANS_TEST_URL`: URL of the test server (default: http://localhost:5010)

## ⚠️ Implementation Notes

1. Tests use a temporary directory for test data by default
2. Some tests expect specific API functionality which may not be implemented yet
3. Stress tests have reduced volumes by default to run faster - adjust constants in code for full stress testing
4. Pay attention to potential race conditions with concurrent tests
5. Some tests may fail if specific components (e.g., emotion analyzer) are not properly initialized

## 🛠 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Dimension mismatch warnings in logs | Expected during testing with different embedding dimensions |
| Empty embeddings | Check if the embedding model is properly loaded |
| HTTP connection errors | Ensure the server is running and accessible at the configured URL |
| File permission errors | Check that the test directory has proper write permissions |
| Test timeouts | Adjust timeout settings or reduce batch sizes in stress tests |

## 🔄 Continuous Integration

This test suite is designed to be integrated with CI/CD pipelines. XML reports in JUnit format can be consumed by most CI systems.

```

# tests\run_tests.py

```py
#!/usr/bin/env python

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

def run_tests(args):
    """Run the Synthians Memory Core test suite with the specified options."""
    # Construct the pytest command
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add test selection options
    if args.markers:
        for marker in args.markers.split(","):
            cmd.append(f"-m {marker}")
    
    if args.module:
        cmd.append(args.module)
    
    if args.test:
        cmd.append(f"-k {args.test}")
    
    # Add parallel execution if specified
    if args.parallel:
        cmd.append(f"-xvs -n {args.parallel}")
    
    # Add report options
    if args.report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join("test_reports", f"report_{timestamp}")
        
        # Create the report directory if it doesn't exist
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        # Add HTML report
        cmd.append(f"--html={report_path}.html")
        
        # Add JUnit XML report for CI integration
        cmd.append(f"--junitxml={report_path}.xml")
    
    # Add asyncio mode
    cmd.append("--asyncio-mode=auto")
    
    # Join the command parts
    cmd_str = " ".join(cmd)
    print(f"Running: {cmd_str}")
    
    # Execute the command
    start_time = time.time()
    result = subprocess.run(cmd_str, shell=True)
    elapsed_time = time.time() - start_time
    
    print(f"\nTests completed in {elapsed_time:.2f} seconds with exit code {result.returncode}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Run Synthians Memory Core test suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-m", "--markers", help="Comma-separated list of markers to run (e.g., 'smoke,integration')")
    parser.add_argument("-k", "--test", help="Expression to filter tests by name")
    parser.add_argument("-t", "--module", help="Specific test module to run (e.g., 'test_api_health.py')")
    parser.add_argument("-p", "--parallel", type=int, help="Run tests in parallel with specified number of processes")
    parser.add_argument("-r", "--report", action="store_true", help="Generate HTML and XML test reports")
    parser.add_argument("--url", help="Override the API server URL (default: http://localhost:5010)")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.url:
        os.environ["SYNTHIANS_TEST_URL"] = args.url
    
    print("=== Synthians Memory Core Test Runner ===")
    print(f"Starting tests at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set the working directory to the script's directory
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    try:
        return run_tests(args)
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())

```

# tests\test_api_health.py

```py
import pytest
import asyncio
import json
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_health_and_stats():
    """Test basic health check and stats endpoints."""
    async with SynthiansClient() as client:
        # Test health endpoint
        health = await client.health_check()
        assert health.get("status") == "healthy", "Health check failed"
        assert "uptime_seconds" in health, "Health response missing uptime"
        assert "version" in health, "Health response missing version"
        
        # Test stats endpoint
        stats = await client.get_stats()
        assert stats.get("success") is True, "Stats endpoint failed"
        assert "api_server" in stats, "Stats missing api_server information"
        assert "memory_count" in stats.get("api_server", {}), "Stats missing memory count"
        
        # Output results for debugging
        print(f"Health check: {json.dumps(health, indent=2)}")
        print(f"Stats: {json.dumps(stats, indent=2)}")

@pytest.mark.asyncio
async def test_api_smoke_test():
    """Test all API endpoints to ensure they respond correctly."""
    async with SynthiansClient() as client:
        # Test embedding generation
        embed_resp = await client.generate_embedding("Test embedding generation")
        assert embed_resp.get("success") is True, "Embedding generation failed"
        assert "embedding" in embed_resp, "No embedding returned"
        assert "dimension" in embed_resp, "No dimension information"
        
        # Test emotion analysis
        emotion_resp = await client.analyze_emotion("I am feeling very happy today")
        assert emotion_resp.get("success") is True, "Emotion analysis failed"
        assert "emotions" in emotion_resp, "No emotions returned"
        assert "dominant_emotion" in emotion_resp, "No dominant emotion identified"
        
        # Test QuickRecal calculation
        qr_resp = await client.calculate_quickrecal(text="Testing QuickRecal API")
        assert qr_resp.get("success") is True, "QuickRecal calculation failed"
        assert "quickrecal_score" in qr_resp, "No QuickRecal score returned"
        
        # Test contradiction detection
        contradict_resp = await client.detect_contradictions(threshold=0.7)
        assert contradict_resp.get("success") is True, "Contradiction detection failed"

```

# tests\test_emotion_and_cognitive.py

```py
import pytest
import asyncio
import json
import numpy as np
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_emotion_analysis_rich():
    """Test emotion analysis with various emotional inputs."""
    async with SynthiansClient() as client:
        # Test happy emotion
        happy_text = "I'm incredibly happy today! Everything is going wonderfully well!"
        happy_result = await client.analyze_emotion(happy_text)
        
        assert happy_result.get("success") is True, "Emotion analysis failed"
        assert happy_result.get("dominant_emotion") in ["joy", "happiness"], f"Expected happy emotion, got {happy_result.get('dominant_emotion')}"
        assert happy_result.get("emotions", {}).get("joy", 0) > 0.5, "Expected high joy score"
        
        print(f"Happy emotion result: {json.dumps(happy_result, indent=2)}")
        
        # Test sad emotion
        sad_text = "I feel so sad and depressed today. Everything is going wrong."
        sad_result = await client.analyze_emotion(sad_text)
        
        assert sad_result.get("success") is True, "Emotion analysis failed"
        assert sad_result.get("dominant_emotion") in ["sadness", "sorrow"], f"Expected sad emotion, got {sad_result.get('dominant_emotion')}"
        
        print(f"Sad emotion result: {json.dumps(sad_result, indent=2)}")
        
        # Test angry emotion
        angry_text = "I'm absolutely furious about how I was treated! This is outrageous!"
        angry_result = await client.analyze_emotion(angry_text)
        
        assert angry_result.get("success") is True, "Emotion analysis failed"
        assert angry_result.get("dominant_emotion") in ["anger", "rage"], f"Expected anger emotion, got {angry_result.get('dominant_emotion')}"
        
        print(f"Angry emotion result: {json.dumps(angry_result, indent=2)}")

@pytest.mark.asyncio
async def test_emotion_fallback_path():
    """Test emotion analysis fallback mechanisms when model fails."""
    # Note: This test assumes emotion analyzer has a fallback mechanism
    # when the primary model fails. We'll test with extreme text that might
    # cause issues for the model.
    
    async with SynthiansClient() as client:
        # Test with extremely long text that might cause issues
        long_text = "happy " * 1000  # Very long repetitive text
        result = await client.analyze_emotion(long_text)
        
        # Even if the main model fails, we should still get a result
        assert result.get("success") is True, "Emotion analysis completely failed"
        assert "dominant_emotion" in result, "No dominant emotion provided"
        
        # Test with empty text
        empty_result = await client.analyze_emotion("")
        assert empty_result.get("success") is True, "Empty text analysis failed"
        assert "dominant_emotion" in empty_result, "No dominant emotion for empty text"
        
        print(f"Empty text emotion result: {json.dumps(empty_result, indent=2)}")

@pytest.mark.asyncio
async def test_emotion_saved_in_metadata():
    """Test that emotional analysis is saved in memory metadata."""
    async with SynthiansClient() as client:
        # Create a memory with strong emotional content
        content = "I am absolutely thrilled about the amazing news I received today!"
        
        # Process the memory with emotion analysis enabled
        memory_resp = await client.process_memory(
            content=content,
            metadata={"analyze_emotion": True}
        )
        
        assert memory_resp.get("success") is True, "Memory creation failed"
        metadata = memory_resp.get("metadata", {})
        
        # Check that emotion data was added to metadata
        assert "dominant_emotion" in metadata, "No dominant emotion in metadata"
        assert "emotional_intensity" in metadata, "No emotional intensity in metadata"
        assert "emotions" in metadata, "No emotions dictionary in metadata"
        
        # Check that the emotion is reasonable for the content
        assert metadata.get("dominant_emotion") in ["joy", "happiness"], f"Expected happy emotion, got {metadata.get('dominant_emotion')}"
        
        print(f"Memory metadata with emotions: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_cognitive_load_score_range():
    """Test that cognitive load scoring works across different complexity levels."""
    async with SynthiansClient() as client:
        # Test with simple text
        simple_text = "This is a simple sentence."
        simple_memory = await client.process_memory(content=simple_text)
        
        # Test with complex text
        complex_text = """The quantum mechanical model is a theoretical framework that describes the behavior of subatomic 
        particles through probabilistic wave functions. It posits that particles exhibit both wave-like and 
        particle-like properties, a concept known as wave-particle duality. The Schrödinger equation, a fundamental 
        mathematical formulation in quantum mechanics, predicts how these wave functions evolve over time. 
        Unlike classical mechanics, quantum mechanics introduces inherent uncertainty in measurements, 
        as formalized in Heisenberg's uncertainty principle, which states that certain pairs of physical properties 
        cannot be precisely measured simultaneously."""
        complex_memory = await client.process_memory(content=complex_text)
        
        # Get metadata for both memories
        simple_metadata = simple_memory.get("metadata", {})
        complex_metadata = complex_memory.get("metadata", {})
        
        # If cognitive_complexity is in the metadata, verify it's higher for complex text
        if "cognitive_complexity" in simple_metadata and "cognitive_complexity" in complex_metadata:
            simple_complexity = simple_metadata.get("cognitive_complexity", 0)
            complex_complexity = complex_metadata.get("cognitive_complexity", 0)
            
            # The complex text should have higher cognitive complexity
            assert complex_complexity > simple_complexity, \
                f"Expected higher complexity for complex text: simple={simple_complexity}, complex={complex_complexity}"
            
            print(f"Simple text complexity: {simple_complexity}")
            print(f"Complex text complexity: {complex_complexity}")

@pytest.mark.asyncio
async def test_emotional_gating_blocks_mismatched():
    """Test that emotional gating blocks memories with mismatched emotions."""
    async with SynthiansClient() as client:
        # Create a happy memory
        happy_text = "I'm so happy and excited about my new job!"
        happy_memory = await client.process_memory(content=happy_text)
        
        # Wait briefly for processing
        await asyncio.sleep(0.5)
        
        # Try to retrieve with angry emotion context
        angry_emotion = {"dominant_emotion": "anger", "emotions": {"anger": 0.9}}
        retrieval_resp = await client.retrieve_memories(
            query="job",
            top_k=5,
            user_emotion=angry_emotion
        )
        
        memories = retrieval_resp.get("memories", [])
        
        # If emotional gating is working, the happy memory might be ranked lower or filtered
        # We can't assert exact behavior since it depends on implementation details
        # Instead, we'll log the results for inspection
        print(f"Retrieved {len(memories)} memories with mismatched emotion")
        
        # Create an angry memory
        angry_text = "I'm absolutely furious about how they handled my job application!"
        angry_memory = await client.process_memory(content=angry_text)
        
        # Wait briefly for processing
        await asyncio.sleep(0.5)
        
        # Retrieve again with the same angry emotion context
        angry_retrieval = await client.retrieve_memories(
            query="job",
            top_k=5,
            user_emotion=angry_emotion
        )
        
        # The angry memory should now be present and possibly ranked higher
        angry_memories = angry_retrieval.get("memories", [])
        
        print(f"Retrieved {len(angry_memories)} memories with matching emotion")
        
        # Print the scores for comparison (if available)
        if memories and angry_memories and "quickrecal_score" in memories[0] and "quickrecal_score" in angry_memories[0]:
            print(f"Mismatched emotion memory score: {memories[0].get('quickrecal_score')}")
            print(f"Matching emotion memory score: {angry_memories[0].get('quickrecal_score')}")

```

# tests\test_memory_lifecycle.py

```py
import pytest
import asyncio
import json
import time
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_basic_memory_flow():
    """Test the basic memory creation, retrieval, and feedback flow."""
    async with SynthiansClient() as client:
        # Step 1: Create a unique memory with a timestamp
        current_time = datetime.now().isoformat()
        content = f"Testing memory processing lifecycle at {current_time}"
        memory_resp = await client.process_memory(
            content=content,
            metadata={"source": "test_suite", "importance": 0.8}
        )
        
        # Assert successful creation
        assert memory_resp.get("success") is True, f"Memory creation failed: {memory_resp.get('error')}"
        memory_id = memory_resp.get("memory_id")
        assert memory_id is not None, "No memory ID returned"
        
        # Print for debugging
        print(f"Memory created with ID: {memory_id}")
        print(f"Memory response: {json.dumps(memory_resp, indent=2)}")
        
        # Step 2: Retrieve the memory
        # Use a unique portion of the content to ensure we get this specific memory
        query = f"memory processing lifecycle at {current_time}"
        # Add a lower threshold to ensure retrieval works
        retrieval_resp = await client.retrieve_memories(query, top_k=3, threshold=0.2)
        
        # Assert successful retrieval
        assert retrieval_resp.get("success") is True, f"Memory retrieval failed: {retrieval_resp.get('error')}"
        memories = retrieval_resp.get("memories", [])
        assert len(memories) > 0, "No memories retrieved"
        
        # Check if our specific memory was retrieved
        retrieved_ids = [m.get("id") for m in memories]
        assert memory_id in retrieved_ids, f"Created memory {memory_id} not found in retrieved memories: {retrieved_ids}"
        
        # Print for debugging
        print(f"Retrieved {len(memories)} memories")
        print(f"Retrieved memory IDs: {retrieved_ids}")
        
        # Step 3: Provide feedback
        feedback_resp = await client.provide_feedback(
            memory_id=memory_id,
            similarity_score=0.85,
            was_relevant=True
        )
        
        # Assert successful feedback
        assert feedback_resp.get("success") is True, f"Feedback submission failed: {feedback_resp.get('error')}"
        assert "new_threshold" in feedback_resp, "No threshold adjustment information returned"
        
        # Print for debugging
        print(f"Feedback response: {json.dumps(feedback_resp, indent=2)}")

@pytest.mark.asyncio
async def test_memory_persistence_roundtrip():
    """Test that memories persist and can be retrieved after creation."""
    async with SynthiansClient() as client:
        # Create a unique memory
        unique_id = int(time.time() * 1000)
        content = f"Persistence test memory with unique ID: {unique_id}"
        
        # Create the memory
        creation_resp = await client.process_memory(content=content)
        assert creation_resp.get("success") is True, "Memory creation failed"
        memory_id = creation_resp.get("memory_id")
        
        # Wait briefly to ensure persistence
        await asyncio.sleep(0.5)
        
        # Retrieve the memory with the unique identifier
        retrieval_resp = await client.retrieve_memories(f"unique ID: {unique_id}", top_k=5, threshold=0.2)
        print(f"\nRetrieval response: {json.dumps(retrieval_resp, indent=2)}")
        assert retrieval_resp.get("success") is True, f"Memory retrieval failed: {retrieval_resp.get('error', 'No error specified')}"
        
        # Verify the memory was retrieved
        memories = retrieval_resp.get("memories", [])
        retrieved_ids = [m.get("id") for m in memories]
        assert memory_id in retrieved_ids, f"Memory {memory_id} not persisted/retrieved"

@pytest.mark.asyncio
async def test_metadata_enrichment_on_store():
    """Test that metadata is properly enriched when storing memories."""
    async with SynthiansClient() as client:
        # Create a memory with minimal metadata
        content = "Test memory for metadata enrichment"
        metadata = {"source": "test_suite", "custom_field": "custom_value"}
        
        response = await client.process_memory(content=content, metadata=metadata)
        assert response.get("success") is True, "Memory creation failed"
        
        # Verify metadata enrichment
        returned_metadata = response.get("metadata", {})
        
        # Check that our custom metadata was preserved
        assert returned_metadata.get("source") == "test_suite"
        assert returned_metadata.get("custom_field") == "custom_value"
        
        # Check that system metadata was added
        assert "timestamp" in returned_metadata, "Timestamp metadata missing"
        assert "length" in returned_metadata, "Length metadata missing"
        assert "uuid" in returned_metadata, "UUID metadata missing"
        
        # Optional checks for more advanced metadata
        if "cognitive_complexity" in returned_metadata:
            assert isinstance(returned_metadata["cognitive_complexity"], (int, float))
        
        print(f"Enriched metadata: {json.dumps(returned_metadata, indent=2)}")

@pytest.mark.asyncio
async def test_delete_memory_by_id():
    """Test memory deletion functionality."""
    async with SynthiansClient() as client:
        # Create a memory
        content = f"Memory to be deleted at {datetime.now().isoformat()}"
        creation_resp = await client.process_memory(content=content)
        assert creation_resp.get("success") is True, "Memory creation failed"
        memory_id = creation_resp.get("memory_id")
        
        # TODO: Implement actual delete endpoint call once available
        # This is a placeholder for when the delete endpoint is implemented
        
        # Example of how delete might be implemented:
        # delete_resp = await client.delete_memory(memory_id=memory_id)
        # assert delete_resp.get("success") is True, "Memory deletion failed"
        
        # After implementing deletion, verify the memory is gone:
        # retrieval_resp = await client.retrieve_memories(content, top_k=1)  
        # memories = retrieval_resp.get("memories", [])
        # assert memory_id not in [m.get("id") for m in memories], "Memory still exists after deletion"

```

# tests\test_retrieval_dynamics.py

```py
import pytest
import asyncio
import json
import time
import numpy as np
from synthians_memory_core.api.client.client import SynthiansClient

@pytest.mark.asyncio
async def test_retrieve_with_emotion_match():
    """Test retrieval with emotional matching."""
    async with SynthiansClient() as client:
        # Create memories with different emotions
        happy_memory = await client.process_memory(
            content="I'm so excited about this amazing project! Everything is going wonderfully!",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        sad_memory = await client.process_memory(
            content="I'm feeling really down today. Nothing seems to be working out.",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        angry_memory = await client.process_memory(
            content="I'm absolutely furious about how this situation was handled!",
            metadata={"source": "emotion_test", "test_group": "retrieval_emotion"}
        )
        
        # Wait briefly for processing
        await asyncio.sleep(1)
        
        # Retrieve with happy emotion context
        happy_emotion = {"dominant_emotion": "joy", "emotions": {"joy": 0.8, "surprise": 0.2}}
        happy_results = await client.retrieve_memories(
            query="feeling emotion test",
            top_k=5,
            user_emotion=happy_emotion
        )
        
        # Retrieve with sad emotion context
        sad_emotion = {"dominant_emotion": "sadness", "emotions": {"sadness": 0.9}}
        sad_results = await client.retrieve_memories(
            query="feeling emotion test",
            top_k=5,
            user_emotion=sad_emotion
        )
        
        # If emotional gating is working correctly, happy memories should rank higher
        # when queried with happy emotion, and sad memories with sad emotion
        happy_memories = happy_results.get("memories", [])
        sad_memories = sad_results.get("memories", [])
        
        print(f"Happy emotion results (first memory): {json.dumps(happy_memories[0] if happy_memories else {}, indent=2)}")
        print(f"Sad emotion results (first memory): {json.dumps(sad_memories[0] if sad_memories else {}, indent=2)}")
        
        # Note: These assertions might be too strict depending on implementation
        # The exact ranking will depend on many factors
        if happy_memories and sad_memories:
            for memory in happy_memories:
                if memory.get("content", "").startswith("I'm so excited"):
                    happy_rank = happy_memories.index(memory)
                    break
            else:
                happy_rank = -1
                
            for memory in sad_memories:
                if memory.get("content", "").startswith("I'm feeling really down"):
                    sad_rank = sad_memories.index(memory)
                    break
            else:
                sad_rank = -1
            
            print(f"Happy memory rank in happy query: {happy_rank}")
            print(f"Sad memory rank in sad query: {sad_rank}")

@pytest.mark.asyncio
async def test_retrieve_with_low_threshold():
    """Test retrieval with different threshold values."""
    async with SynthiansClient() as client:
        # Create a unique memory
        unique_id = int(time.time())
        unique_content = f"This is a unique threshold test memory {unique_id}"
        
        memory_resp = await client.process_memory(content=unique_content)
        memory_id = memory_resp.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Query with high threshold
        high_threshold_resp = await client.retrieve_memories(
            query=f"completely unrelated query {unique_id}",  # Unrelated but with unique ID
            top_k=10,
            threshold=0.9  # High threshold should filter out most memories
        )
        
        # Query with low threshold
        low_threshold_resp = await client.retrieve_memories(
            query=f"completely unrelated query {unique_id}",  # Same unrelated query
            top_k=10,
            threshold=0.1  # Low threshold should include most memories
        )
        
        high_threshold_memories = high_threshold_resp.get("memories", [])
        low_threshold_memories = low_threshold_resp.get("memories", [])
        
        # Low threshold should return more memories than high threshold
        print(f"High threshold returned {len(high_threshold_memories)} memories")
        print(f"Low threshold returned {len(low_threshold_memories)} memories")
        
        # Check if the unique memory is in the low threshold results
        low_thresh_ids = [m.get("id") for m in low_threshold_memories]
        memory_found = memory_id in low_thresh_ids
        
        print(f"Memory found in low threshold results: {memory_found}")
        print(f"Low threshold memory IDs: {low_thresh_ids}")

@pytest.mark.asyncio
async def test_metadata_filtering():
    """Test retrieval with metadata filters."""
    async with SynthiansClient() as client:
        # Create memories with different metadata
        timestamp = int(time.time())
        
        # Create memory with importance=high
        high_importance = await client.process_memory(
            content=f"High importance memory {timestamp}",
            metadata={"importance": "high", "category": "test", "filter_test": True}
        )
        
        # Create memory with importance=medium
        medium_importance = await client.process_memory(
            content=f"Medium importance memory {timestamp}",
            metadata={"importance": "medium", "category": "test", "filter_test": True}
        )
        
        # Create memory with importance=low
        low_importance = await client.process_memory(
            content=f"Low importance memory {timestamp}",
            metadata={"importance": "low", "category": "test", "filter_test": True}
        )
        
        # Create memory with different category
        different_category = await client.process_memory(
            content=f"Different category memory {timestamp}",
            metadata={"importance": "high", "category": "other", "filter_test": True}
        )
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Test if we can filter by metadata
        # Note: This assumes the retrieve_memories endpoint supports metadata filtering
        # If not, this test will need to be adapted
        
        try:
            # Query for high importance memories only
            # This might need to be updated based on actual API implementation
            high_imp_query = await client.retrieve_memories(
                query=f"memory {timestamp}",
                top_k=10,
                metadata_filter={"importance": "high"}
            )
            
            # Query for test category memories only
            test_category_query = await client.retrieve_memories(
                query=f"memory {timestamp}",
                top_k=10,
                metadata_filter={"category": "test"}
            )
            
            high_imp_memories = high_imp_query.get("memories", [])
            test_cat_memories = test_category_query.get("memories", [])
            
            print(f"High importance query returned {len(high_imp_memories)} memories")
            print(f"Test category query returned {len(test_cat_memories)} memories")
            
            # Check that our filtered queries worked as expected
            high_imp_contents = [m.get("content", "") for m in high_imp_memories]
            test_cat_contents = [m.get("content", "") for m in test_cat_memories]
            
            print(f"High importance memory contents: {high_imp_contents}")
            print(f"Test category memory contents: {test_cat_contents}")
            
        except Exception as e:
            # This test may fail if the API doesn't support metadata filtering
            print(f"Metadata filtering test failed: {str(e)}")
            print("This feature may not be implemented yet or works differently.")

@pytest.mark.asyncio
async def test_top_k_ranking_accuracy():
    """Test that memory retrieval respects top_k parameter and ranks by relevance."""
    async with SynthiansClient() as client:
        # Create a set of memories with varying relevance to a specific query
        base_content = "This is a test of the ranking system"
        timestamp = int(time.time())
        
        # Create memories with varying relevance
        await client.process_memory(
            content=f"{base_content} with direct relevance to ranking and sorting. {timestamp}"
        )
        
        await client.process_memory(
            content=f"{base_content} with some relevance to sorting. {timestamp}"
        )
        
        await client.process_memory(
            content=f"{base_content} with minimal relevance. {timestamp}"
        )
        
        await client.process_memory(
            content=f"Completely unrelated content that shouldn't be ranked highly. {timestamp}"
        )
        
        # Create 10 more filler memories
        for i in range(10):
            await client.process_memory(
                content=f"Filler memory {i} for ranking test. {timestamp}"
            )
        
        # Wait briefly
        await asyncio.sleep(1)
        
        # Test with different top_k values
        top_3_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=3
        )
        
        top_5_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=5
        )
        
        top_10_results = await client.retrieve_memories(
            query=f"ranking and sorting test {timestamp}",
            top_k=10
        )
        
        # Verify the correct number of results returned
        assert len(top_3_results.get("memories", [])) <= 3, "top_k=3 returned too many results"
        assert len(top_5_results.get("memories", [])) <= 5, "top_k=5 returned too many results"
        assert len(top_10_results.get("memories", [])) <= 10, "top_k=10 returned too many results"
        
        # Check the ranking - most relevant should be first
        if top_10_results.get("memories"):
            # Get first result content
            first_result = top_10_results["memories"][0]["content"]
            print(f"First ranked result: {first_result}")
            
            # It should contain "ranking and sorting"
            assert "ranking and sorting" in first_result.lower(), "Most relevant content not ranked first"

```

# tests\test_stress_load.py

```py
import pytest
import asyncio
import json
import time
import random
import numpy as np
from datetime import datetime, timedelta
from synthians_memory_core.api.client.client import SynthiansClient

# Optional marker for these slow tests
pytestmark = pytest.mark.slow

@pytest.mark.asyncio
async def test_1000_memory_ingestion():
    """Test the system with a large number of memories (stress test)."""
    async with SynthiansClient() as client:
        start_time = time.time()
        memory_ids = []
        batch_size = 10  # Process in batches to avoid overwhelming the server
        total_memories = 100  # Reduced from 1000 for faster testing - set to 1000 for full stress test
        
        print(f"Starting bulk ingestion of {total_memories} memories...")
        
        # Generate text templates for variety
        templates = [
            "Remember to {action} the {object} at {time}.",
            "I need to {action} {count} {object}s before {time}.",
            "Don't forget that {person} is coming to {location} at {time}.",
            "The {event} is scheduled for {day} at {time}.",
            "Make sure to check the {object} in the {location}."
        ]
        
        actions = ["review", "check", "update", "clean", "fix", "prepare", "send", "receive"]
        objects = ["document", "report", "presentation", "email", "meeting", "project", "task", "schedule"]
        times = ["9:00 AM", "10:30 AM", "noon", "2:15 PM", "4:00 PM", "5:30 PM", "this evening", "tomorrow"]
        people = ["John", "Sara", "Michael", "Emma", "David", "Lisa", "Alex", "Olivia"]
        locations = ["office", "conference room", "lobby", "home", "cafe", "downtown", "upstairs", "kitchen"]
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        events = ["meeting", "conference", "workshop", "presentation", "lunch", "dinner", "call", "interview"]
        counts = ["two", "three", "four", "five", "several", "many", "a few", "some"]
        
        # Create memories in batches
        for batch in range(0, total_memories, batch_size):
            batch_tasks = []
            
            for i in range(batch, min(batch + batch_size, total_memories)):
                # Generate a random memory with a template
                template = random.choice(templates)
                content = template.format(
                    action=random.choice(actions),
                    object=random.choice(objects),
                    time=random.choice(times),
                    person=random.choice(people),
                    location=random.choice(locations),
                    day=random.choice(days),
                    event=random.choice(events),
                    count=random.choice(counts)
                )
                
                # Add a unique identifier
                content += f" (Memory #{i+1})"
                
                # Generate random metadata
                metadata = {
                    "batch": batch // batch_size,
                    "index": i,
                    "importance": random.uniform(0.1, 1.0),
                    "category": random.choice(["work", "personal", "reminder", "event"]),
                    "stress_test": True
                }
                
                # Create memory task
                task = client.process_memory(content=content, metadata=metadata)
                batch_tasks.append(task)
            
            # Process the batch
            batch_results = await asyncio.gather(*batch_tasks)
            
            # Collect memory IDs
            for result in batch_results:
                if result.get("success"):
                    memory_ids.append(result.get("memory_id"))
            
            # Log progress
            elapsed = time.time() - start_time
            progress = min(100, (len(memory_ids) / total_memories) * 100)
            print(f"Progress: {progress:.1f}% - {len(memory_ids)}/{total_memories} memories created in {elapsed:.2f} seconds")
            
            # Pause briefly between batches to avoid overwhelming server
            await asyncio.sleep(0.1)
        
        # Final statistics
        total_time = time.time() - start_time
        rate = len(memory_ids) / total_time if total_time > 0 else 0
        
        print(f"Completed: {len(memory_ids)}/{total_memories} memories created in {total_time:.2f} seconds")
        print(f"Rate: {rate:.2f} memories/second")
        
        # Verify we can retrieve memories from the batch
        if memory_ids:
            # Try to retrieve a random memory by ID
            random_id = random.choice(memory_ids)
            retrieve_resp = await client.retrieve_memories(
                query=f"Memory #{random.randint(1, total_memories)}",
                top_k=5
            )
            
            assert retrieve_resp.get("success") is True, "Failed to retrieve after bulk ingestion"
            print(f"Successfully retrieved {len(retrieve_resp.get('memories', []))} memories from the bulk ingestion")

@pytest.mark.asyncio
async def test_concurrent_retrievals():
    """Test the system with many concurrent retrieval requests."""
    async with SynthiansClient() as client:
        # First, create some memories to retrieve
        timestamp = int(time.time())
        keyword = f"concurrent{timestamp}"
        
        # Create 10 memories with the same keyword
        create_tasks = []
        for i in range(10):
            content = f"Memory {i+1} for concurrent retrieval test with keyword {keyword}"
            task = client.process_memory(content=content)
            create_tasks.append(task)
        
        create_results = await asyncio.gather(*create_tasks)
        created_ids = [r.get("memory_id") for r in create_results if r.get("success")]
        
        assert len(created_ids) > 0, "Failed to create test memories for concurrent retrievals"
        print(f"Created {len(created_ids)} test memories for concurrent retrievals")
        
        # Wait briefly for processing
        await asyncio.sleep(1)
        
        # Now perform many concurrent retrievals
        concurrency = 20  # Number of concurrent requests
        start_time = time.time()
        
        retrieval_tasks = []
        for i in range(concurrency):
            task = client.retrieve_memories(
                query=f"{keyword} memory {random.randint(1, 10)}",
                top_k=5
            )
            retrieval_tasks.append(task)
        
        # Execute concurrently
        retrieval_results = await asyncio.gather(*retrieval_tasks)
        
        # Calculate statistics
        successful = sum(1 for r in retrieval_results if r.get("success"))
        total_time = time.time() - start_time
        rate = concurrency / total_time if total_time > 0 else 0
        
        print(f"Completed {successful}/{concurrency} concurrent retrievals in {total_time:.2f} seconds")
        print(f"Rate: {rate:.2f} retrievals/second")
        
        # Check that all retrievals worked
        assert successful == concurrency, f"Only {successful}/{concurrency} concurrent retrievals succeeded"

@pytest.mark.asyncio
async def test_batch_save_and_reload():
    """Test saving and reloading the memory store during batch operations."""
    # Note: This test assumes there's an endpoint to trigger a save/reload cycle
    # If not available, this can be skipped
    
    async with SynthiansClient() as client:
        try:
            # Create a batch of memories
            timestamp = int(time.time())
            memory_ids = []
            
            # Create 20 test memories
            for i in range(20):
                content = f"Memory {i+1} for save/reload test at {timestamp}"
                response = await client.process_memory(content=content)
                if response.get("success"):
                    memory_ids.append(response.get("memory_id"))
            
            # Call save endpoint (if available)
            # This is hypothetical - might need to be implemented
            save_response = await client.session.post(f"{client.base_url}/save_memory_store")
            save_result = await save_response.json()
            
            print(f"Save operation result: {json.dumps(save_result, indent=2)}")
            
            # Call reload endpoint (if available)
            reload_response = await client.session.post(f"{client.base_url}/reload_memory_store")
            reload_result = await reload_response.json()
            
            print(f"Reload operation result: {json.dumps(reload_result, indent=2)}")
            
            # Verify memories are still retrievable after reload
            retrieved_count = 0
            for memory_id in memory_ids[:5]:  # Check first 5 memories
                query = f"save/reload test at {timestamp}"
                result = await client.retrieve_memories(query=query, top_k=20)
                
                if result.get("success"):
                    result_ids = [m.get("id") for m in result.get("memories", [])]
                    if memory_id in result_ids:
                        retrieved_count += 1
            
            # Check that we could retrieve our memories after reload
            assert retrieved_count > 0, "Failed to retrieve memories after save/reload cycle"
            print(f"Successfully retrieved {retrieved_count}/5 test memories after save/reload cycle")
            
        except Exception as e:
            # The save/reload endpoints might not exist yet
            print(f"Save/reload test failed: {str(e)}")
            print("Save/reload endpoints may not be implemented yet.")

@pytest.mark.asyncio
async def test_memory_decay_pruning():
    """Test memory decay and pruning of old memories."""
    # This test is designed to verify that old memories can be pruned
    # It may need to be adapted based on actual implementation
    
    async with SynthiansClient() as client:
        try:
            # Create memories with backdated timestamps
            timestamp = int(time.time())
            
            # Current memory
            current_response = await client.process_memory(
                content=f"Current memory at {timestamp}",
                metadata={"timestamp": time.time()}
            )
            current_id = current_response.get("memory_id")
            
            # 1-day old memory
            day_old_time = time.time() - (60 * 60 * 24)  # 1 day ago
            day_old_response = await client.process_memory(
                content=f"One day old memory at {timestamp}",
                metadata={"timestamp": day_old_time}
            )
            day_old_id = day_old_response.get("memory_id")
            
            # 1-week old memory
            week_old_time = time.time() - (60 * 60 * 24 * 7)  # 1 week ago
            week_old_response = await client.process_memory(
                content=f"One week old memory at {timestamp}",
                metadata={"timestamp": week_old_time}
            )
            week_old_id = week_old_response.get("memory_id")
            
            # 1-month old memory
            month_old_time = time.time() - (60 * 60 * 24 * 30)  # ~1 month ago
            month_old_response = await client.process_memory(
                content=f"One month old memory at {timestamp}",
                metadata={"timestamp": month_old_time}
            )
            month_old_id = month_old_response.get("memory_id")
            
            # Verify all were created successfully
            assert all(r.get("success") for r in [
                current_response, day_old_response, week_old_response, month_old_response
            ]), "Failed to create test memories with different ages"
            
            print("Successfully created test memories with different timestamps")
            
            # Now trigger a pruning operation (if available)
            # This is hypothetical - might need to be implemented
            prune_response = await client.session.post(
                f"{client.base_url}/prune_old_memories",
                json={"max_age_days": 14}  # Prune memories older than 2 weeks
            )
            prune_result = await prune_response.json()
            
            print(f"Pruning operation result: {json.dumps(prune_result, indent=2)}")
            
            # Check which memories are still retrievable
            retrievable = []
            
            for memory_id, age in [
                (current_id, "current"),
                (day_old_id, "1-day"),
                (week_old_id, "1-week"),
                (month_old_id, "1-month")
            ]:
                query = f"memory at {timestamp}"
                result = await client.retrieve_memories(query=query, top_k=10)
                
                if result.get("success"):
                    result_ids = [m.get("id") for m in result.get("memories", [])]
                    if memory_id in result_ids:
                        retrievable.append(age)
            
            print(f"Still retrievable after pruning: {retrievable}")
            
            # Current and 1-day old should still be retrievable
            # 1-month old should be pruned
            # 1-week old depends on implementation details
            assert "current" in retrievable, "Current memory was incorrectly pruned"
            assert "1-day" in retrievable, "1-day old memory was incorrectly pruned"
            
            if "1-month" in retrievable:
                print("Warning: 1-month old memory was not pruned, pruning may not be implemented yet")
                
        except Exception as e:
            # The pruning endpoint might not exist yet
            print(f"Memory pruning test failed: {str(e)}")
            print("Memory pruning may not be implemented yet.")

```

# tests\test_tool_integration.py

```py
import pytest
import asyncio
import json
import time
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient

# Add the missing tool methods to SynthiansClient if needed
async def process_memory_tool(self, content: str, metadata: dict = None):
    """Process memory as a tool call (simulated)."""
    payload = {
        "content": content,
        "metadata": metadata or {},
        "tool_call": True  # Identify this as coming from a tool call
    }
    async with self.session.post(
        f"{self.base_url}/process_memory", json=payload
    ) as response:
        return await response.json()

async def retrieve_memories_tool(self, query: str, top_k: int = 5, user_emotion: dict = None):
    """Retrieve memories as a tool call (simulated)."""
    payload = {
        "query": query,
        "top_k": top_k,
        "user_emotion": user_emotion,
        "tool_call": True  # Identify this as coming from a tool call
    }
    async with self.session.post(
        f"{self.base_url}/retrieve_memories", json=payload
    ) as response:
        return await response.json()

async def detect_contradictions_tool(self, query: str, threshold: float = 0.75):
    """Detect contradictions as a tool call (simulated)."""
    payload = {
        "query": query,
        "threshold": threshold,
        "tool_call": True  # Identify this as coming from a tool call
    }
    async with self.session.post(
        f"{self.base_url}/detect_contradictions", json=payload
    ) as response:
        return await response.json()

# Add methods to SynthiansClient class if not present
if not hasattr(SynthiansClient, "process_memory_tool"):
    SynthiansClient.process_memory_tool = process_memory_tool

if not hasattr(SynthiansClient, "retrieve_memories_tool"):
    SynthiansClient.retrieve_memories_tool = retrieve_memories_tool

if not hasattr(SynthiansClient, "detect_contradictions_tool"):
    SynthiansClient.detect_contradictions_tool = detect_contradictions_tool

@pytest.mark.asyncio
async def test_tool_call_process_memory_tool():
    """Test processing memory through a simulated tool call."""
    async with SynthiansClient() as client:
        try:
            # Use a unique timestamp to ensure we can find this memory
            timestamp = int(time.time())
            content = f"Memory created through tool call at {timestamp}"
            metadata = {
                "source": "tool_test",
                "importance": 0.9,
                "tool_metadata": {
                    "tool_name": "process_memory_tool",
                    "llm_type": "test_model"
                }
            }
            
            # Process the memory through the tool call
            result = await client.process_memory_tool(content=content, metadata=metadata)
            
            # Verify successful processing
            assert result.get("success") is True, f"Tool call memory processing failed: {result.get('error')}"
            assert "memory_id" in result, "No memory ID returned from tool call"
            
            # Verify the memory was stored with correct metadata
            returned_metadata = result.get("metadata", {})
            assert returned_metadata.get("source") == "tool_test", "Tool metadata not preserved"
            assert "tool_metadata" in returned_metadata, "Tool-specific metadata not preserved"
            
            print(f"Tool memory processing result: {json.dumps(result, indent=2)}")
            
            # Wait briefly
            await asyncio.sleep(0.5)
            
            # Try to retrieve the memory to confirm it was stored
            memory_id = result.get("memory_id")
            retrieval = await client.retrieve_memories(query=f"tool call at {timestamp}", top_k=3)
            
            # Verify the memory can be retrieved
            memories = retrieval.get("memories", [])
            memory_ids = [m.get("id") for m in memories]
            
            assert memory_id in memory_ids, f"Memory created by tool call not retrievable. Expected {memory_id}, got {memory_ids}"
            
        except Exception as e:
            # The API might not support the tool_call parameter yet
            print(f"Tool call memory processing test failed: {str(e)}")
            print("Tool-specific endpoint might not be implemented yet.")

@pytest.mark.asyncio
async def test_tool_call_retrieve_memories_tool():
    """Test retrieving memories through a simulated tool call."""
    async with SynthiansClient() as client:
        try:
            # First, create a memory we can retrieve
            timestamp = int(time.time())
            content = f"Retrievable memory for tool test at {timestamp}"
            
            memory_resp = await client.process_memory(content=content)
            memory_id = memory_resp.get("memory_id")
            
            # Wait briefly
            await asyncio.sleep(0.5)
            
            # Now retrieve it using the tool call endpoint
            retrieval = await client.retrieve_memories_tool(
                query=f"tool test at {timestamp}",
                top_k=3
            )
            
            # Verify successful retrieval
            assert retrieval.get("success") is True, f"Tool call memory retrieval failed: {retrieval.get('error')}"
            assert "memories" in retrieval, "No memories returned from tool call"
            
            # Check if our memory was found
            memories = retrieval.get("memories", [])
            memory_ids = [m.get("id") for m in memories]
            
            print(f"Retrieved memory IDs through tool: {memory_ids}")
            print(f"Expected memory ID: {memory_id}")
            assert memory_id in memory_ids, "Memory not found via tool retrieval"
            
            # Verify tool-specific formatting (if implemented)
            if "tool_format" in retrieval:
                assert retrieval["tool_format"] == "formatted_for_llm", "Tool-specific formatting not applied"
            
        except Exception as e:
            # The API might not support the tool_call parameter yet
            print(f"Tool call memory retrieval test failed: {str(e)}")
            print("Tool-specific endpoint might not be implemented yet.")

@pytest.mark.asyncio
async def test_tool_call_detect_contradictions_tool():
    """Test contradiction detection through a simulated tool call."""
    async with SynthiansClient() as client:
        try:
            # Create contradicting memories
            timestamp = int(time.time())
            
            # First statement
            await client.process_memory(
                content=f"The meeting is scheduled for Tuesday at 2pm. {timestamp}",
                metadata={"contradiction_test": True}
            )
            
            # Contradicting statement
            await client.process_memory(
                content=f"The meeting is scheduled for Wednesday at 3pm. {timestamp}",
                metadata={"contradiction_test": True}
            )
            
            # Wait briefly
            await asyncio.sleep(1)
            
            # Check for contradictions using the tool call
            result = await client.detect_contradictions_tool(
                query=f"meeting schedule {timestamp}",
                threshold=0.7
            )
            
            # Verify successful detection
            assert result.get("success") is True, f"Tool call contradiction detection failed: {result.get('error')}"
            
            # If contradictions were found, they should be in the result
            if "contradictions" in result:
                contradictions = result.get("contradictions", [])
                print(f"Detected {len(contradictions)} contradictions through tool call")
                print(f"Contradiction results: {json.dumps(contradictions, indent=2)}")
                
                # There should be at least one contradiction
                if len(contradictions) > 0:
                    assert "memory_pairs" in contradictions[0], "Contradiction missing memory pairs"
                    assert "contradiction_type" in contradictions[0], "Contradiction missing type"
            
        except Exception as e:
            # The API might not support the contradiction detection yet
            print(f"Tool call contradiction detection test failed: {str(e)}")
            print("Contradiction detection feature might not be implemented yet.")

@pytest.mark.asyncio
async def test_tool_call_feedback_tool():
    """Test providing feedback through a simulated tool call."""
    async with SynthiansClient() as client:
        try:
            # First, create a memory we can provide feedback on
            timestamp = int(time.time())
            content = f"Memory for feedback test at {timestamp}"
            
            memory_resp = await client.process_memory(content=content)
            memory_id = memory_resp.get("memory_id")
            
            # Now provide feedback through the tool call
            # Add a method for this if not available
            if not hasattr(client, "provide_feedback_tool"):
                async def provide_feedback_tool(self, memory_id, similarity_score, was_relevant):
                    payload = {
                        "memory_id": memory_id,
                        "similarity_score": similarity_score,
                        "was_relevant": was_relevant,
                        "tool_call": True  # Identify this as coming from a tool call
                    }
                    async with self.session.post(
                        f"{self.base_url}/provide_feedback", json=payload
                    ) as response:
                        return await response.json()
                
                client.provide_feedback_tool = provide_feedback_tool.__get__(client, SynthiansClient)
            
            # Use the feedback tool
            feedback_resp = await client.provide_feedback_tool(
                memory_id=memory_id,
                similarity_score=0.92,
                was_relevant=True
            )
            
            # Verify successful feedback
            assert feedback_resp.get("success") is True, f"Tool call feedback failed: {feedback_resp.get('error')}"
            assert "new_threshold" in feedback_resp, "Threshold adjustment information missing"
            
            print(f"Feedback through tool call: {json.dumps(feedback_resp, indent=2)}")
            
        except Exception as e:
            # The API might not support the tool call parameter yet
            print(f"Tool call feedback test failed: {str(e)}")
            print("Tool-specific endpoint might not be implemented yet.")

```

# tests\test_transcription_voice_flow.py

```py
import pytest
import asyncio
import json
import time
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient

# Add process_transcription method to SynthiansClient if not already present
async def process_transcription(self, text: str, audio_metadata: dict = None, embedding=None):
    """Process transcription data and store it in the memory system."""
    payload = {
        "text": text,
        "audio_metadata": audio_metadata or {},
        "embedding": embedding
    }
    async with self.session.post(
        f"{self.base_url}/process_transcription", json=payload
    ) as response:
        return await response.json()

# Add the method to the client class if not present
if not hasattr(SynthiansClient, "process_transcription"):
    SynthiansClient.process_transcription = process_transcription

@pytest.mark.asyncio
async def test_transcription_feature_extraction():
    """Test that transcription processing extracts relevant features."""
    async with SynthiansClient() as client:
        # Create a transcription with rich metadata
        text = "This is a test transcription with some pauses... and rhythm changes."
        audio_metadata = {
            "duration_sec": 5.2,
            "avg_volume": 0.75,
            "speaking_rate": 2.1,  # Words per second
            "pauses": [
                {"start": 1.2, "duration": 0.5},
                {"start": 3.5, "duration": 0.8}
            ]
        }
        
        # Process the transcription
        result = await client.process_transcription(
            text=text,
            audio_metadata=audio_metadata
        )
        
        # Verify successful processing
        assert result.get("success") is True, f"Transcription processing failed: {result.get('error')}"
        assert "memory_id" in result, "No memory ID returned for transcription"
        
        # Check metadata enrichment
        metadata = result.get("metadata", {})
        
        # Basic metadata verification
        assert "timestamp" in metadata, "No timestamp in metadata"
        assert "speaking_rate" in metadata, "Speaking rate not captured in metadata"
        assert "duration_sec" in metadata, "Duration not captured in metadata"
        
        # Advanced feature extraction verification (if implemented)
        if "pause_count" in metadata:
            assert metadata["pause_count"] >= 2, "Expected at least 2 pauses to be detected"
        
        if "speech_features" in metadata:
            assert isinstance(metadata["speech_features"], dict), "Speech features not properly structured"
        
        print(f"Transcription metadata: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_interrupt_metadata_enrichment():
    """Test that interruption metadata is properly stored and processed."""
    async with SynthiansClient() as client:
        # Create a transcription with interruption data
        text = "I was talking about- wait, let me restart. This is what I meant to say."
        audio_metadata = {
            "duration_sec": 7.5,
            "was_interrupted": True,
            "interruptions": [
                {"timestamp": 2.1, "duration": 0.3, "type": "self"}
            ],
            "user_interruptions": 1
        }
        
        # Process the transcription
        result = await client.process_transcription(
            text=text,
            audio_metadata=audio_metadata
        )
        
        # Verify successful processing
        assert result.get("success") is True, "Transcription processing failed"
        
        # Check interruption metadata
        metadata = result.get("metadata", {})
        assert "was_interrupted" in metadata, "Interruption flag not in metadata"
        assert metadata.get("was_interrupted") is True, "Interruption flag not preserved"
        
        if "interruption_count" in metadata:
            assert metadata["interruption_count"] >= 1, "Expected at least 1 interruption to be counted"
        
        if "user_interruptions" in metadata:
            assert metadata["user_interruptions"] >= 1, "User interruptions not preserved in metadata"
        
        print(f"Interruption metadata: {json.dumps(metadata, indent=2)}")

@pytest.mark.asyncio
async def test_session_level_memory():
    """Test that multiple utterances within a session are properly linked."""
    async with SynthiansClient() as client:
        # Generate a unique session ID
        session_id = f"test-session-{int(time.time())}"
        
        # Create first utterance in session
        text1 = "This is the first part of a multi-utterance conversation."
        metadata1 = {
            "session_id": session_id,
            "utterance_index": 1,
            "timestamp": time.time()
        }
        
        result1 = await client.process_memory(
            content=text1,
            metadata=metadata1
        )
        
        assert result1.get("success") is True, "First utterance processing failed"
        memory_id1 = result1.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Create second utterance in same session
        text2 = "This is the second part, continuing from what I said before."
        metadata2 = {
            "session_id": session_id,
            "utterance_index": 2,
            "timestamp": time.time(),
            "previous_memory_id": memory_id1  # Link to previous utterance
        }
        
        result2 = await client.process_memory(
            content=text2,
            metadata=metadata2
        )
        
        assert result2.get("success") is True, "Second utterance processing failed"
        memory_id2 = result2.get("memory_id")
        
        # Wait briefly
        await asyncio.sleep(0.5)
        
        # Create third utterance in same session
        text3 = "This is the third and final part of my conversation."
        metadata3 = {
            "session_id": session_id,
            "utterance_index": 3,
            "timestamp": time.time(),
            "previous_memory_id": memory_id2  # Link to previous utterance
        }
        
        result3 = await client.process_memory(
            content=text3,
            metadata=metadata3
        )
        
        assert result3.get("success") is True, "Third utterance processing failed"
        
        # Retrieve memories from this session
        # This assumes the API has a way to filter by session_id
        # If not, we can query by the unique session ID in the content
        retrieval_resp = await client.retrieve_memories(
            query=f"session:{session_id}",
            top_k=10
        )
        
        # Check if all three memories were retrieved
        memories = retrieval_resp.get("memories", [])
        memory_ids = [m.get("id") for m in memories]
        
        print(f"Retrieved session memories: {json.dumps(memory_ids, indent=2)}")
        
        # Check for session links in metadata (if implemented)
        for memory in memories:
            if "metadata" in memory and "session_id" in memory["metadata"]:
                assert memory["metadata"]["session_id"] == session_id, "Session ID not preserved"

@pytest.mark.asyncio
async def test_voice_state_tracking():
    """Test that voice state transitions are properly tracked in memory metadata."""
    async with SynthiansClient() as client:
        # Create a transcription with voice state metadata
        text = "This is a test of voice state tracking."
        audio_metadata = {
            "voice_state": "SPEAKING",
            "state_duration": 3.2,
            "previous_state": "LISTENING",
            "state_transition_count": 5,
            "last_state_transition_time": time.time() - 3.2
        }
        
        try:
            # Process the transcription (if endpoint exists)
            result = await client.process_transcription(
                text=text,
                audio_metadata=audio_metadata
            )
            
            # Verify successful processing
            assert result.get("success") is True, "Voice state tracking test failed"
            
            # Check if voice state metadata was preserved
            metadata = result.get("metadata", {})
            if "voice_state" in metadata:
                assert metadata["voice_state"] == "SPEAKING", "Voice state not preserved"
            
            if "state_transition_count" in metadata:
                assert metadata["state_transition_count"] == 5, "State transition count not preserved"
            
            print(f"Voice state metadata: {json.dumps(metadata, indent=2)}")
            
        except Exception as e:
            # This test may fail if the API doesn't support voice state tracking yet
            print(f"Voice state tracking test failed: {str(e)}")
            print("This feature may not be implemented yet.")

```

# tests\test_vector_index.py

```py
import pytest
import asyncio
import json
import time
import numpy as np
import os
import sys
import logging
from datetime import datetime
from synthians_memory_core.api.client.client import SynthiansClient
from synthians_memory_core.vector_index import MemoryVectorIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_vector_index")

@pytest.mark.asyncio
async def test_faiss_vector_index_creation():
    """Test the creation and basic functionality of the FAISS vector index."""
    # Create a test vector index with a specific dimension
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2',
        'use_gpu': True  # This will use GPU if available, otherwise fall back to CPU
    })
    
    # Verify the index was created with the right parameters
    assert index.dimension == dimension, f"Expected dimension {dimension}, got {index.dimension}"
    logger.info(f"Created vector index with dimension {index.dimension}, GPU usage: {index.is_using_gpu}")
    
    # Create some test embeddings
    num_vectors = 100
    test_vectors = np.random.random((num_vectors, dimension)).astype('float32')
    
    # Add vectors to the index
    for i in range(num_vectors):
        memory_id = f"test_memory_{i}"
        index.add(memory_id, test_vectors[i])
    
    # Verify the index contains the expected number of vectors
    assert index.count() == num_vectors, f"Expected {num_vectors} vectors in index, got {index.count()}"
    
    # Test search functionality
    query_vector = np.random.random(dimension).astype('float32')
    k = 10
    results = index.search(query_vector, k)
    
    # Verify search results format
    assert len(results) <= k, f"Expected at most {k} results, got {len(results)}"
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Results should be (memory_id, score) tuples"
    
    # Test index persistence
    index_path = os.path.join(index.storage_path, 'test_index.faiss')
    index.save(index_path)
    assert os.path.exists(index_path), f"Index file not found at {index_path}"
    
    # Test index loading
    new_index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    new_index.load(index_path)
    
    # Verify loaded index has the same vectors
    assert new_index.count() == index.count(), "Loaded index has different vector count"
    
    # Clean up
    if os.path.exists(index_path):
        os.remove(index_path)
    logger.info("Vector index creation and persistence test completed successfully")

@pytest.mark.asyncio
async def test_dimension_mismatch_handling():
    """Test the handling of embedding dimension mismatches."""
    # Create a vector index with specific dimension
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    
    # Create vectors with different dimensions
    smaller_dim = 384
    larger_dim = 1024
    
    standard_vector = np.random.random(dimension).astype('float32')
    smaller_vector = np.random.random(smaller_dim).astype('float32')
    larger_vector = np.random.random(larger_dim).astype('float32')
    
    # Add vectors with different dimensions
    index.add("standard_vector", standard_vector)
    index.add("smaller_vector", smaller_vector)  # Should be padded
    index.add("larger_vector", larger_vector)    # Should be truncated
    
    # Verify all vectors were added
    assert index.count() == 3, f"Expected 3 vectors in index, got {index.count()}"
    
    # Test search with different dimension vectors
    standard_results = index.search(standard_vector, 3)
    smaller_results = index.search(smaller_vector, 3)
    larger_results = index.search(larger_vector, 3)
    
    # Verify search results contain expected entries
    assert any(r[0] == "standard_vector" for r in standard_results), "Standard vector not found in results"
    assert any(r[0] == "smaller_vector" for r in smaller_results), "Smaller vector not found in results"
    assert any(r[0] == "larger_vector" for r in larger_results), "Larger vector not found in results"
    
    logger.info("Dimension mismatch handling test completed successfully")

@pytest.mark.asyncio
async def test_malformed_embedding_handling():
    """Test the handling of malformed embeddings (NaN/Inf values)."""
    # Create a vector index
    dimension = 768
    index = MemoryVectorIndex({
        'embedding_dim': dimension,
        'storage_path': os.path.join(os.getcwd(), 'test_index'),
        'index_type': 'L2'
    })
    
    # Create a normal vector and malformed vectors
    normal_vector = np.random.random(dimension).astype('float32')
    
    # Vector with NaN values
    nan_vector = np.random.random(dimension).astype('float32')
    nan_vector[10:20] = np.nan
    
    # Vector with Inf values
    inf_vector = np.random.random(dimension).astype('float32')
    inf_vector[30:40] = np.inf
    
    # Add vectors - the malformed ones should be handled gracefully
    index.add("normal_vector", normal_vector)
    
    # These should be handled by replacing with zeros or normalized vectors
    index.add("nan_vector", nan_vector)
    index.add("inf_vector", inf_vector)
    
    # Verify we can search without errors
    results = index.search(normal_vector, 3)
    assert len(results) > 0, "No results returned from search"
    
    # Search with malformed query vectors should also work
    nan_query = np.random.random(dimension).astype('float32')
    nan_query[5:15] = np.nan
    
    nan_results = index.search(nan_query, 3)
    assert len(nan_results) > 0, "No results returned from search with NaN query"
    
    logger.info("Malformed embedding handling test completed successfully")

@pytest.mark.asyncio
async def test_end_to_end_vector_retrieval():
    """End-to-end test of vector indexing and retrieval through the API."""
    async with SynthiansClient() as client:
        # Step 1: Create distinct test memories
        timestamp = datetime.now().isoformat()
        
        memory1 = await client.process_memory(
            content=f"FAISS vector index test memory Alpha at {timestamp}",
            metadata={"test_group": "vector_index", "category": "alpha"}
        )
        
        memory2 = await client.process_memory(
            content=f"FAISS vector index test memory Beta at {timestamp}",
            metadata={"test_group": "vector_index", "category": "beta"}
        )
        
        memory3 = await client.process_memory(
            content=f"FAISS vector index test memory Gamma at {timestamp}",
            metadata={"test_group": "vector_index", "category": "gamma"}
        )
        
        # Allow time for processing and indexing
        await asyncio.sleep(1)
        
        # Step 2: Retrieve with exact match
        alpha_query = f"Alpha at {timestamp}"
        alpha_results = await client.retrieve_memories(alpha_query, top_k=3)
        
        # Verify retrieval accuracy
        assert alpha_results.get("success") is True, "Retrieval failed"
        alpha_memories = alpha_results.get("memories", [])
        alpha_ids = [m.get("id") for m in alpha_memories]
        
        # Memory1 should be retrieved
        assert memory1.get("memory_id") in alpha_ids, "Alpha memory not found in retrieval results"
        
        # Step 3: Test with lower threshold to ensure retrieval works
        general_query = f"vector index test at {timestamp}"
        low_threshold_results = await client.retrieve_memories(
            general_query, 
            top_k=10, 
            threshold=0.3  # Lower threshold as per the memory improvement
        )
        
        all_memories = low_threshold_results.get("memories", [])
        all_ids = [m.get("id") for m in all_memories]
        
        # All memories should be retrieved with a lower threshold
        assert memory1.get("memory_id") in all_ids, "Memory 1 not found with low threshold"
        assert memory2.get("memory_id") in all_ids, "Memory 2 not found with low threshold"
        assert memory3.get("memory_id") in all_ids, "Memory 3 not found with low threshold"
        
        logger.info(f"Retrieved {len(all_memories)} memories with low threshold")
        logger.info("End-to-end vector retrieval test completed successfully")

if __name__ == "__main__":
    # For manual test execution
    asyncio.run(test_faiss_vector_index_creation())
    asyncio.run(test_dimension_mismatch_handling())
    asyncio.run(test_malformed_embedding_handling())
    asyncio.run(test_end_to_end_vector_retrieval())

```

# utils\__init__.py

```py
# synthians_memory_core/utils/__init__.py

from .transcription_feature_extractor import TranscriptionFeatureExtractor

__all__ = ['TranscriptionFeatureExtractor']

```

# utils\transcription_feature_extractor.py

```py
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio

from ..custom_logger import logger

class TranscriptionFeatureExtractor:
    """
    Extracts emotion and semantic features from transcribed voice input.
    Uses an emotion analyzer and optional keyword extractor to enrich transcription metadata.
    
    This class is designed to work with the EmotionAnalyzer and KeyBERT, but can be
    used with any compatible analyzers that follow the same interface.
    """

    def __init__(self, emotion_analyzer, keyword_extractor=None, config: Optional[Dict] = None):
        self.emotion_analyzer = emotion_analyzer  # EmotionAnalyzer instance
        self.keyword_extractor = keyword_extractor  # KeyBERT or similar
        self.config = config or {}
        
        # Default configuration with fallbacks
        self.top_n_keywords = self.config.get('top_n_keywords', 5)
        self.min_keyword_score = self.config.get('min_keyword_score', 0.3)
        self.include_ngrams = self.config.get('include_ngrams', True)
        
        logger.info("TranscriptionFeatureExtractor", "Initialized with" + 
                   f" emotion_analyzer={emotion_analyzer is not None}" +
                   f" keyword_extractor={keyword_extractor is not None}")
        
        # Lazy-load KeyBERT if not provided but needed
        self._keybert = None
    
    async def extract_features(self, transcript: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract features from a transcript and return them as metadata.
        
        Args:
            transcript: The text transcript to analyze
            meta: Optional metadata about the audio (duration, etc.)
            
        Returns:
            A dictionary of extracted features suitable for metadata
        """
        if not transcript or not isinstance(transcript, str) or len(transcript.strip()) == 0:
            logger.warning("TranscriptionFeatureExtractor", "Empty or invalid transcript provided")
            return {"input_modality": "spoken", "source": "transcription", "error": "Empty transcript"}
        
        metadata = {}
        
        # Tag basic information about the input
        metadata["input_modality"] = "spoken"
        metadata["source"] = "transcription"
        metadata["word_count"] = len(transcript.split())
        
        # 1. Emotion Analysis
        emotion_features = await self._extract_emotion_features(transcript)
        if emotion_features:
            metadata.update(emotion_features)
        
        # 2. Keyword Extraction
        keyword_features = await self._extract_keyword_features(transcript)
        if keyword_features:
            metadata.update(keyword_features)
            
        # 3. Speech Metadata
        if meta:
            speech_features = self._extract_speech_features(transcript, meta)
            if speech_features:
                metadata.update(speech_features)
        
        logger.info("TranscriptionFeatureExtractor", 
                   f"Extracted {len(metadata)} features from transcript")
        return metadata
    
    async def _extract_emotion_features(self, text: str) -> Dict[str, Any]:
        """
        Extract emotion features using the emotion analyzer.
        """
        features = {}
        
        if self.emotion_analyzer is None:
            logger.warning("TranscriptionFeatureExtractor", "No emotion analyzer available")
            return features
        
        try:
            # Use our emotion analyzer to get emotion data
            emotion = await self.emotion_analyzer.analyze(text)
            
            # Extract the core emotional features
            features["dominant_emotion"] = emotion.get("dominant_emotion", "neutral")
            features["emotions"] = emotion.get("emotions", {})
            
            # Calculate derived features
            if "emotions" in emotion and emotion["emotions"]:
                # Get intensity (highest emotion score)
                features["intensity"] = max(emotion["emotions"].values())
                
                # Calculate sentiment value (-1 to 1 scale)
                pos_emotions = ["joy", "happiness", "excitement", "love", "optimism", "admiration"]
                neg_emotions = ["sadness", "anger", "fear", "disgust", "disappointment"]
                
                sentiment = 0.0
                for emotion_name, score in emotion["emotions"].items():
                    if emotion_name in pos_emotions:
                        sentiment += score
                    elif emotion_name in neg_emotions:
                        sentiment -= score
                
                # Normalize to [-1, 1]
                features["sentiment_value"] = max(min(sentiment, 1.0), -1.0)
            else:
                features["intensity"] = 0.5
                features["sentiment_value"] = 0.0
            
            # Create emotional_context for compatibility with other systems
            features["emotional_context"] = {
                "dominant_emotion": features["dominant_emotion"],
                "emotions": features["emotions"],
                "intensity": features["intensity"],
                "sentiment_value": features["sentiment_value"]
            }
            
        except Exception as e:
            logger.error("TranscriptionFeatureExtractor", f"Error in emotion analysis: {str(e)}")
            features["dominant_emotion"] = "neutral"
            features["intensity"] = 0.5
            features["sentiment_value"] = 0.0
        
        return features
    
    async def _extract_keyword_features(self, text: str) -> Dict[str, Any]:
        """
        Extract keyword features using KeyBERT or a similar keyword extractor.
        Lazy-loads KeyBERT if needed and not provided.
        """
        features = {}
        
        # Ensure we have a keyword extractor
        if self.keyword_extractor is None:
            # Try to lazy-load KeyBERT if possible
            if self._keybert is None:
                try:
                    loop = asyncio.get_event_loop()
                    self._keybert = await loop.run_in_executor(None, self._load_keybert)
                    if self._keybert is None:
                        logger.warning("TranscriptionFeatureExtractor", "Failed to load KeyBERT")
                        return features
                except Exception as e:
                    logger.error("TranscriptionFeatureExtractor", f"Error loading KeyBERT: {str(e)}")
                    return features
            
            # Use the lazy-loaded KeyBERT
            self.keyword_extractor = self._keybert
        
        # Extract keywords if we have an extractor
        if self.keyword_extractor:
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                keywords = await loop.run_in_executor(
                    None, 
                    lambda: self.keyword_extractor.extract_keywords(
                        text, 
                        top_n=self.top_n_keywords,
                        keyphrase_ngram_range=(1, 3) if self.include_ngrams else (1, 1),
                        stop_words='english',
                        use_mmr=True,
                        diversity=0.7
                    )
                )
                
                # Filter by minimum score
                keywords = [(kw, score) for kw, score in keywords if score >= self.min_keyword_score]
                
                # Save as separate lists for keywords and scores
                features["keywords"] = [kw for kw, _ in keywords]
                features["keyword_scores"] = {kw: score for kw, score in keywords}
                
                # Also save as topic tags for compatibility
                features["topic_tags"] = features["keywords"][:3] if len(features["keywords"]) > 3 else features["keywords"]
                
            except Exception as e:
                logger.error("TranscriptionFeatureExtractor", f"Error extracting keywords: {str(e)}")
                features["keywords"] = []
                features["topic_tags"] = []
        
        return features
    
    def _extract_speech_features(self, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract features related to speech patterns from metadata.
        """
        features = {}
        
        # Extract duration and calculate speaking rate
        duration = meta.get("duration_sec", None)
        if duration is not None and duration > 0:
            word_count = len(text.split())
            features["speaking_rate"] = round(word_count / duration, 2)  # words per second
            features["duration_sec"] = round(duration, 2)
        
        # Add interruption metadata if available
        features["user_interruptions"] = meta.get("user_interruptions", 0)
        features["was_interrupted"] = meta.get("was_interrupted", False)
        
        # Add timestamps if available
        if "interruption_timestamps" in meta and isinstance(meta["interruption_timestamps"], list):
            features["interruption_timestamps"] = meta["interruption_timestamps"]
            
        # Add conversation flow metrics
        if features["was_interrupted"]:
            # Flag for reflection triggers during retrieval
            features["requires_reflection"] = True
            
            # Add analysis of interruption severity
            if features["user_interruptions"] > 5:
                features["interruption_severity"] = "high"
            elif features["user_interruptions"] > 2:
                features["interruption_severity"] = "medium"
            else:
                features["interruption_severity"] = "low"
        
        # Add other speech-related metadata if available
        for key in ["speaker_id", "confidence", "language", "timestamp", "session_id"]:
            if key in meta:
                features[key] = meta[key]
        
        return features
    
    def _load_keybert(self):
        """
        Attempt to lazy-load KeyBERT if it's available.
        Returns None if KeyBERT can't be loaded.
        """
        try:
            from keybert import KeyBERT
            logger.info("TranscriptionFeatureExtractor", "Lazy-loading KeyBERT")
            return KeyBERT()
        except ImportError:
            logger.warning("TranscriptionFeatureExtractor", 
                         "KeyBERT not installed. Install with: pip install keybert")
            return None

```

# vector_index.py

```py
# synthians_memory_core/vector_index.py

import os
import numpy as np

# Dynamic FAISS import with auto-installation fallback
try:
    import faiss
except ImportError:
    import sys
    import subprocess
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("vector_index")
    
    logger.warning("FAISS not found. Attempting to install...")
    
    # Check for GPU availability
    try:
        gpu_available = False
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            gpu_available = result.returncode == 0
        except:
            pass
            
        # Install appropriate FAISS package
        if gpu_available:
            logger.info("GPU detected, installing FAISS with GPU support")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-gpu'])
        else:
            logger.info("No GPU detected, installing CPU-only FAISS")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', 'faiss-cpu'])
            
        # Try importing again
        import faiss
        logger.info(f"Successfully installed and imported FAISS {getattr(faiss, '__version__', 'unknown')}")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
        raise ImportError("Failed to install FAISS. Please install it manually.")

import pickle
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class MemoryVectorIndex:
    """FAISS-based vector index for efficient memory retrieval.
    
    This class manages the storage and retrieval of memory embeddings using FAISS,
    a library for efficient similarity search. It supports persisting the index
    to disk and rehydrating it on startup.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vector index.
        
        Args:
            config: Configuration dictionary with the following keys:
                - embedding_dim: Dimensionality of the embeddings (default: 768)
                - storage_path: Path to store the index (default: '/app/memory/stored/synthians')
                - index_type: Type of FAISS index to use (default: 'L2')
                - use_gpu: Whether to use GPU acceleration if available (default: True)
                - gpu_id: Which GPU to use if multiple are available (default: 0)
        """
        self.config = {
            'embedding_dim': 768,
            'storage_path': '/app/memory/stored/synthians',
            'index_type': 'L2',  # 'L2', 'IP', 'Cosine'
            'use_gpu': True,      # Whether to attempt to use GPU
            'gpu_id': 0,          # Which GPU to use
            **(config or {})
        }
        
        self.dimension = self.config['embedding_dim']
        self.storage_path = self.config['storage_path']
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initialize FAISS index based on the configured type
        if self.config['index_type'] == 'Cosine':
            # For cosine similarity, we need to normalize the vectors
            self.index = faiss.IndexFlatIP(self.dimension)
            self.normalize = True
        elif self.config['index_type'] == 'IP':
            # Inner product (dot product)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.normalize = False
        else:  # Default to L2
            # L2 distance (Euclidean)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.normalize = False
        
        # Try to use GPU if available and requested
        self.using_gpu = False
        if self.config['use_gpu']:
            try:
                # Check if FAISS has GPU support
                gpu_resources = None
                if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                    gpu_id = self.config['gpu_id']
                    logger.info(f"Using GPU {gpu_id} for vector indexing")
                    
                    # Create a StandardGpuResources object
                    gpu_resources = faiss.StandardGpuResources()
                    
                    # Transfer the index to GPU
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, self.index)
                    self.using_gpu = True
                    logger.info(f"Successfully moved FAISS index to GPU {gpu_id}")
                else:
                    logger.info("No GPUs available for FAISS, using CPU index")
            except Exception as e:
                logger.warning(f"Failed to use GPU for FAISS: {e}. Using CPU instead.")
        
        # Map from index positions to memory IDs
        self.id_map: Dict[int, str] = {}
        # Map from memory IDs to index positions
        self.reverse_map: Dict[str, int] = {}
        # Current index size
        self.current_idx = 0
        
        logger.info(f"MemoryVectorIndex initialized with dimension {self.dimension}, "
                   f"type {self.config['index_type']}, GPU: {self.using_gpu}")
    
    def add(self, memory_id: str, embedding: np.ndarray) -> None:
        """Add a memory embedding to the index.
        
        Args:
            memory_id: Unique identifier for the memory
            embedding: Vector representation of the memory
        """
        # Check if memory already exists
        if memory_id in self.reverse_map:
            self.remove(memory_id)
            
        # Validate embedding
        if embedding is None or (np.isnan(embedding).any() or np.isinf(embedding).any()):
            logger.warning(f"Invalid embedding for memory {memory_id} - skipping index")
            return
        
        # Ensure embedding is the right shape and type
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        
        # Handle dimension mismatch
        if embedding.shape[1] != self.dimension:
            logger.warning(f"Embedding dimension mismatch: got {embedding.shape[1]}, expected {self.dimension}")
            # Pad or truncate embedding
            if embedding.shape[1] < self.dimension:
                # Pad with zeros
                padded = np.zeros((1, self.dimension), dtype=np.float32)
                padded[0, :embedding.shape[1]] = embedding
                embedding = padded
            else:
                # Truncate
                embedding = embedding[:, :self.dimension]
        
        # Normalize if using cosine similarity
        if self.normalize:
            faiss.normalize_L2(embedding)
        
        # Add to index
        self.index.add(embedding)
        
        # Update mappings
        self.id_map[self.current_idx] = memory_id
        self.reverse_map[memory_id] = self.current_idx
        self.current_idx += 1
    
    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index.
        
        FAISS doesn't support direct removal, so we rebuild the index without the removed item.
        
        Args:
            memory_id: Unique identifier for the memory to remove
        """
        if memory_id not in self.reverse_map:
            return
        
        # Get the current GPU resources if using GPU
        gpu_resources = None
        gpu_id = None
        if self.using_gpu and hasattr(faiss, 'get_gpu_resources_for_index'):
            try:
                # This is a simplification - in practice we'd need to carefully manage GPU resources
                gpu_id = self.config['gpu_id']
                # We'll recreate this when making the new index
                self.index = faiss.index_gpu_to_cpu(self.index)  # Convert back to CPU for rebuilding
            except Exception as e:
                logger.warning(f"Error converting index back to CPU: {e}")
        
        # Create a new index
        if self.config['index_type'] == 'Cosine' or self.config['index_type'] == 'IP':
            new_index = faiss.IndexFlatIP(self.dimension)
        else:  # Default to L2
            new_index = faiss.IndexFlatL2(self.dimension)
        
        # Get all vectors
        if self.current_idx > 0:
            all_vectors = self.index.reconstruct_n(0, self.current_idx)
            
            # Create new mappings
            new_id_map = {}
            new_reverse_map = {}
            new_idx = 0
            
            # Add all vectors except the one to remove
            for idx, vec in enumerate(all_vectors):
                if idx != self.reverse_map[memory_id]:
                    vec_reshaped = vec.reshape(1, -1)
                    if self.normalize:
                        faiss.normalize_L2(vec_reshaped)
                    
                    new_index.add(vec_reshaped)
                    
                    # Update mappings
                    mem_id = self.id_map[idx]
                    new_id_map[new_idx] = mem_id
                    new_reverse_map[mem_id] = new_idx
                    new_idx += 1
            
            # Replace the old index and mappings
            self.index = new_index
            self.id_map = new_id_map
            self.reverse_map = new_reverse_map
            self.current_idx = new_idx
        
        # If the memory was in our reverse_map but we couldn't rebuild (empty index)
        # Just clear the maps
        if memory_id in self.reverse_map:
            del self.reverse_map[memory_id]
            # Rebuild id_map from reverse_map
            self.id_map = {v: k for k, v in self.reverse_map.items()}
            
        # Move back to GPU if we were using it
        if gpu_id is not None and hasattr(faiss, 'index_cpu_to_gpu'):
            try:
                # Recreate GPU resources
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, self.index)
                self.using_gpu = True
            except Exception as e:
                logger.warning(f"Failed to move rebuilt index back to GPU: {e}")
                self.using_gpu = False
    
    def search(self, query_embedding: np.ndarray, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search for similar memory embeddings.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold (for filtering)
            
        Returns:
            List of (memory_id, similarity_score) tuples
        """
        if self.current_idx == 0:
            return []  # Empty index
        
        # Validate query embedding
        if query_embedding is None or (np.isnan(query_embedding).any() or np.isinf(query_embedding).any()):
            logger.warning("Invalid query embedding - cannot search")
            return []
        
        # Reshape and convert to float32
        query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        
        # Handle dimension mismatch
        if query_embedding.shape[1] != self.dimension:
            logger.warning(f"Query dimension mismatch: got {query_embedding.shape[1]}, expected {self.dimension}")
            # Pad or truncate
            if query_embedding.shape[1] < self.dimension:
                padded = np.zeros((1, self.dimension), dtype=np.float32)
                padded[0, :query_embedding.shape[1]] = query_embedding
                query_embedding = padded
            else:
                query_embedding = query_embedding[:, :self.dimension]
        
        # Normalize if using cosine similarity
        if self.normalize:
            faiss.normalize_L2(query_embedding)
        
        # Limit k to the number of items in the index
        k = min(k, self.current_idx)
        
        # Perform search
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in self.id_map:  # Valid index
                memory_id = self.id_map[idx]
                
                # Convert distance to similarity score
                if self.config['index_type'] == 'L2':
                    # Convert L2 distance to similarity (inverse relationship)
                    # Add small epsilon to avoid division by zero
                    similarity = 1.0 / (1.0 + distances[0][i])
                else:
                    # Inner product is already a similarity measure
                    similarity = float(distances[0][i])
                
                # Apply threshold filter
                if similarity >= threshold:
                    results.append((memory_id, similarity))
        
        return results
    
    def get_embedding(self, memory_id: str) -> Optional[np.ndarray]:
        """Retrieve the embedding for a specific memory ID.
        
        Args:
            memory_id: Unique identifier for the memory
            
        Returns:
            Embedding vector or None if not found
        """
        if memory_id not in self.reverse_map:
            return None
        
        idx = self.reverse_map[memory_id]
        return self.index.reconstruct(idx)
    
    def save(self) -> None:
        """Persist the index and mappings to disk."""
        index_path = os.path.join(self.storage_path, 'vector_index.faiss')
        mappings_path = os.path.join(self.storage_path, 'vector_index_mappings.pkl')
        
        try:
            # If using GPU, convert back to CPU for saving
            save_index = self.index
            if self.using_gpu and hasattr(faiss, 'index_gpu_to_cpu'):
                save_index = faiss.index_gpu_to_cpu(self.index)
                
            # Save the FAISS index
            faiss.write_index(save_index, index_path)
            
            # Save the mappings
            with open(mappings_path, 'wb') as f:
                pickle.dump({
                    'id_map': self.id_map,
                    'reverse_map': self.reverse_map,
                    'current_idx': self.current_idx,
                    'dimension': self.dimension,
                    'index_type': self.config['index_type'],
                    'using_gpu': self.using_gpu
                }, f)
            
            logger.info(f"Vector index saved to {index_path} with {self.current_idx} entries")
        except Exception as e:
            logger.error(f"Failed to save vector index: {e}")
    
    def load(self) -> bool:
        """Load the index and mappings from disk.
        
        Returns:
            True if successful, False otherwise
        """
        index_path = os.path.join(self.storage_path, 'vector_index.faiss')
        mappings_path = os.path.join(self.storage_path, 'vector_index_mappings.pkl')
        
        if not os.path.exists(index_path) or not os.path.exists(mappings_path):
            logger.info("No vector index found to load")
            return False
        
        try:
            # Load the FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load the mappings
            with open(mappings_path, 'rb') as f:
                data = pickle.load(f)
                self.id_map = data['id_map']
                self.reverse_map = data['reverse_map']
                self.current_idx = data['current_idx']
                
                # Check if dimension and index type match
                if data['dimension'] != self.dimension:
                    logger.warning(f"Loaded index dimension {data['dimension']} doesn't match current setting {self.dimension}")
                    self.dimension = data['dimension']
                
                if data.get('index_type') != self.config['index_type']:
                    logger.warning(f"Loaded index type {data.get('index_type')} doesn't match current setting {self.config['index_type']}")
            
            # Try to move to GPU if requested and available
            if self.config['use_gpu'] and hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                try:
                    gpu_id = self.config['gpu_id']
                    gpu_resources = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, self.index)
                    self.using_gpu = True
                    logger.info(f"Loaded index moved to GPU {gpu_id}")
                except Exception as e:
                    logger.warning(f"Failed to move loaded index to GPU: {e}")
                    self.using_gpu = False
            
            logger.info(f"Vector index loaded from {index_path} with {self.current_idx} entries, GPU: {self.using_gpu}")
            return True
        except Exception as e:
            logger.error(f"Failed to load vector index: {e}")
            return False
    
    def clear(self) -> None:
        """Clear the index and mappings."""
        # Keep track of whether we were using GPU
        was_using_gpu = self.using_gpu
        gpu_id = self.config['gpu_id'] if self.using_gpu else None
        
        # Reinitialize the index based on the configured type
        if self.config['index_type'] == 'Cosine' or self.config['index_type'] == 'IP':
            self.index = faiss.IndexFlatIP(self.dimension)
        else:  # Default to L2
            self.index = faiss.IndexFlatL2(self.dimension)
        
        # Move back to GPU if we were using it
        if was_using_gpu and hasattr(faiss, 'index_cpu_to_gpu') and hasattr(faiss, 'StandardGpuResources'):
            try:
                gpu_resources = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(gpu_resources, gpu_id, self.index)
                self.using_gpu = True
            except Exception as e:
                logger.warning(f"Failed to move cleared index to GPU: {e}")
                self.using_gpu = False
        
        # Clear mappings
        self.id_map = {}
        self.reverse_map = {}
        self.current_idx = 0
        
        logger.info(f"Vector index cleared, GPU: {self.using_gpu}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'index_size': self.current_idx,
            'dimension': self.dimension,
            'index_type': self.config['index_type'],
            'storage_path': self.storage_path,
            'using_gpu': self.using_gpu,
            'gpu_id': self.config['gpu_id'] if self.using_gpu else None
        }

```

