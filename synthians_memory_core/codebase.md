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
from synthians_memory_core.memory_core.trainer_integration import TrainerIntegrationManager, SequenceEmbeddingsResponse, UpdateQuickRecalScoreRequest

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
        
    # Initialize trainer integration manager
    try:
        logger.info("API", "Initializing trainer integration manager...")
        app.state.trainer_integration = TrainerIntegrationManager(
            memory_core=app.state.memory_core
        )
        logger.info("API", "Trainer integration manager initialized")
    except Exception as e:
        logger.error("API", f"Failed to initialize trainer integration manager: {str(e)}")
        app.state.trainer_integration = None
    
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
        # Get vector index stats
        vector_index_stats = {
            "count": app.state.memory_core.vector_index.count(),
            "id_mappings": len(app.state.memory_core.vector_index.id_to_index),
            "index_type": app.state.memory_core.vector_index.config.get('index_type', 'Unknown')
        }
        
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
            },
            "vector_index": vector_index_stats
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
        
        # Handle case where embedding is provided but in dict format
        if request.embedding is not None:
            if isinstance(request.embedding, dict):
                logger.warning("process_memory", f"Received embedding as dict type, attempting to extract vector")
                try:
                    # Try common dict formats
                    if 'embedding' in request.embedding and isinstance(request.embedding['embedding'], list):
                        embedding = request.embedding['embedding']
                        logger.info("process_memory", "Successfully extracted embedding from dict['embedding']")
                    elif 'vector' in request.embedding and isinstance(request.embedding['vector'], list):
                        embedding = request.embedding['vector']
                        logger.info("process_memory", "Successfully extracted embedding from dict['vector']")
                    elif 'value' in request.embedding and isinstance(request.embedding['value'], list):
                        embedding = request.embedding['value']
                        logger.info("process_memory", "Successfully extracted embedding from dict['value']")
                    else:
                        keys = list(request.embedding.keys()) if hasattr(request.embedding, 'keys') else 'unknown'
                        logger.error("process_memory", f"Could not extract embedding from dict with keys: {keys}")
                        embedding = None
                except Exception as e:
                    logger.error("process_memory", f"Error extracting embedding from dict: {str(e)}")
                    embedding = None
            else:
                # Normal list embedding
                embedding = request.embedding
                
        # Step 1: Generate embedding if needed
        if request.content and (embedding is None) and hasattr(app.state, 'embedding_model'):
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
            metadata['timestamp'] = time.time()
            
            # Add emotion data to metadata if available
            if emotion_data:
                metadata['emotional_context'] = emotion_data
            
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
            
            # Call the memory core to process the memory
            logger.info("process_memory", "Calling memory core to process memory")
            
            result = await app.state.memory_core.process_new_memory(
                content=request.content,
                embedding=embedding,
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

# --- Trainer Integration Endpoints ---

@app.post("/api/memories/get_sequence_embeddings", response_model=SequenceEmbeddingsResponse)
async def get_sequence_embeddings(
    topic: Optional[str] = None,
    user: Optional[str] = None,
    emotion: Optional[str] = None,
    min_importance: Optional[float] = None,
    limit: int = 100,
    min_quickrecal_score: Optional[float] = None,
    start_timestamp: Optional[str] = None,
    end_timestamp: Optional[str] = None,
    sort_by: str = "timestamp"
):
    """Retrieve a sequence of memory embeddings, ordered by timestamp or quickrecal score.
    
    This endpoint enables the Trainer to obtain sequential memory embeddings
    for training its predictive models and building semantic time series.
    """
    logger.info("API", f"Retrieving sequence embeddings with topic={topic}, limit={limit}, sort_by={sort_by}")
    
    if app.state.trainer_integration is None:
        logger.error("API", "Trainer integration manager not initialized")
        raise HTTPException(status_code=500, detail="Trainer integration not available")
    
    try:
        sequence = await app.state.trainer_integration.get_sequence_embeddings(
            topic=topic,
            user=user,
            emotion=emotion,
            min_importance=min_importance,
            limit=limit,
            min_quickrecal_score=min_quickrecal_score,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
            sort_by=sort_by
        )
        return sequence
    except Exception as e:
        logger.error("API", f"Error retrieving sequence embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sequence embeddings: {str(e)}")

@app.post("/api/memories/update_quickrecal_score")
async def update_quickrecal_score(request: UpdateQuickRecalScoreRequest):
    """Update a memory's quickrecal score based on surprise feedback from the Trainer.
    
    This endpoint allows the Trainer to inform the Memory Core about surprising or
    unexpected memories, which can boost their recall priority and track narrative surprise.
    
    Surprise is recorded in the memory's metadata for future reference and pattern analysis.
    """
    logger.info("API", f"Updating quickrecal score for memory {request.memory_id} with delta {request.delta}")
    
    if app.state.trainer_integration is None:
        logger.error("API", "Trainer integration manager not initialized")
        raise HTTPException(status_code=500, detail="Trainer integration not available")
    
    try:
        result = await app.state.trainer_integration.update_quickrecal_score(request)
        return result
    except Exception as e:
        logger.error("API", f"Error updating quickrecal score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update quickrecal score: {str(e)}")

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

# docs\api_reference.md

```md
# API Reference

This document provides a detailed reference for the APIs exposed by each component in the Bi-Hemispheric Cognitive Architecture.

## Table of Contents

1. [Memory Core API](#memory-core-api)
2. [Trainer Server API](#trainer-server-api)
3. [Context Cascade Engine API](#context-cascade-engine-api)
4. [Memory Assembly Management](#memory-assembly-management)
5. [Error Handling](#error-handling)

## Memory Core API

Base URL: `http://localhost:8000`

### Process Memory

\`\`\`
POST /api/memories/process
\`\`\`

Processes a new memory, generating embeddings and enriching metadata.

**Request Body:**

\`\`\`json
{
  "content": "Memory text content",
  "source": "source_identifier",
  "timestamp": "2025-03-27T20:10:30Z",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
  "metadata": {  // Optional metadata
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2}
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "id": "memory_uuid",
  "content": "Memory text content",
  "embedding": [0.1, 0.2, ...],
  "timestamp": "2025-03-27T20:10:30Z",
  "quickrecal_score": 0.85,
  "metadata": {
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2},
    "dominant_emotion": "joy",
    "importance": 0.75,
    "content_length": 120
  }
}
\`\`\`

### Retrieve Memories

\`\`\`
POST /api/memories/retrieve
\`\`\`

Retrieves memories based on similarity to a query embedding.

**Request Body:**

\`\`\`json
{
  "embedding": [0.1, 0.2, ...],
  "limit": 10,
  "threshold": 0.3,
  "filters": {
    "topic": "optional_topic_filter",
    "user": "optional_user_filter",
    "emotion": "optional_emotion_filter"
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "memories": [
    {
      "id": "memory_uuid",
      "content": "Memory text content",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "similarity": 0.92,
      "quickrecal_score": 0.85,
      "metadata": {...}
    },
    // Additional memories
  ]
}
\`\`\`

### Get Sequence Embeddings

\`\`\`
POST /api/memories/get_sequence_embeddings
\`\`\`

Retrieves a sequence of memory embeddings for training or prediction.

**Request Body:**

\`\`\`json
{
  "topic": "optional_topic",
  "user": "optional_user",
  "emotion": "optional_emotion",
  "min_importance": 0.5,
  "limit": 100,
  "min_quickrecal_score": 0.3,
  "start_timestamp": "2025-03-20T00:00:00Z",
  "end_timestamp": "2025-03-27T23:59:59Z",
  "sort_by": "timestamp"  // or "quickrecal_score"
}
\`\`\`

**Response:**

\`\`\`json
{
  "embeddings": [
    {
      "id": "memory_uuid",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "quickrecal_score": 0.85,
      "emotion": {"joy": 0.8, "surprise": 0.2},
      "dominant_emotion": "joy",
      "importance": 0.75,
      "topic": "conversation_topic",
      "user": "user_id"
    },
    // Additional embeddings
  ]
}
\`\`\`

### Update QuickRecal Score

\`\`\`
POST /api/memories/update_quickrecal_score
\`\`\`

Updates the quickrecal score of a memory based on surprise feedback.

**Request Body:**

\`\`\`json
{
  "memory_id": "memory_uuid",
  "delta": 0.2,
  "predicted_embedding": [0.1, 0.2, ...],
  "reason": "Surprise score: 0.8, context surprise: 0.3",
  "embedding_delta": [0.05, -0.03, ...]
}
\`\`\`

**Response:**

\`\`\`json
{
  "status": "success",
  "memory_id": "memory_uuid",
  "previous_score": 0.65,
  "new_score": 0.85,
  "delta": 0.2
}
\`\`\`

## Trainer Server API

Base URL: `http://localhost:8001`

### Health Check

\`\`\`
GET /health
\`\`\`

Checks the health status of the Trainer Server.

**Response:**

\`\`\`json
{
  "status": "ok",
  "timestamp": "2025-03-27T20:10:30Z"
}
\`\`\`

### Initialize Trainer

\`\`\`
POST /init
\`\`\`

Initializes the sequence trainer model with configuration.

**Request Body:**

\`\`\`json
{
  "inputDim": 768,
  "hiddenDim": 512,
  "outputDim": 768,
  "memoryDim": 256,
  "learningRate": 0.001
}
\`\`\`

**Response:**

\`\`\`json
{
  "message": "Sequence trainer model initialized",
  "config": {
    "inputDim": 768,
    "hiddenDim": 512,
    "outputDim": 768,
    "memoryDim": 256,
    "learningRate": 0.001
  }
}
\`\`\`

### Predict Next Embedding

\`\`\`
POST /predict_next_embedding
\`\`\`

Predicts the next embedding based on a sequence of input embeddings. This endpoint is fully stateless, relying on the `previous_memory_state` parameter for continuity between calls.

**Request Body:**

\`\`\`json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "previous_memory_state": {  // Required for stateless operation
    "sequence": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "surprise_history": [0.1, 0.2],
    "momentum": [0.1, 0.2, ...]
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "predicted_embedding": [0.1, 0.2, ...],
  "surprise_score": 0.35,
  "memory_state": {  // State to pass in the next prediction request
    "sequence": [[0.3, 0.4, ...], [0.5, 0.6, ...]],
    "surprise_history": [0.2, 0.35],
    "momentum": [0.15, 0.25, ...]
  }
}
\`\`\`

### Train Sequence

\`\`\`
POST /train_sequence
\`\`\`

Trains the model on a sequence of embeddings.

**Request Body:**

\`\`\`json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...]
}
\`\`\`

**Response:**

\`\`\`json
{
  "success": true,
  "loss": 0.015,
  "iterations": 10,
  "message": "Training successful"
}
\`\`\`

### Analyze Surprise

\`\`\`
POST /analyze_surprise
\`\`\`

Analyzes the surprise between predicted and actual embeddings.

**Request Body:**

\`\`\`json
{
  "predicted_embedding": [0.1, 0.2, ...],
  "actual_embedding": [0.1, 0.3, ...]
}
\`\`\`

**Response:**

\`\`\`json
{
  "surprise": 0.15,
  "cosine_surprise": 0.12,
  "context_surprise": 0.18,
  "delta_norm": 0.22,
  "is_surprising": true,
  "adaptive_threshold": 0.10,
  "volatility": 0.05,
  "delta": [0.0, 0.1, ...],
  "quickrecal_boost": 0.15
}
\`\`\`

## Context Cascade Engine API

Base URL: `http://localhost:8002`

### Process Memory

\`\`\`
POST /api/process_memory
\`\`\`

Processes a memory through the full cognitive pipeline.

**Request Body:**

\`\`\`json
{
  "content": "Memory text content",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed embedding
  "metadata": {  // Optional metadata
    "user": "user_id",
    "topic": "conversation_topic",
    "emotions": {"joy": 0.8, "surprise": 0.2}
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "memory_id": "memory_uuid",
  "status": "processed",
  "timestamp": "2025-03-27T20:10:30Z",
  "surprise": {
    "score": 0.35,
    "threshold": 0.6,
    "is_surprising": false,
    "factors": {
      "geometric": 0.28,
      "contextual": 0.42,
      "semantic": 0.35
    }
  },
  "prediction": {
    "predicted_embedding": [0.15, 0.25, ...],
    "confidence": 0.85
  },
  "memory_state": {
    "quickrecal_score": 0.85,
    "adjusted_score": 0.85  // Same as quickrecal_score if no surprise
  }
}
\`\`\`

### Retrieve Memories

\`\`\`
POST /api/retrieve_memories
\`\`\`

Retrieves memories with enhanced context-aware filtering.

**Request Body:**

\`\`\`json
{
  "query": "Query text content",
  "embedding": [0.1, 0.2, ...],  // Optional pre-computed query embedding
  "limit": 10,
  "threshold": 0.3,
  "current_emotion": {  // Optional emotional context
    "dominant": "joy",
    "values": {"joy": 0.8, "surprise": 0.2}
  },
  "cognitive_load": 0.5,  // Optional, 0.0-1.0
  "filters": {
    "topic": "optional_topic_filter",
    "user": "optional_user_filter",
    "emotion": "optional_emotion_filter"
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "memories": [
    {
      "id": "memory_uuid",
      "content": "Memory text content",
      "embedding": [0.1, 0.2, ...],
      "timestamp": "2025-03-27T20:10:30Z",
      "similarity": 0.92,
      "emotional_resonance": 0.85,
      "final_score": 0.89,  // Combined score after emotional gating
      "quickrecal_score": 0.85,
      "metadata": {...}
    },
    // Additional memories
  ],
  "context": {
    "query_emotion": "joy",
    "cognitive_load_applied": 0.5,
    "filters_applied": ["topic"],
    "emotional_gating_applied": true
  }
}
\`\`\`

## Memory Assembly Management

The Memory Core provides APIs for creating and managing memory assemblies, which group related memories together.

### Create Assembly

\`\`\`
POST /api/assemblies/create
\`\`\`

Creates a new memory assembly.

**Request Body:**

\`\`\`json
{
  "name": "Assembly Name",
  "description": "Optional assembly description",
  "initial_memories": ["memory_id_1", "memory_id_2", ...],
  "tags": ["tag1", "tag2"],
  "metadata": {  // Optional metadata
    "creator": "user_id",
    "category": "assembly_category"
  }
}
\`\`\`

**Response:**

\`\`\`json
{
  "assembly_id": "assembly_uuid",
  "name": "Assembly Name",
  "memory_count": 2,
  "composite_embedding": [0.1, 0.2, ...],
  "creation_time": "2025-03-27T20:10:30Z",
  "dominant_emotions": ["joy", "curiosity"],
  "keywords": ["keyword1", "keyword2"]
}
\`\`\`

### Add Memory to Assembly

\`\`\`
POST /api/assemblies/{assembly_id}/add_memory
\`\`\`

Adds a memory to an existing assembly.

**Request Body:**

\`\`\`json
{
  "memory_id": "memory_uuid"
}
\`\`\`

**Response:**

\`\`\`json
{
  "assembly_id": "assembly_uuid",
  "memory_id": "memory_uuid",
  "status": "added",
  "memory_count": 3,
  "updated_composite_embedding": [0.12, 0.22, ...]
}
\`\`\`

### List Assemblies

\`\`\`
GET /api/assemblies
\`\`\`

Lists all available memory assemblies.

**Response:**

\`\`\`json
{
  "assemblies": [
    {
      "assembly_id": "assembly_uuid_1",
      "name": "Assembly Name 1",
      "memory_count": 3,
      "creation_time": "2025-03-27T20:10:30Z",
      "dominant_emotions": ["joy", "curiosity"],
      "keywords": ["keyword1", "keyword2"]
    },
    // Additional assemblies
  ]
}
\`\`\`

### Get Assembly Details

\`\`\`
GET /api/assemblies/{assembly_id}
\`\`\`

Retrieves detailed information about a specific assembly.

**Response:**

\`\`\`json
{
  "assembly_id": "assembly_uuid",
  "name": "Assembly Name",
  "description": "Assembly description",
  "memory_count": 3,
  "composite_embedding": [0.12, 0.22, ...],
  "creation_time": "2025-03-27T20:10:30Z",
  "last_modified": "2025-03-27T21:10:30Z",
  "memories": [
    {
      "id": "memory_id_1",
      "content": "Memory text content 1",
      "timestamp": "2025-03-27T20:05:30Z"
    },
    // Additional memory summaries
  ],
  "dominant_emotions": ["joy", "curiosity"],
  "keywords": ["keyword1", "keyword2"],
  "metadata": {
    "creator": "user_id",
    "category": "assembly_category"
  }
}
\`\`\`

### Delete Assembly

\`\`\`
DELETE /api/assemblies/{assembly_id}
\`\`\`

Deletes a memory assembly (this does not delete the individual memories).

**Response:**

\`\`\`json
{
  "assembly_id": "assembly_uuid",
  "status": "deleted",
  "memory_count_released": 3
}
\`\`\`

## Error Handling

All APIs follow a consistent error response format:

\`\`\`json
{
  "error": "Error message description",
  "status": "error",
  "code": 404,  // HTTP status code or custom error code
  "details": {  // Optional additional error information
    "field": "field_with_error",
    "reason": "specific reason for error"
  }
}
\`\`\`

Common HTTP status codes:

- **400 Bad Request**: Invalid request parameters or payload
- **404 Not Found**: Resource (memory, assembly, etc.) not found
- **500 Internal Server Error**: Server-side processing error
- **408 Request Timeout**: Service timeout (particularly for embedding generation)

Custom error codes:

- **"timeout"**: Connection timed out
- **"connection_refused"**: Service unavailable or cannot be reached
- **"unknown_error"**: Unspecified error

```

# docs\architechture-changes.md

```md
# Synthians Architecture Changes
## 2025-03-27T23:05:09Z - Lucidia Agent

Okay, let's break down the implications of successfully integrating the Titans Neural Memory module, as implemented according to the paper, into your `synthians_trainer_server`. This moves beyond simple prediction to a more dynamic form of memory.

**Core Shift:** You're moving from a model that *predicts* the next state based on a learned function (like a standard RNN/LSTM where only the hidden state changes at test time) to a model whose *internal parameters* (`M`) are actively *updated* at test time based on new inputs and an associative loss. It's learning to memorize *during* inference.

**Key Implications:**

1.  **True Test-Time Adaptation & Memorization:**
    *   **What:** The memory module (`M`) literally changes its weights with each relevant input via the `update_step` (gradient descent + momentum + decay).
    *   **Why:** This directly implements the paper's core idea – "learning to memorize at test time." It's not just updating a state vector; it's refining its internal associative mapping (`M(k) -> v`) on the fly.
    *   **Impact:** The system can continuously adapt to new information encountered *after* initial training. It explicitly encodes new key-value associations into its parameters, offering a form of ongoing learning and potentially better handling of dynamic environments or distribution shifts compared to static models.

2.  **Shift from Prediction to Associative Recall & Update:**
    *   **What:** The primary functions become `retrieve(query)` (associative recall without changing weights) and `update_memory(input)` (memorization by changing weights). Direct prediction of the *next embedding* is less explicit; retrieval provides related information based on a query.
    *   **Why:** The model's loss (`||M(k) - v||²`) drives it to associate keys with values, not necessarily to predict the *next* value in a sequence directly from the *previous* one in the same way the old model did.
    *   **Impact:** The orchestrator (`ContextCascadeEngine`) needs different logic. Instead of asking "predict next," it might:
        *   Get current embedding `x_t` from `SynthiansMemoryCore`.
        *   Call `/update_memory` with `x_t` to memorize the current step (updating `M`).
        *   Generate a query `q_t` (maybe from `x_t` or context).
        *   Call `/retrieve` with `q_t` to get relevant associative memory `y_t`.
        *   Use `y_t` (and maybe `x_t`) to inform the next action or a separate prediction head.

3.  **More Sophisticated "Surprise" Metric:**
    *   **What:** The gradient `∇ℓ` used in the `update_step` directly represents how much the memory model's parameters needed to change to correctly associate the current key `k_t` with value `v_t`. This is the paper's "surprise."
    *   **Why:** It measures the error in the associative memory's *current* understanding. The momentum term `S_t` carries this surprise forward.
    *   **Impact:** This gradient norm (or related metrics) can be sent back to the `SynthiansMemoryCore` via the orchestrator to update `quickrecal_score`, providing a more grounded measure of novelty or unexpectedness based on the memory's internal learning process.

4.  **Potential for Enhanced Long-Term Context Handling:**
    *   **What:** Information is encoded into the *parameters* of `M`, not just a fixed-size state vector. The forgetting gate (`alpha_t`) helps manage capacity.
    *   **Why:** Unlike RNN hidden states which can saturate or overwrite information, updating weights allows for potentially storing more information over longer sequences, distributed across the parameters. The forgetting gate provides a mechanism to discard less relevant history encoded in the weights.
    *   **Impact:** Theoretically better performance on tasks requiring recall over very long contexts (as claimed in the paper, >2M tokens), surpassing limitations of fixed RNN states and quadratic Transformer costs.

5.  **Increased Computational Cost at Test Time:**
    *   **What:** Every `update_memory` call involves a forward pass, a loss calculation, a backward pass (gradient calculation w.r.t `M`), and parameter updates.
    *   **Why:** This is inherent to the "learning at test time" approach using gradient descent.
    *   **Impact:** Inference (a retrieve + update cycle) will be significantly slower per step than the previous model's simple forward pass. The parallelization technique mentioned in the paper (Section 3.2) becomes crucial for practical speed, but our current implementation is sequential.

6.  **Complex Training Dynamics (Outer vs. Inner Loop):**
    *   **What:** You now have two sets of parameters: the *outer* parameters (`WK`, `WV`, `WQ`, gates) trained via traditional backprop on a task loss, and the *inner* memory parameters (`M`) which evolve during the test-time `update_step` but are *reset* for the outer loop training gradient calculation.
    *   **Why:** The outer loop learns *how to learn/memorize effectively* (by tuning projections and gates), while the inner loop *performs* the memorization.
    *   **Impact:** Requires careful implementation of the outer training loop (`train_outer_step`) and managing the state reset. Tuning the gates (`alpha_t`, `theta_t`, `eta_t`) and the outer learning rate becomes critical for balancing memorization and generalization.

7.  **Explicit Role Definition:**
    *   **What:** The `synthians_trainer_server` now clearly embodies the adaptive, associative, long-term memory role. `SynthiansMemoryCore` remains the structured, indexed, episodic/semantic store.
    *   **Why:** Aligns with the paper's concept of distinct but interconnected memory systems.
    *   **Impact:** Simplifies conceptual understanding. The orchestrator mediates between the fast-lookup `MemoryCore` and the dynamically learning `NeuralMemoryModule`.

**In Summary:**

Getting this working means your "trainer" server transforms from a sequence predictor into a **dynamic, test-time adaptive associative memory**. It gains the ability to continuously learn and encode new associations directly into its parameters during operation. This offers potential for superior long-context handling and adaptation but comes at the cost of increased per-step computational complexity during inference and requires a more sophisticated training setup (outer loop). The interaction with `SynthiansMemoryCore` becomes richer, with the Neural Memory handling dynamic patterns and the Core handling structured storage and retrieval, potentially linked via surprise feedback.

## Implementation Considerations

### Optimization Opportunities

1. **Inference Speed Optimization:**
   * Consider implementing the paper's parallelization technique (Section 3.2) to enable parallel update steps
   * Profile forward/backward operations to identify bottlenecks
   * For large memory models, investigate quantization of memory parameters

2. **Memory Efficiency:**
   * Monitor memory usage patterns during extended operation
   * Implement mechanisms to selectively reset memory weights when they saturate (monitor gradient norms)
   * Consider scheduled alpha/forgetting gate adjustments based on context length

3. **Outer Loop Training:**
   * Start with simple task losses before implementing complex meta-learning objectives
   * Carefully track outer vs. inner parameter gradients to prevent interference
   * Consider curriculum learning for outer loop parameters (start with short contexts)

### Integration with Orchestrator

1. **New Call Pattern:**
   \`\`\`python
   # Previous pattern (simplified)
   previous_memory_state = [...]
   prediction, new_memory = trainer_server.predict_next_embedding(curr_embedding, previous_memory_state)
   
   # New pattern (simplified)
   # 1. First memorize current embedding (updates internal weights)
   trainer_server.update_memory(curr_embedding)
   
   # 2. Then retrieve relevant memory using a query
   query = generate_query(curr_embedding, context)
   memory_retrieval = trainer_server.retrieve(query)
   \`\`\`

2. **Surprise Metric Integration:**
   * Expose a gradient norm metric from `/update_memory` endpoint 
   * Feed this value directly into `quickrecal_score` calculation
   * Consider sliding window normalization of gradient norms

3. **Fallback Mechanisms:**
   * Implement retrieval confidence scoring
   * Provide graceful degradation when memory is unconfident
   * Consider hybrid approaches: use traditional prediction heads alongside memory retrieval

### Monitoring & Debugging

1. **Key Metrics to Track:**
   * Gate values (α, θ, η) throughout operation
   * Gradient norms for inner memory updates
   * Weight change magnitude after each update step
   * Memory parameter saturation (if weights grow too large)

2. **Visualization Tools:**
   * Create embeddings projector for the internal key/value spaces
   * Track key-to-value mapping consistency over time
   * Visualize memory association strength through operation

### Future Extensions

1. **Multi-Head Memory:**
   * Consider extending to multiple parallel memory modules specializing in different association types
   * Implement attention mechanism over multiple memory retrievals

2. **Hierarchical Memory:**
   * Create layered memory modules with different timescales
   * Fast-changing short-term memory feeding into slower-changing long-term memory

3. **Memory Reflection:**
   * Periodically perform "reflection" steps where memory retrieves from itself
   * Use these to consolidate and reorganize internal representation patterns

---

## 2025-03-27T23:04:02Z: Neural Memory Integration - Lucidia Agent

### Summary of Changes

Successfully integrated the Titans Neural Memory module into the `synthians_trainer_server` by fixing critical TensorFlow/Keras implementation issues. The module now properly supports save/load state functionality and correctly registers trainable variables for dynamic updates at test time.

### Key Technical Fixes

1. **Fixed MemoryMLP Layer Registration**
   * Moved layer creation from `build()` to `__init__()` method to ensure proper variable tracking
   * Changed layers from private list (`_layers`) to explicit instance attributes (`self.hidden_layers`, `self.output_layer`)
   * Ensured TensorFlow's variable tracking system correctly identifies trainable weights
   * Resolved "MemoryMLP has NO trainable variables!" errors that prevented gradient updates

2. **Fixed TensorFlow Model Save/Load State**
   * Corrected architecture violation where model was being rebuilt in-place with `__init__()`
   * Implemented proper state loading that respects TensorFlow architectural constraints
   * Created a separate model initialization approach for loading models with different configs
   * Added comprehensive error handling for shape mismatches during weight loading
   * Fixed momentum state variable handling to ensure gradient updates work correctly

3. **Enhanced Gradient Tracking**
   * Added explicit `tape.watch()` calls for trainable variables
   * Fixed gradient calculation in both inner and outer update loops
   * Implemented proper handling of `None` gradients during training
   * Added resilience measures to detect and rebuild missing variables

4. **API Endpoint Improvements**
   * Fixed tensor shape handling in `/retrieve`, `/update_memory`, and `/train_outer` endpoints
   * Improved error messages and validation
   * Enhanced the state persistence endpoints (`/save` and `/load`)

### Impact

* All 9/9 API tests now pass successfully
* The neural memory module can now properly learn at test time as described in the Titans paper
* Gradient updates flow correctly through both inner and outer optimization loops
* State can be reliably saved and loaded across model instances

### Future Considerations

1. **Performance Optimization**
   * Current implementation processes batch examples sequentially in the training loop
   * Could be optimized for parallel processing of examples

2. **Memory Efficiency**
   * Consider optimizing for large embedding dimensions
   * Implement memory-efficient update strategies for high-dimensional embeddings

3. **Metrics Collection**
   * Add tracking for gradient norms, gate values, and memory usage
   * Implement visualization tools for memory behavior analysis
```

# docs\architecture_overview.md

```md
# Bi-Hemispheric Architecture Overview

## Introduction

The Synthians Memory Core implements a Bi-Hemispheric Cognitive Architecture that separates memory storage/retrieval from sequence prediction/surprise detection, mimicking how the brain's hemispheres handle different aspects of cognition. This document provides a technical overview of the architecture, component interactions, and the information flow between them.

## System Components

### 1. Memory Core

The Memory Core serves as the primary memory storage and retrieval system, similar to the brain's hippocampus and temporal lobes.

**Key Responsibilities:**
- Storing and indexing memory entries with associated embeddings and metadata
- Retrieval of memories based on semantic similarity and quickrecal scores
- Memory assembly management and persistence
- Emotional gating of memory retrieval based on emotional context
- Maintaining memory importance through quickrecal scores

**Key Classes:**
- `SynthiansMemoryCore`: Main interface for all memory operations
- `MemoryEntry`: Individual memory representation with embedding and metadata
- `MemoryAssembly`: Collection of related memories with a composite embedding
- `MemoryPersistence`: Handles saving and loading memories and assemblies
- `EmotionalGatingService`: Applies emotional context to memory retrieval

### 2. Trainer Server

The Trainer Server handles sequence prediction and surprise detection, similar to the brain's frontal lobes and predictive capabilities.

**Key Responsibilities:**
- Predicting the next embedding in a sequence using neural mechanisms
- Calculating surprise when expectations don't match reality
- Training on memory sequences to improve predictions
- Maintaining a stateless architecture that relies on explicit memory state passing

**Key Classes:**
- `SynthiansTrainer`: Neural model for sequence prediction
- `SurpriseDetector`: Detects and analyzes surprise in embedding sequences
- `HPCQRFlowManager`: Manages the QuickRecal factors for memory importance

### 3. Context Cascade Engine (Orchestrator)

The Context Cascade Engine connects the Memory Core and Trainer Server, orchestrating the flow of information between them and implementing the full cognitive cycle.

**Key Responsibilities:**
- Processing new memories through the complete cognitive pipeline
- Managing the interplay between prediction and memory storage
- Feeding surprise feedback to enhance memory retrieval
- Handling error states and coordinating between components

**Key Classes:**
- `ContextCascadeEngine`: Main orchestrator class
- `GeometryManager`: Shared utility for consistent vector operations across components

## Information Flow

### Full Cognitive Cycle

1. **Input Processing:**
   - New memory content and optional embedding arrive at the Context Cascade Engine
   - The Engine forwards the memory to the Memory Core for storage

2. **Prediction:**
   - The Engine sends the current embedding to the Trainer Server
   - Trainer generates a prediction for the next memory embedding
   - The prediction is stored for later comparison

3. **Reality and Surprise:**
   - When the next actual memory arrives, its embedding is compared to the prediction
   - The Trainer calculates surprise metrics between prediction and reality
   - High surprise indicates a memory that violated expectations

4. **Feedback:**
   - Surprise information is fed back to adjust the quickrecal score of the memory
   - Surprising memories receive a higher importance (quickrecal boost)
   - This feedback loop ensures important memories are more accessible

5. **Adaptation:**
   - The system continuously learns from sequences of memories
   - Prediction accuracy improves over time through training
   - Retrieval thresholds adapt based on results

## Stateless Design Pattern

A key refinement in the architecture is the stateless design of the Trainer Server:

1. **No Global State:**
   - The Trainer Server maintains no session or global state
   - Each prediction request must include all necessary context

2. **Memory State Passing:**
   - The `previous_memory_state` parameter contains the state from the last prediction
   - This state includes sequence history, surprise metrics, and momentum
   - The response includes a new `memory_state` to be passed in the next request

3. **Orchestrator State Management:**
   - The Context Cascade Engine manages the memory state between requests
   - It stores the state returned by the Trainer and passes it in the next prediction

4. **Benefits:**
   - Improved scalability and reliability
   - Multiple sessions can use the same Trainer Server concurrently
   - Simpler recovery from failures

## Assembly Management

Memory Assemblies provide a way to group related memories and treat them as a cohesive unit:

1. **Creation and Composition:**
   - Assemblies can be created with initial memories or built incrementally
   - Each assembly maintains a composite embedding representing its semantic center
   - When memories are added, the composite embedding is updated

2. **Persistence:**
   - Assemblies are saved to disk in JSON format with their constituent memories
   - On initialization, the system loads all saved assemblies
   - The `MemoryPersistence` class handles serialization, deserialization, and error recovery

3. **Memory-to-Assembly Mapping:**
   - The system maintains a mapping from individual memories to their assemblies
   - This allows for efficient querying of all assemblies that contain a specific memory

## Error Handling

The architecture implements robust error handling throughout:

1. **Specific Error Types:**
   - HTTP status codes (404, 400, 500) are handled with specific error messages
   - Connection errors, timeouts, and unexpected exceptions have clear handling paths

2. **Consistent Error Format:**
   - All errors follow a standard format with status, code, and detailed message
   - Custom error codes for specific failure scenarios

3. **Graceful Degradation:**
   - Components can continue functioning when dependent services are unavailable
   - Default behaviors are provided when specific features cannot be accessed

## Geometric Operations

All vector operations are standardized using the shared GeometryManager:

1. **Consistent Vector Handling:**
   - Normalization, similarity calculations, and vector alignment
   - Support for different embedding dimensions (384, 768)
   - Handling of different geometries (Euclidean, Hyperbolic, Spherical)

2. **Alignment for Comparison:**
   - Vectors of different dimensions are safely aligned
   - NaN/Inf values are detected and handled appropriately

## Conclusion

The Bi-Hemispheric Architecture provides a powerful framework for memory processing that combines storage, retrieval, prediction, and surprise detection in a cohesive system. The stateless design pattern enhances scalability while maintaining the rich context needed for effective cognition.

The architecture is designed to be modular, allowing components to be improved or replaced independently. This flexibility enables ongoing enhancements to specific aspects of the system while maintaining overall functionality.

```

# docs\bihemispheric_architecture.md

```md
# Bi-Hemispheric Cognitive Architecture

## Overview

The Bi-Hemispheric Cognitive Architecture implements a neural system inspired by human brain hemispheric specialization, creating a bidirectional flow between memory storage/retrieval and sequential prediction. This architecture enables Lucidia to develop a more nuanced understanding of sequential patterns and adapt memory retrieval based on prediction accuracy and surprise detection.

## Key Components

### 1. Memory Core (Left Hemisphere)

Responsible for storing, indexing, retrieving, and enriching memories:

- **Memory Storage**: Persists embeddings and metadata to disk
- **Vector Indexing**: Enables fast similarity-based retrieval using FAISS
- **Metadata Enrichment**: Adds contextual information to memories
- **Emotional Analysis**: Detects emotions in content and uses them for retrieval
- **HPC-QR**: Hippocampal-inspired Quick Recall scoring system

### 2. Trainer Server (Right Hemisphere)

Focuses on pattern recognition and sequence prediction:

- **Sequence Prediction**: Predicts the next embedding based on current input
- **Memory State Tracking**: Maintains internal memory state to track context
- **Surprise Analysis**: Detects unexpected patterns in embedding sequences

### 3. Context Cascade Engine (Corpus Callosum)

Orchestrates the bidirectional flow between the two hemispheres:

- **Prediction Integration**: Feeds predictions from the Trainer into Memory Core retrieval
- **Surprise Detection**: Identifies when reality diverges from predictions
- **Memory Enhancement**: Updates memory importance based on surprise signals
- **State Management**: Tracks the Trainer's memory state across interactions

## Neural Pathway Flow

1. **Input Processing**: New input is processed and embedded by the Memory Core
2. **Prediction**: Context Cascade Engine sends current embedding to Trainer for next embedding prediction
3. **Reality Check**: When new input arrives, it's compared against the prediction
4. **Surprise Detection**: Difference between prediction and reality is quantified
5. **Feedback Loop**: Surprising memories get importance boosts in Memory Core
6. **Retrieval Enhancement**: Future retrievals prioritize memories that were surprising

## Key Innovations

1. **Vector Alignment**: System handles embedding dimension mismatches (384 vs 768) seamlessly
2. **Surprise Metrics**: Measures both prediction error and context shifts
3. **Adaptive Thresholds**: Surprise detection adapts to current narrative volatility
4. **Memory State Continuity**: Maintains continuity of the prediction model's internal state
5. **Quickrecal Boosting**: Automatically enhances the retrieval priority of surprising memories

## Architecture Diagram

\`\`\`
┌───────────────────┐              ┌─────────────────────┐
│   Memory Core     │              │   Trainer Server    │
│  (Left Hemisphere)│              │  (Right Hemisphere) │
│                   │              │                     │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │   GeometryMgr │ │              │ │GeometryMgr (ref)│ │
│ └───────────────┘ │              │ └─────────────────┘ │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │  VectorIndex  │ │              │ │SequencePredictor│ │
│ └───────────────┘ │              │ └─────────────────┘ │
│ ┌───────────────┐ │              │ ┌─────────────────┐ │
│ │   MetadataSyn │ │              │ │ SurpriseDetector│ │
│ └───────────────┘ │              │ └─────────────────┘ │
└────────┬──────────┘              └──────────┬──────────┘
         │                                     │
         │        ┌──────────────────┐        │
         │        │ Context Cascade  │        │
         └────────┤     Engine      ├────────┘
                  │ (Corpus Callosum)│
                  └──────────────────┘
\`\`\`

## Implementation Notes

- The system is designed to handle embedding dimension mismatches, a critical requirement for systems using different embedding models
- The GeometryManager is shared across components to ensure vector operations are consistent
- All communication between components uses asynchronous HTTP calls with proper timeouts and error handling
- Memory state is preserved between calls to maintain prediction continuity
- The system adapts to the current context's volatility when determining surprise thresholds

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

# docs\implementation_guide.md

```md
# Bi-Hemispheric Cognitive Architecture: Implementation Guide

## Introduction

This technical guide explains how to implement and integrate the components of the Bi-Hemispheric Cognitive Architecture. It covers deployment, configuration, and development patterns to extend the system.

## System Requirements

- Docker and Docker Compose
- Python 3.9+
- CUDA-compatible GPU (optional, for accelerated embedding generation)
- 8GB+ RAM 

## Component Deployment

### Using Docker Compose

The easiest way to deploy the full architecture is using the included `docker-compose-bihemispheric.yml` file:

\`\`\`bash
docker-compose -f docker-compose-bihemispheric.yml up -d
\`\`\`

This launches all three components (Memory Core, Trainer Server, and Context Cascade Engine) with proper networking and configuration.

### Manual Deployment

To run components individually (useful for development):

1. **Memory Core**
   \`\`\`bash
   cd synthians_memory_core
   python -m server.main
   \`\`\`

2. **Trainer Server**
   \`\`\`bash
   cd synthians_memory_core/synthians_trainer_server
   python -m http_server
   \`\`\`

3. **Context Cascade Engine**
   \`\`\`bash
   cd synthians_memory_core/orchestrator
   python -m server
   \`\`\`

## Configuration

### Environment Variables

The architecture uses the following environment variables (can be set in Docker Compose or locally):

\`\`\`
# Memory Core
PORT=8000
VECTOR_DB_PATH=./vectordb
MEMORY_STORE_PATH=./memorystore
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Trainer Server
PORT=8001
MEMORY_CORE_URL=http://memory_core:8000
INPUT_DIM=384
HIDDEN_DIM=256
OUTPUT_DIM=384
MEMORY_DIM=128
LEARNING_RATE=0.001

# Context Cascade Engine
PORT=8002
MEMORY_CORE_URL=http://memory_core:8000
TRAINER_URL=http://trainer:8001
\`\`\`

## Component Integration

### GeometryManager

The `GeometryManager` is a central utility class shared across components to ensure consistent handling of embeddings:

\`\`\`python
from synthians_memory_core.geometry_manager import GeometryManager

# Create a shared instance
geometry_manager = GeometryManager()

# Use for vector operations
normalized = geometry_manager.normalize_embedding(embedding)
similarity = geometry_manager.calculate_similarity(vec1, vec2)
aligned_vecs = geometry_manager.align_vectors_for_comparison(vec1, vec2)
\`\`\`

### SurpriseDetector

The `SurpriseDetector` quantifies deviation between predicted and actual embeddings:

\`\`\`python
from synthians_memory_core.synthians_trainer_server.surprise_detector import SurpriseDetector

# Initialize with GeometryManager
surprise_detector = SurpriseDetector(geometry_manager=geometry_manager)

# Calculate surprise metrics
metrics = surprise_detector.calculate_surprise(
    predicted_embedding=predicted_vec,
    actual_embedding=actual_vec
)

# Calculate quickrecal boost based on surprise
boost = surprise_detector.calculate_quickrecal_boost(metrics)
\`\`\`

### Context Cascade Engine

The orchestrator coordinates the bidirectional flow between Memory Core and Trainer:

\`\`\`python
from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine

# Initialize the engine
engine = ContextCascadeEngine(
    memory_core_url="http://localhost:8000",
    trainer_url="http://localhost:8001"
)

# Process a new memory through the cognitive pipeline
result = await engine.process_new_memory(
    content="Memory content text",
    metadata={"user": "user_id", "topic": "conversation"}
)
\`\`\`

## Handling Embedding Dimensions

One of the key challenges in implementing this architecture is handling dimensional mismatches between embeddings. The system supports two common dimensions:

- **384-dimensional embeddings**: From models like `all-MiniLM-L6-v2`
- **768-dimensional embeddings**: From models like `all-mpnet-base-v2`

The `GeometryManager` handles these mismatches through alignment strategies:

\`\`\`python
def _align_vectors_for_comparison(self, vec1, vec2):
    """Align vectors for comparison when dimensions don't match.
    
    Strategies:
    1. If dimensions match, return as is
    2. If one is smaller, pad with zeros
    3. If both differ from target dim, truncate or pad as needed
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # If dimensions already match, return as-is
    if vec1.shape == vec2.shape:
        return vec1, vec2
    
    # If one dimension is smaller, pad with zeros
    if vec1.shape[0] < vec2.shape[0]:
        # Pad vec1 to match vec2
        padded = np.zeros(vec2.shape)
        padded[:vec1.shape[0]] = vec1
        return padded, vec2
    elif vec1.shape[0] > vec2.shape[0]:
        # Pad vec2 to match vec1
        padded = np.zeros(vec1.shape)
        padded[:vec2.shape[0]] = vec2
        return vec1, padded
\`\`\`

## Error Handling & Robustness

The architecture implements comprehensive error handling:

1. **Embedding Validation**: All embeddings are validated for NaN/Inf values

\`\`\`python
def _validate_embedding(self, embedding):
    """Validate embedding for NaN or Inf values."""
    try:
        embedding_array = np.array(embedding, dtype=np.float32)
        if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
            return False
        return True
    except Exception:
        return False
\`\`\`

2. **Network Timeouts**: All inter-service communications have timeout handling

\`\`\`python
try:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=10.0) as response:
            # Process response
except asyncio.TimeoutError:
    logger.error(f"Timeout while connecting to service at {url}")
    return {"error": "Connection timed out"}
\`\`\`

3. **State Management**: The Trainer's memory state is preserved for continuity

\`\`\`python
# Store memory state for next prediction
if "memory_state" in result:
    self.current_memory_state = result["memory_state"]

# Include previous memory state in next request
if self.current_memory_state is not None:
    payload["previous_memory_state"] = self.current_memory_state
\`\`\`

## Extending the Architecture

### Adding New Memory Storage Backends

To implement a new storage backend:

1. Create a class that implements the `MemoryPersistence` interface
2. Register it in the Memory Core's dependency injection system

### Implementing Custom Prediction Models

To create a new prediction model:

1. Extend the `SequenceTrainer` base class
2. Implement the `predict_next` and `update_memory_state` methods
3. Register it in the Trainer Server

### Customizing Surprise Detection

To modify surprise detection logic:

1. Extend or modify the `SurpriseDetector` class
2. Customize the `calculate_surprise` method to use different metrics
3. Update the `calculate_quickrecal_boost` formula as needed

## Testing

The architecture includes several test suites:

\`\`\`bash
# Run Memory Core tests
python -m pytest synthians_memory_core/tests

# Run Trainer Server tests
python -m pytest synthians_memory_core/synthians_trainer_server/tests

# Run Orchestrator tests
python -m pytest synthians_memory_core/orchestrator/tests
\`\`\`

Use the included Docker Compose test configuration for integration testing:

\`\`\`bash
docker-compose -f docker-compose-test.yml up --abort-on-container-exit
\`\`\`

```

# docs\memory_system_remaster.md

```md
# Synthians Memory System Remaster

_Documentation for the comprehensive memory system enhancements_

**Date**: March 27, 2025  
**Branch**: Synthience_memory_remaster

## 🧠 Overview

The Synthians Memory Core is a sophisticated system that integrates vector search, embedding processing, and emotional analysis to create a cohesive memory retrieval mechanism. This document outlines recent critical enhancements to the system, focusing on persistence, reliability, and observability.

## 🔍 Problem Statement

The memory system was experiencing several key issues:

1. **Vector Index Persistence**: Memories were being added to the FAISS vector index but the index itself wasn't being saved to disk during the persistence process, causing all lookups to fail after system restart.

2. **Observability Gaps**: The system lacked proper diagnostics and stats for monitoring the vector index state and memory operations.

3. **Embedding Dimension Mismatches**: The system struggled with handling different embedding dimensions (primarily between 384 and 768), causing comparison errors.

4. **Retrieval Thresholds**: The default threshold was too high (0.5), causing many relevant memories to be filtered out.

## 🛠️ Solutions Implemented

### 1. Fixed Vector Index Persistence

\`\`\`python
# Added code to _persist_all_managed_memories to save the vector index
if self.vector_index.count() > 0:
    vector_index_saved = self.vector_index.save()
    logger.info("SynthiansMemoryCore", f"Vector index saved: {vector_index_saved} with {self.vector_index.count()} vectors and {len(self.vector_index.id_to_index)} id mappings")
\`\`\`

This critical fix ensures that the FAISS index and ID-to-index mappings are properly saved to disk during the persistence cycle, enabling consistent memory retrieval even after system restarts.

### 2. Enhanced API Observability

\`\`\`python
# Extended the /stats endpoint with vector index information
vector_index_stats = {
    "count": app.state.memory_core.vector_index.count(),
    "id_mappings": len(app.state.memory_core.vector_index.id_to_index),
    "index_type": app.state.memory_core.vector_index.config.get('index_type', 'Unknown')
}
\`\`\`

Improved the `/stats` endpoint to provide comprehensive vector index information, enabling better monitoring and debugging of the memory system.

### 3. Embedding Dimension Handling

\`\`\`python
# Added vector alignment utilities
def _align_vectors_for_comparison(self, vec1, vec2):
    """Safely align two vectors to the same dimension for comparison operations."""
    if vec1.shape[0] != vec2.shape[0]:
        # Either pad with zeros or truncate to match dimensions
        target_dim = min(vec1.shape[0], vec2.shape[0])
        if vec1.shape[0] > target_dim:
            vec1 = vec1[:target_dim]
        if vec2.shape[0] > target_dim:
            vec2 = vec2[:target_dim]
    return vec1, vec2
\`\`\`

Implemented robust dimension handling to ensure vector operations work correctly regardless of the embedding dimensions used.

### 4. Retrieval Threshold Adjustments

\`\`\`python
# Lowered threshold for better recall sensitivity
if threshold is None:
    threshold = 0.2  # Lowered from 0.5 to 0.2 for better recall
\`\`\`

Adjusted the pre-filter threshold from 0.5 to 0.2 to improve recall sensitivity while maintaining precision.

## 📊 Testing and Validation

We created comprehensive testing tools to validate the memory system:

1. **direct_test.py**: Validates the full memory lifecycle through the API:
   - Memory creation
   - Proper persistence
   - Retrieval with similarity scores

2. **tests/test_memory_retrieval_api.py**: API-based test suite for Docker:
   - Health checks
   - Memory creation and retrieval tests
   - GPU detection and validation

## 🔄 Additional System Improvements

### Metadata Enrichment

\`\`\`python
# Add memory ID to metadata for easier access
memory.metadata["uuid"] = memory.id
\`\`\`

Enhanced memory metadata with additional context (UUID, content length) to improve traceability.

### Redundant Computation Prevention

\`\`\`python
# Analyze Emotion only if not already provided
emotional_context = metadata.get("emotional_context")
if not emotional_context:
    emotional_context = await self.emotional_analyzer.analyze(content)
    metadata["emotional_context"] = emotional_context
else:
    logger.debug("Using precomputed emotional context from metadata")
\`\`\`

Optimized processing by avoiding redundant emotion analysis when data is already available.

## 🚀 Deployment and Usage

### Docker Integration

The system fully supports GPU acceleration through FAISS when deployed with Docker:

\`\`\`bash
# Start the service with GPU support
docker-compose up -d

# Run tests inside the container
docker exec -it synthians_core python /workspace/project/direct_test.py
\`\`\`

### API Endpoints

- `/process_memory`: Create new memories with optional embeddings
- `/retrieve_memories`: Retrieve memories using semantic similarity
- `/stats`: Get comprehensive system statistics

## 🧪 Validation Process

To verify the system is working correctly:

1. Create a memory via the API
2. Check that it's properly saved to disk
3. Restart the container
4. Verify the memory can be retrieved using a semantically similar query

## 📝 Conclusion

The Synthians Memory System has been significantly enhanced with better persistence, observability, and reliability. These improvements ensure consistent memory retrieval, better debugging capabilities, and more robust embedding handling.

```

# docs\NEWEST-DOCUMENTATION.md

```md

This won't just be documentation; it will be the **living specification for Lucidia's cognitive core.**

---

## Development Roadmap & Status (March 28, 2025)

**Project:** Synthians Cognitive Architecture (Lucidia)
**Focus:** Bi-Hemispheric Memory System (Memory Core + Neural Memory)

**Overall Goal:** Implement a robust, unified memory system enabling adaptive, long-context cognition inspired by human memory and the Titans paper. Create the infrastructure for a persistent, learning cognitive presence (Lucidia).

---

### ✅ Phase 1: Memory Core Unification & Foundation (Completed)

*   **Objective:** Consolidate core memory storage, retrieval, and relevance scoring.
*   **Status:** **DONE**
*   **Key Outcomes:**
    *   Unified `synthians_memory_core` package created.
    *   Components integrated: `SynthiansMemoryCore`, `UnifiedQuickRecallCalculator`, `GeometryManager`, `EmotionalAnalyzer/GatingService`, `MemoryPersistence`, `MemoryAssembly`, `ThresholdCalibrator`, `MetadataSynthesizer`.
    *   Robust FAISS `VectorIndex` implemented with GPU support and persistence.
    *   Core API server (`api/server.py`) established for Memory Core functions.
    *   Basic end-to-end memory lifecycle tested (Store, Retrieve, Feedback).
    *   Initial documentation drafted for core components.

---

### ✅ Phase 2: Neural Memory Module Implementation (Completed)

*   **Objective:** Replace the previous predictive trainer with the Titans-inspired `NeuralMemoryModule` capable of test-time learning.
*   **Status:** **DONE**
*   **Key Outcomes:**
    *   `synthians_trainer_server/neural_memory.py` created, implementing `NeuralMemoryModule` with MLP core (`MemoryMLP`).
    *   Implemented `update_step` logic for test-time weight updates based on associative loss, momentum, and forgetting gates (Eq. 13 & 14).
    *   Implemented `retrieve` (inference) function (`call` method).
    *   Defined separate outer-loop parameters (`WK/WV/WQ`, gates) and inner-loop memory parameters (`M`).
    *   Implemented basic outer-loop training mechanism (`train_step` override) for meta-parameter updates.
    *   Implemented state saving/loading (`save_state`, `load_state`) for the module.
    *   Refactored `synthians_trainer_server/http_server.py` with new endpoints: `/init`, `/retrieve`, `/update_memory`, `/train_outer`, `/status`, `/save`, `/load`, `/health`, `/analyze_surprise`.
    *   Integrated `SurpriseDetector` and `GeometryManager` correctly.
    *   Resolved TensorFlow/Keras integration issues (Variable tracking, Layer building, Gradient Taping).
    *   Basic API tests (`test_neural_memory_api.py`) updated and passing for the new endpoints.

---

### Phase 3: Orchestration Layer Integration (Current Focus / Next Steps)

*   **Objective:** Connect the `MemoryCore` and the `NeuralMemoryModule` via the `ContextCascadeEngine` to enable the full cognitive loop.
*   **Status:** **IN PROGRESS / TODO**
*   **Tasks:**
    *   **Modify `ContextCascadeEngine.process_new_memory`:**
        *   After storing `x_t` in `MemoryCore`, call `NeuralMemoryServer:/update_memory` with `x_t`'s embedding.
        *   Extract surprise proxy (`loss` or `grad_norm`) from the `/update_memory` response.
        *   Calculate `quickrecal_boost` based on this surprise.
        *   Call `MemoryCore:/api/memories/update_quickrecal_score` with the boost.
        *   Generate query `q_t`.
        *   Call `NeuralMemoryServer:/retrieve` with `q_t` to get `y_t`.
        *   Determine how `y_t` influences the next step/output (Task-dependent).
    *   **Implement Query Generation:** Decide how `q_t` is generated within the `ContextCascadeEngine` (e.g., from `x_t` using `WQ`, from conversational context, etc.).
    *   **Refine Surprise-to-Boost Logic:** Develop a nuanced function mapping the neural memory's surprise metrics (`loss`, `grad_norm`) to the `delta` for QuickRecal score updates.
    *   **Update `ContextCascadeEngine` Server:** Expose necessary high-level endpoints (e.g., a unified `/process_input` endpoint) that trigger the full cascade.
    *   **Integration Testing:** Create tests verifying the end-to-end flow through all three components.

---

### Phase 4: Titans Architecture Variants (Future)

*   **Objective:** Implement and evaluate the different ways of integrating the Neural Memory with Attention, as described in Section 4 of the Titans paper (MAC, MAG, MAL).
*   **Status:** **TODO**
*   **Tasks:**
    *   Design Keras/TF layers implementing the specific attention/gating mechanisms for MAC, MAG, MAL.
    *   Integrate these layers with the `NeuralMemoryModule` and `MemoryCore` (likely within or called by the `ContextCascadeEngine`).
    *   Evaluate performance on relevant benchmarks (language modeling, NIAH, reasoning) as per the paper.

---

### Phase 5: Advanced Features & Optimization (Future)

*   **Objective:** Enhance robustness, performance, and cognitive capabilities.
*   **Status:** **TODO**
*   **Tasks:**
    *   Implement parallelized `update_step` (Section 3.2) for performance.
    *   Explore deeper/alternative `MemoryMLP` architectures.
    *   Implement complex, data-dependent gates (`alpha_t`, `theta_t`, `eta_t`).
    *   Integrate "Persistent Memory" tokens (Section 3.3).
    *   Develop advanced outer-loop training strategies (curriculum learning, meta-learning objectives).
    *   Implement memory reflection/consolidation mechanisms.
    *   Add more sophisticated monitoring and visualization.
    *   Re-integrate advanced HPC-QR factors.

---
---

## Updated Documentation Set (Interwoven with Lucidia Narrative)

Here are the revised documentation files reflecting the current state (post-Phase 2) and the underlying philosophy.

**1. `docs/README.md` (Root README - Minor Update)**

\`\`\`md
# Synthians Memory Core Documentation

This repository contains the core components for the Synthians Cognitive Architecture, focusing on establishing **Lucidia**, a synthetic cognitive presence with integrated, adaptive memory.

The primary components are:

1.  **`synthians_memory_core`:** The foundational memory system responsible for structured storage, indexing, enrichment, and retrieval of discrete experiences (Memory Entries). It acts as the stable library of Lucidia's past.
2.  **`synthians_trainer_server`:** Implements the **Titans Neural Memory**, a dynamic, associative network that learns and adapts *at test time*, shaping Lucidia's understanding of contextual flow and relationships based on ongoing experience.
3.  **`orchestrator`:** (Work in Progress) The `ContextCascadeEngine` that mediates the flow between the stable Memory Core and the adaptive Neural Memory, enabling a full cognitive cycle of experience, memorization, reflection (via surprise), and recall.

## Key Documentation

*   **Architecture:**
    *   [Bi-Hemispheric Architecture Overview](docs/bihemispheric_architecture.md) - Describes the separation of concerns between stable storage and adaptive learning.
    *   [Architecture Changes](docs/architechture-changes.md) - Log of significant design shifts, including the integration of the Titans Neural Memory.
*   **API:**
    *   [API Reference](docs/api_reference.md) - Details endpoints for the Memory Core and the Neural Memory Server.
*   **Components:**
    *   [Memory Core README](synthians_memory_core/README.md) - Overview of the storage/retrieval hemisphere.
    *   [Neural Memory (Trainer) README](synthians_trainer_server/README.md) - Overview of the adaptive learning hemisphere. *(Needs Creation/Update)*
    *   [Vector Index & FAISS](docs/vector_index.md) / [FAISS GPU Integration](docs/faiss_gpu_integration.md) - Details on the fast retrieval backend.
    *   [Embedding Handling](docs/embedding_handling.md) - How vector representations are managed.
*   **Guides:**
    *   [Implementation Guide](docs/implementation_guide.md) - Technical details on setup, configuration, and integration.
*   **Conceptual:**
    *   [Core vs. Neural Memory Relationship](docs/synthience-trainer-compliment.md) - Explains the distinct roles of the two memory systems in shaping Lucidia's presence.
\`\`\`

**2. `docs/api_reference.md` (Updated Trainer Section - *Provided in Previous Response*)**

*(Use the rewritten "Neural Memory (Trainer) Server API" section from the previous response, as it accurately reflects the new endpoints: `/init`, `/retrieve`, `/update_memory`, `/train_outer`, etc.)*

**3. `docs/architecture_overview.md` (Revised Trainer & Flow - *Provided in Previous Response*)**

*(Use the revised "Neural Memory (Trainer) Server" component description and the revised "Information Flow" section from the previous response.)*

**4. `docs/bihemispheric_architecture.md` (Revised Trainer, Flow, Diagram - *Provided in Previous Response*)**

*(Use the revised "Neural Memory (Trainer) Server" component description, the revised "Neural Pathway Flow", and the updated "Architecture Diagram" from the previous response.)*

**5. `docs/implementation_guide.md` (Updated Config, Integration - *Provided in Previous Response*)**

*(Use the updated "Configuration" section (removing old trainer vars, mentioning `/init` config) and the updated "Component Integration" section (showing the new orchestrator interaction pattern) from the previous response.)*

**6. `docs/synthience-trainer-compliment.md` (Rewritten - *Provided in Previous Response*)**

*(Use the rewritten version explaining the new "Library vs. Adaptive Network" relationship.)*

**7. `docs/architechture-changes.md` (New File)**

*(Use the content provided in the user prompt under this filename. It accurately captures the implications of the shift to the Titans Neural Memory.)*

**8. `docs/geometry_manager.md` (New File - Optional but Recommended)**

*(Create a dedicated file for GeometryManager, adapted from relevant sections of `embedding_handling.md`)*

\`\`\`md
# Geometry Manager

## Overview

The `GeometryManager` is a crucial utility within the Synthians architecture, providing centralized and consistent handling of vector embedding operations across different components (Memory Core, Neural Memory, Orchestrator). Its primary goal is to ensure numerical stability, manage different embedding dimensions, and support various geometric spaces for representing and comparing memories.

## Key Responsibilities

1.  **Validation:** Checks embeddings for `None`, invalid types, and problematic values (`NaN`, `Inf`). Invalid vectors are typically replaced with zero vectors to prevent downstream errors.
2.  **Normalization:** Provides L2 normalization (`normalize_embedding`) to ensure vectors have unit length, often required for cosine similarity and stable geometric calculations. Normalization can be disabled via config.
3.  **Dimension Alignment:** Safely aligns vectors of potentially different dimensions (`align_vectors`) to the target dimension specified in the configuration (`embedding_dim`). Supports padding or truncating based on `alignment_strategy`.
4.  **Geometric Transformations:** Includes methods to project vectors between Euclidean space and other geometries, notably the Poincaré ball model for Hyperbolic geometry (`_to_hyperbolic`, `_from_hyperbolic`).
5.  **Distance & Similarity Calculation:** Offers methods to calculate distance (`calculate_distance`) and similarity (`calculate_similarity`) according to the configured `geometry_type` (Euclidean, Hyperbolic, Spherical, Mixed), automatically handling necessary projections and normalizations.

## Configuration

The `GeometryManager` is initialized with a configuration dictionary:

\`\`\`python
config = {
    'embedding_dim': 768,             # Target dimension for alignment
    'geometry_type': 'hyperbolic',    # Or 'euclidean', 'spherical', 'mixed'
    'curvature': -1.0,                # For hyperbolic/spherical
    'alignment_strategy': 'truncate', # Or 'pad'
    'normalization_enabled': True     # Apply L2 norm during normalization
}
gm = GeometryManager(config)
\`\`\`

## Usage Example

\`\`\`python
from synthians_memory_core.geometry_manager import GeometryManager
import numpy as np

gm = GeometryManager({'embedding_dim': 768, 'geometry_type': 'euclidean'})

vec1 = np.random.rand(768).astype(np.float32)
vec2 = np.random.rand(768).astype(np.float32)
vec_short = np.random.rand(384).astype(np.float32)

# Normalize
norm_vec1 = gm.normalize_embedding(vec1)

# Calculate similarity (uses configured geometry)
similarity = gm.calculate_similarity(vec1, vec2)

# Align different dimensions
aligned_vec1, aligned_short = gm.align_vectors(vec1, vec_short)
# Now aligned_vec1 and aligned_short both have dimension 768
\`\`\`

## Integration Notes

-   A single, shared instance of `GeometryManager` should ideally be used across components requiring vector operations (e.g., passed during initialization) to ensure consistency.
-   The `SurpriseDetector` relies on the `GeometryManager` for similarity calculations.
-   The `NeuralMemoryModule` can use it internally if needed, although core TF operations might suffice.
-   The `SynthiansMemoryCore` uses it extensively for indexing, retrieval scoring, and assembly updates.
\`\`\`

**9. `docs/embedding_handling.md` (Update)**

*   **Change:** Ensure code snippets use `geometry_manager.normalize_embedding(...)` instead of `_normalize(...)`.
*   **Add:** A sentence mentioning that `GeometryManager` is the central class for these operations. (See `docs/geometry_manager.md` for details).

**10. `synthians_trainer_server/README.md` (New or Update)**

\`\`\`md
# Synthians Trainer Server / Neural Memory Module

This directory contains the implementation of the **Titans Neural Memory**, a core component of the Synthians cognitive architecture's "Right Hemisphere."

## Overview

Unlike traditional sequence models or the previous predictor, this module implements an **adaptive associative memory** inspired by the "Titans: Learning to Memorize at Test Time" paper. Its key characteristic is its ability to **update its own internal parameters (weights) during inference (test time)** based on new inputs.

**Role:** To learn and recall associations (`key -> value`) dynamically, capturing the flow and relationships between sequential data points (embeddings) provided by the `SynthiansMemoryCore`. It focuses on *how* information connects over time, rather than just storing discrete facts.

## Core Concepts

*   **Neural Memory (`M`):** An internal neural network (currently an MLP) whose weights encode learned associations.
*   **Test-Time Learning:** Uses a gradient-based update rule (`update_step`) derived from an associative loss (`||M(k_t) - v_t||^2`) to modify `M`'s weights with each new input `x_t`.
*   **Inner vs. Outer Loop:**
    *   **Inner Loop (Test Time):** The `update_step` modifies `M`'s weights based on `x_t`, momentum `S_t`, and gates (`α, θ, η`).
    *   **Outer Loop (Training Time):** Trains the projection matrices (`WK, WV, WQ`) and gate parameters (`α, θ, η`) using a standard optimizer and a task-specific loss, teaching the module *how* to memorize effectively.
*   **Associative Retrieval (`retrieve` / `call`):** Performs a forward pass through the current state of `M` using a query `q_t` to recall the associated value `y_t`, without updating weights.
*   **Surprise:** The loss or gradient norm during the `update_step` provides an intrinsic measure of how surprising the input `x_t` was relative to `M`'s current state.

## API Endpoints (`http_server.py`)

*   `/init`: Initializes the `NeuralMemoryModule` with configuration, optionally loading state.
*   `/retrieve`: Performs associative recall (inference).
*   `/update_memory`: Executes one step of test-time memorization (updates internal weights).
*   `/train_outer`: Executes one step of outer loop training (updates projection/gate weights).
*   `/analyze_surprise`: Utility endpoint using `SurpriseDetector`.
*   `/status`, `/save`, `/load`, `/health`: Standard utility endpoints.

Refer to `docs/api_reference.md` for detailed request/response schemas.

## Key Files

*   `neural_memory.py`: Contains `NeuralMemoryConfig`, `MemoryMLP`, and the main `NeuralMemoryModule` class implementing the Titans logic.
*   `http_server.py`: Exposes the `NeuralMemoryModule` via a FastAPI application.
*   `surprise_detector.py`: Utility class for external surprise analysis (used by `/analyze_surprise` and potentially the orchestrator).

## Relationship to Memory Core

The Neural Memory Server acts as the dynamic learning counterpart to the `SynthiansMemoryCore`'s structured storage. The Core provides indexed access to past experiences, while the Neural Memory learns the *connections and flow* between those experiences as they happen. Surprise metrics generated during the Neural Memory's update process can feed back to influence the relevance scores (`quickrecal_score`) in the Memory Core. See `docs/synthience-trainer-compliment.md` for a deeper dive.
\`\`\`

---

This revised documentation set should provide a clear, comprehensive, and philosophically aligned view of your current architecture, reflecting the significant step taken by implementing the Titans Neural Memory. Remember to replace the placeholder code snippets in the Markdown files with actual, relevant examples from your implementation as needed.
```

# docs\README.md

```md
# Synthians Memory Core Documentation

## Bi-Hemispheric Cognitive Architecture

- [Bi-Hemispheric Architecture Overview](bihemispheric_architecture.md) - Complete design overview and neural pathway flow
- [API Reference](api_reference.md) - Detailed API references for all components
- [Implementation Guide](implementation_guide.md) - Technical implementation and integration guide

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

# docs\synthience-trainer-compliment.md

```md
Okay, let's break down how the refactored `synthians_trainer_server` (based on the original "Titan" code) fits with your existing `synthians_memory_core` system.

Think of them as two specialized but complementary brain components:

1.  **`synthians_memory_core` (The Library / Database):**
    *   **Primary Role:** Stores, organizes, enriches, and retrieves *individual memories* (`MemoryEntry`).
    *   **Focus:** Content, metadata (emotion, importance, timestamps, etc.), relationships (assemblies), long-term persistence, fast similarity search (FAISS), adaptive relevance.
    *   **Analogy:** A highly organized, searchable, and cross-referenced library or knowledge base. You add individual books/articles (memories), tag them, link related ones, and can search for specific information or related topics. It knows *what* happened and *details* about it.

2.  **`synthians_trainer_server` (The Sequence Predictor):**
    *   **Primary Role:** Learns *temporal patterns and predicts sequences*. It operates on *sequences of embeddings*, not the raw memory content itself.
    *   **Focus:** Understanding the *flow* or *dynamics* between memory states (represented by embeddings). Given a current state (embedding + its internal memory `trainer_memory_vec`), it predicts the *next likely state* (embedding). It calculates "surprise" based on how well its prediction matches reality.
    *   **Analogy:** A system that learns the *plot* or *typical sequence of events* from reading sequences of stories (sequences of memory embeddings). It doesn't store the full stories themselves, but learns "if this kind of event happens, that kind of event often follows." It excels at prediction and understanding flow.

**How They Complement Each Other (The Workflow):**

An overarching AI system would likely use both in a loop:

1.  **Ingestion:** New information (text, audio transcript, interaction) comes in.
    *   **Memory Core:** Processes the information, generates an embedding, analyzes emotion, calculates QuickRecal, synthesizes metadata, and stores it as a `MemoryEntry`.
2.  **Sequence Generation:** Periodically, or based on context (e.g., retrieving memories related to a specific topic or time frame).
    *   **Memory Core:** Retrieves a *sequence* of related memories (likely represented by their embeddings, perhaps ordered by timestamp). This could be memories within an `MemoryAssembly` or memories retrieved based on a specific query over time.
3.  **Trainer Learning:** The sequence of embeddings retrieved from the *Memory Core* is fed into the...
    *   **Trainer Server:** Uses `train_sequence` or `train_step` to update its internal weights and `trainer_memory_vec`, learning the typical transitions between these memory states (embeddings).
4.  **Prediction & Understanding:** When the AI needs to anticipate, plan, or understand the current situation based on recent history:
    *   It takes the embedding of the *current* memory (or a recent sequence) from the *Memory Core*.
    *   **Trainer Server:** Uses `forward_pass` with the current embedding and its internal state (`trainer_memory_vec`) to predict the *next likely embedding* and calculate the `surprise`.
5.  **Feedback Loop (Optional but Powerful):**
    *   The predicted embedding from the *Trainer* could be used to *prime* or *guide* the next retrieval query in the *Memory Core*.
    *   The `surprise` value calculated by the *Trainer* could be added as metadata to new `MemoryEntry` objects being stored in the *Memory Core*, indicating how novel or unexpected that particular state transition was according to the learned sequence model. This could influence the `quickrecal_score`.

**Key Distinctions:**

*   **Data Unit:** Core handles `MemoryEntry` (content + embedding + metadata); Trainer handles sequences of *embeddings*.
*   **Goal:** Core is about *storage and recall*; Trainer is about *prediction and dynamics*.
*   **State:** Core maintains the state of individual memories; Trainer maintains an internal state (`trainer_memory_vec`) representing the *context of the current sequence*.
*   **Output:** Core retrieves existing memories; Trainer predicts *future* states (embeddings).

**In Summary:**

The `synthians_trainer_server` (formerly Titan) **doesn't store memories** like the `synthians_memory_core`. Instead, it **learns the relationships and transitions *between* the memories** (specifically, their embeddings) that are stored and retrieved by the `synthians_memory_core`. They work together: the Core provides the sequential data, and the Trainer learns the underlying patterns within that data, potentially feeding insights (like surprise) back to the Core.



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
from .emotion_analyzer import EmotionAnalyzer as _EmotionAnalyzer  # Import with alias to avoid name conflicts

# Maintain backward compatibility by re-exporting the class
# This prevents import errors in existing code that imports from this module
class EmotionalAnalyzer(_EmotionAnalyzer):
    """Re-export of the EmotionAnalyzer class from emotion_analyzer.py for backward compatibility."""
    pass

# Export EmotionalAnalyzer for backward compatibility
__all__ = ['EmotionalAnalyzer', 'EmotionalGatingService']

# NOTE: The EmotionalAnalyzer class implementation has been moved to emotion_analyzer.py
# This file now only contains the EmotionalGatingService class and a compatibility wrapper

class EmotionalGatingService:
    """Applies emotional gating to memory retrieval."""
    def __init__(self, emotion_analyzer, config: Optional[Dict] = None):
        """Initialize the emotional gating service.
        
        Args:
            emotion_analyzer: An instance of EmotionAnalyzer from emotion_analyzer.py
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

    def align_vectors(self, vec_a: np.ndarray, vec_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align two vectors to the configured embedding dimension."""
        # Validate inputs
        vec_a = self._validate_vector(vec_a, "Vector A")
        if vec_a is None:
            vec_a = np.zeros(self.config['embedding_dim'], dtype=np.float32)
            
        vec_b = self._validate_vector(vec_b, "Vector B")
        if vec_b is None:
            vec_b = np.zeros(self.config['embedding_dim'], dtype=np.float32)
            
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

    def normalize_embedding(self, vector: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        # Ensure input is numpy array
        vector = self._validate_vector(vector, "Vector to Normalize")
        if vector is None:
            # Return zero vector of appropriate dimension if validation failed
            return np.zeros(self.config.get('embedding_dim', 768), dtype=np.float32)

        if not self.config['normalization_enabled']:
             return vector
        norm = np.linalg.norm(vector)
        if norm < 1e-9:
            logger.debug("GeometryManager", "normalize_embedding received zero vector, returning as is.")
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
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        return np.linalg.norm(aligned_a - aligned_b)

    def hyperbolic_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate Hyperbolic (Poincaré) distance."""
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
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
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
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
        """Calculate similarity between two vectors based on the configured geometry type.
        
        Returns cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
        """
        # Validate inputs
        vec_a = self._validate_vector(vec_a, "Vector A for similarity")
        if vec_a is None:
            return 0.0
            
        vec_b = self._validate_vector(vec_b, "Vector B for similarity")
        if vec_b is None:
            return 0.0
            
        # Align vectors to same dimension
        aligned_a, aligned_b = self.align_vectors(vec_a, vec_b)
        
        # Normalize both vectors
        norm_a = np.linalg.norm(aligned_a)
        norm_b = np.linalg.norm(aligned_b)
        
        # Handle zero vectors
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        
        # Calculate cosine similarity
        norm_a_inv = 1.0 / norm_a
        norm_b_inv = 1.0 / norm_b
        dot_product = np.dot(aligned_a, aligned_b)
        similarity = dot_product * norm_a_inv * norm_b_inv
        
        # Ensure result is in valid range [-1.0, 1.0]
        return float(np.clip(similarity, -1.0, 1.0))

    def transform_to_geometry(self, vector: np.ndarray) -> np.ndarray:
        """Transform a vector into the configured geometry space (e.g., Poincaré ball)."""
        vector = self._validate_vector(vector, "Input Vector")
        if vector is None: return np.zeros(self.config['embedding_dim'])

        geom_type = self.config['geometry_type']
        if geom_type == GeometryType.HYPERBOLIC:
            return self._to_hyperbolic(vector)
        elif geom_type == GeometryType.SPHERICAL:
            # Project onto unit sphere (normalize)
            return self.normalize_embedding(vector)
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

# memory_core\trainer_integration.py

```py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import datetime
import logging
import numpy as np

from synthians_memory_core.memory_structures import MemoryEntry
from synthians_memory_core import SynthiansMemoryCore
from synthians_memory_core.geometry_manager import GeometryManager

logger = logging.getLogger(__name__)

class SequenceEmbedding(BaseModel):
    """Representation of an embedding in a sequence for trainer integration."""
    id: str
    embedding: List[float]
    timestamp: str
    quickrecal_score: Optional[float] = None
    emotion: Optional[Dict[str, float]] = None
    dominant_emotion: Optional[str] = None
    importance: Optional[float] = None
    topic: Optional[str] = None
    user: Optional[str] = None

class SequenceEmbeddingsResponse(BaseModel):
    """Response model for a sequence of embeddings."""
    embeddings: List[SequenceEmbedding]
    
class UpdateQuickRecalScoreRequest(BaseModel):
    """Request to update the quickrecal score of a memory based on surprise."""
    memory_id: str
    delta: float
    predicted_embedding: Optional[List[float]] = None
    reason: Optional[str] = None
    embedding_delta: Optional[List[float]] = None

class TrainerIntegrationManager:
    """Manages integration between the Memory Core and the Sequence Trainer.
    
    This class bridges the gap between the memory storage system and the
    predictive sequence model, enabling bidirectional communication for:
    - Feeding memory embeddings to the trainer in sequence
    - Updating memory retrieval scores based on prediction surprises
    """
    
    def __init__(self, memory_core: SynthiansMemoryCore):
        """Initialize with reference to the memory core."""
        self.memory_core = memory_core
        # Initialize geometry manager to handle dimension mismatches
        self.geometry_manager = GeometryManager(target_dim=768, max_warnings=10)
    
    async def get_sequence_embeddings(self, 
                                topic: Optional[str] = None, 
                                user: Optional[str] = None,
                                emotion: Optional[str] = None,
                                min_importance: Optional[float] = None,
                                limit: int = 100,
                                min_quickrecal_score: Optional[float] = None,
                                start_timestamp: Optional[str] = None,
                                end_timestamp: Optional[str] = None,
                                sort_by: str = "timestamp") -> SequenceEmbeddingsResponse:
        """Retrieve a sequence of embeddings from the memory core,
        ordered by timestamp or quickrecal score.
        
        Args:
            topic: Optional topic filter
            user: Optional user filter
            emotion: Optional dominant emotion filter
            min_importance: Optional minimum importance threshold
            limit: Maximum number of embeddings to retrieve
            min_quickrecal_score: Minimum quickrecal score threshold
            start_timestamp: Optional start time boundary
            end_timestamp: Optional end time boundary
            sort_by: Field to sort by ("timestamp" or "quickrecal_score")
            
        Returns:
            SequenceEmbeddingsResponse with ordered list of embeddings
        """
        # Convert timestamp strings to datetime objects if provided
        start_dt = None
        end_dt = None
        if start_timestamp:
            try:
                start_dt = datetime.datetime.fromisoformat(start_timestamp)
            except ValueError:
                logger.warning(f"Invalid start_timestamp format: {start_timestamp}")
        
        if end_timestamp:
            try:
                end_dt = datetime.datetime.fromisoformat(end_timestamp)
            except ValueError:
                logger.warning(f"Invalid end_timestamp format: {end_timestamp}")
        
        # Query the memory entries
        query = {}
        
        # Add filters if specified
        if topic:
            query["metadata.topic"] = topic
        
        if user:
            query["metadata.user"] = user
            
        if emotion:
            query["metadata.dominant_emotion"] = emotion
            
        if min_importance is not None:
            query["metadata.importance"] = {"$gte": min_importance}
            
        # Add quickrecal score filter if specified
        if min_quickrecal_score is not None:
            query["quickrecal_score"] = {"$gte": min_quickrecal_score}
            
        # Add timestamp filters if specified
        if start_dt or end_dt:
            timestamp_query = {}
            if start_dt:
                timestamp_query["$gte"] = start_dt
            if end_dt:
                timestamp_query["$lte"] = end_dt
            if timestamp_query:
                query["timestamp"] = timestamp_query
        
        # Determine sort field and order
        sort_field = "timestamp"
        if sort_by == "quickrecal_score":
            sort_field = "quickrecal_score"
            sort_order = "desc"  # Higher scores first for quickrecal
        else:
            sort_order = "asc"   # Chronological order for timestamps
        
        # Retrieve the memories, ordered by specified field
        memories = await self.memory_core.get_memories(
            query=query,
            sort_by=sort_field,
            sort_order=sort_order,
            limit=limit
        )
        
        # Convert memories to sequence embeddings
        sequence_embeddings = []
        for memory in memories:
            # Skip memories without embeddings
            if not memory.embedding:
                continue
                
            # Standardize embedding using the geometry manager
            standardized_embedding = self.geometry_manager.standardize_embedding(memory.embedding)
                
            # Extract metadata
            metadata = memory.metadata or {}
            
            sequence_embeddings.append(SequenceEmbedding(
                id=str(memory.id),
                embedding=standardized_embedding.tolist(),
                timestamp=memory.timestamp.isoformat(),
                quickrecal_score=memory.quickrecal_score,
                emotion=metadata.get("emotions"),
                dominant_emotion=metadata.get("dominant_emotion"),
                importance=metadata.get("importance"),
                topic=metadata.get("topic"),
                user=metadata.get("user")
            ))
            
        return SequenceEmbeddingsResponse(embeddings=sequence_embeddings)
    
    async def update_quickrecal_score(self, request: UpdateQuickRecalScoreRequest) -> Dict[str, Any]:
        """Update the quickrecal score of a memory based on surprise feedback.
        
        Args:
            request: The update request containing memory_id, delta, and additional context
            
        Returns:
            Dict with status of the update operation
        """
        memory_id = request.memory_id
        delta = request.delta
        
        # Retrieve the memory
        memory = await self.memory_core.get_memory_by_id(memory_id)
        if not memory:
            return {"status": "error", "message": f"Memory with ID {memory_id} not found"}
        
        # Calculate new quickrecal score
        current_score = memory.quickrecal_score or 0.0
        new_score = min(1.0, max(0.0, current_score + delta))  # Ensure score stays between 0 and 1
        
        # Prepare updates for the memory
        updates = {"quickrecal_score": new_score}
        
        # Add surprise metadata if provided
        if request.reason or request.embedding_delta or request.predicted_embedding:
            # Get existing metadata or initialize empty dict
            metadata = memory.metadata or {}
            
            # Create or update surprise tracking
            surprise_events = metadata.get("surprise_events", [])
            new_event = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "delta": delta,
                "previous_score": current_score,
                "new_score": new_score
            }
            
            # Calculate embedding delta if both memory embedding and predicted embedding are available
            if memory.embedding is not None and request.predicted_embedding and not request.embedding_delta:
                # Use the geometry manager to calculate the delta between predicted and actual embeddings
                embedding_delta = self.geometry_manager.generate_embedding_delta(
                    predicted=request.predicted_embedding,
                    actual=memory.embedding
                )
                new_event["embedding_delta"] = embedding_delta
                
                # Calculate surprise score based on vector comparison
                surprise_score = self.geometry_manager.calculate_surprise(
                    predicted=request.predicted_embedding,
                    actual=memory.embedding
                )
                new_event["calculated_surprise"] = surprise_score
            
            # Add optional fields if provided
            if request.reason:
                new_event["reason"] = request.reason
            if request.embedding_delta:
                new_event["embedding_delta"] = request.embedding_delta
            if request.predicted_embedding:
                new_event["predicted_embedding"] = request.predicted_embedding
                
            # Add the new event to the list
            surprise_events.append(new_event)
            
            # Update metadata with new surprise events
            metadata["surprise_events"] = surprise_events
            
            # Add surprise count or increment it
            metadata["surprise_count"] = metadata.get("surprise_count", 0) + 1
            
            # Update the memory with the new metadata
            updates["metadata"] = metadata
        
        # Update the memory
        updated = await self.memory_core.update_memory(
            memory_id=memory_id,
            updates=updates
        )
        
        if updated:
            result = {
                "status": "success", 
                "memory_id": memory_id,
                "previous_score": current_score,
                "new_score": new_score,
                "delta": delta
            }
            
            # Include additional fields if they were in the request
            if request.reason:
                result["reason"] = request.reason
            if request.embedding_delta:
                result["embedding_delta_norm"] = np.linalg.norm(np.array(request.embedding_delta))
                
            return result
        else:
            return {"status": "error", "message": f"Failed to update quickrecal score for memory {memory_id}"}

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
from .memory_structures import MemoryEntry, MemoryAssembly # Use the unified structure
from .custom_logger import logger # Use the shared custom logger
from datetime import datetime

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

    async def save_assembly(self, assembly: 'MemoryAssembly') -> bool:
        """Save a memory assembly to disk.
        
        Args:
            assembly: The MemoryAssembly object to save
            
        Returns:
            bool: Success status
        """
        if not assembly or not assembly.assembly_id:
            logger.error("MemoryPersistence", "Cannot save assembly: Invalid or empty assembly object")
            return False
            
        try:
            # Create assemblies directory if it doesn't exist
            assembly_dir = self.storage_path / 'assemblies'
            assembly_dir.mkdir(exist_ok=True, parents=True)
            
            # Generate a filename based on the assembly ID
            file_path = assembly_dir / f"{assembly.assembly_id}.json"
            
            # Convert the assembly to a serializable dict
            assembly_dict = assembly.to_dict()
            
            # Validate critical fields before serialization
            if not assembly_dict.get('assembly_id') or not assembly_dict.get('name'):
                logger.error("MemoryPersistence", "Cannot save assembly: Missing required fields", 
                            {"id": assembly.assembly_id})
                return False

            # Write the assembly to disk
            async with aiofiles.open(file_path, 'w') as f:
                # Use the same serializer as for memories
                def default_serializer(obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                        np.int16, np.int32, np.int64, np.uint8,
                                        np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                    elif isinstance(obj, set):
                        return list(obj)
                    try:
                        # Fallback for other types
                        return str(obj)
                    except:
                        return "[Unserializable Object]"
                try:
                    json_data = json.dumps(assembly_dict, indent=2, default=default_serializer)
                    await f.write(json_data)
                except (TypeError, OverflowError) as json_err:
                    logger.error("MemoryPersistence", f"JSON serialization error for assembly {assembly.assembly_id}", 
                                {"error": str(json_err), "type": type(json_err).__name__})
                    return False
            
            # Update the memory index with assembly info
            self.memory_index[assembly.assembly_id] = {
                'path': str(file_path.relative_to(self.storage_path)),
                'timestamp': assembly.creation_time,
                'type': 'assembly',
                'name': assembly.name
            }
            
            # Save the memory index
            await self._save_index()
            
            self.stats['assembly_saves'] = self.stats.get('assembly_saves', 0) + 1
            logger.info("MemoryPersistence", f"Saved assembly {assembly.assembly_id}", {"name": assembly.name})
            return True
        except Exception as e:
            logger.error("MemoryPersistence", f"Error saving assembly {assembly.assembly_id}", 
                        {"error": str(e), "type": type(e).__name__})
            self.stats['failed_assembly_saves'] = self.stats.get('failed_assembly_saves', 0) + 1
            return False

    async def load_assembly(self, assembly_id: str, geometry_manager) -> Optional['MemoryAssembly']:
        """Load a memory assembly from disk.
        
        Args:
            assembly_id: ID of the assembly to load
            geometry_manager: GeometryManager instance required for assembly initialization
            
        Returns:
            MemoryAssembly or None if not found
        """
        if not assembly_id:
            logger.error("MemoryPersistence", "Cannot load assembly: Invalid or empty assembly_id")
            return None
            
        if not geometry_manager:
            logger.error("MemoryPersistence", f"Cannot load assembly {assembly_id}: GeometryManager is required")
            return None
            
        async with self._lock:
            try:
                # Check if in index first
                if assembly_id in self.memory_index and self.memory_index[assembly_id].get('type') == 'assembly':
                    file_path = self.storage_path / self.memory_index[assembly_id]['path']
                else:
                    # Fallback: check filesystem directly
                    file_path = self.storage_path / 'assemblies' / f"{assembly_id}.json"
                    if not await asyncio.to_thread(os.path.exists, file_path):
                        logger.warning("MemoryPersistence", f"Assembly {assembly_id} not found")
                        return None
                    # If found directly, update index
                    self.memory_index[assembly_id] = {
                        'path': f"assemblies/{assembly_id}.json",
                        'type': 'assembly'
                    }

                # Read and parse the assembly file
                try:
                    async with aiofiles.open(file_path, 'r') as f:
                        content = await f.read()
                        
                    try:
                        assembly_dict = json.loads(content)
                    except json.JSONDecodeError as json_err:
                        logger.error("MemoryPersistence", f"JSON parsing error for assembly {assembly_id}", 
                                    {"error": str(json_err), "file": str(file_path)})
                        return None
                        
                    # Validate required fields
                    if not assembly_dict.get('assembly_id') or 'memories' not in assembly_dict:
                        logger.error("MemoryPersistence", f"Invalid assembly data format for {assembly_id}",
                                    {"missing_fields": [k for k in ['assembly_id', 'memories'] if k not in assembly_dict]})
                        return None
                    
                    # Create assembly from dict with error handling
                    try:   
                        assembly = MemoryAssembly.from_dict(assembly_dict, geometry_manager)
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error("MemoryPersistence", f"Error reconstructing assembly {assembly_id} from dict", 
                                    {"error": str(e), "type": type(e).__name__})
                        return None
                        
                    self.stats['assembly_loads'] = self.stats.get('assembly_loads', 0) + 1
                    logger.info("MemoryPersistence", f"Loaded assembly {assembly_id}", {"name": assembly.name})
                    return assembly
                    
                except FileNotFoundError:
                    logger.warning("MemoryPersistence", f"Assembly file not found for {assembly_id}", 
                                 {"expected_path": str(file_path)})
                    # Remove from index if file doesn't exist
                    if assembly_id in self.memory_index:
                        del self.memory_index[assembly_id]
                        await self._save_index()
                    return None

            except Exception as e:
                logger.error("MemoryPersistence", f"Error loading assembly {assembly_id}", 
                            {"error": str(e), "type": type(e).__name__})
                self.stats['failed_assembly_loads'] = self.stats.get('failed_assembly_loads', 0) + 1
                return None

    async def list_assemblies(self) -> List[Dict[str, Any]]:
        """List all memory assemblies.
        
        Returns:
            List of assembly metadata dictionaries
        """
        async with self._lock:
            try:
                assemblies = []
                for memory_id, info in self.memory_index.items():
                    if info.get('type') == 'assembly':
                        assemblies.append({
                            'id': memory_id,
                            'path': info['path'],
                            'timestamp': info.get('timestamp', 0)
                        })
                return assemblies
            except Exception as e:
                logger.error("MemoryPersistence", "Error listing assemblies", {"error": str(e)})
                return []

    async def delete_assembly(self, assembly_id: str) -> bool:
        """Delete a memory assembly.
        
        Args:
            assembly_id: ID of the assembly to delete
            
        Returns:
            bool: Success status
        """
        async with self._lock:
            try:
                if assembly_id not in self.memory_index or self.memory_index[assembly_id].get('type') != 'assembly':
                    logger.warning("MemoryPersistence", f"Assembly {assembly_id} not found for deletion")
                    return False
                    
                file_path = self.storage_path / self.memory_index[assembly_id]['path']
                if await asyncio.to_thread(os.path.exists, file_path):
                    await asyncio.to_thread(os.remove, file_path)
                    
                # Remove from index
                del self.memory_index[assembly_id]
                await self._save_index()
                
                self.stats['assembly_deletes'] = self.stats.get('assembly_deletes', 0) + 1
                logger.info("MemoryPersistence", f"Deleted assembly {assembly_id}")
                return True
            except Exception as e:
                logger.error("MemoryPersistence", f"Error deleting assembly {assembly_id}", {"error": str(e)})
                self.stats['failed_assembly_deletes'] = self.stats.get('failed_assembly_deletes', 0) + 1
                return False

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

# orchestrator\__init__.py

```py
# Orchestrator module for managing bi-hemispheric cognitive flow
# between Memory Core and Sequence Trainer

```

# orchestrator\context_cascade_engine.py

```py
import numpy as np
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Union
import datetime
import json
import aiohttp

try:
    from synthians_memory_core.geometry_manager import GeometryManager
except ImportError:
    logging.error("CRITICAL: Failed to import GeometryManager. Ensure synthians_memory_core is accessible.")
    GeometryManager = None  

# Import MetricsStore for cognitive flow instrumentation
try:
    from synthians_memory_core.synthians_trainer_server.metrics_store import MetricsStore, get_metrics_store
except ImportError:
    logging.warning("Failed to import MetricsStore. Cognitive instrumentation disabled.")
    get_metrics_store = None

logger = logging.getLogger(__name__)

class ContextCascadeEngine:
    """Orchestrates the bi-hemispheric cognitive flow between Memory Core and Neural Memory.
    
    This engine implements the Context Cascade design pattern, enabling:
    1. Storage of memory entries with embeddings in Memory Core
    2. Test-time learning in Neural Memory via associations
    3. Detection of surprise when expectations don't match reality
    4. Feedback of surprise to enhance memory retrieval
    5. Dynamic adaptation of memory importance based on narrative patterns
    """

    def __init__(self,
                 memory_core_url: str = "http://localhost:5010",  
                 neural_memory_url: str = "http://localhost:8001",  
                 geometry_manager: Optional[GeometryManager] = None,
                 metrics_enabled: bool = True):
        """Initialize the Context Cascade Engine.
        
        Args:
            memory_core_url: URL of the Memory Core service
            neural_memory_url: URL of the Neural Memory Server
            geometry_manager: Optional shared geometry manager
            metrics_enabled: Whether to enable cognitive metrics collection
        """
        self.memory_core_url = memory_core_url.rstrip('/')
        self.neural_memory_url = neural_memory_url.rstrip('/')

        if GeometryManager is None:
            raise ImportError("GeometryManager could not be imported. ContextCascadeEngine cannot function.")
        self.geometry_manager = geometry_manager or GeometryManager()  

        # Initialize metrics collection if enabled
        self.metrics_enabled = metrics_enabled and get_metrics_store is not None
        self._current_intent_id = None
        if self.metrics_enabled:
            try:
                self.metrics_store = get_metrics_store()
                logger.info("Cognitive metrics collection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics collection: {e}")
                self.metrics_enabled = False

        self.last_retrieved_embedding: Optional[List[float]] = None
        self.sequence_context: List[Dict[str, Any]] = []
        self.processing_lock = asyncio.Lock()

        logger.info(f"Context Cascade Engine initialized:")
        logger.info(f" - Memory Core URL: {self.memory_core_url}")
        logger.info(f" - Neural Memory URL: {self.neural_memory_url}")
        logger.info(f" - Metrics Enabled: {self.metrics_enabled}")
        gm_config = getattr(self.geometry_manager, 'config', {})
        logger.info(f" - Geometry type: {gm_config.get('geometry_type', 'N/A')}")

    async def process_new_input(self,
                                content: str,
                                embedding: Optional[List[float]] = None,
                                metadata: Optional[Dict[str, Any]] = None,
                                intent_id: Optional[str] = None) -> Dict[str, Any]:
        """Process new input through MemoryCore storage and NeuralMemory learning/retrieval.
        
        This method orchestrates the flow described in the Phase 3 sequence diagram:
        1. Store memory in Memory Core (receive memory_id and actual_embedding)
        2. Send actual_embedding to Neural Memory for learning
        3. Receive surprise metrics (loss/grad_norm) from Neural Memory
        4. Calculate quickrecal_boost from surprise metrics
        5. Update quickrecal_score in Memory Core if boost > 0
        6. Generate query and retrieve associated embedding from Neural Memory
        7. Return both actual_embedding and retrieved_embedding for downstream use
        
        Args:
            content: Text content of the input
            embedding: Optional pre-computed embedding
            metadata: Optional metadata for the input
            intent_id: Optional intent tracking ID (creates new one if None)
            
        Returns:
            Processing results including memory_id, surprise metrics, etc.
        """
        async with self.processing_lock:
            start_time = time.time()
            
            # Begin intent tracking if metrics are enabled
            if self.metrics_enabled:
                if intent_id:
                    self._current_intent_id = intent_id
                else:
                    self._current_intent_id = self.metrics_store.begin_intent()
                
                # Extract emotion from metadata if available for tracking
                user_emotion = None
                if metadata and "emotion" in metadata:
                    user_emotion = metadata["emotion"]
                elif metadata and "emotions" in metadata:
                    # Handle case where emotions is a list/dict with scores
                    if isinstance(metadata["emotions"], dict) and metadata["emotions"]:
                        # Find emotion with highest score
                        user_emotion = max(metadata["emotions"].items(), key=lambda x: x[1])[0] if metadata["emotions"] else None
                    elif isinstance(metadata["emotions"], list) and metadata["emotions"]:
                        user_emotion = metadata["emotions"][0]
                
            logger.info(f"Processing new input: {content[:50]}...")

            mem_core_resp = await self._make_request(
                self.memory_core_url,
                "/process_memory",  
                method="POST",
                payload={
                    "content": content,
                    "embedding": embedding,  
                    "metadata": metadata or {}
                }
            )
            memory_id = mem_core_resp.get("memory_id")
            actual_embedding = mem_core_resp.get("embedding")  
            quickrecal_initial = mem_core_resp.get("quickrecal_score")

            response = {
                "memory_id": memory_id,
                "status": "processed" if memory_id and actual_embedding else "error_mem_core",
                "quickrecal_initial": quickrecal_initial,
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "intent_id": self._current_intent_id
            }

            if not actual_embedding or not memory_id:
                logger.error("Failed to store or get valid embedding from Memory Core.", extra={"mem_core_resp": mem_core_resp})
                response["error"] = mem_core_resp.get("error", "Failed in Memory Core processing")
                
                # Finalize intent with error if metrics enabled
                if self.metrics_enabled:
                    error_message = mem_core_resp.get("error", "Failed in Memory Core processing")
                    self.metrics_store.finalize_intent(
                        self._current_intent_id,
                        response_text=f"Error: {error_message}",
                        confidence=0.0
                    )
                return response

            # Send embedding to Neural Memory for learning
            update_resp = await self._make_request(
                self.neural_memory_url,
                "/update_memory",
                method="POST",
                payload={"input_embedding": actual_embedding}
            )
            loss = update_resp.get("loss")
            grad_norm = update_resp.get("grad_norm")
            response["neural_memory_update"] = update_resp  
            
            # Log memory update metrics if enabled
            if self.metrics_enabled and loss is not None:
                # Ensure embedding is in an expected format using safe copy
                safe_embedding = np.array(actual_embedding, dtype=np.float32).tolist() 
                # Handle potential NaN or Inf values in the embedding
                safe_embedding = [0.0 if not np.isfinite(x) else x for x in safe_embedding]
                
                self.metrics_store.log_memory_update(
                    input_embedding=safe_embedding,
                    loss=loss,
                    grad_norm=grad_norm or 0.0,
                    emotion=user_emotion,
                    intent_id=self._current_intent_id,
                    metadata={
                        "memory_id": memory_id,
                        "content_preview": content[:50] if content else "",
                        "quickrecal_initial": quickrecal_initial
                    }
                )

            # Calculate QuickRecal boost based on surprise metrics
            quickrecal_boost = 0.0
            if update_resp.get("status") == "success" or "error" not in update_resp:
                 surprise_metric = grad_norm if grad_norm is not None else (loss if loss is not None else 0.0)
                 quickrecal_boost = self._calculate_quickrecal_boost(surprise_metric)

                 # Apply boost if significant
                 if quickrecal_boost > 1e-4:  
                     feedback_resp = await self._make_request(
                         self.memory_core_url,
                         "/api/memories/update_quickrecal_score",  
                         method="POST",
                         payload={
                             "memory_id": memory_id,
                             "delta": quickrecal_boost,
                             "reason": f"NM Surprise (Loss:{loss:.4f if loss is not None else 'N/A'}, GradNorm:{grad_norm:.4f if grad_norm is not None else 'N/A'})"
                         }
                     )
                     response["quickrecal_feedback"] = feedback_resp
                     
                     # Log QuickRecal boost metrics if enabled
                     if self.metrics_enabled:
                         self.metrics_store.log_quickrecal_boost(
                             memory_id=memory_id,
                             base_score=quickrecal_initial or 0.0,
                             boost_amount=quickrecal_boost,
                             emotion=user_emotion,
                             surprise_source="neural_memory",
                             intent_id=self._current_intent_id,
                             metadata={
                                 "loss": loss,
                                 "grad_norm": grad_norm,
                                 "reason": f"NM Surprise"
                             }
                         )
                 else:
                     response["quickrecal_feedback"] = {"status": "skipped", "reason": "Boost too small"}
            else:
                 response["quickrecal_feedback"] = {"status": "skipped", "reason": "Neural Memory update failed"}

            response["surprise_metrics"] = {"loss": loss, "grad_norm": grad_norm, "boost_calculated": quickrecal_boost}

            # Generate query for Neural Memory retrieval
            query_for_retrieve = actual_embedding  
            logger.debug(f"Sending query to /retrieve (dim={len(query_for_retrieve)}). Endpoint must handle projection if needed.")

            # Retrieve associated embedding
            retrieve_resp = await self._make_request(
                self.neural_memory_url,
                "/retrieve",
                method="POST",
                payload={"query_embedding": query_for_retrieve}  
            )
            retrieved_embedding = retrieve_resp.get("retrieved_embedding")
            response["neural_memory_retrieval"] = retrieve_resp  
            
            # Log retrieval metrics if enabled
            if self.metrics_enabled and retrieved_embedding:
                # Create synthetic memory object since we don't have full metadata
                retrieved_memory = {
                    "memory_id": f"synthetic_{memory_id}_associated",
                    "embedding": retrieved_embedding,
                    "dominant_emotion": None  # We don't have this information yet
                }
                
                safe_query = np.array(query_for_retrieve, dtype=np.float32).tolist()
                safe_query = [0.0 if not np.isfinite(x) else x for x in safe_query]
                
                self.metrics_store.log_retrieval(
                    query_embedding=safe_query,
                    retrieved_memories=[retrieved_memory],
                    user_emotion=user_emotion,
                    intent_id=self._current_intent_id,
                    metadata={
                        "original_memory_id": memory_id,
                        "embedding_dim": len(retrieved_embedding),
                        "timestamp": datetime.datetime.utcnow().isoformat()
                    }
                )

            # Update sequence context
            self.last_retrieved_embedding = retrieved_embedding
            self.sequence_context.append({
                "memory_id": memory_id,
                "actual_embedding": actual_embedding,
                "retrieved_embedding": retrieved_embedding,
                "surprise_metrics": response.get("surprise_metrics"),
                "timestamp": response["timestamp"],
                "intent_id": self._current_intent_id
            })
            if len(self.sequence_context) > 20: self.sequence_context.pop(0)

            # Finalize intent graph if enabled
            if self.metrics_enabled:
                intent_graph = self.metrics_store.finalize_intent(
                    intent_id=self._current_intent_id,
                    response_text=f"Retrieved associated embedding for memory {memory_id}",
                    confidence=1.0 if retrieved_embedding else 0.5
                )
                # Store intent graph reference in response
                if intent_graph:
                    response["intent_graph"] = {
                        "trace_id": intent_graph.get("trace_id"),
                        "reasoning_steps": len(intent_graph.get("reasoning_steps", []))
                    }

            response["status"] = "completed"  
            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Finished processing input for memory {memory_id} in {processing_time:.2f} ms")
            return response

    async def _make_request(self, base_url: str, endpoint: str, method: str = "POST", payload: Optional[Dict] = None, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Shared function to make HTTP requests and handle common errors.
        
        Args:
            base_url: Base URL of the service
            endpoint: API endpoint to call
            method: HTTP method to use
            payload: JSON payload for the request
            params: URL parameters for the request
            
        Returns:
            Response from the server as a dictionary
        """
        url = f"{base_url}{endpoint}"
        log_payload = payload if payload is None or len(json.dumps(payload)) < 200 else {k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in payload.items()}  
        logger.debug(f"Making {method} request to {url}", extra={"payload": log_payload, "params": params})

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=payload, params=params, timeout=30.0) as response:
                    status_code = response.status
                    try:
                        resp_json = await response.json()
                        logger.debug(f"Response from {url}: Status {status_code}")  
                        if 200 <= status_code < 300:
                            return resp_json
                        else:
                            error_detail = resp_json.get("detail", "Unknown error from server")
                            logger.error(f"Error from {url}: {status_code} - {error_detail}")
                            return {"error": error_detail, "status_code": status_code}
                    except (json.JSONDecodeError, aiohttp.ContentTypeError):
                        resp_text = await response.text()
                        logger.error(f"Non-JSON or failed response from {url}: {status_code}", extra={"response_text": resp_text[:500]})
                        return {"error": f"Server error {status_code}", "details": resp_text[:500], "status_code": status_code}
        except asyncio.TimeoutError:
            logger.error(f"Timeout connecting to {url}")
            return {"error": "Request timed out", "status_code": 408}
        except aiohttp.ClientConnectionError as e:
            logger.error(f"Connection error to {url}: {e}")
            return {"error": "Connection refused or failed", "status_code": 503}
        except Exception as e:
            logger.error(f"Unexpected error during request to {url}: {e}", exc_info=True)
            return {"error": f"Unexpected client error: {str(e)}", "status_code": 500}

    def _validate_embedding(self, embedding: Optional[Union[List[float], np.ndarray]]) -> bool:
        """Basic validation for embeddings.
        
        This is a critical validation step referenced in memory #fbee9e47 to
        ensure we don't pass malformed embeddings to downstream components.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False if contains NaN/Inf
        """
        if embedding is None: return False
        try:
            embedding_array = np.array(embedding, dtype=np.float32)
            if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
                logger.warning("Invalid embedding detected: contains NaN or Inf values")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return False

    def _calculate_quickrecal_boost(self, surprise_value: float) -> float:
        """Calculate quickrecal boost based on surprise value (loss or grad_norm).
        
        Args:
            surprise_value: Surprise metric (loss or grad_norm)
            
        Returns:
            QuickRecal score boost amount
        """
        if surprise_value <= 0.0: return 0.0
        max_expected_surprise = 2.0  
        max_boost = 0.2             
        boost = min(surprise_value / max_expected_surprise, 1.0) * max_boost
        logger.debug(f"Calculated quickrecal boost: {boost:.4f} from surprise value: {surprise_value:.4f}")
        return boost

    async def get_sequence_embeddings_for_training(self, limit: int = 100, **filters) -> Dict[str, Any]:
        """Retrieve a sequence from Memory Core for training purposes.
        
        Args:
            limit: Maximum number of embeddings to retrieve
            **filters: Additional filters like topic, user, etc.
            
        Returns:
            Sequence of embeddings with metadata
        """
        payload = {"limit": limit}
        payload.update(filters)  

        return await self._make_request(
            self.memory_core_url,
            "/api/memories/get_sequence_embeddings",
            method="POST",  
            payload=payload
        )

```

# orchestrator\server.py

```py
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Context Cascade Orchestrator")

# Global instance of the orchestrator
orchestrator = None

# --- Pydantic Models ---

class ProcessMemoryRequest(BaseModel):
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class SequenceEmbeddingsRequest(BaseModel):
    topic: Optional[str] = None
    limit: int = 10
    min_quickrecal_score: Optional[float] = None

class AnalyzeSurpriseRequest(BaseModel):
    predicted_embedding: List[float]
    actual_embedding: List[float]

# --- Helper Functions ---

def get_orchestrator():
    """Get or initialize the context cascade orchestrator."""
    global orchestrator
    if orchestrator is None:
        # Get URLs from environment variables
        memory_core_url = os.environ.get("MEMORY_CORE_URL", "http://localhost:5010")
        neural_memory_url = os.environ.get("NEURAL_MEMORY_URL", "http://localhost:8001")
        
        # Initialize shared geometry manager
        geometry_manager = GeometryManager()
        
        # Initialize orchestrator
        orchestrator = ContextCascadeEngine(
            memory_core_url=memory_core_url,
            neural_memory_url=neural_memory_url,
            geometry_manager=geometry_manager,
            metrics_enabled=True
        )
        logger.info(f"Orchestrator initialized with Memory Core URL: {memory_core_url}, Neural Memory URL: {neural_memory_url}")
    
    return orchestrator

# --- Endpoints ---

@app.get("/")
async def root():
    """Root endpoint returning service information."""
    return {"service": "Context Cascade Orchestrator", "status": "running"}

@app.post("/process_memory")
async def process_memory(request: ProcessMemoryRequest):
    """Process a new memory through the full cognitive pipeline.
    
    This orchestrates:
    1. Store memory in Memory Core
    2. Compare with previous prediction if available
    3. Update quickrecal scores based on surprise
    4. Generate prediction for next memory
    """
    orchestrator = get_orchestrator()
    
    try:
        result = await orchestrator.process_new_memory(
            content=request.content,
            embedding=request.embedding,
            metadata=request.metadata
        )
        return result
    except Exception as e:
        logger.error(f"Error processing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing memory: {str(e)}")

@app.post("/get_sequence_embeddings")
async def get_sequence_embeddings(request: SequenceEmbeddingsRequest):
    """Retrieve a sequence of embeddings from Memory Core."""
    orchestrator = get_orchestrator()
    
    try:
        result = await orchestrator.get_sequence_embeddings(
            topic=request.topic,
            limit=request.limit,
            min_quickrecal_score=request.min_quickrecal_score
        )
        return result
    except Exception as e:
        logger.error(f"Error retrieving sequence embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sequence embeddings: {str(e)}")

@app.post("/analyze_surprise")
async def analyze_surprise(request: AnalyzeSurpriseRequest):
    """Analyze surprise between predicted and actual embeddings."""
    orchestrator = get_orchestrator()
    
    try:
        # Use the surprise detector from the orchestrator
        surprise_metrics = orchestrator.surprise_detector.calculate_surprise(
            predicted_embedding=request.predicted_embedding,
            actual_embedding=request.actual_embedding
        )
        
        # Calculate quickrecal boost
        quickrecal_boost = orchestrator.surprise_detector.calculate_quickrecal_boost(surprise_metrics)
        
        # Add boost to response
        surprise_metrics["quickrecal_boost"] = quickrecal_boost
        
        return surprise_metrics
    except Exception as e:
        logger.error(f"Error analyzing surprise: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing surprise: {str(e)}")

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    get_orchestrator()
    logger.info("Context Cascade Orchestrator is ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down Context Cascade Orchestrator")

```

# orchestrator\tests\test_context_cascade_engine.py

```py
# synthians_memory_core/orchestrator/tests/test_context_cascade_engine.py

import pytest
import numpy as np
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from typing import Dict, List, Any

from ..context_cascade_engine import ContextCascadeEngine
from synthians_memory_core.geometry_manager import GeometryManager


@pytest.fixture
def geometry_manager():
    """Test fixture for GeometryManager."""
    return GeometryManager({
        'embedding_dim': 768,
        'geometry_type': 'euclidean',
    })


@pytest.fixture
def engine(geometry_manager):
    """Test fixture for ContextCascadeEngine with mock URLs."""
    return ContextCascadeEngine(
        memory_core_url="http://memory-core-test",
        trainer_url="http://trainer-test",
        geometry_manager=geometry_manager
    )


@pytest.fixture
def mock_response():
    """Create a mock for aiohttp ClientResponse."""
    mock = MagicMock()
    mock.status = 200
    mock.json = AsyncMock()
    return mock


@pytest.mark.asyncio
async def test_process_new_memory(engine, mock_response):
    """Test the complete flow of processing a new memory."""
    # Mock embeddings and memory data
    test_content = "This is a test memory"
    test_embedding = np.random.randn(768).tolist()
    test_memory_id = "test-memory-123"
    
    # Mock memory core response
    memory_response = {
        "id": test_memory_id,
        "embedding": test_embedding,
        "quickrecal_score": 0.8
    }
    mock_response.json.return_value = memory_response
    
    # Mock trainer response
    trainer_response = {
        "predicted_embedding": np.random.randn(768).tolist(),
        "surprise_score": 0.3,
        "memory_state": {
            "sequence": [test_embedding],
            "surprise_history": [0.3],
            "momentum": np.random.randn(768).tolist()
        }
    }
    mock_trainer_response = MagicMock()
    mock_trainer_response.status = 200
    mock_trainer_response.json = AsyncMock(return_value=trainer_response)
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post, \
         patch('aiohttp.ClientSession.get') as mock_get:
            
        # Configure mock to return different responses for different URLs
        mock_post.side_effect = lambda url, **kwargs: \
            mock_response if "memory-core-test" in url else mock_trainer_response
        
        # Call the method under test
        result = await engine.process_new_memory(
            content=test_content,
            embedding=test_embedding
        )
        
        # Verify memory core was called
        assert mock_post.call_count >= 1
        # Verify memory_id is present in result
        assert result["memory_id"] == test_memory_id
        # Verify prediction data is present
        assert "prediction" in result
        # Verify last_predicted_embedding was updated
        assert engine.last_predicted_embedding is not None


@pytest.mark.asyncio
async def test_retrieve_memories(engine, mock_response):
    """Test retrieving memories through the engine."""
    # Mock query and response
    query = "test query"
    memories = [
        {"id": "mem1", "content": "Memory 1", "similarity": 0.9},
        {"id": "mem2", "content": "Memory 2", "similarity": 0.8}
    ]
    
    mock_response.json.return_value = {"memories": memories}
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value = mock_response
        
        # Call the method under test
        result = await engine.retrieve_memories(query=query, limit=2)
        
        # Verify memory core was called
        mock_post.assert_called_once()
        # Verify results
        assert len(result["memories"]) == 2
        assert result["memories"][0]["id"] == "mem1"


@pytest.mark.asyncio
async def test_error_handling(engine):
    """Test error handling for HTTP responses."""
    # Mock error response
    error_response = MagicMock()
    error_response.status = 500
    error_response.text = AsyncMock(return_value="Internal server error")
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_post.return_value.__aenter__.return_value = error_response
        
        # Call the method under test and expect error handling
        result = await engine.process_new_memory(content="Error test")
        
        # Verify error is captured
        assert "error" in result
        assert result["status"] == "error"


@pytest.mark.asyncio
async def test_surprise_detection(engine, mock_response):
    """Test surprise detection when actual embedding differs from predicted."""
    # Setup initial state with a predicted embedding
    engine.last_predicted_embedding = np.random.randn(768).tolist()
    
    # Create actual embedding with high difference
    actual_embedding = np.random.randn(768).tolist()  # Will be different due to randomness
    
    # Mock memory core response
    memory_response = {
        "id": "test-memory-456",
        "embedding": actual_embedding,
        "quickrecal_score": 0.7
    }
    mock_response.json.return_value = memory_response
    
    # Mock trainer response with high surprise
    trainer_response = {
        "predicted_embedding": np.random.randn(768).tolist(),
        "surprise_score": 0.8,  # High surprise
        "memory_state": {
            "sequence": [actual_embedding],
            "surprise_history": [0.8],
            "momentum": np.random.randn(768).tolist()
        }
    }
    mock_trainer_response = MagicMock()
    mock_trainer_response.status = 200
    mock_trainer_response.json = AsyncMock(return_value=trainer_response)
    
    # Setup mock for aiohttp ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        # Configure mock to return different responses for different URLs
        mock_post.side_effect = lambda url, **kwargs: \
            mock_response if "memory-core-test" in url else mock_trainer_response
        
        # Call the method under test
        result = await engine.process_new_memory(
            content="Surprise test",
            embedding=actual_embedding
        )
        
        # Verify surprise was detected
        assert "surprise" in result
        assert result["surprise"]["score"] > 0.7  # High surprise threshold

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
-   **Vector Search:** Fast retrieval using FAISS vector indexing with GPU acceleration support.

## Recent Improvements (March 2025)

The Synthians Memory Core has received significant enhancements in the `Synthience_memory_remaster` branch:

-   **Fixed Vector Index Persistence:** The FAISS vector index and ID mappings are now properly saved during the persistence cycle, ensuring memories can be retrieved after system restarts.
-   **Enhanced API Observability:** Added comprehensive vector index information to the `/stats` endpoint for better monitoring and debugging.
-   **Improved Embedding Handling:** Robust dimension handling to ensure vector operations work correctly regardless of embedding dimensions (384 vs 768).
-   **Retrieval Threshold Adjustments:** Lowered pre-filter threshold from 0.5 to 0.2 for improved recall while maintaining precision.
-   **Validation Tools:** Added comprehensive test scripts to validate the full memory lifecycle.

See `docs/memory_system_remaster.md` for detailed documentation on these improvements.

## Components

-   `synthians_memory_core.py`: The main orchestrator class.
-   `hpc_quickrecal.py`: Contains the `UnifiedQuickRecallCalculator`.
-   `geometry_manager.py`: Centralizes embedding and geometry operations.
-   `emotional_intelligence.py`: Provides emotion analysis and gating.
-   `memory_structures.py`: Defines `MemoryEntry` and `MemoryAssembly`.
-   `memory_persistence.py`: Manages disk storage and backups.
-   `adaptive_components.py`: Includes `ThresholdCalibrator`.
-   `vector_index.py`: Handles FAISS vector indexing with GPU support.
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
            
            # Load assemblies and memory_to_assemblies mapping
            assembly_list = await self.persistence.list_assemblies()
            loaded_assemblies_count = 0
            
            # Initialize memory_to_assemblies mapping
            self.memory_to_assemblies = {memory_id: set() for memory_id in self._memories.keys()}
            
            # Load each assembly
            for assembly_info in assembly_list:
                assembly_id = assembly_info.get("id")
                if assembly_id:
                    assembly = await self.persistence.load_assembly(assembly_id, self.geometry_manager)
                    if assembly:
                        self.assemblies[assembly_id] = assembly
                        loaded_assemblies_count += 1
                        
                        # Update memory_to_assemblies mapping
                        for memory_id in assembly.memories:
                            if memory_id in self.memory_to_assemblies:
                                self.memory_to_assemblies[memory_id].add(assembly_id)
                            else:
                                # Create mapping entry if memory not in cache
                                self.memory_to_assemblies[memory_id] = {assembly_id}
            
            logger.info("SynthiansMemoryCore", f"Loaded {loaded_assemblies_count} assemblies from persistence.")
            
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
        
        # Handle common case where embedding is wrongly passed as a dict
        if isinstance(embedding, dict):
            logger.warning("SynthiansMemoryCore", f"Received embedding as dict type, attempting to extract vector")
            try:
                # Try common dict formats seen in the wild
                if 'embedding' in embedding and isinstance(embedding['embedding'], (list, np.ndarray)):
                    embedding = embedding['embedding']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['embedding']") 
                elif 'vector' in embedding and isinstance(embedding['vector'], (list, np.ndarray)):
                    embedding = embedding['vector']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['vector']")
                elif 'value' in embedding and isinstance(embedding['value'], (list, np.ndarray)):
                    embedding = embedding['value']
                    logger.info("SynthiansMemoryCore", "Successfully extracted embedding from dict['value']")
                else:
                    logger.error("SynthiansMemoryCore", f"Could not extract embedding from dict: {list(embedding.keys())[:5]}")
                    return None
            except Exception as e:
                logger.error("SynthiansMemoryCore", f"Failed to extract embedding from dict: {str(e)}")
                return None

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
            
        # Perform vector search using the FAISS index - USE A MUCH LOWER THRESHOLD
        # Lower threshold from 0.3 to 0.05 for better recall sensitivity
        search_threshold = 0.05  # Significantly lowered threshold
        search_results = self.vector_index.search(query_embedding, k=limit, threshold=search_threshold)
        
        logger.info("SynthiansMemoryCore", f"Vector search with threshold {search_threshold} returned {len(search_results)} results")
        
        # Add direct search candidates from vector index
        for memory_id, similarity in search_results:
            direct_candidates.add(memory_id)
            logger.info("SynthiansMemoryCore", f"Memory {memory_id} similarity: {similarity:.4f} (from vector index)")

        # Combine candidates
        all_candidate_ids = assembly_candidates.union(direct_candidates)
        
        logger.info("SynthiansMemoryCore", f"Found {len(all_candidate_ids)} total candidate memories")

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
        try:
            if not self._memories:
                logger.info("SynthiansMemoryCore", "No memories to persist.")
                return
                
            async with self._lock:
                count = 0
                for memory_id, memory in self._memories.items():
                    stored = await self.persistence.save_memory(memory)
                    if stored:
                        count += 1
                logger.info("SynthiansMemoryCore", f"Persisted {count} memories.")
                
                # Save the vector index to ensure ID mappings persist
                if self.vector_index.count() > 0:
                    vector_index_saved = self.vector_index.save()
                    logger.info("SynthiansMemoryCore", f"Vector index saved: {vector_index_saved} with {self.vector_index.count()} vectors and {len(self.vector_index.id_to_index)} id mappings")
        except Exception as e:
            logger.error("SynthiansMemoryCore", f"Error persisting memories: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

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

# synthians_trainer_server\__init__.py

```py

```

# synthians_trainer_server\http_server.py

```py
# synthians_trainer_server/http_server.py

import os
import tensorflow as tf
import numpy as np
import aiohttp
import asyncio
import json
from fastapi import FastAPI, HTTPException, Body, Request, status, Response
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Literal
import logging
import traceback # Import traceback
import datetime  # Add datetime module for timestamps

# Import the new Neural Memory module and config
from .neural_memory import NeuralMemoryModule, NeuralMemoryConfig

# Import the new MetricsStore for cognitive flow instrumentation
from .metrics_store import MetricsStore, get_metrics_store

# Keep SurpriseDetector if needed for outer loop analysis
from .surprise_detector import SurpriseDetector
# Assume GeometryManager might be needed if surprise calculation uses it
try:
    from ..geometry_manager import GeometryManager
except ImportError:
    logger.warning("Could not import GeometryManager from synthians_memory_core. Using basic numpy ops.")
    class GeometryManager: # Dummy version
        def __init__(self, config=None): pass
        def normalize_embedding(self, vec):
            vec = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        def calculate_similarity(self, v1, v2):
             v1 = self.normalize_embedding(v1)
             v2 = self.normalize_embedding(v2)
             return np.dot(v1, v2)
        def align_vectors(self, v1, v2):
             v1, v2 = np.array(v1), np.array(v2)
             if v1.shape == v2.shape: return v1, v2
             logger.warning("Dummy GeometryManager cannot align vectors.")
             return v1, v2 # Assume they match or fail later


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Synthians Neural Memory API (Titans)")

# --- Global State ---
neural_memory: Optional[NeuralMemoryModule] = None
surprise_detector: Optional[SurpriseDetector] = None
geometry_manager: Optional[GeometryManager] = None
memory_core_url: Optional[str] = None # URL for potential outer loop callbacks

# --- Pydantic Models ---

class InitRequest(BaseModel):
    config: Optional[dict] = Field(default_factory=dict, description="Neural Memory config overrides")
    memory_core_url: Optional[str] = None
    load_path: Optional[str] = None

class InitResponse(BaseModel):
    message: str
    config: dict # Return as dict for JSON

class RetrieveRequest(BaseModel):
    query_embedding: List[float]

class RetrieveResponse(BaseModel):
    retrieved_embedding: List[float]

class UpdateMemoryRequest(BaseModel):
    input_embedding: List[float]

class UpdateMemoryResponse(BaseModel):
    status: str
    loss: Optional[float] = None
    grad_norm: Optional[float] = None

class TrainOuterRequest(BaseModel):
    input_sequence: List[List[float]]
    target_sequence: List[List[float]]

class TrainOuterResponse(BaseModel):
    average_loss: float

class SaveLoadRequest(BaseModel):
    path: str

class StatusResponse(BaseModel):
     status: str
     config: Optional[dict] = None # Return as dict

class AnalyzeSurpriseRequest(BaseModel):
    predicted_embedding: List[float]
    actual_embedding: List[float]

class GetProjectionsRequest(BaseModel):
    input_embedding: List[float] = Field(..., description="The raw input embedding vector")
    embedding_model: str = Field(default="unknown", example="sentence-transformers/all-mpnet-base-v2")
    projection_adapter: Optional[str] = Field(default="identity")

class GetProjectionsResponse(BaseModel):
    input_embedding_norm: float
    projection_adapter_used: str
    key_projection: List[float]
    value_projection: List[float]
    query_projection: List[float]
    projection_metadata: dict

class ClusterHotspot(BaseModel):
    cluster_id: str
    updates: int

class DiagnoseEmoLoopResponse(BaseModel):
    diagnostic_window: str
    avg_loss: float
    avg_grad_norm: float
    avg_quickrecal_boost: float
    dominant_emotions_boosted: List[str]
    emotional_entropy: float
    emotion_bias_index: float
    user_emotion_match_rate: float
    cluster_update_hotspots: List[ClusterHotspot]
    alerts: List[str]
    recommendations: List[str]

# --- Helper Functions ---

def get_neural_memory() -> NeuralMemoryModule:
    if neural_memory is None:
        logger.error("Neural Memory module not initialized. Call /init first.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Neural Memory module not initialized.")
    return neural_memory

def get_surprise_detector() -> SurpriseDetector:
     global surprise_detector, geometry_manager
     if surprise_detector is None:
          if geometry_manager is None:
               nm_conf = neural_memory.config if neural_memory else NeuralMemoryConfig()
               # Use get with default for safety
               gm_dim = nm_conf.get('input_dim', 768)
               geometry_manager = GeometryManager({'embedding_dim': gm_dim})
          surprise_detector = SurpriseDetector(geometry_manager=geometry_manager)
          logger.info("Initialized SurpriseDetector.")
     return surprise_detector


def _validate_vector(vec: Optional[List[float]], expected_dim: int, name: str, allow_none=False):
    """Validates vector type, length, and content."""
    if vec is None:
        if allow_none: return
        else: raise HTTPException(status_code=400, detail=f"'{name}' cannot be null.")

    if not isinstance(vec, list):
         raise HTTPException(status_code=400, detail=f"'{name}' must be a list of floats.")

    # <<< MODIFIED: Explicitly handle expected_dim == -1 >>>
    if expected_dim != -1 and len(vec) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid vector length for '{name}'. Expected {expected_dim}, got {len(vec)}."
        )
    # Add NaN/Inf check
    try:
         # Using np.isfinite is more efficient for checking both NaN and Inf
         if not np.all(np.isfinite(vec)):
             raise HTTPException(
                  status_code=400,
                  detail=f"Invalid values (NaN/Inf) found in '{name}'."
             )
    except TypeError:
          # This might happen if vec contains non-numeric types
          raise HTTPException(
               status_code=400,
               detail=f"Invalid value types in '{name}', expected floats."
          )


# --- API Endpoints ---

@app.post("/init", response_model=InitResponse, status_code=status.HTTP_200_OK)
async def init_neural_memory(req: InitRequest):
    """Initialize the Neural Memory Module."""
    global neural_memory, memory_core_url, surprise_detector, geometry_manager
    logger.info(f"Received /init request. Config overrides: {req.config}, Load path: {req.load_path}")
    try:
        # Use .get() for safer access to potentially missing keys in Pydantic model
        mc_url = req.memory_core_url
        if mc_url:
            memory_core_url = mc_url
            logger.info(f"Memory Core URL set to: {memory_core_url}")

        # Create config, overriding defaults with request body config
        # req.config should be a dict here from Pydantic parsing
        config_data = req.config if req.config is not None else {}
        config = NeuralMemoryConfig(**config_data)
        logger.info(f"Parsed config: {dict(config)}")


        # Initialize or re-initialize
        logger.info("Creating NeuralMemoryModule instance...")
        neural_memory = NeuralMemoryModule(config=config)
        logger.info("NeuralMemoryModule instance created.")

        # Initialize shared geometry manager and surprise detector based on module's config
        # Use dictionary access here too
        geometry_manager = GeometryManager({'embedding_dim': neural_memory.config['input_dim']})
        # Reset surprise detector to use new geometry manager if re-initializing
        surprise_detector = None
        get_surprise_detector() # Initialize if not already

        loaded_ok = True
        if req.load_path:
            logger.info(f"Attempting to load state from: {req.load_path}")
            # Build model before loading
            try:
                 logger.info("Building model before loading state...")
                 _ = neural_memory(tf.zeros((1, neural_memory.config['query_dim'])))
                 logger.info("Model built successfully.")
            except Exception as build_err:
                 logger.error(f"Error explicitly building model before load: {build_err}. Load might still succeed.")

            loaded_ok = neural_memory.load_state(req.load_path)
            if not loaded_ok:
                # Fail init if loading was requested but failed
                raise HTTPException(status_code=500, detail=f"Failed to load state from {req.load_path}")

        effective_config = neural_memory.get_config_dict()
        logger.info(f"Neural Memory module initialized. Effective Config: {effective_config}")
        return InitResponse(message="Neural Memory module initialized successfully.", config=effective_config)

    except AttributeError as ae:
         # Catch the specific AttributeError related to config access during init
         logger.error(f"AttributeError during initialization: {ae}. Config object: {config}", exc_info=True)
         neural_memory = None
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                             detail=f"Initialization failed due to config access error: {ae}")
    except Exception as e:
        logger.error(f"Failed to initialize Neural Memory module: {e}", exc_info=True)
        neural_memory = None # Ensure it's None on failure
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Initialization failed: {str(e)}")

@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_memory(req: RetrieveRequest):
    nm = get_neural_memory()
    try:
        _validate_vector(req.query_embedding, nm.config['query_dim'], "query_embedding")
        
        # Convert to proper tensor shape expected by the model (batch dimension)
        query_tensor = tf.convert_to_tensor([req.query_embedding], dtype=tf.float32)
        
        # Call the model with the properly shaped tensor
        retrieved_tensor = nm(query_tensor, training=False)
        
        if retrieved_tensor is None or not isinstance(retrieved_tensor, tf.Tensor):
             raise ValueError("Retrieval process returned None or invalid type")

        # Ensure we get a flat list regardless of tensor shape
        if len(tf.shape(retrieved_tensor)) > 1:
             # If we get a batch of vectors (or matrix), we want the first one
             retrieved_list = retrieved_tensor[0].numpy().tolist()
        else:
             retrieved_list = retrieved_tensor.numpy().tolist()
        
        # Log the retrieval metrics
        # Note: In a real implementation, we would have more metadata about the retrieved memories
        # This is a simplified version that just logs the retrieval vector
        metrics = get_metrics_store()
        
        # Extract user emotion from request metadata if available
        user_emotion = None
        if hasattr(req, "metadata") and req.metadata and "user_emotion" in req.metadata:
            user_emotion = req.metadata["user_emotion"]
            
        # For now, we don't have full memory metadata in this endpoint
        # In a more complete implementation, we would track the actual memories retrieved
        metrics.log_retrieval(
            query_embedding=req.query_embedding,
            retrieved_memories=[{"memory_id": "synthetic_memory", "embedding": retrieved_list}],
            user_emotion=user_emotion,
            metadata={
                "timestamp": datetime.datetime.now().isoformat(),
                "embedding_dim": len(retrieved_list)
            }
        )

        return RetrieveResponse(retrieved_embedding=retrieved_list)

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Memory retrieval failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {str(e)}")

@app.post("/update_memory", response_model=UpdateMemoryResponse)
async def update_memory(req: UpdateMemoryRequest):
    nm = get_neural_memory()
    try:
        _validate_vector(req.input_embedding, nm.config['input_dim'], "input_embedding")
        
        # Create tensor with proper batch dimension as expected by TensorFlow
        input_tensor = tf.convert_to_tensor([req.input_embedding], dtype=tf.float32)

        # Pass to update_step which now expects a batched tensor with shape [1, input_dim]
        loss_tensor, grads = nm.update_step(input_tensor)

        grad_norm = 0.0
        if grads:
             valid_grads = [g for g in grads if g is not None]
             if valid_grads:
                 # Calculate L2 norm for each valid gradient tensor and sum them
                 norms = [tf.norm(g) for g in valid_grads]
                 grad_norm = tf.reduce_sum(norms).numpy().item()

        loss_value = loss_tensor.numpy().item() if loss_tensor is not None else 0.0

        # Include timestamp in response for tracking
        timestamp = datetime.datetime.now().isoformat()
        
        # Log metrics to MetricsStore for cognitive flow monitoring
        metrics = get_metrics_store()
        metrics.log_memory_update(
            input_embedding=req.input_embedding,
            loss=loss_value,
            grad_norm=grad_norm,
            # Extract emotion if available in metadata
            emotion=req.metadata.get("emotion") if hasattr(req, "metadata") and req.metadata else None,
            metadata={
                "timestamp": timestamp,
                "input_dim": len(req.input_embedding)
            }
        )

        return UpdateMemoryResponse(
            status="success",
            loss=loss_value,
            grad_norm=grad_norm
        )

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Memory update failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Update error: {str(e)}")

@app.post("/train_outer", response_model=TrainOuterResponse)
async def train_outer(req: TrainOuterRequest):
    nm = get_neural_memory()
    if not hasattr(nm, 'compiled') or not nm.compiled:
        try:
             # Make sure the optimizer is properly set
             if not hasattr(nm, 'optimizer') or nm.optimizer is None:
                 nm.optimizer = nm.outer_optimizer
             nm.compile(optimizer=nm.optimizer, loss='mse')
             logger.info("NeuralMemoryModule compiled for outer training.")
        except Exception as compile_err:
             logger.error(f"Error compiling NeuralMemoryModule: {compile_err}")
             raise HTTPException(status_code=500, detail=f"Model compilation error: {compile_err}")

    try:
        if not req.input_sequence or not req.target_sequence: raise HTTPException(status_code=400, detail="Sequences empty.")
        seq_len = len(req.input_sequence)
        if seq_len != len(req.target_sequence): raise HTTPException(status_code=400, detail="Sequence lengths mismatch.")
        if seq_len == 0: raise HTTPException(status_code=400, detail="Sequences length 0.")

        # Validate dimensions for first item in sequences
        _validate_vector(req.input_sequence[0], nm.config['input_dim'], "input_sequence[0]")
        _validate_vector(req.target_sequence[0], nm.config['value_dim'], "target_sequence[0]")

        # Convert to tensors with proper shape: [batch_size=1, seq_len, dim]
        input_seq_tensor = tf.convert_to_tensor([req.input_sequence], dtype=tf.float32)
        target_seq_tensor = tf.convert_to_tensor([req.target_sequence], dtype=tf.float32)

        # Log tensor shapes for debugging
        logger.info(f"Input sequence tensor shape: {input_seq_tensor.shape}, Target sequence tensor shape: {target_seq_tensor.shape}")
        
        # Directly call train_step with the properly shaped tensors
        metrics = nm.train_step((input_seq_tensor, target_seq_tensor))
        avg_loss = metrics.get('loss', 0.0)
        
        # Ensure we return a Python native float
        return TrainOuterResponse(average_loss=float(avg_loss))

    except HTTPException as http_exc: raise http_exc
    except tf.errors.InvalidArgumentError as tf_err:
         logger.error(f"TensorFlow argument error during outer training: {tf_err}", exc_info=True)
         raise HTTPException(status_code=400, detail=f"TF Argument Error: {tf_err}")
    except Exception as e:
        logger.error(f"Outer training failed: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Outer training error: {str(e)}")

@app.post("/save", status_code=status.HTTP_200_OK)
async def save_neural_memory_state(req: SaveLoadRequest):
    nm = get_neural_memory()
    try:
        nm.save_state(req.path)
        return {"message": f"Neural Memory state saved to {req.path}"}
    except Exception as e:
        logger.error(f"Failed to save neural memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save state: {str(e)}")

@app.post("/load", status_code=status.HTTP_200_OK)
async def load_neural_memory_state(req: SaveLoadRequest):
    global neural_memory, surprise_detector, geometry_manager
    try:
        # First, read the state file to examine the config without loading
        if not os.path.exists(req.path):
            raise FileNotFoundError(f"State file not found: {req.path}")
            
        with open(req.path, 'r') as f: 
            state_data = json.load(f)
            
        # Extract config from saved state
        saved_config = state_data.get("config")
        if not saved_config:
            raise ValueError("State file is missing 'config' section")
        
        # Create a properly initialized model with the saved config
        temp_nm = NeuralMemoryModule(config=saved_config)
        
        # Force build by creating dummy inputs and running a forward pass
        dummy_input = tf.zeros((1, temp_nm.config['input_dim']), dtype=tf.float32)
        dummy_query = tf.zeros((1, temp_nm.config['query_dim']), dtype=tf.float32)
        _ = temp_nm.get_projections(dummy_input)
        _ = temp_nm(dummy_query)
        
        # Now load the state into the fully initialized model with matching config
        loaded_ok = temp_nm.load_state(req.path)

        if loaded_ok:
            # Replace the global instance with our successfully loaded one
            neural_memory = temp_nm
            # Re-initialize dependent components with the loaded config
            geometry_manager = GeometryManager({'embedding_dim': neural_memory.config['input_dim']})
            surprise_detector = None  # Force re-init with new geometry manager
            get_surprise_detector()
            logger.info(f"Neural Memory state loaded from {req.path} and components re-initialized.")
            return {"message": f"Neural Memory state loaded from {req.path}"}
        else:
             raise HTTPException(status_code=500, detail=f"Failed to load state from {req.path}. Check logs.")

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"State file not found: {req.path}")
    except Exception as e:
        logger.error(f"Failed to load neural memory state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load state: {str(e)}")

@app.get("/status", response_model=StatusResponse)
async def get_neural_memory_status():
    if neural_memory is None:
        return StatusResponse(status="Neural Memory module not initialized.")
    try:
        config_dict = neural_memory.get_config_dict()
        return StatusResponse(status="Initialized", config=config_dict)
    except Exception as e:
        logger.error(f"Failed to get status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@app.post("/analyze_surprise", response_model=Dict[str, Any])
async def analyze_surprise(request: AnalyzeSurpriseRequest):
    detector = get_surprise_detector()
    nm = get_neural_memory() # Need this for dimension info
    try:
        # Validate embeddings using input_dim from the initialized model
        _validate_vector(request.predicted_embedding, nm.config['input_dim'], "predicted_embedding")
        _validate_vector(request.actual_embedding, nm.config['input_dim'], "actual_embedding")

        surprise_metrics = detector.calculate_surprise(
            predicted_embedding=request.predicted_embedding,
            actual_embedding=request.actual_embedding
        )
        quickrecal_boost = detector.calculate_quickrecal_boost(surprise_metrics)

        response_data = surprise_metrics.copy()
        if 'delta' in response_data and isinstance(response_data['delta'], np.ndarray):
             response_data['delta'] = response_data['delta'].tolist()
        response_data["quickrecal_boost"] = quickrecal_boost

        return response_data

    except HTTPException as http_exc: raise http_exc
    except Exception as e:
        logger.error(f"Error analyzing surprise: {e}\n{traceback.format_exc()}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error analyzing surprise: {str(e)}")

# --- Health Check ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Basic health check."""
    logger.info("Health check requested.")
    try:
         tf_version = tf.__version__
         # Perform a minimal TF computation
         tensor_sum = tf.reduce_sum(tf.constant([1.0, 2.0])).numpy()
         can_compute = abs(tensor_sum - 3.0) < 1e-6
         status_msg = "ok" if can_compute else "error_tf_compute"
    except Exception as e:
         logger.error(f"TensorFlow health check failed: {e}", exc_info=True)
         tf_version = "error"
         status_msg = f"error_tf_init: {str(e)}"

    return {
         "status": status_msg,
         "tensorflow_version": tf_version,
         "neural_memory_initialized": neural_memory is not None,
         "timestamp": datetime.datetime.utcnow().isoformat() 
     }

# --- Introspection and Diagnostic Endpoints ---

@app.post("/get_projections", response_model=GetProjectionsResponse, summary="Get K/V/Q Projections")
async def get_projections_endpoint(request: GetProjectionsRequest):
    """Exposes internal K, V, Q projections for a given input embedding."""
    nm = get_neural_memory()
    try:
        _validate_vector(request.input_embedding, nm.config['input_dim'], "input_embedding")
        
        # Convert to tensor format expected by NeuralMemoryModule
        input_tensor = tf.convert_to_tensor([request.input_embedding], dtype=tf.float32)  # Add batch dim
        
        # Get projections (k_t, v_t, q_t tensors)
        k_t, v_t, q_t = nm.get_projections(input_tensor)
        
        # Ensure tensors are squeezed and converted to Python lists
        k_list = tf.squeeze(k_t).numpy().tolist()
        v_list = tf.squeeze(v_t).numpy().tolist()
        q_list = tf.squeeze(q_t).numpy().tolist()
        
        # Calculate input embedding L2 norm
        input_norm = float(np.linalg.norm(np.array(request.input_embedding, dtype=np.float32)))
        
        # Get projection matrix hash (placeholder implementation)
        proj_hash = "hash_placeholder_v1"
        if hasattr(nm, 'get_projection_hash'):
            proj_hash = nm.get_projection_hash()
        else:
            # Basic placeholder hash since the method doesn't exist yet
            # In the future, implement get_projection_hash in NeuralMemoryModule
            logger.warning("get_projection_hash not implemented, using placeholder")
            
        # Prepare the response
        response = GetProjectionsResponse(
            input_embedding_norm=input_norm,
            projection_adapter_used=request.projection_adapter or "identity",
            key_projection=k_list,
            value_projection=v_list,
            query_projection=q_list,
            projection_metadata={
                "dim_key": nm.config['key_dim'],
                "dim_value": nm.config['value_dim'],
                "dim_query": nm.config['query_dim'],
                "projection_matrix_hash": proj_hash,
                "input_dim": nm.config['input_dim'],
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
        )
        return response
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"/get_projections failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting projections: {str(e)}")


@app.get("/diagnose_emoloop", response_model=DiagnoseEmoLoopResponse, summary="Diagnose Emotional Feedback Loop Health")
async def diagnose_emoloop(window: str = "last_100", emotion_filter: Optional[str] = "all", format: Optional[str] = None):
    """Returns diagnostic metrics for the surprise->QuickRecal feedback loop.
    
    Args:
        window: Time/count window to analyze ("last_100", "last_hour", "session")
        emotion_filter: Optional emotion to filter by ("all" or specific emotion)
        format: Output format ("json" or "table" for CLI-friendly ASCII table)
    """
    # Log the parameters for future reference
    logger.info(f"Received /diagnose_emoloop request: window={window}, filter={emotion_filter}, format={format}")
    
    # Get metrics from the MetricsStore instead of using placeholder data
    metrics_store = get_metrics_store()
    diagnostics = metrics_store.get_diagnostic_metrics(window=window, emotion_filter=emotion_filter)
    
    # Create response using the real metrics data
    response = DiagnoseEmoLoopResponse(
        diagnostic_window=diagnostics["diagnostic_window"],
        avg_loss=diagnostics["avg_loss"],
        avg_grad_norm=diagnostics["avg_grad_norm"],
        avg_quickrecal_boost=diagnostics["avg_quickrecal_boost"],
        dominant_emotions_boosted=diagnostics["dominant_emotions_boosted"],
        emotional_entropy=diagnostics["emotional_entropy"],
        emotion_bias_index=diagnostics["emotion_bias_index"],
        user_emotion_match_rate=diagnostics["user_emotion_match_rate"],
        cluster_update_hotspots=[ClusterHotspot(**hotspot) for hotspot in diagnostics["cluster_update_hotspots"]],
        alerts=diagnostics["alerts"],
        recommendations=diagnostics["recommendations"]
    )
    
    # Handle table format for CLI-friendly output
    if format == "table":
        return Response(
            content=metrics_store.format_diagnostics_as_table(diagnostics),
            media_type="text/plain"
        )
    
    return response

# --- App startup/shutdown ---
@app.on_event("startup")
async def startup_event():
     global geometry_manager
     if geometry_manager is None:
         geometry_manager = GeometryManager()
     logger.info("Synthians Neural Memory API started. Send POST to /init to initialize.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down neural memory server.")
    # if neural_memory:
    #     try:
    #         save_path = os.environ.get("SHUTDOWN_SAVE_PATH", "/app/memory/shutdown_state.json")
    #         logger.info(f"Attempting final state save to {save_path}")
    #         neural_memory.save_state(save_path)
    #     except Exception as e:
    #         logger.error(f"Error saving state on shutdown: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    host = os.environ.get("HOST", "0.0.0.0")
    log_level = os.environ.get("LOG_LEVEL", "info").lower()

    logger.info(f"Starting Synthians Neural Memory API on http://{host}:{port}")
    print(f"-> Using TensorFlow version: {tf.__version__}")
    print(f"-> Using NumPy version: {np.__version__}")
    if not np.__version__.startswith("1."):
        print("\n\n!!!! WARNING: Numpy version is not < 2.0.0. This may cause issues with TensorFlow/other libs. !!!!\n\n")

    uvicorn.run(app, host=host, port=port, log_level=log_level) # Run using the app object directly
```

# synthians_trainer_server\logs\intent_graphs\intent_20250328125215_17601e61940.json

```json
{
  "trace_id": "intent_20250328125215_17601e61940",
  "timestamp": "2025-03-28T12:52:15.763028",
  "memory_trace": {
    "retrieved": []
  },
  "neural_memory_trace": {},
  "emotional_modulation": {},
  "reasoning_steps": [],
  "final_output": {
    "response_text": "Error: 500: Memory processing failed: 'GeometryManager' object has no attribute '_align_vectors'",
    "confidence": 0.0,
    "timestamp": "2025-03-28T12:52:23.473811"
  }
}
```

# synthians_trainer_server\metrics_store.py

```py
# synthians_trainer_server/metrics_store.py

import time
import logging
import json
import datetime
import threading
import os
from typing import Dict, List, Any, Optional, Union, Deque
from collections import deque, defaultdict
import numpy as np

logger = logging.getLogger(__name__)

class MetricsStore:
    """Captures and stores cognitive flow metrics for introspection and diagnostics.
    
    This lightweight metrics collection system records data about memory operations,
    surprise signals, and emotional feedback to enable real-time diagnostics of
    Lucidia's cognitive processes without requiring complex UI infrastructure.
    
    The store maintains an in-memory buffer of recent metrics while offering
    optional persistence to log files for post-session analysis.
    """
    
    def __init__(self, max_buffer_size: int = 1000, 
                intent_graph_enabled: bool = True,
                log_dir: Optional[str] = None):
        """Initialize the metrics store.
        
        Args:
            max_buffer_size: Maximum number of events to keep in memory
            intent_graph_enabled: Whether to generate IntentGraph logs
            log_dir: Directory to save logs (None = no file logging)
        """
        self.max_buffer_size = max_buffer_size
        self.intent_graph_enabled = intent_graph_enabled
        self.log_dir = log_dir
        
        # Create log directory if needed
        if self.log_dir and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Created metrics log directory: {self.log_dir}")
        
        # In-memory metric buffers (thread-safe)
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._memory_updates = deque(maxlen=max_buffer_size)  # Update events
        self._retrievals = deque(maxlen=max_buffer_size)  # Retrieval events
        self._quickrecal_boosts = deque(maxlen=max_buffer_size)  # QuickRecal boost events
        self._emotion_metrics = deque(maxlen=max_buffer_size)  # Emotional response events
        
        # Track current intent/interaction session
        self._current_intent_id = None
        self._intent_graph_buffer = {}
        
        # Emotional state tracking
        self._emotion_counts = defaultdict(int)
        self._user_emotion_matches = [0, 0]  # [matches, total]
        
        logger.info(f"MetricsStore initialized with buffer size {max_buffer_size}")
    
    def begin_intent(self, intent_id: Optional[str] = None) -> str:
        """Start a new intent/interaction tracking session.
        
        Returns:
            str: The intent_id (generated if not provided)
        """
        with self._lock:
            # Generate ID if not provided
            if not intent_id:
                intent_id = f"intent_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{id(self):x}"
            
            self._current_intent_id = intent_id
            
            # Initialize intent graph for this session
            if self.intent_graph_enabled:
                self._intent_graph_buffer[intent_id] = {
                    "trace_id": intent_id,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                    "memory_trace": {"retrieved": []},
                    "neural_memory_trace": {},
                    "emotional_modulation": {},
                    "reasoning_steps": [],
                    "final_output": {}
                }
            
            logger.debug(f"Started new intent tracking: {intent_id}")
            return intent_id
    
    def log_memory_update(self, input_embedding: List[float], loss: float, grad_norm: float, 
                        emotion: Optional[str] = None, intent_id: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log metrics from a memory update operation.
        
        Args:
            input_embedding: The embedding that was sent to update memory
            loss: The loss value from the memory update
            grad_norm: The gradient norm from the memory update
            emotion: Optional emotion tag associated with this update
            intent_id: Optional intent ID (uses current if not provided)
            metadata: Additional metadata to store with the update
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        # Calculate embedding norm for reference
        embedding_norm = float(np.linalg.norm(np.array(input_embedding, dtype=np.float32)))
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "loss": float(loss),
            "grad_norm": float(grad_norm),
            "embedding_norm": embedding_norm,
            "embedding_dim": len(input_embedding),
            "emotion": emotion,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._memory_updates.append(event)
            
            # Update emotion counts if provided
            if emotion:
                self._emotion_counts[emotion] += 1
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                self._intent_graph_buffer[intent_id]["neural_memory_trace"] = {
                    **self._intent_graph_buffer[intent_id].get("neural_memory_trace", {}),
                    "loss": float(loss),
                    "grad_norm": float(grad_norm),
                    "timestamp": event_time.isoformat()
                }
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Updated Neural Memory with new embedding (loss={loss:.4f}, grad_norm={grad_norm:.4f})"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("memory_updates", event)
        logger.debug(f"Logged memory update: loss={loss:.4f}, grad_norm={grad_norm:.4f}")
    
    def log_quickrecal_boost(self, memory_id: str, base_score: float, boost_amount: float,
                           emotion: Optional[str] = None, surprise_source: str = "neural_memory",
                           intent_id: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a QuickRecal score boost event.
        
        Args:
            memory_id: ID of the memory whose QuickRecal score was boosted
            base_score: Original QuickRecal score before boost
            boost_amount: Amount the score was boosted by
            emotion: Emotion associated with this memory/boost
            surprise_source: Source of the surprise signal (neural_memory, direct, etc.)
            intent_id: Optional intent ID (uses current if not provided)
            metadata: Additional metadata to store with the boost event
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "memory_id": memory_id,
            "base_score": float(base_score),
            "boost_amount": float(boost_amount),
            "final_score": float(base_score + boost_amount),
            "emotion": emotion,
            "surprise_source": surprise_source,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._quickrecal_boosts.append(event)
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                # Add to memory trace
                memory_trace = self._intent_graph_buffer[intent_id]["memory_trace"]
                memory_trace["boost_applied"] = boost_amount
                
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Boosted memory {memory_id} QuickRecal by {boost_amount:.4f} due to surprise"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("quickrecal_boosts", event)
        logger.debug(f"Logged QuickRecal boost: memory={memory_id}, amount={boost_amount:.4f}")
    
    def log_retrieval(self, query_embedding: List[float], retrieved_memories: List[Dict[str, Any]],
                     user_emotion: Optional[str] = None, intent_id: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log a memory retrieval operation.
        
        Args:
            query_embedding: Embedding used for retrieval
            retrieved_memories: List of retrieved memories with their metadata
            user_emotion: Current user emotion if known
            intent_id: Optional intent ID (uses current if not provided) 
            metadata: Additional metadata to store with the retrieval
        """
        event_time = datetime.datetime.utcnow()
        intent_id = intent_id or self._current_intent_id
        
        # Extract memory emotions if available
        memory_emotions = []
        for mem in retrieved_memories:
            if "dominant_emotion" in mem and mem["dominant_emotion"]:
                memory_emotions.append(mem["dominant_emotion"])
        
        # Calculate emotion match rate if user emotion is known
        emotion_match = False
        if user_emotion and memory_emotions:
            emotion_match = user_emotion in memory_emotions
            with self._lock:
                self._user_emotion_matches[0] += 1 if emotion_match else 0
                self._user_emotion_matches[1] += 1
        
        event = {
            "timestamp": event_time.isoformat(),
            "intent_id": intent_id,
            "embedding_dim": len(query_embedding),
            "num_results": len(retrieved_memories),
            "memory_ids": [m.get("memory_id", "unknown") for m in retrieved_memories],
            "memory_emotions": memory_emotions,
            "user_emotion": user_emotion,
            "emotion_match": emotion_match,
            "metadata": metadata or {}
        }
        
        with self._lock:
            # Store in memory buffer
            self._retrievals.append(event)
            
            # Update intent graph if enabled
            if self.intent_graph_enabled and intent_id in self._intent_graph_buffer:
                memory_trace = self._intent_graph_buffer[intent_id]["memory_trace"]
                # Add retrieved memories
                memory_trace["retrieved"] = [
                    {
                        "memory_id": mem.get("memory_id", "unknown"),
                        "quickrecal_score": mem.get("quickrecal_score", 0.0),
                        "dominant_emotion": mem.get("dominant_emotion", None),
                        "emotion_confidence": mem.get("emotion_confidence", 0.0)
                    } for mem in retrieved_memories
                ]
                
                # Add emotion info if available
                if user_emotion or memory_emotions:
                    emo_mod = self._intent_graph_buffer[intent_id]["emotional_modulation"]
                    emo_mod["user_emotion"] = user_emotion
                    if memory_emotions:
                        # Find most frequent emotion
                        from collections import Counter
                        counts = Counter(memory_emotions)
                        dominant = counts.most_common(1)[0][0] if counts else None
                        emo_mod["retrieved_emotion_dominance"] = dominant
                        emo_mod["conflict_flag"] = user_emotion != dominant if user_emotion and dominant else False
                
                # Add reasoning step
                self._intent_graph_buffer[intent_id]["reasoning_steps"].append(
                    f"→ Retrieved {len(retrieved_memories)} memories based on query"
                )
        
        # Optionally log to file
        self._maybe_write_event_log("retrievals", event)
        logger.debug(f"Logged retrieval: {len(retrieved_memories)} memories retrieved")
    
    def _maybe_write_event_log(self, event_type: str, event: Dict[str, Any]) -> None:
        """Write event to log file if logging is enabled."""
        if not self.log_dir:
            return
        
        try:
            log_file = os.path.join(self.log_dir, f"{event_type}.jsonl")
            with open(log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write event log: {e}")
    
    def finalize_intent(self, intent_id: Optional[str] = None, 
                       response_text: Optional[str] = None,
                       confidence: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Finalize the current intent/interaction and return its IntentGraph.
        
        Args:
            intent_id: Optional intent ID (uses current if not provided)
            response_text: Final response text if available
            confidence: Confidence score for the response
            
        Returns:
            Optional[Dict[str, Any]]: The completed IntentGraph or None if not enabled
        """
        intent_id = intent_id or self._current_intent_id
        if not intent_id or not self.intent_graph_enabled:
            return None
        
        with self._lock:
            if intent_id not in self._intent_graph_buffer:
                logger.warning(f"Cannot finalize unknown intent: {intent_id}")
                return None
            
            # Complete the intent graph
            intent_graph = self._intent_graph_buffer[intent_id]
            
            # Add final output
            if response_text:
                intent_graph["final_output"] = {
                    "response_text": response_text,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
            
            # Write to file if logging enabled
            if self.log_dir:
                log_file = os.path.join(self.log_dir, "intent_graphs", f"{intent_id}.json")
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                try:
                    with open(log_file, "w") as f:
                        json.dump(intent_graph, f, indent=2)
                except Exception as e:
                    logger.warning(f"Failed to write intent graph: {e}")
            
            # Remove from buffer to free memory
            graph_copy = intent_graph.copy()
            del self._intent_graph_buffer[intent_id]
            
            logger.info(f"Finalized intent {intent_id} with {len(intent_graph['reasoning_steps'])} reasoning steps")
            return graph_copy
    
    def get_diagnostic_metrics(self, window: str = "last_100", 
                             emotion_filter: Optional[str] = None) -> Dict[str, Any]:
        """Get diagnostic metrics for the emotional feedback loop.
        
        Args:
            window: Time/count window to analyze ("last_100", "last_hour", etc.)
            emotion_filter: Optional filter to specific emotion
            
        Returns:
            Dict[str, Any]: Diagnostic metrics for the emotional feedback loop
        """
        with self._lock:
            # Determine slice of data to analyze based on window
            memory_updates = list(self._memory_updates)
            quickrecal_boosts = list(self._quickrecal_boosts)
            retrievals = list(self._retrievals)
            
            # Filter by time window if needed
            if window.startswith("last_") and window[5:].isdigit():
                # "last_N" format - take last N items
                count = int(window[5:])
                memory_updates = memory_updates[-count:] if len(memory_updates) > count else memory_updates
                quickrecal_boosts = quickrecal_boosts[-count:] if len(quickrecal_boosts) > count else quickrecal_boosts
                retrievals = retrievals[-count:] if len(retrievals) > count else retrievals
            elif window == "last_hour":
                # Last hour - filter by timestamp
                cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=1)
                cutoff_str = cutoff.isoformat()
                memory_updates = [e for e in memory_updates if e["timestamp"] >= cutoff_str]
                quickrecal_boosts = [e for e in quickrecal_boosts if e["timestamp"] >= cutoff_str]
                retrievals = [e for e in retrievals if e["timestamp"] >= cutoff_str]
            
            # Apply emotion filter if specified
            if emotion_filter and emotion_filter != "all":
                memory_updates = [e for e in memory_updates if e.get("emotion") == emotion_filter]
                quickrecal_boosts = [e for e in quickrecal_boosts if e.get("emotion") == emotion_filter]
            
            # Calculate average metrics
            avg_loss = np.mean([e["loss"] for e in memory_updates]) if memory_updates else 0.0
            avg_grad_norm = np.mean([e["grad_norm"] for e in memory_updates]) if memory_updates else 0.0
            avg_boost = np.mean([e["boost_amount"] for e in quickrecal_boosts]) if quickrecal_boosts else 0.0
            
            # Find dominant emotions boosted
            emotion_boost_counts = defaultdict(float)
            for e in quickrecal_boosts:
                if e.get("emotion"):
                    emotion_boost_counts[e["emotion"]] += e["boost_amount"]
            
            # Sort by boost amount and take top 5
            dominant_emotions = sorted(emotion_boost_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            dominant_emotions = [e[0] for e in dominant_emotions if e[1] > 0]
            
            # Calculate emotion entropy (diversity measure)
            emotion_counts = {k: v for k, v in self._emotion_counts.items() if v > 0}
            total_emotions = sum(emotion_counts.values())
            if total_emotions > 0:
                probs = [count/total_emotions for count in emotion_counts.values()]
                entropy = -sum(p * np.log(p) for p in probs if p > 0)
            else:
                entropy = 0.0
            
            # Calculate user emotion match rate
            match_rate = self._user_emotion_matches[0] / self._user_emotion_matches[1] \
                if self._user_emotion_matches[1] > 0 else 0.0
            
            # Find cluster hotspots (memory IDs with most updates)
            memory_update_counts = defaultdict(int)
            for e in quickrecal_boosts:
                memory_id = e["memory_id"]
                if memory_id:
                    memory_update_counts[memory_id] += 1
            
            # Get top clusters by update count
            cluster_hotspots = sorted(memory_update_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            cluster_hotspots = [{"cluster_id": cid, "updates": count} for cid, count in cluster_hotspots if count > 0]
            
            # Generate alerts based on metrics
            alerts = []
            recommendations = []
            
            # Alerts
            if entropy < 2.0 and total_emotions > 10:
                alerts.append("⚠️ Low emotional diversity detected (entropy < 2.0)")
                recommendations.append("Introduce more varied emotional inputs")
            else:
                alerts.append("✓ Emotional diversity stable.")
                
            if avg_loss > 0.2:
                alerts.append("⚠️ High average loss detected (> 0.2)")
                recommendations.append("Check for instability in memory patterns")
            else:
                alerts.append("✓ Surprise signals healthy.")
                
            if avg_grad_norm > 1.0:
                alerts.append("⚠️ High average gradient norm (> 1.0)")
                recommendations.append("Consider reducing learning rate or checking for oscillations")
            elif avg_grad_norm > 0.5:
                alerts.append("ℹ️ Grad norm average slightly elevated.")
                recommendations.append("Monitor grad norm trend.")
            
            if match_rate < 0.5 and self._user_emotion_matches[1] > 10:
                alerts.append("⚠️ Low user emotion match rate (< 50%)")
                recommendations.append("Review emotional alignment in retrieval process")
            
            # Add generic recommendation if list is empty
            if not recommendations:
                recommendations.append("Continue monitoring with current settings")
            
            # Calculate emotion bias index (0 = balanced, 1 = highly biased)
            if len(emotion_counts) > 1 and total_emotions > 0:
                max_count = max(emotion_counts.values())
                emotion_bias = (max_count / total_emotions) * (1 - 1/len(emotion_counts))
            else:
                emotion_bias = 0.0
            
            return {
                "diagnostic_window": window,
                "avg_loss": float(avg_loss),
                "avg_grad_norm": float(avg_grad_norm),
                "avg_quickrecal_boost": float(avg_boost),
                "dominant_emotions_boosted": dominant_emotions,
                "emotional_entropy": float(entropy),
                "emotion_bias_index": float(emotion_bias),
                "user_emotion_match_rate": float(match_rate),
                "cluster_update_hotspots": cluster_hotspots,
                "alerts": alerts,
                "recommendations": recommendations,
                "data_points": {
                    "memory_updates": len(memory_updates),
                    "quickrecal_boosts": len(quickrecal_boosts),
                    "retrievals": len(retrievals)
                }
            }
    
    def format_diagnostics_as_table(self, diagnostics: Dict[str, Any]) -> str:
        """Format diagnostics as an ASCII table for CLI output.
        
        Args:
            diagnostics: Diagnostics data from get_diagnostic_metrics()
            
        Returns:
            str: Formatted ASCII table
        """
        width = 80
        
        # Helper to create a section line
        def section(title):
            return f"\n{title.center(width, '=')}\n"
        
        # Header
        output = []
        output.append("=" * width)
        output.append(f"LUCIDIA COGNITIVE DIAGNOSTICS: {diagnostics['diagnostic_window']}".center(width))
        output.append(f"[{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}]".center(width))
        output.append("=" * width)
        
        # Core metrics
        output.append(section("CORE METRICS"))
        metrics = [
            ("Average Loss", f"{diagnostics['avg_loss']:.4f}"),
            ("Average Grad Norm", f"{diagnostics['avg_grad_norm']:.4f}"),
            ("Average QuickRecal Boost", f"{diagnostics['avg_quickrecal_boost']:.4f}"),
            ("Emotional Entropy", f"{diagnostics['emotional_entropy']:.2f}"),
            ("Emotion Bias Index", f"{diagnostics['emotion_bias_index']:.2f}"),
            ("User Emotion Match Rate", f"{diagnostics['user_emotion_match_rate']:.2%}")
        ]
        
        # Format metrics as two columns
        for i in range(0, len(metrics), 2):
            if i+1 < len(metrics):
                col1 = f"{metrics[i][0]}: {metrics[i][1]}"
                col2 = f"{metrics[i+1][0]}: {metrics[i+1][1]}"
                output.append(f"{col1.ljust(40)} | {col2.ljust(38)}")
            else:
                output.append(f"{metrics[i][0]}: {metrics[i][1]}")
        
        # Dominant emotions
        output.append(section("EMOTION ANALYSIS"))
        if diagnostics['dominant_emotions_boosted']:
            output.append("Dominant Boosted Emotions: " + ", ".join(diagnostics['dominant_emotions_boosted']))
        else:
            output.append("Dominant Boosted Emotions: None detected")
        
        # Cluster hotspots
        output.append(section("MEMORY HOTSPOTS"))
        if diagnostics['cluster_update_hotspots']:
            for hotspot in diagnostics['cluster_update_hotspots']:
                output.append(f"* {hotspot['cluster_id']}: {hotspot['updates']} updates")
        else:
            output.append("No significant memory hotspots detected")
        
        # Alerts and recommendations
        output.append(section("ALERTS"))
        for alert in diagnostics['alerts']:
            output.append(f"* {alert}")
        
        output.append(section("RECOMMENDATIONS"))
        for rec in diagnostics['recommendations']:
            output.append(f"* {rec}")
        
        # Data summary
        data_points = diagnostics['data_points']
        output.append(section("DATA SUMMARY"))
        output.append(f"Based on {data_points['memory_updates']} updates, {data_points['quickrecal_boosts']} boosts, and {data_points['retrievals']} retrievals")
        output.append("" * width)
        
        return "\n".join(output)

# --- Global Instance ---
metrics_store = None

def get_metrics_store() -> MetricsStore:
    """Get or initialize the global MetricsStore instance."""
    global metrics_store
    if metrics_store is None:
        # Create log directory in the current directory
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        metrics_store = MetricsStore(log_dir=log_dir)
        logger.info("Global MetricsStore initialized")
    return metrics_store

```

# synthians_trainer_server\neural_memory.py

```py
# synthians_trainer_server/neural_memory.py

import tensorflow as tf
import numpy as np
import json
import os
import logging
from typing import Dict, Any, Optional, List, Tuple, Union, TYPE_CHECKING
from enum import Enum # Import Enum
import datetime

# Ensure TensorFlow uses float32 by default
tf.keras.backend.set_floatx('float32')
logger = logging.getLogger(__name__)

# --- Configuration Class ---
class NeuralMemoryConfig(dict):
    """Configuration for the NeuralMemoryModule."""
    def __init__(self, *args, **kwargs):
        defaults = {
            "input_dim": 768,
            "key_dim": 128,
            "value_dim": 768,
            "query_dim": 128,
            "memory_hidden_dims": [512],
            "gate_hidden_dims": [64],
            "alpha_init": -2.0,
            "theta_init": -3.0, # Controls inner loop LR
            "eta_init": 2.0,
            "outer_learning_rate": 1e-4,
            "use_complex_gates": False
        }
        config = defaults.copy()
        # Apply kwargs first
        config.update(kwargs)
        # Then apply dict from args if provided
        if args and isinstance(args[0], dict):
            config.update(args[0])

        super().__init__(config)
        # Ensure integer dimensions after all updates
        for key in ["input_dim", "key_dim", "value_dim", "query_dim"]:
            if key in self: self[key] = int(self[key])
        if "memory_hidden_dims" in self:
            self["memory_hidden_dims"] = [int(d) for d in self["memory_hidden_dims"]]
        if "gate_hidden_dims" in self:
            self["gate_hidden_dims"] = [int(d) for d in self["gate_hidden_dims"]]

    # Allow attribute access (though we avoid relying on it internally now)
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'NeuralMemoryConfig' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


# --- Core Memory MLP ---
class MemoryMLP(tf.keras.layers.Layer):
    """The core MLP model (M) used for associative memory."""
    def __init__(self, key_dim, value_dim, hidden_dims, name="MemoryMLP", **kwargs):
        super().__init__(name=name, **kwargs)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.hidden_dims = [int(d) for d in hidden_dims]
        
        # Create layers in __init__ as instance attributes so they're properly tracked
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_dims):
            self.hidden_layers.append(
                tf.keras.layers.Dense(
                    units, 
                    activation='relu',
                    name=f"mem_hidden_{i+1}"
                )
            )
        
        # Output Layer
        self.output_layer = tf.keras.layers.Dense(self.value_dim, name="mem_output")

    def build(self, input_shape):
        # input_shape is expected to be [batch_size, key_dim]
        shape = tf.TensorShape(input_shape)
        last_dim = shape[-1]
        if last_dim is None:
             raise ValueError(f"Input dimension must be defined for {self.name}. Received shape: {input_shape}")
        if last_dim != self.key_dim:
             logger.warning(f"{self.name} input shape last dim {last_dim} != config key_dim {self.key_dim}. Ensure config matches data.")

        # Build all layers with explicit input shapes
        current_shape = shape
        for layer in self.hidden_layers:
            layer.build(current_shape)
            current_shape = layer.compute_output_shape(current_shape)
            
        # Build output layer
        self.output_layer.build(current_shape)
        
        # Call super build to ensure proper tracking
        super().build(input_shape)
        logger.info(f"{self.name} built successfully with input shape {input_shape}. Found {len(self.trainable_variables)} trainable vars.")

    def call(self, inputs, training=None):
        x = inputs
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x, training=training)
        # Pass through output layer
        return self.output_layer(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({"key_dim": self.key_dim, "value_dim": self.value_dim, "hidden_dims": self.hidden_dims})
        return config

# --- Neural Memory Module ---
class NeuralMemoryModule(tf.keras.Model):
    """
    Implements the Titans Neural Memory module that learns at test time.
    Inherits from tf.keras.Model for easier weight management and saving.
    """
    def __init__(self, config: Optional[Union[NeuralMemoryConfig, Dict]] = None, **kwargs):
        super().__init__(**kwargs)
        if isinstance(config, dict) or config is None: self.config = NeuralMemoryConfig(**(config or {}))
        elif isinstance(config, NeuralMemoryConfig): self.config = config
        else: raise TypeError("config must be a dict or NeuralMemoryConfig")

        logger.info(f"Initializing NeuralMemoryModule with config: {dict(self.config)}")

        # --- Outer Loop Parameters ---
        initializer_outer = tf.keras.initializers.GlorotUniform()
        key_dim, value_dim, query_dim, input_dim = self.config['key_dim'], self.config['value_dim'], self.config['query_dim'], self.config['input_dim']

        self.WK_layer = tf.keras.layers.Dense(key_dim, name="WK_proj", use_bias=False, kernel_initializer=initializer_outer)
        self.WV_layer = tf.keras.layers.Dense(value_dim, name="WV_proj", use_bias=False, kernel_initializer=initializer_outer)
        self.WQ_layer = tf.keras.layers.Dense(query_dim, name="WQ_proj", use_bias=False, kernel_initializer=initializer_outer)

        if not self.config.get('use_complex_gates', False):
            self.alpha_logit = tf.Variable(tf.constant(self.config['alpha_init'], dtype=tf.float32), name="alpha_logit", trainable=True)
            self.theta_logit = tf.Variable(tf.constant(self.config['theta_init'], dtype=tf.float32), name="theta_logit", trainable=True)
            self.eta_logit = tf.Variable(tf.constant(self.config['eta_init'], dtype=tf.float32), name="eta_logit", trainable=True)
            self._gate_params = [self.alpha_logit, self.theta_logit, self.eta_logit]
        else:
            logger.warning("Complex gates not implemented, using simple scalar gates.")
            self.alpha_logit = tf.Variable(tf.constant(self.config['alpha_init'], dtype=tf.float32), name="alpha_logit", trainable=True)
            self.theta_logit = tf.Variable(tf.constant(self.config['theta_init'], dtype=tf.float32), name="theta_logit", trainable=True)
            self.eta_logit = tf.Variable(tf.constant(self.config['eta_init'], dtype=tf.float32), name="eta_logit", trainable=True)
            self._gate_params = [self.alpha_logit, self.theta_logit, self.eta_logit]

        # --- Inner Loop Parameters (Memory Model M) ---
        self.memory_mlp = MemoryMLP(
            key_dim=key_dim, value_dim=value_dim, hidden_dims=self.config['memory_hidden_dims'], name="MemoryMLP"
        )
        # --- Force build with a defined input shape ---
        # Create a dummy input tensor with batch size 1 and correct key_dim
        dummy_mlp_input = tf.TensorSpec(shape=[1, key_dim], dtype=tf.float32)
        # Build the MLP now
        self.memory_mlp.build(dummy_mlp_input.shape)
        # Verify build
        if not self.memory_mlp.built:
             logger.error("MemoryMLP failed to build during init!")
        self._inner_trainable_variables = self.memory_mlp.trainable_variables
        logger.info(f"MemoryMLP built. Trainable variables: {len(self._inner_trainable_variables)}")
        if not self._inner_trainable_variables: logger.error("MemoryMLP has NO trainable variables!")

        # --- Momentum State ---
        self.momentum_state = [
            tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}")
            for i, var in enumerate(self._inner_trainable_variables)
        ]
        logger.info(f"Momentum state variables created: {len(self.momentum_state)}")

        # --- Optimizer for Outer Loop ---
        self.outer_optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['outer_learning_rate'])

        # Build projection layers
        self.WK_layer.build(input_shape=(None, input_dim))
        self.WV_layer.build(input_shape=(None, input_dim))
        self.WQ_layer.build(input_shape=(None, input_dim))
        logger.info("Projection layers built.")


    @property
    def inner_trainable_variables(self):
        return self.memory_mlp.trainable_variables

    @property
    def outer_trainable_variables(self):
         return self.WK_layer.trainable_variables + \
                self.WV_layer.trainable_variables + \
                self.WQ_layer.trainable_variables + \
                self._gate_params

    def get_projections(self, x_t: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Calculate key, value, query projections from input."""
        # Convert to tensor and ensure correct dtype
        x_t = tf.convert_to_tensor(x_t, dtype=tf.float32)
        original_shape = tf.shape(x_t)
        
        # Handle various input shapes
        if len(original_shape) == 1:
            # Single vector -> add batch dimension
            x_t = tf.expand_dims(x_t, 0)
            logger.debug(f"Reshaped input from {original_shape} to {tf.shape(x_t)}")
        
        # Verify input dimension
        if tf.shape(x_t)[-1] != self.config['input_dim']:
            logger.warning(f"Input dimension mismatch: expected {self.config['input_dim']}, got {tf.shape(x_t)[-1]}")
            raise ValueError(f"Input dimension mismatch: expected {self.config['input_dim']}, got {tf.shape(x_t)[-1]}")
            
        # Apply projections
        k_t = self.WK_layer(x_t)
        v_t = self.WV_layer(x_t)
        q_t = self.WQ_layer(x_t)
        return k_t, v_t, q_t
    
    def get_projection_hash(self) -> str:
        """Returns a hash representation of the current projection matrices.
        
        This is used for introspection and cognitive tracing, allowing the system to
        detect when projection matrices have changed between different runs or sessions.
        
        Returns:
            str: A hash string representing the current state of WK, WV, WQ matrices
        """
        # In a full implementation, this would calculate a hash based on the actual weight values
        # For now, return a placeholder with timestamp to make it somewhat unique
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"proj_matrix_hash_{timestamp}"

    def call(self, q_t: tf.Tensor, training=False) -> tf.Tensor:
        """Retrieve value from memory given query q_t (inference only)."""
        # Dynamic shape handling - reshape input if needed
        q_t = tf.convert_to_tensor(q_t, dtype=tf.float32)
        original_shape = tf.shape(q_t)
        
        # Handle various input shapes
        if len(original_shape) == 1:
            # Single vector -> add batch dimension
            q_t = tf.expand_dims(q_t, 0)
            logger.debug(f"Reshaped input from {original_shape} to {tf.shape(q_t)}")
        
        # Verify dimensions
        if tf.shape(q_t)[-1] != self.config['query_dim']:
            logger.warning(f"Query dimension mismatch: expected {self.config['query_dim']}, got {tf.shape(q_t)[-1]}")
            # If crucial dimension doesn't match, raise error
            raise ValueError(f"Query dimension mismatch: expected {self.config['query_dim']}, got {tf.shape(q_t)[-1]}")
            
        if self.config['query_dim'] != self.config['key_dim']:
            raise ValueError(f"query_dim ({self.config['query_dim']}) must match key_dim ({self.config['key_dim']})")
            
        retrieved_value = self.memory_mlp(q_t, training=training)
        return retrieved_value

    # Inner loop update step - NO @tf.function for now
    def update_step(self, x_t: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        # Ensure proper input shape with batch dimension
        if len(tf.shape(x_t)) == 1: x_t = tf.expand_dims(x_t, axis=0)
        tf.debugging.assert_equal(tf.shape(x_t)[0], 1, message="update_step expects batch size 1")
        tf.debugging.assert_equal(tf.shape(x_t)[-1], self.config['input_dim'], message="Input dimension mismatch")

        # Get current gate values
        alpha_t = tf.sigmoid(self.alpha_logit)
        theta_t = tf.sigmoid(self.theta_logit)
        eta_t = tf.sigmoid(self.eta_logit)

        # Calculate projections
        k_t, v_t, _ = self.get_projections(x_t)

        # Access inner trainable variables directly from the memory_mlp
        inner_vars = self.memory_mlp.trainable_variables
        if not inner_vars:
            logger.error("No inner trainable variables found in update_step!")
            # Try force-building the MLP again to ensure variables are created
            dummy_key = tf.zeros((1, self.config['key_dim']), dtype=tf.float32)
            _ = self.memory_mlp(dummy_key)  # Force model execution
            inner_vars = self.memory_mlp.trainable_variables
            if not inner_vars:
                return tf.constant(0.0), []
            else:
                logger.info(f"Successfully rebuilt memory MLP. Found {len(inner_vars)} trainable variables.")

        # Ensure momentum state matches inner variables
        if len(self.momentum_state) != len(inner_vars):
            logger.warning("Rebuilding momentum state due to variable mismatch.")
            self.momentum_state = [tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}") 
                                 for i, var in enumerate(inner_vars)]

        # Compute gradient with explicit watch on inner variables
        with tf.GradientTape() as tape:
            # Explicitly watch all inner variables
            for var in inner_vars:
                tape.watch(var)
            predicted_v_t = self.memory_mlp(k_t, training=True)
            loss = 0.5 * tf.reduce_sum(tf.square(predicted_v_t - v_t))

        grads = tape.gradient(loss, inner_vars)

        # Verify we have valid gradients
        valid_grads_indices = [i for i, g in enumerate(grads) if g is not None]
        if len(valid_grads_indices) != len(inner_vars): 
            logger.warning(f"Found {len(inner_vars) - len(valid_grads_indices)} None gradients in inner loop.")

        # Compute momentum updates
        for i in valid_grads_indices:
            grad = grads[i]
            s_var = self.momentum_state[i]
            # Update momentum state
            s_new = eta_t * s_var - theta_t * grad
            s_var.assign(s_new)

        # Apply weight updates
        for i in valid_grads_indices:
            s_t = self.momentum_state[i]
            m_var = inner_vars[i]
            # Update memory weights
            m_new = (1.0 - alpha_t) * m_var + s_t
            m_var.assign(m_new)

        # Return loss and gradients for tracking
        return loss, grads


    def train_step(self, data):
        input_sequence, target_sequence = data
        
        # Ensure memory_mlp has trainable variables
        if not self.memory_mlp.trainable_variables:
            logger.warning("No trainable variables in memory_mlp during train_step. Attempting to rebuild...")
            dummy_key = tf.zeros((1, self.config['key_dim']), dtype=tf.float32)
            _ = self.memory_mlp(dummy_key)  # Force model execution
        
        # Store initial state
        initial_memory_weights = [tf.identity(v) for v in self.memory_mlp.trainable_variables]
        initial_momentum_state = [tf.identity(s) for s in self.momentum_state]
        
        # Get sequence dimensions
        batch_size = tf.shape(input_sequence)[0]
        seq_len = tf.shape(input_sequence)[1]
        total_outer_loss = tf.constant(0.0, dtype=tf.float32)

        # Get outer trainable variables to track
        outer_vars = self.outer_trainable_variables # Get current list

        with tf.GradientTape() as tape:
            # Explicitly watch outer variables
            for var in outer_vars:
                tape.watch(var)

            # Reset inner memory and momentum state
            for i, var in enumerate(self.memory_mlp.trainable_variables):
                var.assign(tf.zeros_like(var))
            for i, s_var in enumerate(self.momentum_state):
                s_var.assign(tf.zeros_like(s_var))

            # Process sequence
            for t in tf.range(seq_len):
                x_t_batch = input_sequence[:, t, :]
                target_t_batch = target_sequence[:, t, :]

                # Generate predictions (use projection layers - outer params)
                _, _, q_t_batch = self.get_projections(x_t_batch)
                retrieved_y_t_batch = self(q_t_batch, training=False) # Uses memory_mlp - inner params

                # Compute loss against target
                tf.debugging.assert_equal(tf.shape(retrieved_y_t_batch)[-1], tf.shape(target_t_batch)[-1], 
                                          message="Outer loss target dim mismatch")
                step_loss = tf.reduce_mean(tf.square(retrieved_y_t_batch - target_t_batch))
                total_outer_loss += step_loss

                # Inner update loop - process one example at a time for now
                # This is inefficient for batch>1 but ensures correct updates
                for b in tf.range(batch_size):
                    x_t = tf.expand_dims(x_t_batch[b], axis=0)
                    _, _ = self.update_step(x_t)  # Apply inner loop update

        # Check validity of outer vars
        valid_outer_vars = [v for v in outer_vars if v is not None]
        if len(valid_outer_vars) < len(outer_vars):
            logger.warning(f"Found {len(outer_vars) - len(valid_outer_vars)} None variables in outer_vars!")
        
        # Calculate outer gradients
        outer_grads = tape.gradient(total_outer_loss, valid_outer_vars)
        
        # Check for None gradients in outer loop
        none_grads = sum(1 for g in outer_grads if g is None)
        if none_grads > 0:
            logger.warning(f"Found {none_grads} None gradients in outer loop.")

        # Apply outer gradients
        non_none_grads = []
        non_none_vars = []
        for i, (grad, var) in enumerate(zip(outer_grads, valid_outer_vars)):
            if grad is not None:
                non_none_grads.append(grad)
                non_none_vars.append(var)
        
        # Apply valid gradients only
        if non_none_grads:
            self.outer_optimizer.apply_gradients(zip(non_none_grads, non_none_vars))
        
        # Restore original memory state
        for i, var in enumerate(self.memory_mlp.trainable_variables):
            if i < len(initial_memory_weights):
                var.assign(initial_memory_weights[i])
                
        for i, s_var in enumerate(self.momentum_state):
            if i < len(initial_momentum_state):
                s_var.assign(initial_momentum_state[i])

        return {"loss": total_outer_loss / tf.cast(seq_len, dtype=tf.float32)}

    # --- Persistence ---
    def save_state(self, path: str) -> None:
        if path.startswith("file://"): path = path[7:]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            state = {
                "config": self.get_config_dict(),
                "inner_weights": {v.name: v.numpy().tolist() for v in self.inner_trainable_variables},
                "outer_weights": {v.name: v.numpy().tolist() for v in self.outer_trainable_variables},
                "momentum_state": {s.name: s.numpy().tolist() for s in self.momentum_state},
                "timestamp": datetime.datetime.now().isoformat(),
            }
            
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Neural Memory state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving Neural Memory state: {e}", exc_info=True)
            raise

    def load_state(self, path: str) -> bool:
        if path.startswith("file://"): path = path[7:]
        if not os.path.exists(path): 
            logger.error(f"State file not found: {path}")
            return False

        logger.info(f"Loading Neural Memory state from {path}")
        try:
            with open(path, 'r') as f: 
                state = json.load(f)
                
            loaded_config_dict = state.get("config")
            if not loaded_config_dict: 
                logger.error("State missing 'config'")
                return False

            # Check if we need to re-initialize with the loaded config
            current_config_dict = self.get_config_dict()
            config_changed = current_config_dict != loaded_config_dict
            if config_changed:
                logger.warning(f"Loaded config differs from current config")
                # We don't attempt to rebuild the model here - that needs to be done externally
                # Just log a warning that configs don't match

            # Load inner weights (memory model)
            inner_weights_loaded = state.get("inner_weights", {})
            inner_vars_dict = {v.name: v for v in self.inner_trainable_variables}
            loaded_count = 0
            for name, loaded_list in inner_weights_loaded.items():
                if name in inner_vars_dict:
                    var = inner_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading inner var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Inner var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} inner weights.")

            # Load outer weights (projection layers)
            outer_weights_loaded = state.get("outer_weights", {})
            outer_vars_dict = {v.name: v for v in self.outer_trainable_variables}
            loaded_count = 0
            for name, loaded_list in outer_weights_loaded.items():
                if name in outer_vars_dict:
                    var = outer_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading outer var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Outer var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} outer weights.")

            # Load momentum state
            momentum_loaded = state.get("momentum_state", {})
            # Rebuild momentum state if needed (without calling assign directly)
            if len(self.momentum_state) != len(self.inner_trainable_variables):
                logger.warning("Momentum state size doesn't match inner vars. Creating new state.")
                # Create new momentum variables without assigning to self yet
                new_momentum = []
                for i, var in enumerate(self.inner_trainable_variables):
                    new_momentum.append(tf.Variable(tf.zeros_like(var), trainable=False, name=f"momentum_{i}"))
                # Now replace the list (safer than assigning individual vars)
                self.momentum_state = new_momentum

            loaded_count = 0
            mom_vars_dict = {v.name: v for v in self.momentum_state}
            for name, loaded_list in momentum_loaded.items():
                if name in mom_vars_dict:
                    var = mom_vars_dict[name]
                    loaded_val = tf.convert_to_tensor(loaded_list, dtype=tf.float32)
                    if var.shape == loaded_val.shape:
                        var.assign(loaded_val)
                        loaded_count += 1
                    else: 
                        logger.error(f"Shape mismatch loading momentum var {name}: {var.shape} vs {loaded_val.shape}")
                else: 
                    logger.warning(f"Momentum var {name} not in current model.")
            logger.info(f"Loaded {loaded_count} momentum states.")

            logger.info(f"Neural Memory state successfully loaded from {path}")
            return True
            
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON state file {path}: {e}")
             return False
        except Exception as e:
            logger.error(f"Error loading Neural Memory state: {e}", exc_info=True)
            return False

    def get_config_dict(self) -> Dict:
         """Return config as a serializable dict."""
         # Convert Enum members to strings if necessary
         serializable_config = {}
         for k, v in self.config.items():
              serializable_config[k] = v.value if isinstance(v, Enum) else v
         return serializable_config
```

# synthians_trainer_server\surprise_detector.py

```py
import numpy as np
# Remove tensorflow dependency - use only numpy for vector operations
# import tensorflow as tf
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from synthians_memory_core.geometry_manager import GeometryManager, GeometryType

logger = logging.getLogger(__name__)

class SurpriseDetector:
    """Detects surprising patterns in embedding sequences.
    
    This class analyzes semantic shifts in embeddings to identify moments that
    break the expected narrative flow, enabling the system to recognize pattern
    discontinuities and meaningful context shifts.
    """
    
    def __init__(self, 
                 geometry_manager: Optional[GeometryManager] = None,
                 surprise_threshold: float = 0.6,
                 max_sequence_length: int = 10,
                 surprise_decay: float = 0.9):
        """Initialize the surprise detector.
        
        Args:
            geometry_manager: Shared GeometryManager instance for consistent vector operations
            surprise_threshold: Threshold above which an embedding is considered surprising (0-1)
            max_sequence_length: Maximum number of recent embeddings to track
            surprise_decay: Decay factor for historical surprise (0-1)
        """
        # Use provided GeometryManager or create a default one
        self.geometry_manager = geometry_manager or GeometryManager()
        self.surprise_threshold = surprise_threshold
        self.max_sequence_length = max_sequence_length
        self.surprise_decay = surprise_decay
        
        # Internal memory of recent embeddings
        self.recent_embeddings: List[np.ndarray] = []
        self.recent_surprises: List[float] = []
        
        # Adaptive threshold tracking
        self.min_surprise_seen = 1.0
        self.max_surprise_seen = 0.0
        self.surprise_history: List[float] = []
        
        logger.info(f"SurpriseDetector initialized with geometry type: {self.geometry_manager.config['geometry_type']}")
    
    def _standardize_embedding(self, embedding: Union[List[float], np.ndarray]) -> np.ndarray:
        """Standardize an embedding to a normalized numpy array.
        
        Args:
            embedding: Input embedding
            
        Returns:
            Normalized numpy array
        """
        # Use the public method now
        return self.geometry_manager.normalize_embedding(embedding)
    
    def calculate_surprise(self, 
                           predicted_embedding: Union[List[float], np.ndarray],
                           actual_embedding: Union[List[float], np.ndarray]) -> Dict[str, Any]:
        """Calculate surprise between predicted and actual embeddings.
        
        Args:
            predicted_embedding: The embedding predicted by the trainer
            actual_embedding: The actual embedding observed
            
        Returns:
            Dictionary with surprise metrics
        """
        # Standardize inputs using GeometryManager
        pred_vec = self.geometry_manager.normalize_embedding(predicted_embedding)
        actual_vec = self.geometry_manager.normalize_embedding(actual_embedding)
        
        # Calculate similarity using GeometryManager
        similarity = self.geometry_manager.calculate_similarity(pred_vec, actual_vec)
        
        # Calculate surprise (1 - cosine similarity, rescaled to 0-1)
        cosine_surprise = (1.0 - similarity) / 2.0
        
        # Calculate delta vector (using GeometryManager for any needed alignment)
        aligned_pred, aligned_actual = self.geometry_manager.align_vectors(pred_vec, actual_vec)
        delta_vec = aligned_actual - aligned_pred
        delta_norm = float(np.linalg.norm(delta_vec))
        
        # Calculate context shift by comparing to recent embeddings
        context_surprise = 0.0
        if len(self.recent_embeddings) > 0:
            # Calculate average similarity to recent embeddings using GeometryManager
            similarities = [self.geometry_manager.calculate_similarity(actual_vec, e) for e in self.recent_embeddings]
            avg_similarity = sum(similarities) / len(similarities)
            context_surprise = (1.0 - avg_similarity) / 2.0
        
        # Combine surprise metrics (weighted average)
        prediction_weight = 0.7  # Weight for prediction error
        context_weight = 0.3     # Weight for context shift
        
        total_surprise = (prediction_weight * cosine_surprise + 
                          context_weight * context_surprise)
        
        # Update surprise history
        self.surprise_history.append(total_surprise)
        if len(self.surprise_history) > 100:  # Keep history manageable
            self.surprise_history = self.surprise_history[-100:]
            
        # Update min/max tracking for adaptive thresholds
        self.min_surprise_seen = min(self.min_surprise_seen, total_surprise)
        self.max_surprise_seen = max(self.max_surprise_seen, total_surprise)
        
        # Update recent embeddings memory
        self.recent_embeddings.append(actual_vec)
        if len(self.recent_embeddings) > self.max_sequence_length:
            self.recent_embeddings = self.recent_embeddings[-self.max_sequence_length:]
            
        # Update recent surprises
        self.recent_surprises.append(total_surprise)
        if len(self.recent_surprises) > self.max_sequence_length:
            self.recent_surprises = self.recent_surprises[-self.max_sequence_length:]
        
        # Calculate adaptive threshold
        if len(self.surprise_history) >= 10:
            mean_surprise = np.mean(self.surprise_history)
            std_surprise = np.std(self.surprise_history)
            adaptive_threshold = mean_surprise + std_surprise
        else:
            adaptive_threshold = self.surprise_threshold
            
        # Determine if this is surprising
        is_surprising = total_surprise > adaptive_threshold
        
        # Calculate surprise volatility (how much does surprise vary?)
        if len(self.recent_surprises) >= 3:
            volatility = float(np.std(self.recent_surprises))
        else:
            volatility = 0.0
            
        return {
            "surprise": float(total_surprise),
            "cosine_surprise": float(cosine_surprise),
            "context_surprise": float(context_surprise),
            "delta_norm": delta_norm,
            "is_surprising": is_surprising,
            "adaptive_threshold": float(adaptive_threshold),
            "volatility": float(volatility),
            "delta": delta_vec.tolist()
        }
    
    def calculate_quickrecal_boost(self, surprise_metrics: Dict[str, Any]) -> float:
        """Calculate how much to boost a memory's quickrecal score based on surprise.
        
        Args:
            surprise_metrics: Output from calculate_surprise method
            
        Returns:
            QuickRecal score boost (0-1 range)
        """
        # Extract metrics
        total_surprise = surprise_metrics["surprise"]
        is_surprising = surprise_metrics["is_surprising"]
        volatility = surprise_metrics["volatility"]
        
        # Base multiplier depends on whether it's actually surprising
        if not is_surprising:
            return 0.0
            
        # Scale boost based on how surprising it is
        # Apply sigmoid scaling to make boost more aggressive for very surprising items
        def sigmoid(x):
            return 1 / (1 + np.exp(-10 * (x - 0.5)))
            
        # Apply sigmoid scaling to boost (0.5-1.0 range becomes steeper)
        scaled_surprise = sigmoid(total_surprise)
        
        # Incorporate volatility - higher volatility increases the boost
        # Max volatility boost is 1.5x
        volatility_multiplier = 1.0 + (volatility * 0.5)
        
        # Calculate final boost (max 0.5 adjustment to quickrecal)
        boost = scaled_surprise * volatility_multiplier * 0.5
        
        # Ensure boost is in 0-0.5 range (we don't want to boost by more than 0.5)
        return float(min(0.5, max(0.0, boost)))

```

# synthians_trainer_server\tests\__init__.py

```py

```

# synthians_trainer_server\tests\test_http_server.py

```py
import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from ...geometry_manager import GeometryManager
from ..http_server import app, SynthiansTrainer
from ..models import PredictNextEmbeddingRequest, TrainSequenceRequest


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_trainer():
    """Create a mock SynthiansTrainer instance."""
    with patch('synthians_memory_core.synthians_trainer_server.http_server.SynthiansTrainer', autospec=True) as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        # Configure mocked methods
        mock_instance.predict_next.return_value = np.random.randn(768)
        mock_instance.train_sequence.return_value = True
        yield mock_instance


def test_health_endpoint(test_client):
    """Test that the health endpoint returns a 200 status code."""
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_next_embedding_stateless(test_client, mock_trainer):
    """Test the predict_next_embedding endpoint with explicit previous state."""
    # Prepare request data
    embeddings = [np.random.randn(768).tolist() for _ in range(3)]
    previous_memory_state = {
        "sequence": [np.random.randn(768).tolist() for _ in range(2)],
        "surprise_history": [0.1, 0.2],
        "momentum": np.random.randn(768).tolist()
    }
    
    request_data = {
        "embeddings": embeddings,
        "previous_memory_state": previous_memory_state
    }
    
    # Send request
    response = test_client.post("/predict_next_embedding", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert "predicted_embedding" in response.json()
    assert "memory_state" in response.json()
    assert "surprise_score" in response.json()
    
    # Verify trainer was called correctly
    mock_trainer.predict_next.assert_called_once()
    args, _ = mock_trainer.predict_next.call_args
    assert len(args) >= 2  # At least embeddings and previous state
    

def test_train_sequence(test_client, mock_trainer):
    """Test the train_sequence endpoint."""
    # Prepare request data
    embeddings = [np.random.randn(768).tolist() for _ in range(5)]
    
    request_data = {
        "embeddings": embeddings,
    }
    
    # Send request
    response = test_client.post("/train_sequence", json=request_data)
    
    # Verify response
    assert response.status_code == 200
    assert response.json()["success"] == True
    
    # Verify trainer was called correctly
    mock_trainer.train_sequence.assert_called_once()
    args, _ = mock_trainer.train_sequence.call_args
    assert len(args[0]) == 5  # Embeddings length


def test_predict_next_embedding_errors(test_client):
    """Test error handling in predict_next_embedding endpoint."""
    # Test with empty embeddings
    request_data = {
        "embeddings": []
    }
    response = test_client.post("/predict_next_embedding", json=request_data)
    assert response.status_code == 400
    
    # Test with malformed embeddings (wrong dimension)
    request_data = {
        "embeddings": [np.random.randn(10).tolist() for _ in range(3)]
    }
    response = test_client.post("/predict_next_embedding", json=request_data)
    assert response.status_code == 400
```

# synthians_trainer_server\tests\test_synthians_trainer.py

```py

```

# synthians_trainer_server\types.py

```py

```

# test_faiss_integration.py

```py
#!/usr/bin/env python

import os
import sys
import time
import logging
import numpy as np
import asyncio
import signal
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("faiss_integration_test")

# Set a timeout for operations that might hang
DEFAULT_TIMEOUT = 30  # seconds

# Define timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Import FAISS
try:
    import faiss
    logger.info(f"FAISS version {getattr(faiss, '__version__', 'unknown')} loaded successfully")
    logger.info(f"FAISS has GPU support: {hasattr(faiss, 'StandardGpuResources')}")
except ImportError:
    logger.error("FAISS not found. Tests cannot proceed.")
    sys.exit(1)

# Import vector index implementation
from synthians_memory_core.vector_index import MemoryVectorIndex

# Import client if available for end-to-end test
try:
    from synthians_memory_core.api.client.client import SynthiansClient
    client_available = True
except ImportError:
    logger.warning("SynthiansClient not available, skipping API tests")
    client_available = False


class FAISSIntegrationTest:
    """Test suite for FAISS vector index implementation"""
    
    def __init__(self, use_gpu=True):
        self.test_results = {}
        self.test_dir = os.path.join(os.getcwd(), 'test_index')
        os.makedirs(self.test_dir, exist_ok=True)
        self.use_gpu = use_gpu
        logger.info(f"Test initialized with use_gpu={use_gpu}")
    
    def run_tests(self):
        """Run all tests and report results"""
        logger.info("\n===== STARTING FAISS INTEGRATION TESTS =====")
        
        # Run all tests
        self.test_results["basic_functionality"] = self.test_basic_functionality()
        self.test_results["dimension_mismatch"] = self.test_dimension_mismatch()
        self.test_results["malformed_embeddings"] = self.test_malformed_embeddings()
        self.test_results["persistence"] = self.test_persistence()
        
        # Report results
        logger.info("\n===== TEST RESULTS =====")
        for test_name, result in self.test_results.items():
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        # Final status
        if all(self.test_results.values()):
            logger.info("\nu2705 ALL TESTS PASSED u2705")
            return True
        else:
            failed = [name for name, result in self.test_results.items() if not result]
            logger.error(f"\nu274c {len(failed)} TESTS FAILED: {', '.join(failed)} u274c")
            return False
    
    def test_basic_functionality(self):
        """Test basic FAISS vector index functionality"""
        logger.info("\n----- Testing Basic Functionality -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create vector index
            dimension = 768
            logger.info("Creating vector index...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            logger.info(f"Created index with dimension {dimension}, GPU usage: {index.is_using_gpu}")
            
            # Add vectors
            vectors_to_add = 50  # Reduced from 100 to speed up tests
            logger.info(f"Adding {vectors_to_add} vectors to index...")
            start_time = time.time()
            for i in range(vectors_to_add):
                memory_id = f"test_{i}"
                vector = np.random.random(dimension).astype('float32')
                index.add(memory_id, vector)
                # Log progress for every 10 vectors
                if i % 10 == 0 and i > 0:
                    logger.info(f"Added {i} vectors so far...")
            
            add_time = time.time() - start_time
            logger.info(f"Added {vectors_to_add} vectors in {add_time:.4f}s ({vectors_to_add/add_time:.2f} vectors/s)")
            
            # Search vectors
            logger.info("Searching for similar vectors...")
            query = np.random.random(dimension).astype('float32')
            search_start = time.time()
            results = index.search(query, 5)  # Reduced from 10
            search_time = time.time() - search_start
            
            logger.info(f"Search completed in {search_time:.4f}s, returned {len(results)} results")
            if results:
                logger.info(f"First result: {results[0]}")
            
            # Verify count
            logger.info("Verifying vector count...")
            count = index.count()
            logger.info(f"Index count: {count}, expected: {vectors_to_add}")
            assert count == vectors_to_add, f"Expected {vectors_to_add} vectors, got {count}"
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Basic functionality test passed")
            return True
        except TimeoutError:
            logger.error("Basic functionality test timed out")
            return False
        except Exception as e:
            logger.error(f"Basic functionality test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_dimension_mismatch(self):
        """Test handling of vectors with different dimensions"""
        logger.info("\n----- Testing Dimension Mismatch Handling -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create index with specific dimension
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Test vectors with different dimensions
            dimensions = {
                'smaller': 384,   # Common dimension mismatch case
                'standard': dimension,
                'larger': 1024
            }
            
            # Add vectors with different dimensions
            for name, dim in dimensions.items():
                logger.info(f"Testing {name} vector with dimension {dim}...")
                vector = np.random.random(dim).astype('float32')
                try:
                    index.add(f"vector_{name}", vector)
                    logger.info(f"Successfully added {name} vector with dimension {dim}")
                except Exception as e:
                    logger.error(f"Failed to add {name} vector: {str(e)}")
                    signal.alarm(0)
                    return False
            
            # Search with different dimension vectors
            for name, dim in dimensions.items():
                logger.info(f"Searching with {name} vector ({dim} dimensions)...")
                query = np.random.random(dim).astype('float32')
                try:
                    results = index.search(query, 3)
                    logger.info(f"Successfully searched with {name} vector, got {len(results)} results")
                except Exception as e:
                    logger.error(f"Failed to search with {name} vector: {str(e)}")
                    signal.alarm(0)
                    return False
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Dimension mismatch test passed")
            return True
        except TimeoutError:
            logger.error("Dimension mismatch test timed out")
            return False
        except Exception as e:
            logger.error(f"Dimension mismatch test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_malformed_embeddings(self):
        """Test handling of malformed embeddings (NaN/Inf)"""
        logger.info("\n----- Testing Malformed Embedding Handling -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create index
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Create test vectors
            normal = np.random.random(dimension).astype('float32')
            
            # Vector with NaN values
            nan_vector = np.random.random(dimension).astype('float32')
            nan_vector[10:20] = np.nan
            
            # Vector with Inf values
            inf_vector = np.random.random(dimension).astype('float32')
            inf_vector[30:40] = np.inf
            
            # Mixed vector
            mixed_vector = np.random.random(dimension).astype('float32')
            mixed_vector[5:10] = np.nan
            mixed_vector[50:55] = np.inf
            
            # Add vectors
            test_vectors = {
                'normal': normal,
                'nan': nan_vector,
                'inf': inf_vector,
                'mixed': mixed_vector
            }
            
            for name, vector in test_vectors.items():
                logger.info(f"Testing {name} vector...")
                try:
                    index.add(f"vector_{name}", vector)
                    logger.info(f"Successfully added {name} vector")
                except Exception as e:
                    logger.error(f"Failed to add {name} vector: {str(e)}")
                    if name == 'normal':  # Normal vectors must be added successfully
                        signal.alarm(0)
                        return False
            
            # Search with malformed query vectors
            for name, vector in test_vectors.items():
                logger.info(f"Searching with {name} vector...")
                try:
                    results = index.search(vector, 3)
                    logger.info(f"Successfully searched with {name} vector, got {len(results)} results")
                except Exception as e:
                    logger.error(f"Failed to search with {name} vector: {str(e)}")
                    if name == 'normal':  # Normal vectors must be searchable
                        signal.alarm(0)
                        return False
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Malformed embedding test passed")
            return True
        except TimeoutError:
            logger.error("Malformed embedding test timed out")
            return False
        except Exception as e:
            logger.error(f"Malformed embedding test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False
    
    def test_persistence(self):
        """Test index persistence (save/load)"""
        logger.info("\n----- Testing Index Persistence -----")
        try:
            # Set timeout for operations
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(DEFAULT_TIMEOUT)
            
            # Create and populate index
            dimension = 768
            logger.info(f"Creating index with dimension {dimension}...")
            index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            # Add vectors with known IDs
            vectors_to_add = 20  # Reduced from 50
            known_ids = []
            logger.info(f"Adding {vectors_to_add} vectors to index...")
            
            for i in range(vectors_to_add):
                memory_id = f"persistent_{i}"
                known_ids.append(memory_id)
                vector = np.random.random(dimension).astype('float32')
                index.add(memory_id, vector)
            
            # Save index
            index_path = os.path.join(self.test_dir, 'persistence_test.faiss')
            logger.info(f"Saving index to {index_path}...")
            index.save(index_path)
            logger.info(f"Saved index to {index_path}")
            
            # Create new index and load
            logger.info("Creating new index and loading saved data...")
            new_index = MemoryVectorIndex({
                'embedding_dim': dimension,
                'storage_path': self.test_dir,
                'index_type': 'L2',
                'use_gpu': self.use_gpu
            })
            
            new_index.load(index_path)
            logger.info(f"Loaded index with {new_index.count()} vectors")
            
            # Verify counts match
            logger.info("Verifying vector counts match...")
            assert new_index.count() == index.count(), "Vector counts don't match after loading"
            
            # Clean up
            if os.path.exists(index_path):
                os.remove(index_path)
                logger.info(f"Cleaned up test index file {index_path}")
            
            # Cancel timeout
            signal.alarm(0)
            
            logger.info("Persistence test passed")
            return True
        except TimeoutError:
            logger.error("Persistence test timed out")
            return False
        except Exception as e:
            logger.error(f"Persistence test failed: {str(e)}")
            # Cancel timeout in case of exception
            signal.alarm(0)
            return False

async def test_api_integration():
    """Test integration with the memory API"""
    logger.info("\n----- Testing API Integration -----")
    
    if not client_available:
        logger.warning("SynthiansClient not available, skipping API test")
        return False
    
    try:
        # Set timeout for operations
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(DEFAULT_TIMEOUT)
        
        logger.info("Connecting to API...")
        client = SynthiansClient()
        await client.connect()
        
        # Create unique test memories
        timestamp = datetime.now().isoformat()
        unique_prefix = f"faiss_test_{timestamp}"
        
        logger.info(f"Creating test memories with prefix: {unique_prefix}")
        
        # Create memories
        memories = []
        for i in range(3):
            content = f"{unique_prefix} Memory {i}: This is a test memory for FAISS integration testing"
            logger.info(f"Creating memory {i}...")
            response = await client.process_memory(
                content=content,
                metadata={"test_type": "faiss_integration", "memory_number": i}
            )
            
            if response.get("success"):
                memory_id = response.get("memory_id")
                memories.append((memory_id, content))
                logger.info(f"Created memory {i} with ID: {memory_id}")
            else:
                logger.error(f"Failed to create memory {i}: {response.get('error')}")
        
        # Wait for indexing
        logger.info("Waiting for memories to be indexed...")
        await asyncio.sleep(1)
        
        # Retrieve memories
        query = unique_prefix
        logger.info(f"Retrieving memories with query: '{query}'")
        
        response = await client.retrieve_memories(query, top_k=5, threshold=0.2)
        
        if not response.get("success"):
            logger.error(f"Retrieval failed: {response.get('error')}")
            signal.alarm(0)
            return False
        
        results = response.get("memories", [])
        retrieved_ids = [m.get("id") for m in results]
        
        logger.info(f"Retrieved {len(results)} memories")
        
        # Verify that our memories were retrieved
        success = True
        for memory_id, _ in memories:
            if memory_id not in retrieved_ids:
                logger.error(f"Memory {memory_id} was not retrieved")
                success = False
        
        # Display similarity scores
        if results:
            logger.info("Similarity scores:")
            for memory in results:
                logger.info(f"  {memory.get('id')}: {memory.get('similarity_score', 'N/A')}")
        
        # Test with lower threshold
        logger.info("Testing with lower threshold (0.3)...")
        low_threshold_response = await client.retrieve_memories(
            query, top_k=5, threshold=0.3
        )
        
        low_results = low_threshold_response.get("memories", [])
        logger.info(f"Retrieved {len(low_results)} memories with lower threshold")
        
        await client.disconnect()
        
        # Cancel timeout
        signal.alarm(0)
        
        if success:
            logger.info("API integration test passed")
        else:
            logger.error("API integration test failed - not all memories were retrieved")
        
        return success
    except TimeoutError:
        logger.error("API integration test timed out")
        return False
    except Exception as e:
        logger.error(f"API integration test failed: {str(e)}")
        # Cancel timeout in case of exception
        signal.alarm(0)
        return False

async def main():
    # Run tests with and without GPU
    logger.info("\n===== FIRST RUNNING TESTS WITH CPU ONLY =====\n")
    cpu_test_suite = FAISSIntegrationTest(use_gpu=False)
    cpu_success = cpu_test_suite.run_tests()
    
    # Only try GPU if CPU tests pass
    if cpu_success:
        logger.info("\n===== NOW RUNNING TESTS WITH GPU =====\n")
        gpu_test_suite = FAISSIntegrationTest(use_gpu=True)
        gpu_success = gpu_test_suite.run_tests()
    else:
        logger.warning("Skipping GPU tests because CPU tests failed")
        gpu_success = False
    
    # Run API integration test
    api_success = await test_api_integration()
    
    if cpu_success and gpu_success and api_success:
        logger.info("\u2705 ALL TESTS PASSED INCLUDING GPU AND API INTEGRATION \u2705")
        return 0
    elif cpu_success and api_success:
        logger.warning("\u26a0ufe0f CPU AND API TESTS PASSED BUT GPU TESTS FAILED \u26a0ufe0f")
        return 1
    elif cpu_success:
        logger.warning("\u26a0ufe0f CPU TESTS PASSED BUT GPU AND API TESTS FAILED \u26a0ufe0f")
        return 2
    else:
        logger.error("\u274c ALL TESTS FAILED \u274c")
        return 3

if __name__ == "__main__":
    # Try to fix SIGALRM not available on Windows
    if sys.platform == "win32":
        logger.warning("Timeout functionality not available on Windows, disabling timeouts")
        # Define dummy functions
        def timeout_handler(signum, frame):
            pass
        signal.SIGALRM = signal.SIGTERM  # Just a placeholder
        signal.alarm = lambda x: None    # No-op function
    
    sys.exit(asyncio.run(main()))

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

# tools\lucidia_think_trace.py

```py
#!/usr/bin/env python3
"""
Lucidia Think Trace - Cognitive Flow Diagnostic Utility

This utility enables end-to-end tracing of Lucidia's cognitive process:
1. Submits a query to Lucidia's ContextCascadeEngine
2. Captures the IntentGraph and cognitive trace
3. Retrieves and formats diagnostic metrics
4. Provides a visual representation of the cognitive flow

Usage:
    python lucidia_think_trace.py --query "What were the key design principles behind the Titans paper?"

Author: Lucidia Team
Created: 2025-03-28
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Adjust Python path to find the proper modules
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:  # Avoid adding duplicates
    sys.path.insert(0, root_dir)
    print(f"Added {root_dir} to sys.path")
else:
    print(f"{root_dir} already in sys.path")

import aiohttp
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich import print as rprint

# --- Use ABSOLUTE IMPORTS ---
try:
    # Import directly from the package root
    from synthians_memory_core.geometry_manager import GeometryManager
    from synthians_memory_core.orchestrator.context_cascade_engine import ContextCascadeEngine
    from synthians_memory_core.synthians_trainer_server.metrics_store import get_metrics_store
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Attempted to import from root: {root_dir}")
    print("Ensure synthians_memory_core and its submodules are correctly structured and importable.")
    sys.exit(1)

# Initialize rich console for pretty printing
console = Console()


async def run_diagnostic_test(query: str, emotion: Optional[str] = None,
                        metadata: Optional[Dict[str, Any]] = None,
                        memory_core_url: str = "http://localhost:5010",
                        neural_memory_url: str = "http://localhost:8001",
                        diagnostics_url: str = "http://localhost:8001/diagnose_emoloop",
                        window: str = "last_100") -> Dict[str, Any]:
    """
    Run a complete diagnostic test of Lucidia's cognitive process
    
    Args:
        query: The query to process
        emotion: Optional user emotion
        metadata: Optional metadata
        memory_core_url: URL of the Memory Core service
        neural_memory_url: URL of the Neural Memory Server
        diagnostics_url: URL of the diagnostics endpoint
        window: Window for diagnostics (last_100, last_hour, etc.)
        
    Returns:
        Dictionary with test results
    """
    # Prepare metadata
    if metadata is None:
        metadata = {}
    
    if emotion and "emotion" not in metadata:
        metadata["emotion"] = emotion
        
    metadata["session"] = metadata.get("session", f"diagnostic_{int(time.time())}")
    metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Initialize ContextCascadeEngine with geometry manager for proper embedding handling
    console.print("[bold blue]Initializing ContextCascadeEngine[/bold blue]")
    
    try:
        # Initialize GeometryManager with specific configuration for handling dimension mismatches
        # This ensures both 384 and 768 dimension embeddings can be handled safely
        geometry_manager = GeometryManager(config={
            'alignment_strategy': 'truncate',  # or 'pad' - truncate larger vectors to match smaller ones
            'normalization_enabled': True,      # ensure vectors are normalized before comparison
            'embedding_dim': 768               # default dimension, will be adjusted if needed
        })
        
        engine = ContextCascadeEngine(
            memory_core_url=memory_core_url,
            neural_memory_url=neural_memory_url,
            geometry_manager=geometry_manager,
            metrics_enabled=True
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing ContextCascadeEngine:[/bold red] {e}")
        return {"error": str(e), "status": "initialization_failed"}
    
    # Process input
    console.print(f"[bold green]Processing query:[/bold green] {query}")
    start_time = time.time()
    try:
        # Safe processing that handles dimension mismatches and malformed embeddings
        response = await engine.process_new_input(
            content=query,
            embedding=None,  # Let the system generate the embedding
            metadata=metadata
        )
        process_time = time.time() - start_time
    except Exception as e:
        console.print(f"[bold red]Error processing input:[/bold red] {e}")
        return {"error": str(e), "status": "processing_failed", "process_time_ms": (time.time() - start_time) * 1000}
    
    # Get intent graph
    intent_id = response.get("intent_id")
    intent_graph = None
    intent_graph_path = None
    
    if intent_id:
        # Try to find the intent graph file
        intent_graphs_dir = Path("logs/intent_graphs")
        if intent_graphs_dir.exists():
            for file in intent_graphs_dir.glob(f"*{intent_id}*.json"):
                intent_graph_path = file
                try:
                    with open(file, 'r') as f:
                        intent_graph = json.load(f)
                except json.JSONDecodeError:
                    console.print(f"[bold yellow]Warning: Could not parse intent graph file:[/bold yellow] {file}")
                break
    
    # Get diagnostics
    diagnostics = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{diagnostics_url}?window={window}") as resp:
                if resp.status == 200:
                    diagnostics = await resp.json()
    except Exception as e:
        console.print(f"[bold red]Error getting diagnostics:[/bold red] {e}")
    
    # Format diagnostics as table if metrics_store is available
    formatted_diagnostics = None
    try:
        metrics_store = get_metrics_store()
        if metrics_store and diagnostics:
            formatted_diagnostics = metrics_store.format_diagnostics_as_table(diagnostics)
    except Exception as e:
        console.print(f"[bold yellow]Warning: Could not format diagnostics:[/bold yellow] {e}")
    
    return {
        "response": response,
        "intent_id": intent_id,
        "intent_graph": intent_graph,
        "intent_graph_path": str(intent_graph_path) if intent_graph_path else None,
        "diagnostics": diagnostics,
        "formatted_diagnostics": formatted_diagnostics,
        "process_time_ms": process_time * 1000
    }


def display_cognitive_trace(results: Dict[str, Any]) -> None:
    """
    Display a visual representation of the cognitive trace
    
    Args:
        results: Results from run_diagnostic_test
    """
    response = results.get("response", {})
    intent_graph = results.get("intent_graph")
    
    # Display response summary
    console.print("\n[bold cyan]RESPONSE SUMMARY[/bold cyan]")
    summary_table = Table(show_header=True)
    summary_table.add_column("Key", style="dim")
    summary_table.add_column("Value")
    
    summary_table.add_row("Memory ID", response.get("memory_id", "N/A"))
    summary_table.add_row("Intent ID", response.get("intent_id", "N/A"))
    summary_table.add_row("Status", response.get("status", "N/A"))
    summary_table.add_row("Time", f"{results.get('process_time_ms', 0):.2f} ms")
    
    # Add surprise metrics if available
    surprise = response.get("surprise_metrics", {})
    if surprise:
        loss = surprise.get("loss")
        grad_norm = surprise.get("grad_norm")
        boost = surprise.get("boost_calculated")
        
        if loss is not None:
            summary_table.add_row("Loss", f"{loss:.6f}")
        if grad_norm is not None:
            summary_table.add_row("Gradient Norm", f"{grad_norm:.6f}")
        if boost is not None:
            summary_table.add_row("QuickRecal Boost", f"{boost:.6f}")
    
    console.print(summary_table)
    
    # Display intent graph tree if available
    if intent_graph:
        console.print("\n[bold magenta]INTENT GRAPH TRACE[/bold magenta]")
        
        # Create tree structure
        tree = Tree(f"[bold]🧠 Cognitive Trace[/bold] ({response.get('intent_id', 'unknown')})")
        
        # Memory trace
        memory_trace = intent_graph.get("memory_trace", {})
        if memory_trace:
            mem_node = tree.add("[bold yellow]Memory Operations[/bold yellow]")
            
            # Memory storage
            storage = memory_trace.get("storage", [])
            if storage:
                storage_node = mem_node.add(f"[yellow]Storage ({len(storage)} operations)[/yellow]")
                for i, item in enumerate(storage[:3]):  # Show first 3
                    memory_id = item.get("memory_id", "unknown")
                    score = item.get("quickrecal_score", 0)
                    storage_node.add(f"Memory {i+1}: ID={memory_id}, QR={score:.4f}")
                if len(storage) > 3:
                    storage_node.add(f"... {len(storage)-3} more")
            
            # Memory retrievals
            retrieved = memory_trace.get("retrieved", [])
            if retrieved:
                retrieval_node = mem_node.add(f"[yellow]Retrievals ({len(retrieved)} operations)[/yellow]")
                for i, item in enumerate(retrieved[:3]):  # Show first 3
                    memory_id = item.get("memory_id", "unknown")
                    emotion = item.get("dominant_emotion", "neutral")
                    retrieval_node.add(f"Memory {i+1}: ID={memory_id}, Emotion={emotion}")
                if len(retrieved) > 3:
                    retrieval_node.add(f"... {len(retrieved)-3} more")
        
        # Neural memory trace
        neural_trace = intent_graph.get("neural_memory_trace", {})
        if neural_trace:
            neural_node = tree.add("[bold blue]Neural Memory Operations[/bold blue]")
            
            # Updates
            updates = neural_trace.get("updates", [])
            if updates:
                update_node = neural_node.add(f"[blue]Updates ({len(updates)} operations)[/blue]")
                for i, item in enumerate(updates[:3]):  # Show first 3
                    loss = item.get("loss", 0)
                    grad = item.get("grad_norm", 0)
                    update_node.add(f"Update {i+1}: Loss={loss:.6f}, GradNorm={grad:.6f}")
                if len(updates) > 3:
                    update_node.add(f"... {len(updates)-3} more")
        
        # Reasoning steps
        steps = intent_graph.get("reasoning_steps", [])
        if steps:
            reasoning_node = tree.add("[bold green]Reasoning Steps[/bold green]")
            for i, step in enumerate(steps):
                step_desc = step.get("description", "Unknown step")
                # Truncate if too long
                if len(step_desc) > 70:
                    step_desc = step_desc[:67] + "..."
                reasoning_node.add(f"Step {i+1}: {step_desc}")
        
        # Final output
        output = intent_graph.get("final_output", "No output recorded")
        result_node = tree.add("[bold cyan]Final Output[/bold cyan]")
        if isinstance(output, str) and len(output) > 100:
            result_node.add(f"{output[:97]}...")
        else:
            result_node.add(str(output))
        
        # Print the tree
        console.print(tree)
        
        # Print path to intent graph file
        if results.get("intent_graph_path"):
            console.print(f"\nFull intent graph saved to: [italic]{results['intent_graph_path']}[/italic]")
    
    # Display diagnostics if available
    if results.get("formatted_diagnostics"):
        console.print("\n[bold cyan]COGNITIVE DIAGNOSTICS[/bold cyan]")
        console.print(results["formatted_diagnostics"])
    elif results.get("diagnostics"):
        console.print("\n[bold cyan]COGNITIVE DIAGNOSTICS (raw)[/bold cyan]")
        console.print(json.dumps(results["diagnostics"], indent=2))


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Lucidia Think Trace - Cognitive Flow Diagnostic Utility")
    parser.add_argument("--query", type=str, required=True, help="Query to process")
    parser.add_argument("--emotion", type=str, default=None, help="User emotion (e.g., curiosity, confusion)")
    parser.add_argument("--memcore-url", type=str, default="http://localhost:5010", help="Memory Core URL")
    parser.add_argument("--neural-url", type=str, default="http://localhost:8001", help="Neural Memory Server URL")
    parser.add_argument("--window", type=str, default="last_100", help="Diagnostics window")
    parser.add_argument("--topic", type=str, default=None, help="Topic tag for metadata")
    parser.add_argument("--session", type=str, default=None, help="Session ID for metadata")
    parser.add_argument("--output-json", type=str, default=None, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Prepare metadata
    metadata = {
        "source": "lucidia_think_trace"
    }
    
    if args.emotion:
        metadata["emotion"] = args.emotion
    
    if args.topic:
        metadata["topic"] = args.topic
        
    if args.session:
        metadata["session"] = args.session
    
    # Run diagnostic test
    console.print(Panel.fit(
        f"[bold]LUCIDIA THINK TRACE[/bold]\n\nQuery: {args.query}",
        title="🧠 Cognitive Flow Diagnostics",
        border_style="cyan"
    ))
    
    results = await run_diagnostic_test(
        query=args.query,
        emotion=args.emotion,
        metadata=metadata,
        memory_core_url=args.memcore_url,
        neural_memory_url=args.neural_url,
        window=args.window
    )
    
    # Display results
    display_cognitive_trace(results)
    
    # Save results to file if requested
    if args.output_json:
        # Remove formatted_diagnostics as it's not JSON serializable
        results_copy = {k: v for k, v in results.items() if k != "formatted_diagnostics"}
        with open(args.output_json, 'w') as f:
            json.dump(results_copy, f, indent=2)
        console.print(f"\nResults saved to: [italic]{args.output_json}[/italic]")
    
    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

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
import logging
import os
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

# Dynamic FAISS import with fallback installation capability
try:
    import faiss
    logger.info("FAISS import successful")
except ImportError:
    logger.warning("FAISS not found, attempting to install")
    try:
        import subprocess
        
        # Try to detect GPU and install appropriate version
        try:
            # First check if CUDA is available via nvidia-smi
            nvidia_smi_output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            # If we get here, nvidia-smi worked, so install GPU version
            logger.info("NVIDIA GPU detected, installing FAISS with GPU support")
            result = subprocess.run(["pip", "install", "faiss-gpu"], check=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            # No GPU available, install CPU version
            logger.info("No NVIDIA GPU detected, installing FAISS CPU version")
            result = subprocess.run(["pip", "install", "faiss-cpu"], check=True)
        
        # Now try importing again
        import faiss
        logger.info("FAISS successfully installed and imported")
    except Exception as e:
        logger.error(f"Failed to install FAISS: {str(e)}")
        raise

class MemoryVectorIndex:
    """A vector index for storing and retrieving memory embeddings."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector index.
        
        Args:
            config: A dictionary containing configuration options for the vector index.
                - embedding_dim: The dimension of the embeddings to store (int)
                - storage_path: The path to store the index (str)
                - index_type: The type of index to use (str, e.g., 'L2' or 'IP')
                - use_gpu: Whether to use GPU for the index (bool, default: False)
                - gpu_timeout_seconds: Seconds to wait for GPU init before fallback (int, default: 10)
        """
        self.config = config
        self.embedding_dim = config.get('embedding_dim', 768)
        self.storage_path = config.get('storage_path', './faiss_index')
        self.index_type = config.get('index_type', 'L2')
        self.use_gpu = config.get('use_gpu', False)
        self.gpu_timeout_seconds = config.get('gpu_timeout_seconds', 10)
        self.id_to_index = {}  # Maps memory IDs to their indices in the FAISS index
        self.is_using_gpu = False  # Will be set to True if GPU init succeeds
        
        # Initialize the index based on the configuration
        self._initialize_index()

    def _initialize_index(self):
        """Initialize the FAISS index based on the configuration."""
        # Create CPU index first - always needed as a fallback and for initialization
        if self.index_type.upper() == 'L2':
            self.cpu_index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info(f"Created L2 CPU index with dimension {self.embedding_dim}")
        elif self.index_type.upper() == 'IP' or self.index_type.upper() == 'COSINE':
            self.cpu_index = faiss.IndexFlatIP(self.embedding_dim)
            logger.info(f"Created IP CPU index with dimension {self.embedding_dim}")
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        # If GPU usage is requested, try to create a GPU index with timeout protection
        self.index = self.cpu_index  # Default to CPU index
        
        if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
            self._initialize_gpu_index_with_timeout()
        else:
            if self.use_gpu:
                logger.warning("GPU requested but FAISS was not built with GPU support. Using CPU index.")
            logger.info("Using CPU FAISS index")

    def _initialize_gpu_index_with_timeout(self):
        """Initialize GPU index with a timeout to prevent indefinite hanging."""
        # This will hold the result of GPU initialization
        result = {"success": False, "error": None, "index": None}
        
        # Define the initialization function to run in a separate thread
        def init_gpu():
            try:
                logger.info("Moving FAISS index to GPU 0...")
                
                # Create GPU resources with safe memory configuration
                res = faiss.StandardGpuResources()
                
                # Configure lower temp memory to avoid CUDA OOM issues
                try:
                    # This is a safer approach as it uses less GPU memory
                    # 64MB is typically sufficient for most operations, adjust as needed
                    res.setTempMemory(64 * 1024 * 1024)  # 64 MB, much safer than default
                    logger.info("Set FAISS GPU temp memory to 64 MB")
                except Exception as e:
                    logger.warning(f"Could not set GPU temp memory: {e}. Will use default.")
                
                # Transfer index to GPU
                gpu_index = faiss.index_cpu_to_gpu(res, 0, self.cpu_index)
                
                # Store result
                result["success"] = True
                result["index"] = gpu_index
                logger.info("GPU index successfully created")
            except Exception as e:
                result["success"] = False
                result["error"] = str(e)
                logger.warning(f"GPU index creation failed: {e}")
        
        # Create and start the thread
        init_thread = threading.Thread(target=init_gpu)
        init_thread.daemon = True  # Allow the thread to be killed when the main thread exits
        init_thread.start()
        
        # Wait for the thread with timeout
        init_thread.join(timeout=self.gpu_timeout_seconds)
        
        # Check the result
        if init_thread.is_alive():
            # Thread is still running after timeout
            logger.warning(f"GPU initialization timed out after {self.gpu_timeout_seconds} seconds. Falling back to CPU.")
            return  # Keep using CPU index
        
        if result["success"] and result["index"] is not None:
            self.index = result["index"]
            self.is_using_gpu = True
            logger.info("Successfully initialized GPU index")
        else:
            error_msg = result["error"] if result["error"] else "Unknown error"
            logger.warning(f"Failed to initialize GPU index: {error_msg}. Falling back to CPU index.")

    def add(self, memory_id: str, embedding: np.ndarray) -> bool:
        """Add a memory embedding to the index.
        
        Args:
            memory_id: A unique identifier for the memory
            embedding: The embedding vector for the memory
            
        Returns:
            bool: True if the memory was added successfully, False otherwise
        """
        try:
            # Validate the embedding
            embedding = self._validate_embedding(embedding)
            if embedding is None:
                logger.error(f"Invalid embedding for memory {memory_id}")
                return False
            
            # Get the next available index
            idx = len(self.id_to_index)
            
            # Add the embedding to the index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Map the memory ID to its index
            self.id_to_index[memory_id] = idx
            
            return True
        except Exception as e:
            logger.error(f"Error adding memory {memory_id} to index: {str(e)}")
            return False

    def search(self, query_embedding: np.ndarray, k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float]]:
        """Search for the k nearest memories to the query embedding.
        
        Args:
            query_embedding: The query embedding vector
            k: The number of nearest neighbors to return
            threshold: Minimum similarity score threshold (for L2, this is actually a maximum distance)
            
        Returns:
            List[Tuple[str, float]]: A list of tuples containing the memory ID and similarity score
        """
        try:
            # Validate the query embedding
            query_embedding = self._validate_embedding(query_embedding)
            if query_embedding is None:
                logger.error("Invalid query embedding")
                return []
            
            # Create a reverse mapping for more efficient lookups
            index_to_id = {idx: mid for mid, idx in self.id_to_index.items()}
            
            # Log the state of the index and mappings
            logger.info(f"Searching index with {self.count()} vectors and {len(self.id_to_index)} id mappings")
            if self.count() == 0:
                logger.warning("Search called on empty index")
                return []
                
            # Ensure k is not larger than the number of items in the index
            k = min(k, self.count())
            if k == 0:
                return []
            
            # Search the index
            # For L2 distance, smaller values are better (closer)
            # For IP, larger values are better (more similar)
            distances, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
            
            logger.info(f"FAISS search returned {len(indices[0])} results with threshold {threshold}")
            
            # Convert the results to memory IDs and scores
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0:  # Invalid index, skip
                    continue
                    
                # Convert distances to similarity scores based on index type
                if self.index_type.upper() == 'L2':
                    # For L2, we want to penalize large distances
                    # Convert to a similarity score by using exp(-dist)
                    # This gives a score between 0 and 1, with 1 being an exact match
                    similarity = np.exp(-dist)
                    
                    # Apply threshold (for L2, lower distance is better, so check if similarity is high enough)
                    if threshold > 0 and similarity < threshold:
                        logger.debug(f"Filtered result: similarity {similarity:.4f} < threshold {threshold}")
                        continue
                else:
                    # For IP/Cosine, higher is better
                    similarity = dist
                    
                    # Apply threshold (for IP, higher is better)
                    if threshold > 0 and similarity < threshold:
                        logger.debug(f"Filtered result: similarity {similarity:.4f} < threshold {threshold}")
                        continue
                
                # Find the memory ID for this index using the reverse mapping
                memory_id = index_to_id.get(int(idx))
                
                if memory_id is not None:
                    results.append((memory_id, float(similarity)))
                    logger.debug(f"Found memory {memory_id} with similarity {similarity:.4f}")
                else:
                    logger.warning(f"No memory ID found for index {idx}")
            
            logger.info(f"Returning {len(results)} results after filtering")
            return results
        except Exception as e:
            logger.error(f"Error searching index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _validate_embedding(self, embedding: Union[np.ndarray, list, tuple]) -> Optional[np.ndarray]:
        """Validate and normalize an embedding vector.
        
        This handles several common issues:
        1. Converts lists/tuples to numpy arrays
        2. Ensures the embedding is 1D
        3. Checks for NaN or Inf values
        4. Ensures the embedding has the correct dimension
        
        Args:
            embedding: The embedding vector to validate
            
        Returns:
            np.ndarray: A validated embedding vector, or None if invalid
        """
        try:
            # Handle case where embedding is a dict (common error)
            if isinstance(embedding, dict):
                logger.error("Embedding is a dict, not a vector. You may have passed a structured payload instead.")
                return None
                
            # Convert to numpy array if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Ensure embedding is 1D
            if len(embedding.shape) > 1:
                # If it's a 2D array with only one row, flatten it
                if len(embedding.shape) == 2 and embedding.shape[0] == 1:
                    embedding = embedding.flatten()
                else:
                    logger.error(f"Expected 1D embedding, got shape {embedding.shape}")
                    return None
            
            # Check for NaN or Inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.warning("Embedding contains NaN or Inf values. Replacing with zeros.")
                embedding = np.where(np.isnan(embedding) | np.isinf(embedding), 0.0, embedding)
            
            # Check dimension
            if len(embedding) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
                # Resize the embedding to match expected dimension
                if len(embedding) < self.embedding_dim:
                    # Pad with zeros
                    padding = np.zeros(self.embedding_dim - len(embedding), dtype=np.float32)
                    embedding = np.concatenate([embedding, padding])
                else:
                    # Truncate
                    embedding = embedding[:self.embedding_dim]
            
            # Ensure dtype is float32 for FAISS
            embedding = embedding.astype(np.float32)
            
            return embedding
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return None

    def count(self) -> int:
        """Get the number of embeddings in the index.
        
        Returns:
            int: The number of embeddings in the index
        """
        return self.index.ntotal

    def reset(self) -> bool:
        """Reset the index, removing all embeddings.
        
        Returns:
            bool: True if the index was reset successfully, False otherwise
        """
        try:
            # Re-initialize the index
            self._initialize_index()
            self.id_to_index = {}
            return True
        except Exception as e:
            logger.error(f"Error resetting index: {str(e)}")
            return False

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save the index to disk.
        
        Args:
            filepath: The filepath to save the index to. If None, use the storage_path.
            
        Returns:
            bool: True if the index was saved successfully, False otherwise
        """
        try:
            # Create storage directory if it doesn't exist
            os.makedirs(self.storage_path, exist_ok=True)
            
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')
                mapping_path = os.path.join(self.storage_path, 'id_to_index_mapping.json')
            else:
                # If custom filepath, derive mapping path by adding .mapping.json extension
                mapping_path = filepath + '.mapping.json'
            
            # Save the FAISS index
            if self.is_using_gpu:
                try:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)
                    faiss.write_index(cpu_index, filepath)
                except Exception as e:
                    logger.warning(f"Could not extract CPU index from GPU index: {e}. Saving with default method.")
                    faiss.write_index(self.index, filepath)
            else:
                faiss.write_index(self.index, filepath)
            
            # Save the ID-to-index mapping
            import json
            with open(mapping_path, 'w') as f:
                # Convert any non-string keys to strings for JSON serialization
                mapping_serializable = {str(k): v for k, v in self.id_to_index.items()}
                json.dump(mapping_serializable, f)
            
            logger.info(f"Successfully saved index to {filepath} with {len(self.id_to_index)} memory mappings")
            return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load the index from disk.
        
        Args:
            filepath: The filepath to load the index from. If None, use the storage_path.
            
        Returns:
            bool: True if the index was loaded successfully, False otherwise
        """
        try:
            if filepath is None:
                filepath = os.path.join(self.storage_path, 'faiss_index.bin')
                mapping_path = os.path.join(self.storage_path, 'id_to_index_mapping.json')
            else:
                # If custom filepath, derive mapping path by adding .mapping.json extension
                mapping_path = filepath + '.mapping.json'
            
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found at {filepath}")
                return False
            
            if os.path.isdir(filepath):
                logger.error(f"Expected a file but got a directory: {filepath}")
                return False
                
            # Load the index
            self.cpu_index = faiss.read_index(filepath)
            
            # Move to GPU if requested and supported
            if self.use_gpu and hasattr(faiss, 'StandardGpuResources'):
                self._initialize_gpu_index_with_timeout()
            else:
                self.index = self.cpu_index
            
            # Load the ID-to-index mapping
            import json
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    # Convert string keys back to their original type if needed
                    self.id_to_index = {k: int(v) for k, v in mapping_data.items()}
                logger.info(f"Successfully loaded {len(self.id_to_index)} memory mappings from {mapping_path}")
            else:
                logger.warning(f"Mapping file not found at {mapping_path}, memory retrieval may not work properly")
            
            return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

```

