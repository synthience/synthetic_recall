import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
import datetime
import json
import aiohttp

from synthians_memory_core.geometry_manager import GeometryManager
from synthians_memory_core.synthians_trainer_server.surprise_detector import SurpriseDetector

logger = logging.getLogger(__name__)

class ContextCascadeEngine:
    """Orchestrates the bi-hemispheric cognitive flow between Memory Core and Trainer.
    
    This engine implements the Context Cascade design pattern, enabling:
    1. Prediction of next embeddings based on memory sequences
    2. Detection of surprise when expectations don't match reality
    3. Feedback of surprise to enhance memory retrieval
    4. Dynamic adaptation of memory importance based on narrative patterns
    """
    
    def __init__(self, 
                 memory_core_url: str = "http://localhost:8000",
                 trainer_url: str = "http://localhost:8001",
                 geometry_manager: Optional[GeometryManager] = None):
        """Initialize the Context Cascade Engine.
        
        Args:
            memory_core_url: URL of the Memory Core service
            trainer_url: URL of the Trainer service
            geometry_manager: Optional shared geometry manager
        """
        self.memory_core_url = memory_core_url
        self.trainer_url = trainer_url
        self.geometry_manager = geometry_manager or GeometryManager()
        
        # Initialize surprise detector with shared geometry manager
        self.surprise_detector = SurpriseDetector(
            geometry_manager=self.geometry_manager,
            surprise_threshold=0.6,
            max_sequence_length=10,
            surprise_decay=0.9
        )
        
        # State tracking
        self.current_memory_state = None
        self.last_predicted_embedding = None
        self.sequence_context = []
        self.processing_lock = asyncio.Lock()
        
        logger.info(f"Context Cascade Engine initialized with:")
        logger.info(f" - Memory Core URL: {memory_core_url}")
        logger.info(f" - Trainer URL: {trainer_url}")
        logger.info(f" - Geometry type: {self.geometry_manager.config['geometry_type']}")
    
    async def process_new_memory(self, 
                               content: str,
                               embedding: Optional[List[float]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a new memory through the full cognitive pipeline.
        
        This method orchestrates:
        1. Store memory in Memory Core
        2. Compare with previous prediction if available
        3. Update quickrecal scores based on surprise
        4. Generate prediction for next memory
        
        Args:
            content: Text content of the memory
            embedding: Optional pre-computed embedding
            metadata: Optional metadata for the memory
            
        Returns:
            Processing results including memory_id, surprise metrics, etc.
        """
        # Use lock to prevent concurrent processing issues
        async with self.processing_lock:
            # Step 1: Process memory in Memory Core
            memory_result = await self._process_memory_core(content, embedding, metadata)
            memory_id = memory_result.get("id")
            actual_embedding = memory_result.get("embedding")
            
            # Initialize response with memory processing result
            response = {
                "memory_id": memory_id,
                "status": "processed",
                "timestamp": datetime.datetime.utcnow().isoformat()
            }
            
            # Step 2: Calculate surprise if we had a previous prediction
            if self.last_predicted_embedding is not None and actual_embedding is not None:
                # Calculate surprise between predicted and actual embedding
                surprise_metrics = self.surprise_detector.calculate_surprise(
                    predicted_embedding=self.last_predicted_embedding,
                    actual_embedding=actual_embedding
                )
                
                # Calculate quickrecal boost based on surprise
                quickrecal_boost = self.surprise_detector.calculate_quickrecal_boost(surprise_metrics)
                
                # Step 3: Send surprise feedback to Memory Core
                if memory_id and quickrecal_boost > 0:
                    await self._send_surprise_feedback(
                        memory_id=memory_id,
                        delta=quickrecal_boost,
                        predicted_embedding=self.last_predicted_embedding,
                        surprise_metrics=surprise_metrics
                    )
                
                # Add surprise metrics to response
                response["surprise_metrics"] = surprise_metrics
                response["quickrecal_boost"] = quickrecal_boost
            
            # Step 4: Generate prediction for next memory
            if actual_embedding:
                prediction_result = await self._predict_next_embedding(actual_embedding)
                self.last_predicted_embedding = prediction_result.get("predicted_embedding")
                response["prediction"] = {
                    "next_embedding_available": self.last_predicted_embedding is not None
                }
            
            # Track this embedding in sequence context
            if actual_embedding:
                self.sequence_context.append({
                    "embedding": actual_embedding,
                    "memory_id": memory_id,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                })
                
                # Keep context manageable
                if len(self.sequence_context) > 20:  # Retain last 20 memories
                    self.sequence_context = self.sequence_context[-20:]
            
            return response
    
    async def _process_memory_core(self, 
                               content: str,
                               embedding: Optional[List[float]] = None,
                               metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a memory using the Memory Core service.
        
        Args:
            content: Text content of the memory
            embedding: Optional pre-computed embedding
            metadata: Optional metadata for the memory
            
        Returns:
            Processing result from Memory Core
        """
        # Prepare request payload
        payload = {
            "content": content,
            "source": "contextcascade",
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Add embedding if provided
        if embedding:
            # Validate embedding for NaN/Inf values (referenced in memory #fbee9e47)
            if embedding and self._validate_embedding(embedding):
                payload["embedding"] = embedding
            else:
                logger.warning("Invalid embedding detected (contains NaN/Inf). Letting Memory Core handle generation.")
            
        # Add metadata if provided
        if metadata:
            payload["metadata"] = metadata
            
        # Send to Memory Core
        endpoint = "/api/memories/process"
        url = f"{self.memory_core_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Add timeout for robustness
                async with session.post(url, json=payload, timeout=10.0) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    elif response.status == 404:
                        error_text = await response.text()
                        logger.error(f"Memory Core endpoint not found: {url} - {error_text}")
                        return {"error": "Memory Core endpoint not found", "status": "error", "code": 404}
                    elif response.status == 400:
                        error_text = await response.text()
                        logger.error(f"Bad request to Memory Core: {error_text}")
                        return {"error": f"Memory Core rejected request: {error_text}", "status": "error", "code": 400}
                    elif response.status == 500:
                        error_text = await response.text()
                        logger.error(f"Memory Core internal error: {error_text}")
                        return {"error": "Memory Core internal server error", "status": "error", "code": 500}
                    else:
                        error_text = await response.text()
                        logger.error(f"Memory Core processing failed: {response.status} - {error_text}")
                        return {"error": f"Memory Core processing failed: {response.status}", "status": "error", "code": response.status}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while connecting to Memory Core at {url}")
            return {"error": "Memory Core connection timed out", "status": "error", "code": "timeout"}
        except aiohttp.ClientConnectionError:
            logger.error(f"Connection error to Memory Core at {url}")
            return {"error": "Memory Core connection refused", "status": "error", "code": "connection_refused"}
        except Exception as e:
            logger.error(f"Exception in Memory Core processing: {str(e)}")
            return {"error": f"Memory Core connection error: {str(e)}", "status": "error", "code": "unknown_error"}
    
    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding for NaN or Inf values.
        
        This is a critical validation step referenced in memory #fbee9e47 to
        ensure we don't pass malformed embeddings to downstream components.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if valid, False if contains NaN/Inf
        """
        try:
            # Convert to numpy array for validation
            embedding_array = np.array(embedding, dtype=np.float32)
            
            # Check for NaN or Inf values
            if np.isnan(embedding_array).any() or np.isinf(embedding_array).any():
                logger.warning("Invalid embedding detected: contains NaN or Inf values")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return False
    
    async def _predict_next_embedding(self, current_embedding: List[float]) -> Dict[str, Any]:
        """Request prediction of next embedding from Trainer.
        
        Args:
            current_embedding: Current embedding to base prediction on
            
        Returns:
            Prediction result including predicted_embedding
        """
        # Prepare request payload
        payload = {
            "embeddings": [current_embedding]
        }
        
        # Include previous memory state if available for stateless operation
        if self.current_memory_state is not None:
            payload["previous_memory_state"] = self.current_memory_state
        
        # Send to Trainer with timeout
        endpoint = "/predict_next_embedding"
        url = f"{self.trainer_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Add timeout for robustness
                async with session.post(url, json=payload, timeout=10.0) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Store the new memory state for next prediction
                        if "memory_state" in result:
                            self.current_memory_state = result["memory_state"]
                            logger.debug("Updated trainer memory state")
                        
                        return result
                    elif response.status == 404:
                        error_text = await response.text()
                        logger.error(f"Trainer endpoint not found: {url} - {error_text}")
                        return {"error": "Trainer endpoint not found", "status": "error", "code": 404}
                    elif response.status == 400:
                        error_text = await response.text()
                        logger.error(f"Bad request to Trainer: {error_text}")
                        return {"error": f"Trainer rejected request: {error_text}", "status": "error", "code": 400}
                    elif response.status == 500:
                        error_text = await response.text()
                        logger.error(f"Trainer internal error: {error_text}")
                        return {"error": "Trainer internal server error", "status": "error", "code": 500}
                    else:
                        error_text = await response.text()
                        logger.error(f"Trainer prediction failed: {response.status} - {error_text}")
                        return {"error": f"Trainer prediction failed: {response.status}", "status": "error", "code": response.status}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while connecting to Trainer at {url}")
            return {"error": "Trainer connection timed out", "status": "error", "code": "timeout"}
        except aiohttp.ClientConnectionError:
            logger.error(f"Connection error to Trainer at {url}")
            return {"error": "Trainer connection refused", "status": "error", "code": "connection_refused"}
        except Exception as e:
            logger.error(f"Exception in Trainer prediction: {str(e)}")
            return {"error": f"Trainer connection error: {str(e)}", "status": "error", "code": "unknown_error"}
    
    async def _send_surprise_feedback(self,
                                    memory_id: str,
                                    delta: float,
                                    predicted_embedding: List[float],
                                    surprise_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Send surprise feedback to Memory Core.
        
        Args:
            memory_id: ID of the memory to update
            delta: QuickRecal score adjustment
            predicted_embedding: What was predicted
            surprise_metrics: Full surprise analysis metrics
            
        Returns:
            Feedback result
        """
        # Prepare request payload
        payload = {
            "memory_id": memory_id,
            "delta": delta,
            "predicted_embedding": predicted_embedding,
            "reason": f"Surprise score: {surprise_metrics['surprise']:.4f}, context surprise: {surprise_metrics['context_surprise']:.4f}",
            "embedding_delta": surprise_metrics["delta"]
        }
        
        # Send to Memory Core
        endpoint = "/api/memories/update_quickrecal_score"
        url = f"{self.memory_core_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                # Add timeout for robustness
                async with session.post(url, json=payload, timeout=10.0) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Successfully sent surprise feedback for memory {memory_id} with delta {delta:.4f}")
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Memory Core feedback failed: {response.status} - {error_text}")
                        return {"error": f"Memory Core feedback failed: {response.status}"}
        except asyncio.TimeoutError:
            logger.error(f"Timeout while sending surprise feedback to Memory Core at {url}")
            return {"error": "Memory Core feedback connection timed out"}
        except Exception as e:
            logger.error(f"Exception in Memory Core feedback: {str(e)}")
            return {"error": f"Memory Core feedback connection error: {str(e)}"}
    
    async def get_sequence_embeddings(self,
                                    topic: Optional[str] = None,
                                    limit: int = 10,
                                    min_quickrecal_score: Optional[float] = None) -> Dict[str, Any]:
        """Retrieve a sequence of embeddings from Memory Core.
        
        Args:
            topic: Optional topic filter
            limit: Maximum number of embeddings to retrieve
            min_quickrecal_score: Minimum quickrecal score
            
        Returns:
            Sequence of embeddings with metadata
        """
        # Prepare request parameters
        params = {
            "limit": limit,
            "sort_by": "quickrecal_score"  # Prioritize important memories
        }
        
        if topic:
            params["topic"] = topic
            
        if min_quickrecal_score is not None:
            params["min_quickrecal_score"] = min_quickrecal_score
        
        # Send to Memory Core
        endpoint = "/api/memories/get_sequence_embeddings"
        url = f"{self.memory_core_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=params) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        logger.error(f"Memory Core sequence retrieval failed: {response.status} - {error_text}")
                        return {"error": f"Memory Core sequence retrieval failed: {response.status}"}
        except Exception as e:
            logger.error(f"Exception in Memory Core sequence retrieval: {str(e)}")
            return {"error": f"Memory Core sequence connection error: {str(e)}"}
