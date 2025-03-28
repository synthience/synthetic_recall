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
                     loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else 'N/A'
                     grad_norm_str = f"{grad_norm:.4f}" if isinstance(grad_norm, (int, float)) else 'N/A'
                     feedback_resp = await self._make_request(
                         self.memory_core_url,
                         "/api/memories/update_quickrecal_score",  
                         method="POST",
                         payload={
                             "memory_id": memory_id,
                             "delta": quickrecal_boost,
                             "reason": f"NM Surprise (Loss:{loss_str}, GradNorm:{grad_norm_str})"
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
            # IMPORTANT: We send the raw embedding and let the Neural Memory module handle the projection
            query_for_retrieve = actual_embedding  
            logger.debug(f"Sending raw embedding to /retrieve (dim={len(query_for_retrieve)}). NeuralMemory will handle projection.")

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
