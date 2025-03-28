import os
import json
import time
import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import aiohttp
from datetime import datetime
from urllib.parse import urljoin

# Import the sequence context manager
from .history import SequenceContextManager

# Import the titans variants - note we're importing the type and factory function
# but not directly importing the variant classes which would trigger TensorFlow import
from .titans_variants import TitansVariantType, create_titans_variant

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
                 geometry_manager: Optional[Any] = None,
                 metrics_enabled: bool = True,
                 sequence_context_length: int = 50):
        """Initialize the Context Cascade Engine.
        
        Args:
            memory_core_url: URL of the Memory Core service
            neural_memory_url: URL of the Neural Memory Server
            geometry_manager: Optional shared geometry manager
            metrics_enabled: Whether to enable cognitive metrics collection
            sequence_context_length: Maximum length of the sequence context buffer
        """
        self.memory_core_url = memory_core_url.rstrip('/')
        self.neural_memory_url = neural_memory_url.rstrip('/')

        if geometry_manager is None:
            raise ImportError("GeometryManager could not be imported. ContextCascadeEngine cannot function.")
        self.geometry_manager = geometry_manager  

        # Initialize metrics collection if enabled
        self.metrics_enabled = metrics_enabled
        self._current_intent_id = None
        if self.metrics_enabled:
            try:
                from synthians_memory_core.synthians_trainer_server.metrics_store import MetricsStore, get_metrics_store
                self.metrics_store = get_metrics_store()
                logger.info("Cognitive metrics collection enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize metrics collection: {e}")
                self.metrics_enabled = False

        self.last_retrieved_embedding: Optional[List[float]] = None
        
        # Initialize sequence context manager for attention history
        self.sequence_context_length = sequence_context_length
        self.sequence_context_manager = SequenceContextManager(max_length=self.sequence_context_length)
        
        # Keep the legacy sequence_context list for backward compatibility
        self.sequence_context: List[Dict[str, Any]] = []
        self.processing_lock = asyncio.Lock()
        
        # Determine active Titans variant from environment
        variant_name_str = os.environ.get("TITANS_VARIANT", "NONE").upper()
        try:
            self.active_variant_type = TitansVariantType(variant_name_str)
        except ValueError:
            logger.warning(f"Invalid TITANS_VARIANT '{variant_name_str}'. Defaulting to NONE.")
            self.active_variant_type = TitansVariantType.NONE
        logger.info(f"Active Titans Variant: {self.active_variant_type.value}")
        
        # Configuration ready flag and event
        self._config_ready = False
        self._config_ready_event = asyncio.Event()
        self.variant_processor = None
        
        # Trigger dynamic configuration
        asyncio.create_task(self._configure_and_set_ready())
        
        logger.info(f"Context Cascade Engine initializing:")
        logger.info(f" - Memory Core URL: {self.memory_core_url}")
        logger.info(f" - Neural Memory URL: {self.neural_memory_url}")
        logger.info(f" - Metrics Enabled: {self.metrics_enabled}")
        logger.info(f" - Sequence Context Length: {self.sequence_context_length}")
        logger.info(f" - Active Titans Variant: {self.active_variant_type.value}")
        gm_config = getattr(self.geometry_manager, 'config', {})
        logger.info(f" - Geometry type: {gm_config.get('geometry_type', 'N/A')}")
        logger.info(f" - Dynamic configuration in progress...")
        
    async def _configure_and_set_ready(self):
        """Initialize configuration and set the ready flag when complete."""
        try:
            await self._configure_attention_and_variant()
            self._config_ready = True
            self._config_ready_event.set()
            logger.info("Dynamic configuration completed successfully.")
        except Exception as e:
            logger.error(f"Error during dynamic configuration: {e}")
            # Set ready flag even on failure to prevent blocking forever
            self._config_ready = True
            self._config_ready_event.set()
            
    async def _configure_attention_and_variant(self):
        """Retrieve configuration from Neural Memory and initialize the attention module and variant processor."""
        try:
            # Retrieve configuration from Neural Memory
            config_resp = await self._make_request(
                self.neural_memory_url,
                "/config",
                method="GET"
            )
            
            if "error" in config_resp:
                logger.warning(f"Failed to retrieve configuration from Neural Memory: {config_resp.get('error')}")
                logger.warning("Using default configuration values.")
                attention_config = {
                    'num_heads': 4,
                    'key_dim': 32,  # Per head dimension (total key_dim is 128)
                    'dropout': 0.0,
                    'use_layer_norm': True,
                    'use_residual': True,
                    'max_dim_mismatch_warnings': 10,
                }
            else:
                logger.info("Retrieved configuration from Neural Memory.")
                # Extract attention configuration from the response
                attention_config = config_resp.get("attention_config", {})
                
                # If we have neural_memory_config, extract relevant dimensions
                if "neural_memory_config" in config_resp:
                    nm_config = config_resp["neural_memory_config"]
                    if not attention_config:
                        # Create attention config from neural memory config
                        attention_config = {
                            'num_heads': 4,
                            'key_dim': nm_config.get('key_dim', 128) // 4,  # Per head dimension (typically 32)
                            'dropout': 0.0,
                            'use_layer_norm': True,
                            'use_residual': True,
                            'max_dim_mismatch_warnings': 10,
                        }
                    # Add dimensions info
                    attention_config["embedding_dimensions"] = {
                        "input_dim": nm_config.get('input_dim', 768),
                        "key_dim": nm_config.get('key_dim', 128),
                        "value_dim": nm_config.get('value_dim', 768),
                        "query_dim": nm_config.get('query_dim', 128)
                    }
                
                # Get variant support information directly from the response
                supports_external_gates = config_resp.get("supports_external_gates", False)
                supports_external_projections = config_resp.get("supports_external_projections", False)
                current_variant = config_resp.get("titans_variant", "NONE")
                
                logger.info(f"Neural Memory active variant: {current_variant}")
                logger.info(f"Neural Memory supports: external gates={supports_external_gates}, external projections={supports_external_projections}")
                
                # No need to check if our variant is supported - the Neural Memory API will handle this
            
            # Initialize the variant processor with the retrieved configuration
            if self.active_variant_type != TitansVariantType.NONE:
                self.variant_processor = create_titans_variant(
                    variant_type=self.active_variant_type,
                    config=attention_config
                )
                
                # Initialize the variant processor with context manager and neural memory URL
                self.variant_processor.set_sequence_context(self.sequence_context_manager)
                self.variant_processor.set_neural_memory_url(self.neural_memory_url)
                logger.info(f"Initialized {self.active_variant_type.value} variant processor")
            else:
                self.variant_processor = None
                logger.info("No Titans Variant active. Using standard Neural Memory flow.")
            
            return attention_config
                
        except Exception as e:
            logger.error(f"Error configuring attention and variant: {e}")
            # Return default configuration
            return {
                'num_heads': 4,
                'key_dim': 32,
                'dropout': 0.0,
                'use_layer_norm': True,
                'use_residual': True,
                'max_dim_mismatch_warnings': 10,
            }

    async def process_new_input(self,
                                content: str,
                                embedding: Optional[List[float]] = None,
                                metadata: Optional[Dict[str, Any]] = None,
                                intent_id: Optional[str] = None) -> Dict[str, Any]:
        """Orchestrates the cognitive cascade for a single input.
        
        This method implements the full cognitive flow with variant-specific processing:
        1. Store input in Memory Core
        2. Get projections from Neural Memory (k_t, v_t, q_t)
        3. Apply variant-specific pre-update processing (MAG/MAL)
        4. Update Neural Memory with appropriate modifications
        5. Update QuickRecal score based on surprise metrics
        6. Retrieve from Neural Memory
        7. Apply variant-specific post-retrieval processing (MAC)
        8. Update sequence history
        9. Return final response
        
        The processing flow differs based on the active Titans variant:
        - NONE: Standard processing without attention mechanisms
        - MAC: Standard update with post-retrieval attention enhancement
        - MAG: Pre-update calculation of gate values via attention
        - MAL: Pre-update modification of value projection via attention
        
        Args:
            content: Text content for the memory
            embedding: Optional embedding for the content (will be generated if not provided)
            metadata: Optional metadata to store with the memory
            intent_id: Optional intent ID for the cognitive operation
            
        Returns:
            Dict containing processing results and memory information
        """

        if not self._config_ready:
            logger.info("Waiting for dynamic configuration...")
            try:
                await asyncio.wait_for(self._config_ready_event.wait(), 10.0)
                logger.info("Configuration ready, proceeding.")
            except asyncio.TimeoutError:
                logger.error("Timed out waiting for configuration. Cannot process input.")
                return self._finalize_error("Configuration timeout", {})

        async with self.processing_lock:
            start_time = time.time()
            # 1. Setup Intent & Metadata
            intent_id, user_emotion = self._setup_intent_and_metadata(intent_id, metadata)
            logger.info(f"Processing input: {content[:50]}... (Intent: {intent_id})")

            # Initialize context dict for this step
            step_context = {
                "content": content,
                "input_embedding": embedding,
                "metadata": metadata,
                "user_emotion": user_emotion,
                "memory_id": None,
                "x_t": None, # Raw embedding from MemCore
                "k_t": None, # Projections
                "v_t": None,
                "q_t": None,
                "v_prime_t": None, # Potentially modified by MAL
                "external_gates": None, # Calculated by MAG
                "loss": None,
                "grad_norm": None,
                "y_t_raw": None, # Raw output from NM retrieve
                "y_t_final": None, # Final output after MAC
                "variant_metrics": {}
            }

            # 2. Store Memory
            store_resp = await self._store_memory(content, embedding, metadata)
            if not store_resp.get("success"):
                return self._finalize_error("Memory storage failed", store_resp, intent_id)
            step_context["memory_id"] = store_resp["memory_id"]
            step_context["x_t"] = store_resp["embedding"] # Store the validated embedding
            quickrecal_initial = store_resp.get("quickrecal_score")

            # 3. Get Projections
            proj_resp = await self._get_projections_from_nm(step_context["x_t"])
            if not proj_resp.get("success"):
                # Log warning but proceed, NM update/retrieve might handle it
                logger.warning(f"Failed to get explicit projections: {proj_resp.get('error')}")
            else:
                step_context["k_t"] = np.array(proj_resp["key_projection"], dtype=np.float32)
                step_context["v_t"] = np.array(proj_resp["value_projection"], dtype=np.float32)
                step_context["q_t"] = np.array(proj_resp["query_projection"], dtype=np.float32)

            # 4. Variant Pre-Update Logic (MAG/MAL)
            if self.variant_processor and self.active_variant_type in [TitansVariantType.MAG, TitansVariantType.MAL]:
                 if step_context["k_t"] is not None and step_context["v_t"] is not None and step_context["q_t"] is not None:
                     variant_pre_result = await self._apply_variant_pre_update(step_context)
                     step_context["external_gates"] = variant_pre_result.get("gates") # For MAG
                     step_context["v_prime_t"] = variant_pre_result.get("v_prime_t") # For MAL
                     step_context["variant_metrics"].update(variant_pre_result.get("metrics", {}))
                 else:
                     logger.warning(f"Skipping {self.active_variant_type.value} pre-update: Missing projections.")

            # 5. Update Neural Memory
            update_resp = await self._update_neural_memory(step_context)
            if not update_resp.get("success"):
                 # Log error but proceed if possible (e.g., maybe retrieval still works)
                 logger.error(f"Neural Memory update failed: {update_resp.get('error')}")
                 # Initialize an error response, but we'll still try to retrieve
                 response_errors = {"update_error": update_resp.get("error")}
            else:
                 step_context["loss"] = update_resp.get("loss")
                 step_context["grad_norm"] = update_resp.get("grad_norm")
                 # Update projections if returned (they should match if not MAL)
                 if update_resp.get("key_projection"): step_context["k_t"] = np.array(update_resp["key_projection"], dtype=np.float32)
                 if update_resp.get("value_projection"): step_context["v_t"] = np.array(update_resp["value_projection"], dtype=np.float32)
                 response_errors = {}

            # 6. Apply QuickRecal Boost (If update succeeded)
            feedback_resp = None
            if "loss" in step_context or "grad_norm" in step_context:
                 feedback_resp = await self._apply_quickrecal_boost(step_context, quickrecal_initial)

            # 7. Retrieve from Neural Memory
            retrieve_resp = await self._retrieve_from_neural_memory(step_context["x_t"])
            if not retrieve_resp.get("success"):
                # Log error and exit - retrieval is critical
                logger.error(f"Neural Memory retrieval failed: {retrieve_resp.get('error')}")
                return self._finalize_error("Neural Memory retrieval failed", 
                                           {"retrieve_error": retrieve_resp.get("error"), **response_errors}, 
                                           intent_id)
            else:
                 step_context["y_t_raw"] = np.array(retrieve_resp["retrieved_embedding"], dtype=np.float32)
                 step_context["y_t_final"] = step_context["y_t_raw"] # Default final to raw
                 # Use query projection returned by /retrieve for consistency
                 if retrieve_resp.get("query_projection"):
                      step_context["q_t"] = np.array(retrieve_resp["query_projection"], dtype=np.float32)


            # 8. Variant Post-Retrieval Logic (MAC)
            if self.variant_processor and self.active_variant_type == TitansVariantType.MAC:
                 if step_context["y_t_raw"] is not None and step_context["q_t"] is not None:
                     variant_post_result = await self._apply_variant_post_retrieval(step_context)
                     if variant_post_result.get("success"):
                         step_context["y_t_final"] = variant_post_result["attended_output"]
                         step_context["variant_metrics"].update(variant_post_result.get("metrics", {}))
                     else:
                         logger.warning(f"MAC post-retrieval processing failed: {variant_post_result.get('error')}")
                 else:
                     logger.warning("Skipping MAC post-retrieval: Missing raw retrieval or query projection.")

            # 9. Update History
            # Use v_t (potentially modified by MAL), raw y_t (before MAC), and final y_t
            await self._update_history(step_context)

            # 10. Finalize Response
            response = self._finalize_response({}, step_context, update_resp, retrieve_resp, feedback_resp)

            processing_time = (time.time() - start_time) * 1000
            logger.info(f"Finished processing input for memory {step_context['memory_id']} in {processing_time:.2f} ms (Variant: {self.active_variant_type.value})")

            # Finalize intent graph
            if self.metrics_enabled:
                 final_text = f"Retrieved: {len(response.get('neural_memory_retrieval',{}).get('retrieved_embedding',[]))} dims" if response.get('status') == 'completed' else f"Error: {response.get('error','Unknown')}"
                 self.metrics_store.finalize_intent(
                     intent_id=intent_id,
                     response_text=final_text,
                     confidence=1.0 if response.get('status') == 'completed' else 0.0
                 )

            return response

    # --- Private Helper Methods for Refactored Flow ---

    def _setup_intent_and_metadata(self, intent_id: Optional[str], metadata: Optional[Dict]) -> Tuple[str, Optional[str]]:
        """Handles intent ID generation and extracts user emotion."""
        metadata = metadata or {}
        user_emotion = None
        if self.metrics_enabled:
            intent_id = intent_id or self.metrics_store.begin_intent()
            self._current_intent_id = intent_id # Store current intent
            if "emotion" in metadata: user_emotion = metadata["emotion"]
            elif "emotions" in metadata:
                # Simplified extraction
                emo_data = metadata["emotions"]
                if isinstance(emo_data, dict) and emo_data: user_emotion = max(emo_data.items(), key=lambda x: x[1])[0]
                elif isinstance(emo_data, list) and emo_data: user_emotion = emo_data[0]
        else:
            intent_id = intent_id or f"intent_{int(time.time())}" # Simple ID if metrics off
        return intent_id, user_emotion

    async def _store_memory(self, content: str, embedding: Optional[List], metadata: Optional[Dict]) -> Dict:
        """Stores input in MemoryCore, returns success status, ID, and validated embedding."""
        logger.debug("Step 1: Storing memory in Memory Core...")
        mem_core_resp = await self._make_request(
            self.memory_core_url, "/process_memory", method="POST",
            payload={"content": content, "embedding": embedding, "metadata": metadata or {}}
        )
        if "error" in mem_core_resp or not mem_core_resp.get("memory_id") or not mem_core_resp.get("embedding"):
            logger.error(f"Memory Core storage failed: {mem_core_resp.get('error', 'Missing ID or embedding')}")
            return {"success": False, "error": mem_core_resp.get('error', 'Store failed'), **mem_core_resp}
        else:
             # Validate embedding received from Memory Core
             is_valid = self._validate_embedding(mem_core_resp["embedding"])
             if not is_valid:
                  logger.error("Memory Core returned an invalid embedding.")
                  return {"success": False, "error": "Invalid embedding from Memory Core", **mem_core_resp}
             logger.info(f"Memory stored successfully: ID {mem_core_resp['memory_id']}")
             return {"success": True, **mem_core_resp}

    async def _get_projections_from_nm(self, actual_embedding: List[float]) -> Dict:
        """Fetches K/V/Q projections from Neural Memory."""
        logger.debug("Step 2: Fetching projections from Neural Memory...")
        if not self._validate_embedding(actual_embedding):
            return {"success": False, "error": "Invalid embedding provided to get_projections"}

        proj_resp = await self._make_request(
            self.neural_memory_url, "/get_projections", method="POST",
            payload={"input_embedding": actual_embedding}
        )
        if "error" in proj_resp or not all(k in proj_resp for k in ["key_projection", "value_projection", "query_projection"]):
             logger.warning(f"Failed to get projections: {proj_resp.get('error', 'Missing projection keys')}")
             return {"success": False, **proj_resp}
        else:
            # Validate received projections
            valid = all(self._validate_embedding(proj_resp[k]) for k in ["key_projection", "value_projection", "query_projection"])
            if not valid:
                 logger.error("Neural Memory returned invalid projections.")
                 return {"success": False, "error": "Invalid projections from Neural Memory", **proj_resp}
            logger.info("Projections fetched successfully.")
            return {"success": True, **proj_resp}

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

        # Special debug logging for important endpoints
        debug_endpoints = ["/get_projections", "/update_memory", "/retrieve", "/config"]
        if endpoint in debug_endpoints:
            logger.info(f"DEBUG: Calling {endpoint} with payload: {log_payload if log_payload != payload else payload}")

        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=payload, params=params, timeout=30.0) as response:
                    status_code = response.status
                    try:
                        resp_json = await response.json()
                        
                        # Enhanced logging for specific endpoints
                        if endpoint in debug_endpoints:
                            resp_sample = {k: (v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v) 
                                          for k, v in resp_json.items()} if isinstance(resp_json, dict) else resp_json
                            logger.info(f"DEBUG: Response from {endpoint}: Status {status_code}, Content sample: {resp_sample}")
                        else:
                            logger.debug(f"Response from {url}: Status {status_code}")  
                            
                        if 200 <= status_code < 300:
                            # For specific endpoints, ensure key fields are present
                            if endpoint == "/get_projections" and isinstance(resp_json, dict):
                                expected_keys = ["key_projection", "value_projection", "query_projection"]
                                missing_keys = [k for k in expected_keys if k not in resp_json]
                                if missing_keys:
                                    logger.warning(f"WARNING: Response from {endpoint} is missing expected keys: {missing_keys}")
                                    resp_json["warning"] = f"Missing expected keys: {missing_keys}"
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

    def _validate_embedding(self, embedding: Union[np.ndarray, List[float], None]) -> bool:
        """Validate that the embedding is in a usable form (valid np.ndarray or list)."""
        if embedding is None:
            return False
        
        # If it's already a list, validate its contents
        if isinstance(embedding, list):
            if not embedding or not all(isinstance(val, (int, float)) for val in embedding):
                return False
            try:
                # Convert to numpy to do further validation
                embedding = np.array(embedding, dtype=np.float32)
            except:
                return False
        
        try:
            # Convert to numpy if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
            
            # Check for NaN and Inf
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                logger.error("Embedding contains NaN or Inf values.")
                return False
                
            # Check for zero vector
            if np.all(embedding == 0):
                logger.warning("Embedding is a zero vector.")
                # We still return True as zero vectors are technically valid
                
            return True
        except Exception as e:
            logger.error(f"Error validating embedding: {str(e)}")
            return False
            
    def _to_list(self, arr):
        """Safely convert numpy arrays or tensors to list."""
        if arr is None:
            return None
        if isinstance(arr, list):
            return arr
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        
        # Try to handle tensorflow tensors with lazy loading
        try:
            # Check if this might be a TensorFlow tensor
            if hasattr(arr, 'numpy'):
                return arr.numpy().tolist()
                
            # Last attempt - import TF and try conversion
            from synthians_memory_core.orchestrator.titans_variants import _get_tf
            tf = _get_tf()
            if tf is not None and tf.is_tensor(arr):
                return tf.make_ndarray(tf.make_tensor_proto(arr)).tolist()
        except Exception as e:
            logger.debug(f"Failed to convert possible tensor to list: {e}")
        
        # Last resort, try direct conversion
        try:
            return list(arr)
        except Exception as e:
            logger.warning(f"Could not convert {type(arr)} to list: {e}")
            return None

    async def _retrieve_from_neural_memory(self, actual_embedding: np.ndarray) -> Dict:
        """Retrieves associated embedding from Neural Memory."""
        logger.debug("Step 6: Retrieving from Neural Memory...")
        if not self._validate_embedding(actual_embedding):
             return {"success": False, "error": "Invalid embedding for retrieval"}

        retrieve_payload = {"input_embedding": self._to_list(actual_embedding)}
        retrieve_resp = await self._make_request(
            self.neural_memory_url, "/retrieve", method="POST", payload=retrieve_payload
        )

        if "error" in retrieve_resp or not retrieve_resp.get("retrieved_embedding"):
             logger.error(f"Neural Memory retrieval failed: {retrieve_resp.get('error', 'Missing retrieved_embedding')}")
             return {"success": False, **retrieve_resp}
        else:
             # Validate retrieved embedding
             if not self._validate_embedding(retrieve_resp["retrieved_embedding"]):
                   logger.error("Neural Memory returned invalid retrieved_embedding.")
                   return {"success": False, "error": "Invalid retrieved_embedding", **retrieve_resp}
             # Validate query projection if returned
             if "query_projection" in retrieve_resp and not self._validate_embedding(retrieve_resp["query_projection"]):
                  logger.warning("Neural Memory returned invalid query_projection.")
                  # Don't fail the whole step, but nullify it
                  retrieve_resp["query_projection"] = None

             # Log retrieval metrics if enabled
             if self.metrics_enabled:
                 # Create synthetic memory object since we don't have full metadata
                 retrieved_memory = {
                     "memory_id": f"synthetic_associated",
                     "embedding": retrieve_resp["retrieved_embedding"],
                     "dominant_emotion": None  # We don't have this information
                 }
                 
                 self.metrics_store.log_retrieval(
                     query_embedding=self._to_list(actual_embedding),
                     retrieved_memories=[retrieved_memory],
                     user_emotion=None,
                     intent_id=self._current_intent_id,
                     metadata={
                         "embedding_dim": len(retrieve_resp["retrieved_embedding"]),
                         "timestamp": datetime.utcnow().isoformat(),
                         "variant_type": self.active_variant_type.value
                     }
                 )

             logger.info("Neural Memory retrieval successful.")
             return {"success": True, **retrieve_resp}

    async def _apply_variant_post_retrieval(self, step_context: Dict) -> Dict:
        """Apply variant-specific post-retrieval processing for MAC variant.
        
        This method handles the variant-specific processing that occurs AFTER
        Neural Memory retrieval:
        
        - MAC Variant: Enhances the retrieved embedding (y_t) by applying attention
          between the current query and historical keys, and using this to create
          a weighted combination of historical values with the retrieved embedding.
          This produces a contextually enhanced memory representation.
        
        Args:
            step_context: Current processing context containing embeddings and projections
        
        Returns:
            Dict containing variant processing results
        """
        if not self.variant_processor or self.active_variant_type != TitansVariantType.MAC:
            return {"success": True, "attended_output": step_context.get("y_t_raw")} # Return raw if not MAC

        logger.debug("Step 7: Applying MAC post-retrieval logic...")
        y_t_raw = step_context.get("y_t_raw")
        q_t = step_context.get("q_t")

        if y_t_raw is None or q_t is None:
             logger.warning("Skipping MAC: Missing raw retrieval or query projection.")
             return {"success": False, "error": "Missing y_t_raw or q_t for MAC"}

        try:
            # Call variant processor - assumes process_input can handle None for some args if needed
            variant_results = await self.variant_processor.process_input(
                memory_id=step_context["memory_id"],
                x_t=step_context["x_t"], k_t=step_context["k_t"],
                v_t=step_context["v_t"], q_t=q_t, y_t=y_t_raw # Provide raw y_t
            )
            if variant_results and "attended_output" in variant_results:
                 attended_y_t = variant_results["attended_output"]
                 if self._validate_embedding(attended_y_t):
                     logger.info("MAC variant successfully applied attention.")
                     return {"success": True, "attended_output": attended_y_t, "metrics": variant_results.get("metrics", {})}
                 else:
                      logger.error("MAC variant returned invalid attended_output.")
                      return {"success": False, "error": "Invalid attended_output from MAC", "attended_output": y_t_raw}
            else:
                 logger.warning("MAC variant did not return 'attended_output'.")
                 return {"success": False, "error": "MAC variant failed", "attended_output": y_t_raw}

        except Exception as e:
            logger.error(f"Error applying MAC variant: {e}", exc_info=True)
            return {"success": False, "error": str(e), "attended_output": y_t_raw}

    async def _update_history(self, step_context: Dict):
        """Adds the completed step context to the history manager."""
        logger.debug("Step 8: Updating sequence history...")
        
        # Early return if memory_id is missing (indicates something went wrong earlier)
        if "memory_id" not in step_context:
            logger.warning("History update skipped: Missing memory_id.")
            return
        
        # Ensure all components are valid numpy arrays before adding
        required_keys = ["x_t", "k_t", "v_t", "q_t", "y_t_final"]
        valid_context = True
        context_tuple_args = {}

        # Extract and validate required components
        for key in required_keys:
            value = step_context.get(key)
            if value is None:
                logger.warning(f"History update skipped: Missing '{key}'")
                valid_context = False
                break
            
            # Convert to numpy array if it's a list
            if isinstance(value, list):
                try:
                    value = np.array(value, dtype=np.float32)
                    step_context[key] = value  # Update in context
                except Exception as e:
                    logger.warning(f"History update skipped: Could not convert '{key}' to numpy array: {e}")
                    valid_context = False
                    break
            
            # Validate numpy array
            if not isinstance(value, np.ndarray):
                logger.warning(f"History update skipped: '{key}' is not a numpy array but {type(value)}")
                valid_context = False
                break
                
            # Further validation (NaN/Inf) - _validate_embedding does this
            if not self._validate_embedding(value):
                logger.warning(f"History update skipped: Invalid data in '{key}'")
                valid_context = False
                break
                
            context_tuple_args[key] = value

        if valid_context:
            try:
                # Log detailed shapes for debugging
                shapes_info = {
                    k: f"{v.shape} ({v.dtype})" for k, v in context_tuple_args.items()
                }
                logger.debug(f"Adding context with shapes: {shapes_info}")
                
                self.sequence_context_manager.add_context(
                    timestamp=time.time(), # Use current time for history entry
                    memory_id=step_context["memory_id"],
                    x_t=context_tuple_args["x_t"],
                    k_t=context_tuple_args["k_t"],
                    v_t=context_tuple_args["v_t"], # Use the v_t that was ACTUALLY used in update
                    q_t=context_tuple_args["q_t"],
                    y_t=context_tuple_args["y_t_final"] # Use the final output y_t
                )
                logger.info(f"Added context to SequenceContextManager. Length: {len(self.sequence_context_manager)}")
            except Exception as e:
                logger.error(f"Failed to add context to history manager: {e}", exc_info=True)
        else:
            logger.error("Failed to update history due to invalid/missing context components.")

    def _finalize_response(self, base_response: Dict, step_context: Dict,
                           update_resp: Dict, retrieve_resp: Dict, feedback_resp: Optional[Dict]) -> Dict:
        """Constructs the final response dictionary."""
        logger.debug("Step 9: Finalizing response.")
        final_response = {
            "memory_id": step_context["memory_id"],
            "intent_id": self._current_intent_id,
            "status": "completed", # Assume completion if we got this far
            "timestamp": datetime.utcnow().isoformat(),
            "neural_memory_update": update_resp,
            "neural_memory_retrieval": { # Structure this more cleanly
                 "success": retrieve_resp.get("success", False),
                 "retrieved_embedding": self._to_list(step_context.get("y_t_final")) if step_context.get("y_t_final") is not None else None,
                 "query_projection": self._to_list(step_context.get("q_t")) if step_context.get("q_t") is not None else None,
                 "error": retrieve_resp.get("error")
            },
            "surprise_metrics": {
                "loss": step_context.get("loss"),
                "grad_norm": step_context.get("grad_norm"),
                "boost_calculated": step_context.get("quickrecal_boost")
            },
            "quickrecal_feedback": feedback_resp or {"status": "N/A"},
            "variant_output": step_context.get("variant_metrics", {}) # Include variant metrics if any
        }
         # Add variant type to variant_output
        final_response["variant_output"]["variant_type"] = self.active_variant_type.value

        # Merge any errors captured earlier
        if "error" in base_response: final_response["error"] = base_response["error"]
        if not update_resp.get("success"): final_response["update_error"] = update_resp.get("error")
        if not retrieve_resp.get("success"): final_response["retrieval_error"] = retrieve_resp.get("error")

        # Update overall status if errors occurred
        if "error" in final_response or "update_error" in final_response or "retrieval_error" in final_response:
             final_response["status"] = "error_partial" if step_context["memory_id"] else "error_total"


        # Update CCE state
        if step_context.get("y_t_final") is not None:
             self.last_retrieved_embedding = self._to_list(step_context["y_t_final"])
        # Add to legacy sequence context (maybe remove later)
        self.sequence_context.append({
             "memory_id": step_context["memory_id"],
             "actual_embedding": self._to_list(step_context["x_t"]) if step_context["x_t"] is not None else None,
             "retrieved_embedding": self.last_retrieved_embedding,
             "surprise_metrics": final_response["surprise_metrics"],
             "timestamp": final_response["timestamp"],
             "intent_id": self._current_intent_id
         })
        if len(self.sequence_context) > 20: self.sequence_context.pop(0)


        return final_response

    def _finalize_error(self, message: str, context: dict, intent_id: Optional[str] = None) -> dict:
        """Constructs a standardized error response and finalizes intent."""
        intent_id = intent_id or self._current_intent_id
        logger.error(f"Finalizing with error: {message}", extra=context)
        response = {
            "status": "error",
            "error": message,
            "details": context.get("error", context.get("details", "No details")),
            "timestamp": datetime.utcnow().isoformat(),
            "intent_id": intent_id
        }
        if self.metrics_enabled:
            self.metrics_store.finalize_intent(
                intent_id=intent_id,
                response_text=f"Error: {message}",
                confidence=0.0
            )
        return response

    def _calculate_quickrecal_boost(self, surprise_value: Optional[float]) -> float:
        """Calculate quickrecal boost based on surprise value (loss or grad_norm)."""
        if surprise_value is None or surprise_value <= 0.0: return 0.0
        # Simple linear scaling for now, capped at 0.2
        # Example: loss/grad_norm of 1.0 gives 0.1 boost, 2.0 gives 0.2 boost
        max_expected_surprise = 2.0
        max_boost = 0.2
        boost = min(surprise_value / max_expected_surprise, 1.0) * max_boost
        logger.debug(f"Calculated quickrecal boost: {boost:.6f} from surprise value: {surprise_value:.6f}")
        return boost

    async def _apply_variant_pre_update(self, step_context: Dict) -> Dict:
        """Apply variant-specific pre-update processing for MAG/MAL variants.
        
        This method handles the variant-specific processing that must occur BEFORE
        the Neural Memory update:
        
        - MAG Variant: Calculates attention-based gate values (alpha_t, theta_t, eta_t)
          that control the Neural Memory update process:
          * alpha_t: Controls forgetting rate (higher = forget more)
          * theta_t: Controls learning rate (higher = learn faster)
          * eta_t: Controls momentum decay (higher = retain more momentum)
        
        - MAL Variant: Calculates a modified value projection (v_prime) by applying
          attention between the current query and historical keys/values. This enhances
          the value representation before it's stored in Neural Memory.
        
        Args:
            step_context: Current processing context containing embeddings and projections
        
        Returns:
            Dict containing variant processing results
        """
        if not self.variant_processor or self.active_variant_type not in [TitansVariantType.MAG, TitansVariantType.MAL]:
            return {"success": True} # No pre-processing needed

        logger.debug(f"Step 3: Applying {self.active_variant_type.value} pre-update logic...")
        variant_results = {}
        try:
            # MAG: Calculate Gates
            if self.active_variant_type == TitansVariantType.MAG:
                # Retrieve K_hist
                k_hist = self.sequence_context_manager.get_recent_keys()
                if not k_hist:
                    logger.info("MAG: Not enough context for gate calculation.")
                    return {"success": True, "gates": None, "metrics": {}}

                # Ensure q_t and k_hist are tensors for attention
                try:
                    from synthians_memory_core.orchestrator.titans_variants import _get_tf
                    tf = _get_tf() # Lazy load TF
                    q_tensor = tf.convert_to_tensor([step_context["q_t"]], dtype=tf.float32)
                    k_hist_tensor = tf.convert_to_tensor(k_hist, dtype=tf.float32)
                    if len(k_hist_tensor.shape) == 2: k_hist_tensor = tf.expand_dims(k_hist_tensor, 0)

                    # Calculate attention output (Query attends to historical Keys)
                    attention_output_tensor = self.attention_module(
                        query=q_tensor, key=k_hist_tensor, value=k_hist_tensor, training=False
                    )
                    attention_output_list = tf.squeeze(attention_output_tensor).numpy().tolist()
                except Exception as e:
                    logger.error(f"Error during MAG attention calculation: {e}")
                    return {"success": False, "error": str(e), "gates": None, "metrics": {}}

                # Call NM API to calculate gates
                gates_resp = await self._make_request(
                    self.neural_memory_url, "/calculate_gates", method="POST",
                    payload={"attention_output": attention_output_list}
                )
                if "error" not in gates_resp:
                     variant_results = {
                         "success": True,
                         "gates": {"alpha_t": gates_resp["alpha"], "theta_t": gates_resp["theta"], "eta_t": gates_resp["eta"]},
                         "metrics": getattr(self.attention_module, 'get_metrics', lambda: {})() # Safe access
                     }
                     logger.info(f"MAG calculated gates: {variant_results['gates']}")
                else:
                     logger.error(f"MAG failed to calculate gates via API: {gates_resp.get('error')}")
                     variant_results = {"success": False, "error": gates_resp.get('error'), "gates": None, "metrics": {}}

            # MAL: Calculate v_prime_t
            elif self.active_variant_type == TitansVariantType.MAL:
                k_hist, v_hist = self.sequence_context_manager.get_recent_kv_pairs()
                if not k_hist or not v_hist:
                     logger.info("MAL: Not enough context for value augmentation.")
                     return {"success": True, "v_prime_t": step_context["v_t"], "metrics": {}} # Return original v_t

                # Call variant processor's method (assuming it exists and handles TF conversion)
                # This requires `titans_variants.MALVariant` to have the calculation logic
                mal_output = await self.variant_processor.calculate_v_prime(
                    q_t=step_context["q_t"],
                    v_t=step_context["v_t"],
                    k_hist=k_hist,
                    v_hist=v_hist
                )
                if mal_output and mal_output.get("success"):
                     v_prime_t = mal_output["v_prime_t"]
                     if self._validate_embedding(v_prime_t):
                         variant_results = {"success": True, "v_prime_t": v_prime_t, "metrics": mal_output.get("metrics", {})}
                         logger.info("MAL calculated v_prime_t.")
                     else:
                          logger.error("MAL variant returned invalid v_prime_t.")
                          variant_results = {"success": False, "error": "Invalid v_prime_t from MAL", "v_prime_t": step_context["v_t"]}
                else:
                     logger.error(f"MAL variant processing failed: {mal_output.get('error')}")
                     variant_results = {"success": False, "error": mal_output.get('error'), "v_prime_t": step_context["v_t"]}


        except Exception as e:
            logger.error(f"Error during variant pre-update ({self.active_variant_type.value}): {e}", exc_info=True)
            return {"success": False, "error": str(e)}

        return {"success": True, **variant_results} # Default success if no relevant variant

    async def _update_neural_memory(self, step_context: Dict) -> Dict:
        """Update Neural Memory with appropriate modifications based on active variant.
        
        This method handles the Neural Memory update process with variant-specific modifications:
        
        - NONE Variant: Standard update with the input embedding only
        - MAC Variant: Standard update (variant processing occurs after retrieval)
        - MAG Variant: Update with externally calculated gate values (alpha_t, theta_t, eta_t)
        - MAL Variant: Update with modified value projection (v_prime)
        
        Args:
            step_context: Current processing context containing embeddings and projections
        
        Returns:
            Dict containing update response with loss and gradient norm
        """
        logger.debug("Step 4: Updating Neural Memory...")
        update_payload = {"input_embedding": self._to_list(step_context["x_t"])} # Base payload

        # Add MAG gates if calculated
        if step_context["external_gates"]:
             gates = step_context["external_gates"]
             # Use the specific keys expected by the updated UpdateMemoryRequest
             update_payload["external_alpha_gate"] = gates.get("alpha_t")
             update_payload["external_theta_gate"] = gates.get("theta_t")
             update_payload["external_eta_gate"] = gates.get("eta_t")
             logger.info("Using MAG external gates for update.")

        # Add MAL projections if calculated (v_prime_t overrides default v_t)
        elif step_context["v_prime_t"] is not None:
             if step_context["k_t"] is None:
                 logger.error("MAL Error: v_prime_t calculated but k_t is missing.")
                 return {"success": False, "error": "k_t missing for MAL update"}
             update_payload = { # Override payload for MAL
                 "input_embedding": self._to_list(step_context["x_t"]),
                 "key_projection": self._to_list(step_context["k_t"]),
                 "value_projection": self._to_list(step_context["v_prime_t"])
             }
             logger.info("Using MAL explicit projections (k_t, v_prime_t) for update.")

        update_resp = await self._make_request(
            self.neural_memory_url, "/update_memory", method="POST", payload=update_payload
        )

        if "error" in update_resp:
            return {"success": False, **update_resp}
        else:
            logger.info(f"Neural Memory updated: Loss={update_resp.get('loss'):.6f}, GradNorm={update_resp.get('grad_norm'):.6f}")
            # Log memory update metrics if enabled
            if self.metrics_enabled:
                self.metrics_store.log_memory_update(
                    input_embedding=self._to_list(step_context["x_t"]),
                    loss=update_resp.get("loss"),
                    grad_norm=update_resp.get("grad_norm", 0.0),
                    emotion=step_context["user_emotion"],
                    intent_id=self._current_intent_id,
                    metadata={
                        "memory_id": step_context["memory_id"],
                        "content_preview": step_context["content"][:50] if step_context["content"] else "",
                        "variant_type": self.active_variant_type.value
                    }
                )
            
        # Check for errors
        if "error" in update_resp:
             logger.error(f"Neural Memory update failed: {update_resp['error']}")
             return {"success": False, **update_resp}
             
        # Extract metrics for subsequent processing
        if "loss" in update_resp:
            step_context["loss"] = update_resp["loss"]
        if "grad_norm" in update_resp:
            step_context["grad_norm"] = update_resp["grad_norm"]
             
        logger.info("Neural Memory update successful")
        return {"success": True, **update_resp}

    async def _apply_quickrecal_boost(self, step_context: Dict, quickrecal_initial: Optional[float]) -> Optional[Dict]:
         """Calculates and applies QuickRecal boost if needed."""
         logger.debug("Step 5: Applying QuickRecal boost...")
         loss = step_context.get("loss")
         grad_norm = step_context.get("grad_norm")
         memory_id = step_context["memory_id"]
         user_emotion = step_context["user_emotion"]

         if memory_id and (loss is not None or grad_norm is not None):
             surprise_metric = grad_norm if grad_norm is not None else loss
             boost = self._calculate_quickrecal_boost(surprise_metric)
             step_context["quickrecal_boost"] = boost # Store calculated boost

             if boost > 1e-4:
                 loss_str = f"{loss:.6f}" if isinstance(loss, (float, int)) else 'N/A'
                 grad_norm_str = f"{grad_norm:.6f}" if isinstance(grad_norm, (float, int)) else 'N/A'
                 feedback_payload = {
                     "memory_id": memory_id, "delta": boost,
                     "reason": f"NM Surprise (Loss:{loss_str}, GradNorm:{grad_norm_str})"
                 }
                 feedback_resp = await self._make_request(
                     self.memory_core_url, "/api/memories/update_quickrecal_score",
                     method="POST", payload=feedback_payload
                 )
                 if "error" in feedback_resp:
                      logger.error(f"QuickRecal boost failed: {feedback_resp.get('error')}")
                      return {"status": "error", "error": feedback_resp.get('error')}
                 else:
                      logger.info(f"QuickRecal boost applied: Delta={boost:.6f}")
                      if self.metrics_enabled:
                          self.metrics_store.log_quickrecal_boost(
                              memory_id=memory_id, base_score=quickrecal_initial or 0.0,
                              boost_amount=boost, emotion=user_emotion, intent_id=self._current_intent_id,
                              metadata={"loss": loss, "grad_norm": grad_norm, "reason": "NM Surprise"}
                          )
                 return feedback_resp
             else:
                  logger.debug("QuickRecal boost skipped (too small).")
                  return {"status": "skipped", "reason": "Boost too small"}
         else:
              logger.warning("Skipping QuickRecal boost: Missing memory_id or surprise metrics.")
              return {"status": "skipped", "reason": "Missing ID or surprise metrics"}

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
