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
        # Wait for configuration to be ready before processing
        if not self._config_ready:
            logger.info("Waiting for dynamic configuration to complete...")
            try:
                # Wait with a timeout to avoid blocking forever if configuration fails
                await asyncio.wait_for(self._config_ready_event.wait(), 10.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for configuration. Proceeding with defaults.")
                self._config_ready = True  # Prevent further waiting
            
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

            # Step 1: Store memory in Memory Core
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
                "timestamp": datetime.utcnow().isoformat(),
                "intent_id": self._current_intent_id,
                "variant_output": {
                    "variant_type": self.active_variant_type.value if hasattr(self, "active_variant_type") else "NONE"
                }
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
            
            # Step 2: Get projections from Neural Memory before updating
            # This is needed for all Titans variants
            key_projection = None
            value_projection = None
            query_projection = None
            original_value_projection = None  # Store the original value for later context management
            
            projections_resp = await self._make_request(
                self.neural_memory_url,
                "/get_projections",
                method="POST",
                payload={"input_embedding": actual_embedding}
            )
            
            # The response should contain direct projection fields, not nested under 'status'
            if "error" not in projections_resp and "key_projection" in projections_resp and "value_projection" in projections_resp and "query_projection" in projections_resp:
                key_projection = projections_resp.get("key_projection")
                value_projection = projections_resp.get("value_projection")
                original_value_projection = value_projection.copy() if isinstance(value_projection, list) else value_projection  # Store original for context
                query_projection = projections_resp.get("query_projection")
                logger.info(f"Got projections from Neural Memory: K:{len(key_projection) if key_projection else None}, "
                            f"V:{len(value_projection) if value_projection else None}, "
                            f"Q:{len(query_projection) if query_projection else None}")
                
                # Log dimensions for debugging
                logger.debug(f"Projection dimensions: key_dim={len(key_projection) if key_projection else 'Unknown'}, "
                            f"value_dim={len(value_projection) if value_projection else 'Unknown'}, "
                            f"query_dim={len(query_projection) if query_projection else 'Unknown'}")
            else:
                error_msg = projections_resp.get("error", "Unknown error")
                logger.warning(f"Failed to get projections from Neural Memory: {error_msg}")
                if "warning" in projections_resp:
                    logger.warning(f"Additional warnings: {projections_resp['warning']}")
            
            # Initialize variant-specific parameters
            external_gates = None
            modified_value_projection = None
            
            # Prepare context data if we have the projections
            x_t = None
            k_t = None
            v_t = None
            q_t = None
            y_t = None
            context_valid = False
            
            if actual_embedding and key_projection and value_projection and query_projection:
                try:
                    # Convert all projections to numpy arrays if they're not already
                    x_t = np.array(actual_embedding, dtype=np.float32)
                    k_t = np.array(key_projection, dtype=np.float32)
                    v_t = np.array(value_projection, dtype=np.float32)
                    q_t = np.array(query_projection, dtype=np.float32)
                    
                    # Validate all embeddings to ensure they don't contain NaN/Inf values
                    valid_embeddings = True
                    for embedding_name, embedding in [("x_t", x_t), ("k_t", k_t), ("v_t", v_t), ("q_t", q_t)]:
                        if not self._validate_embedding(embedding):
                            logger.warning(f"Invalid values detected in {embedding_name} projection. Skipping context processing.")
                            valid_embeddings = False
                            break
                    
                    if valid_embeddings:
                        context_valid = True
                        logger.debug("Projections validated for context processing")
                except Exception as e:
                    logger.error(f"Error preparing projections for context: {e}")
            
            # Step 3: Apply Titans variant logic BEFORE Neural Memory update if context is valid
            # This allows MAG and MAL variants to influence the update process
            if context_valid and self.variant_processor and self.active_variant_type != TitansVariantType.NONE:
                try:
                    logger.info(f"Applying {self.active_variant_type.value} variant logic before update...")
                    
                    # Process input with the variant processor
                    variant_results = self.variant_processor.process_input(
                        memory_id=memory_id,
                        x_t=x_t,
                        k_t=k_t,
                        v_t=v_t,
                        q_t=q_t,
                        y_t=None  # Will be populated after retrieval
                    )
                    
                    response["variant_output"] = {
                        "variant_type": self.active_variant_type.value,
                        "applied_before_update": True
                    }
                    
                    # For MAG variant: Get calculated gates to influence Neural Memory update
                    if self.active_variant_type == TitansVariantType.MAG and "alpha" in variant_results:
                        external_gates = {
                            "alpha": float(variant_results.get("alpha")),
                            "theta": float(variant_results.get("theta")),
                            "eta": float(variant_results.get("eta"))
                        }
                        response["variant_output"].update(external_gates)
                        logger.info(f"MAG variant calculated gates: alpha={external_gates['alpha']}, "
                                  f"theta={external_gates['theta']}, eta={external_gates['eta']}")
                    
                    # For MAL variant: Get modified value projection
                    elif self.active_variant_type == TitansVariantType.MAL and "v_prime" in variant_results:
                        modified_value_projection = variant_results.get("v_prime")
                        if isinstance(modified_value_projection, np.ndarray):
                            modified_value_projection = modified_value_projection.tolist()
                        response["variant_output"]["v_prime_applied"] = True
                        logger.info("MAL variant calculated augmented value projection.")
                    
                    logger.info(f"{self.active_variant_type.value} variant pre-processing applied successfully.")
                    
                except Exception as e:
                    logger.error(f"Error applying {self.active_variant_type.value} variant before update: {e}")
                    response["variant_output"]["pre_processing_error"] = str(e)
            
            # Step 4: Send embedding to Neural Memory for learning with MAG/MAL modifications if applicable
            update_payload = {"input_embedding": actual_embedding}
            
            # Add MAG external gates if available - properly format for the API endpoint
            if external_gates and self.active_variant_type == TitansVariantType.MAG:
                if "alpha" in external_gates and external_gates["alpha"] is not None:
                    update_payload["external_alpha_gate"] = external_gates["alpha"]
                if "theta" in external_gates and external_gates["theta"] is not None:
                    update_payload["external_theta_gate"] = external_gates["theta"]
                if "eta" in external_gates and external_gates["eta"] is not None:
                    update_payload["external_eta_gate"] = external_gates["eta"]
                
                logger.info(f"MAG: Sending external gates to /update_memory: alpha={update_payload.get('external_alpha_gate')}, "
                           f"theta={update_payload.get('external_theta_gate')}, eta={update_payload.get('external_eta_gate')}")
            
            # Add MAL modified value projection if available
            if modified_value_projection is not None and self.active_variant_type == TitansVariantType.MAL:
                update_payload["key_projection"] = key_projection
                update_payload["value_projection"] = modified_value_projection
                # Store the modified value projection for proper context recording
                v_t = np.array(modified_value_projection, dtype=np.float32)
            
            # If we have projections but no MAL-specific modifications, we can still use them
            elif key_projection is not None and value_projection is not None:
                update_payload["key_projection"] = key_projection
                update_payload["value_projection"] = value_projection
            
            update_resp = await self._make_request(
                self.neural_memory_url,
                "/update_memory",
                method="POST",
                payload=update_payload
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
                        "quickrecal_initial": quickrecal_initial,
                        "variant_type": self.active_variant_type.value if hasattr(self, "active_variant_type") else "NONE"
                    }
                )

            # Step 5: Calculate QuickRecal boost based on surprise metrics
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

            # Step 6: Generate query and retrieve associated embedding from Neural Memory
            # Ensure we're sending the correctly formatted request with debug logging
            retrieve_payload = {"input_embedding": actual_embedding}  # API expects raw input, handles projection internally
            
            # Add debug logging for MAG variant
            if self.active_variant_type and self.active_variant_type.value == "MAG":
                logger.info(f"MAG variant: Using input_embedding for /retrieve payload, dim={len(actual_embedding) if actual_embedding else 'None'}")
                logger.debug(f"MAG variant input details: input_dim={len(actual_embedding) if actual_embedding else 'Unknown'}, "
                            f"query_dim={len(q_t) if isinstance(q_t, np.ndarray) else 'Unknown'}")
            
            retrieve_resp = await self._make_request(
                self.neural_memory_url,
                "/retrieve",
                method="POST",
                payload=retrieve_payload
            )
            
            retrieved_embedding = retrieve_resp.get("retrieved_embedding")
            query_projection_from_retrieve = retrieve_resp.get("query_projection")  # May override earlier q_t
            
            # Enhanced logging for the retrieve response
            if retrieved_embedding:
                logger.info(f"Successfully retrieved embedding from Neural Memory, dim={len(retrieved_embedding)}")
                if query_projection_from_retrieve and query_projection_from_retrieve != query_projection:
                    # Update our q_t if the retrieve endpoint returned a different projection
                    logger.info(f"Using query projection from /retrieve endpoint. Dim={len(query_projection_from_retrieve)}")
                    query_projection = query_projection_from_retrieve
                    if context_valid:
                        q_t = np.array(query_projection, dtype=np.float32)
            else:
                logger.warning(f"Failed to retrieve embedding from Neural Memory: {retrieve_resp.get('error', 'Unknown error')}")
                
            response["neural_memory_retrieval"] = retrieve_resp  
            
            # Step 7: Process MAC variant or finalize context
            if context_valid and retrieved_embedding:
                try:
                    # Convert retrieved embedding to numpy
                    y_t_raw = np.array(retrieved_embedding, dtype=np.float32)
                    
                    # For MAC variant: Apply post-retrieval attention processing
                    if self.variant_processor and self.active_variant_type == TitansVariantType.MAC:
                        try:
                            # Process with MAC variant to get attended output
                            variant_results = self.variant_processor.process_input(
                                memory_id=memory_id,
                                x_t=x_t,
                                k_t=k_t,
                                v_t=v_t,
                                q_t=q_t,
                                y_t=y_t_raw
                            )
                            
                            response["variant_output"] = {
                                "variant_type": self.active_variant_type.value,
                                "applied_after_retrieval": True
                            }
                            
                            # Update the retrieved embedding with the attended output
                            if "attended_output" in variant_results:
                                y_t = variant_results["attended_output"]
                                if isinstance(y_t, np.ndarray):
                                    retrieved_embedding = y_t.tolist()
                                else:
                                    retrieved_embedding = y_t
                                    y_t = np.array(y_t, dtype=np.float32)
                                    
                                # Update the response with the attended output
                                response["neural_memory_retrieval"]["retrieved_embedding"] = retrieved_embedding
                                response["variant_output"]["attended_output"] = "applied"
                                logger.info("MAC variant updated retrieved_embedding with attended output.")
                            else:
                                # If no attended output, use the raw retrieved embedding
                                y_t = y_t_raw
                        except Exception as e:
                            logger.error(f"Error applying MAC variant after retrieval: {e}")
                            response["variant_output"]["post_processing_error"] = str(e)
                            y_t = y_t_raw  # Fall back to raw retrieved embedding
                    else:
                        # For other variants, use the raw retrieved embedding
                        y_t = y_t_raw
                    
                    # Now add the full context with all components to the sequence context manager
                    # Use a SINGLE add with the properly prepared v_t (which might be modified by MAL)
                    if self._validate_embedding(y_t):
                        # Record the final projections (including any MAL modifications to v_t)
                        self.sequence_context_manager.add_context(
                            memory_id=memory_id,
                            x_t=x_t,
                            k_t=k_t,
                            v_t=v_t,  # This is the v_t actually used in the update (possibly modified by MAL)
                            q_t=q_t,
                            y_t=y_t   # This is the final y_t (possibly modified by MAC)
                        )
                        logger.debug(f"Added complete context to SequenceContextManager. Current length: {len(self.sequence_context_manager)}")
                        
                        # Also store the original value projection for reference if it differs from what was used
                        if original_value_projection is not None and self.active_variant_type == TitansVariantType.MAL:
                            response["variant_output"]["original_v_preserved"] = True
                except Exception as e:
                    logger.error(f"Error finalizing context: {e}")
            
            # Log retrieval metrics if enabled
            if self.metrics_enabled and retrieved_embedding:
                # Create synthetic memory object since we don't have full metadata
                retrieved_memory = {
                    "memory_id": f"synthetic_{memory_id}_associated",
                    "embedding": retrieved_embedding,
                    "dominant_emotion": None  # We don't have this information yet
                }
                
                safe_query = np.array(actual_embedding, dtype=np.float32).tolist()
                safe_query = [0.0 if not np.isfinite(x) else x for x in safe_query]
                
                self.metrics_store.log_retrieval(
                    query_embedding=safe_query,
                    retrieved_memories=[retrieved_memory],
                    user_emotion=user_emotion,
                    intent_id=self._current_intent_id,
                    metadata={
                        "original_memory_id": memory_id,
                        "embedding_dim": len(retrieved_embedding),
                        "timestamp": datetime.utcnow().isoformat(),
                        "variant_type": self.active_variant_type.value if hasattr(self, "active_variant_type") else "NONE"
                    }
                )

            # Update sequence context for legacy compatibility
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
