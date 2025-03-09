# server/hypersphere_manager.py
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Union

from memory.lucidia_memory_system.core.hypersphere_dispatcher import HypersphereDispatcher
from memory.lucidia_memory_system.core.manifold_geometry import ManifoldGeometryRegistry
from memory.lucidia_memory_system.core.memory_entry import MemoryEntry

class HypersphereManager:
    """
    Manager for integrating the HypersphereDispatcher with the Lucidia memory system.
    
    This manager initializes and provides access to the HypersphereDispatcher,
    ensuring proper geometric compatibility and batch optimization for embedding operations.
    """
    
    def __init__(self, memory_client=None, config: Dict[str, Any] = None):
        """
        Initialize the HypersphereManager.
        
        Args:
            memory_client: Reference to the EnhancedMemoryClient for tensor/HPC operations
            config: Configuration dictionary
        """
        self.logger = logging.getLogger("HypersphereManager")
        self.config = config or {}
        self.memory_client = memory_client
        
        # Initialize geometry registry
        self.geometry_registry = ManifoldGeometryRegistry()
        
        # Configure dispatcher settings
        dispatcher_config = {
            "max_connections": self.config.get("max_connections", 5),
            "min_batch_size": self.config.get("min_batch_size", 1),
            "max_batch_size": self.config.get("max_batch_size", 32),
            "target_latency": self.config.get("target_latency", 100),  # ms
            "default_model_version": self.config.get("default_model_version", "latest"),
            "batch_timeout": self.config.get("batch_timeout", 0.1),  # seconds
            "retry_limit": self.config.get("retry_limit", 3),
            "error_cache_time": self.config.get("error_cache_time", 60),  # seconds
            "use_circuit_breaker": self.config.get("use_circuit_breaker", True),
        }
        
        # Initialize the hypersphere dispatcher
        tensor_server_uri = self.config.get("tensor_server_uri", "ws://nemo_sig_v3:5001")
        hpc_server_uri = self.config.get("hpc_server_uri", "ws://nemo_sig_v3:5005")
        
        self.dispatcher = HypersphereDispatcher(
            tensor_server_uri=tensor_server_uri,
            hpc_server_uri=hpc_server_uri,
            max_connections=dispatcher_config.get("max_connections", 5),
            min_batch_size=dispatcher_config.get("min_batch_size", 4),
            max_batch_size=dispatcher_config.get("max_batch_size", 32),
            target_latency=dispatcher_config.get("target_latency", 0.5),
            reconnect_backoff_min=0.1,
            reconnect_backoff_max=30.0,
            reconnect_backoff_factor=2.0,
            health_check_interval=60.0
        )
        
        self.logger.info("HypersphereManager initialized")
    
    async def initialize(self):
        """
        Complete async initialization tasks.
        
        This method registers the WebSocket interface with the dispatcher
        and performs any needed asynchronous setup.
        """
        try:
            # Register the tensor and HPC WebSocket connection handlers if memory_client is available
            if self.memory_client is not None:
                self.dispatcher.register_tensor_client(self.memory_client)
                self.dispatcher.register_hpc_client(self.memory_client)
                
                # Register supported model versions from config or defaults
                model_versions = self.config.get("supported_model_versions", ["latest", "v1", "v2"])
                for version in model_versions:
                    await self.register_model_version(version)
                
                self.logger.info(f"HypersphereManager registered tensor and HPC clients successfully")
            else:
                self.logger.warning("Memory client not available for HypersphereManager")
                
        except Exception as e:
            self.logger.error(f"Error during HypersphereManager initialization: {e}")
    
    async def register_model_version(self, version: str, dimensions: int = 768):
        """
        Register a model version with the geometry registry.
        
        Args:
            version: Model version identifier
            dimensions: Embedding dimensions for this model version
        """
        try:
            # Create geometric profile for the model version
            model_profile = {
                "dimensions": dimensions,
                "normalization": "unit_hypersphere",
                "distance_metric": "cosine",
                "compatible_versions": [version]  # Initially compatible only with itself
            }
            
            # Register with geometry registry
            self.geometry_registry.register_model_geometry(version, model_profile)
            self.logger.info(f"Registered model version {version} with {dimensions} dimensions")
            
            return {"status": "success", "version": version, "dimensions": dimensions}
        
        except Exception as e:
            self.logger.error(f"Error registering model version {version}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_embedding(self, text: str, model_version: str = "latest"):
        """
        Get embedding for text using the HypersphereDispatcher.
        
        Args:
            text: The text to embed
            model_version: The model version to use
            
        Returns:
            Dict containing the embedding and metadata
        """
        try:
            # Use the dispatcher to get the embedding
            embedding_result = await self.dispatcher.get_embedding(text=text, model_version=model_version)
            return embedding_result
        except Exception as e:
            self.logger.error(f"Error getting embedding: {e}")
            return {"status": "error", "message": str(e)}
    
    async def batch_similarity_search(self, query_embedding, memory_embeddings, memory_ids, model_version="latest", top_k=10):
        """
        Perform similarity search using the HypersphereDispatcher.
        
        Args:
            query_embedding: The query embedding
            memory_embeddings: List of memory embeddings to search against
            memory_ids: List of memory IDs corresponding to the embeddings
            model_version: The model version to use
            top_k: Number of top results to return
            
        Returns:
            List of dicts containing memory_id and similarity score
        """
        try:
            # Use the dispatcher to perform batch similarity search
            results = await self.dispatcher.batch_similarity_search(
                query_embedding=query_embedding,
                memory_embeddings=memory_embeddings,
                memory_ids=memory_ids,
                model_version=model_version,
                top_k=top_k
            )
            return results
        except Exception as e:
            self.logger.error(f"Error in batch similarity search: {e}")
            return []
    
    async def batch_process_embeddings(self, texts: List[str], model_version: str = "latest") -> Dict[str, Any]:
        """
        Process multiple texts into embeddings in a single batch operation.
        
        Args:
            texts: List of texts to embed
            model_version: The model version to use
            
        Returns:
            Dictionary containing all embeddings and metadata
        """
        try:
            self.logger.info(f"Processing batch of {len(texts)} embeddings using model {model_version}")
            
            # Check if dispatcher is properly initialized
            if not self.dispatcher or not hasattr(self.dispatcher, 'batch_get_embeddings'):
                raise ValueError("HypersphereDispatcher not properly initialized")
            
            # Process the batch of texts
            batch_results = await self.dispatcher.batch_get_embeddings(
                texts=texts,
                model_version=model_version
            )
            
            # Format the response
            embeddings = []
            
            # Check if the batch_results is properly structured
            if batch_results["status"] == "success" and "embeddings" in batch_results:
                # Get the embeddings array from the results
                result_embeddings = batch_results["embeddings"]
                
                for i, text in enumerate(texts):
                    # Make sure we have corresponding embedding result
                    if i < len(result_embeddings):
                        result = result_embeddings[i]
                        if result["status"] == "success":
                            embeddings.append({
                                "index": i,
                                "embedding": result["embedding"],
                                "dimensions": len(result["embedding"]),
                                "model_version": result.get("model_version", model_version),
                                "status": "success"
                            })
                        else:
                            embeddings.append({
                                "index": i,
                                "text": text[:50] + "..." if len(text) > 50 else text,
                                "status": "error",
                                "error": result.get("error", "Unknown error")
                            })
                    else:
                        # Handle case where result is missing
                        embeddings.append({
                            "index": i,
                            "text": text[:50] + "..." if len(text) > 50 else text,
                            "status": "error",
                            "error": "No embedding generated"
                        })
            else:
                # Handle error case
                error_msg = batch_results.get("message", "Unknown error in batch processing")
                for i, text in enumerate(texts):
                    embeddings.append({
                        "index": i,
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "status": "error",
                        "error": error_msg
                    })
            
            return {
                "status": "success",
                "model_version": model_version,
                "count": len(texts),
                "successful": sum(1 for e in embeddings if e["status"] == "success"),
                "embeddings": embeddings
            }
            
        except Exception as e:
            self.logger.error(f"Error processing batch embeddings: {e}")
            return {
                "status": "error",
                "model_version": model_version,
                "count": len(texts),
                "successful": 0,
                "message": str(e),
                "embeddings": []
            }
    
    async def shutdown(self):
        """
        Properly shutdown the HypersphereManager and its components.
        """
        try:
            if hasattr(self.dispatcher, "shutdown") and callable(self.dispatcher.shutdown):
                await self.dispatcher.shutdown()
            self.logger.info("HypersphereManager shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during HypersphereManager shutdown: {e}")
