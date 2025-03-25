"""
Embedding Manager Module for Lucidia's Knowledge Graph

This module implements vector embedding transformations, hyperbolic space projections,
similarity calculations, embedding alignment and normalization, and batch conversion processes.
"""

import numpy as np
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from collections import defaultdict

from .base_module import KnowledgeGraphModule

class EmbeddingManager(KnowledgeGraphModule):
    """
    Embedding Manager responsible for vector operations in the knowledge graph.
    
    This module manages embedding transformations, hyperbolic space projections,
    similarity calculations, and batch conversion processes.
    """
    
    def __init__(self, event_bus, module_registry, config=None):
        """Initialize the Embedding Manager."""
        super().__init__(event_bus, module_registry, config)
        
        # Hyperbolic embeddings configuration
        self.hyperbolic_embedding = {
            "enabled": self.get_config("enable_hyperbolic", False),
            "curvature": self.get_config("hyperbolic_curvature", 1.0),
            "initialized": False,
            "dimensions": self.get_config("embedding_dimension", 768)
        }
        
        # Default embedding dimension
        self.default_embedding_dimension = self.get_config("embedding_dimension", 768)
        
        # Initialize embeddings cache
        self.embedding_cache = {}
        
        # Logger
        self.logger = logging.getLogger("EmbeddingManager")
        
        # We'll subscribe to events in _setup_module since that method is async
        # and can properly await the event subscriptions
        
        self.logger.info("Embedding Manager initialized")
    
    async def _subscribe_to_events(self):
        """Subscribe to relevant events."""
        await self.event_bus.subscribe("embedding.transform", self._handle_embedding_transform)
        await self.event_bus.subscribe("embedding.validate", self._handle_embedding_validation)
        await self.event_bus.subscribe("embedding.compare", self._handle_embedding_comparison)
        await self.event_bus.subscribe("embedding.hyperbolic.convert", self._handle_hyperbolic_conversion)
        await self.event_bus.subscribe("embedding.batch.process", self._handle_batch_processing)
        self.logger.info("Subscribed to embedding-related events")
    
    async def _setup_module(self) -> None:
        """Set up module-specific resources and state."""
        # Register operation handlers
        self.module_registry.register_operation_handler("generate_embedding", self.generate_embedding)
        self.module_registry.register_operation_handler("transform_embedding", self.transform_embedding)
        self.module_registry.register_operation_handler("validate_embedding", self.validate_embedding)
        self.module_registry.register_operation_handler("compare_embeddings", self.compare_embeddings)
        self.module_registry.register_operation_handler("to_hyperbolic", self.to_hyperbolic)
        self.module_registry.register_operation_handler("from_hyperbolic", self.from_hyperbolic)
        self.module_registry.register_operation_handler("convert_nodes_to_hyperbolic", self.convert_nodes_to_hyperbolic)
        self.module_registry.register_operation_handler("batch_convert_to_hyperbolic", self.batch_convert_to_hyperbolic)
        self.module_registry.register_operation_handler("align_vectors", self._align_vectors_for_comparison)
        self.module_registry.register_operation_handler("hyperbolic_distance", self.hyperbolic_distance)
        self.module_registry.register_operation_handler("hyperbolic_similarity", self.hyperbolic_similarity)
        self.module_registry.register_operation_handler("euclidean_to_hyperbolic", self.euclidean_to_hyperbolic)
        
        # Initialize the hyperbolic space if enabled
        if self.hyperbolic_embedding["enabled"]:
            self.logger.info(f"Hyperbolic embedding enabled with curvature {self.hyperbolic_embedding['curvature']}")
        
        # Subscribe to events
        await self._subscribe_to_events()
        
        self.logger.info("Embedding Manager setup complete")

    def _validate_embedding(self, embedding):
        """Validate an embedding to ensure it contains no NaN or Inf values.
        
        Args:
            embedding: Vector embedding to validate
            
        Returns:
            bool: True if embedding is valid, False otherwise
        """
        if embedding is None:
            return False
            
        try:
            # Convert embedding to numpy array if it's not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding, dtype=np.float32)
                
            # Check for NaN or Inf values
            if np.isnan(embedding).any() or np.isinf(embedding).any():
                self.logger.warning("Embedding contains NaN or Inf values")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating embedding: {e}")
            return False
    
    def _align_vectors_for_comparison(self, vec1, vec2):
        """Align two vectors to the same dimension for comparison.
        
        This handles cases where embeddings might have different dimensions,
        such as when comparing 384-dimension and 768-dimension vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            tuple: (aligned_vec1, aligned_vec2)
        """
        # Convert inputs to numpy arrays if they aren't already
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1, dtype=np.float32)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2, dtype=np.float32)
        
        # If dimensions match, return as is
        if vec1.shape == vec2.shape:
            return vec1, vec2
        
        # Get dimensions
        dim1 = vec1.shape[0]
        dim2 = vec2.shape[0]
        
        # Determine target dimension (smaller of the two)
        target_dim = min(dim1, dim2)
        
        # Truncate vectors if needed
        aligned_vec1 = vec1[:target_dim] if dim1 > target_dim else vec1
        aligned_vec2 = vec2[:target_dim] if dim2 > target_dim else vec2
        
        # Zero-pad if one vector is smaller than target dimension
        if dim1 < target_dim:
            padding = np.zeros(target_dim - dim1, dtype=np.float32)
            aligned_vec1 = np.concatenate([aligned_vec1, padding])
        
        if dim2 < target_dim:
            padding = np.zeros(target_dim - dim2, dtype=np.float32)
            aligned_vec2 = np.concatenate([aligned_vec2, padding])
        
        return aligned_vec1, aligned_vec2

    def _normalize_embedding(self, embedding):
        """Normalize an embedding vector to unit length.
        
        Args:
            embedding: Vector to normalize
            
        Returns:
            numpy.ndarray: Normalized vector
        """
        if embedding is None:
            return None
        
        # Convert to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Check if embedding is valid
        if not self._validate_embedding(embedding):
            # Return a zero vector of appropriate dimension if invalid
            return np.zeros(self.default_embedding_dimension, dtype=np.float32)
        
        # Handle different embedding dimensions
        if embedding.shape[0] != self.default_embedding_dimension:
            # Pad or truncate to match default dimension
            if embedding.shape[0] < self.default_embedding_dimension:
                # Pad with zeros
                padding = np.zeros(self.default_embedding_dimension - embedding.shape[0], dtype=np.float32)
                embedding = np.concatenate([embedding, padding])
            else:
                # Truncate
                embedding = embedding[:self.default_embedding_dimension]
        
        # Normalize vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        else:
            return embedding

    def _handle_embedding_validation(self, data):
        """Handle embedding validation events.
        
        Args:
            data: Event data containing 'embedding'
            
        Returns:
            Dict containing validation result
        """
        embedding = data.get('embedding')
        is_valid = self._validate_embedding(embedding)
        
        return {
            'is_valid': is_valid,
            'original_embedding': embedding,
            'normalized_embedding': self._normalize_embedding(embedding) if is_valid else None
        }

    async def _handle_node_added(self, data):
        """
        Handle node added events for embedding generation.
        
        Args:
            data: Node added event data
        """
        node_id = data.get("node_id")
        if not node_id:
            return
        
        # Check if we should generate embeddings automatically
        auto_generate = self.get_config("auto_generate_embeddings", False)
        if auto_generate:
            # Get the node from core graph manager
            core_graph = self.module_registry.get_module("core_graph")
            if not core_graph:
                self.logger.error("Core graph module not found")
                return
            
            node_data = await core_graph.get_node(node_id)
            if not node_data:
                self.logger.warning(f"Node {node_id} not found")
                return
            
            # Only generate embeddings if node doesn't already have one
            if "embedding" not in node_data:
                await self.generate_embedding(node_id, node_data)
    
    async def _handle_node_updated(self, data):
        """
        Handle node updated events for embedding updates.
        
        Args:
            data: Node updated event data
        """
        node_id = data.get("node_id")
        if not node_id:
            return
        
        # Check if we should update embeddings automatically
        auto_update = self.get_config("auto_update_embeddings", False)
        if auto_update:
            # Get the node from core graph manager
            core_graph = self.module_registry.get_module("core_graph")
            if not core_graph:
                self.logger.error("Core graph module not found")
                return
            
            node_data = await core_graph.get_node(node_id)
            if not node_data:
                self.logger.warning(f"Node {node_id} not found")
                return
            
            # Check if key attributes changed that would require embedding update
            attributes = data.get("attributes", {})
            update_needed = any(attr in attributes for attr in ['name', 'definition', 'description', 'content'])
            
            if update_needed:
                await self.generate_embedding(node_id, node_data)
    
    async def _handle_embedding_conversion(self, data):
        """
        Handle embedding conversion requests.
        
        Args:
            data: Conversion request data
            
        Returns:
            Conversion result
        """
        embedding = data.get("embedding")
        to_hyperbolic = data.get("to_hyperbolic", True)
        
        if embedding is None:
            return {"success": False, "error": "Embedding required"}
        
        try:
            if to_hyperbolic:
                result = self.to_hyperbolic(embedding)
                return {"success": True, "hyperbolic_embedding": result.tolist()}
            else:
                result = self.from_hyperbolic(embedding)
                return {"success": True, "euclidean_embedding": result.tolist()}
        except Exception as e:
            self.logger.error(f"Error converting embedding: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_embedding_batch(self, data):
        """
        Handle batch embedding requests.
        
        Args:
            data: Batch request data
            
        Returns:
            Batch processing result
        """
        node_ids = data.get("node_ids", [])
        node_types = data.get("node_types", [])
        domains = data.get("domains", [])
        to_hyperbolic = data.get("to_hyperbolic", True)
        
        if not node_ids and not node_types and not domains:
            return {"success": False, "error": "Node IDs, types, or domains required"}
        
        # Get core graph for access to nodes
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Get nodes to process
        nodes_to_process = set()
        
        # Add nodes by ID
        for node_id in node_ids:
            if await core_graph.has_node(node_id):
                nodes_to_process.add(node_id)
        
        # Add nodes by type
        for node_type in node_types:
            type_nodes = await core_graph.get_nodes_by_type(node_type)
            nodes_to_process.update(type_nodes)
        
        # Add nodes by domain
        for domain in domains:
            domain_nodes = await core_graph.get_nodes_by_domain(domain)
            nodes_to_process.update(domain_nodes)
        
        # Process nodes
        processed = 0
        skipped = 0
        errors = 0
        
        for node_id in nodes_to_process:
            node_data = await core_graph.get_node(node_id)
            if not node_data:
                skipped += 1
                continue
            
            try:
                result = await self.generate_embedding(node_id, node_data, to_hyperbolic)
                if result:
                    processed += 1
                else:
                    skipped += 1
            except Exception as e:
                self.logger.error(f"Error generating embedding for {node_id}: {e}")
                errors += 1
        
        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "errors": errors,
            "total": len(nodes_to_process)
        }
    
    async def _handle_calculate_similarity(self, data):
        """
        Handle similarity calculation requests.
        
        Args:
            data: Similarity request data
            
        Returns:
            Similarity result
        """
        # Get parameters
        item1 = data.get("item1")
        item2 = data.get("item2")
        use_hyperbolic = data.get("use_hyperbolic", self.hyperbolic_embedding["use_for_similarity"])
        
        if item1 is None or item2 is None:
            return {"success": False, "error": "Two items required for similarity calculation"}
        
        # Calculate similarity based on item types
        if isinstance(item1, (list, np.ndarray)) and isinstance(item2, (list, np.ndarray)):
            # Both are embeddings
            similarity = self.calculate_similarity(item1, item2, use_hyperbolic)
            return {"success": True, "similarity": similarity}
        
        elif isinstance(item1, str) and isinstance(item2, str):
            # Both are node IDs or text
            similarity = await self.calculate_node_similarity(item1, item2, use_hyperbolic)
            return {"success": True, "similarity": similarity}
        
        else:
            # Mixed types not supported
            return {"success": False, "error": "Unsupported item types for similarity calculation"}
    
    async def _handle_find_nearest(self, data):
        """
        Handle find nearest nodes requests.
        
        Args:
            data: Nearest nodes request data
            
        Returns:
            Nearest nodes result
        """
        # Get parameters
        reference = data.get("reference")
        limit = data.get("limit", 10)
        threshold = data.get("threshold", 0.5)
        use_hyperbolic = data.get("use_hyperbolic", self.hyperbolic_embedding["use_for_similarity"])
        node_types = data.get("node_types", [])
        domains = data.get("domains", [])
        
        if reference is None:
            return {"success": False, "error": "Reference item required"}
        
        # Find nearest nodes
        nearest_nodes = await self.find_nearest_nodes(
            reference, limit, threshold, use_hyperbolic, node_types, domains
        )
        
        return {"success": True, "nearest_nodes": nearest_nodes}
    
    async def generate_embedding(self, node_id: str, node_data: Dict[str, Any], to_hyperbolic: bool = True) -> bool:
        """
        Generate embedding for a node.
        
        Args:
            node_id: ID of the node
            node_data: Node data
            to_hyperbolic: Whether to also generate hyperbolic embedding
            
        Returns:
            Success status
        """
        # Skip if node already has an embedding
        if "embedding" in node_data and not self.get_config("force_regenerate", False):
            self.logger.debug(f"Node {node_id} already has an embedding")
            
            # Still generate hyperbolic if needed
            if to_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" not in node_data:
                embedding = node_data["embedding"]
                if not isinstance(embedding, np.ndarray):
                    embedding = np.array(embedding, dtype=np.float32)
                
                hyperbolic_emb = self.to_hyperbolic(embedding)
                
                # Update node with hyperbolic embedding
                await self._update_node_embedding(node_id, None, hyperbolic_emb.tolist())
                return True
            
            return False
        
        # Extract node text for embedding
        node_text = self._extract_node_text(node_data)
        
        # Generate embedding
        embedding = self._generate_embedding_from_text(node_text)
        
        # Generate hyperbolic embedding if needed
        hyperbolic_embedding = None
        if to_hyperbolic and self.hyperbolic_embedding["enabled"]:
            hyperbolic_embedding = self.to_hyperbolic(embedding).tolist()
        
        # Update node with new embedding
        await self._update_node_embedding(node_id, embedding.tolist(), hyperbolic_embedding)
        
        return True
    
    def _extract_node_text(self, node_data: Dict[str, Any]) -> str:
        """
        Extract text from node data for embedding generation.
        
        Args:
            node_data: Node data
            
        Returns:
            Text for embedding generation
        """
        node_type = node_data.get("type", "unknown")
        
        if node_type == "concept":
            # For concepts, use definition
            text = f"{node_data.get('id', '')} {node_data.get('definition', '')}"
        elif node_type == "entity":
            # For entities, use name and description
            text = f"{node_data.get('id', '')} {node_data.get('name', '')} {node_data.get('description', '')}"
        elif node_type == "dream_insight":
            # For dream insights, use insight text
            text = f"{node_data.get('id', '')} {node_data.get('insight', '')}"
        else:
            # For other types, combine available text fields
            text = f"{node_data.get('id', '')} "
            for key, value in node_data.items():
                if isinstance(value, str) and key not in ["id", "type", "domain", "created", "modified"]:
                    text += f" {value}"
        
        return text
    
    def _generate_embedding_from_text(self, text: str) -> np.ndarray:
        """
        Generate embedding vector from text.
        
        This is a simple implementation that would be replaced with a more
        sophisticated model in a real implementation.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector
        """
        # Check if we have cached this text
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            self.cache_hit_count += 1
            return self.embedding_cache[cache_key]
        
        self.cache_miss_count += 1
        
        # In a real implementation, this would use a proper embedding model
        # Here we use a simple hash-based approach as a placeholder
        
        # Get embedding dimension
        dim = self.default_embedding_dimension
        
        # Generate pseudo-random embedding based on text hash
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Generate embedding from hash bytes
        embedding = np.zeros(dim, dtype=np.float32)
        for i in range(min(16, dim)):
            embedding[i] = (hash_bytes[i] / 255.0) * 2 - 1
            
        # Stretch to fill remaining dimensions with oscillations
        if dim > 16:
            for i in range(16, dim):
                idx = i % 16
                phase = (i // 16) * np.pi / 8
                embedding[i] = embedding[idx] * np.cos(phase)
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        # Update cache
        if len(self.embedding_cache) >= self.cache_limit:
            # Evict a random item if cache is full
            key_to_evict = next(iter(self.embedding_cache))
            del self.embedding_cache[key_to_evict]
        
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    async def _update_node_embedding(self, node_id: str, embedding: Optional[List[float]], 
                            hyperbolic_embedding: Optional[List[float]]) -> bool:
        """
        Update node with new embedding.
        
        Args:
            node_id: ID of the node
            embedding: Standard embedding (None to keep existing)
            hyperbolic_embedding: Hyperbolic embedding (None to keep existing)
            
        Returns:
            Success status
        """
        # Get core graph for node updates
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return False
        
        # Prepare updates
        updates = {}
        
        if embedding is not None:
            updates["embedding"] = embedding
        
        if hyperbolic_embedding is not None:
            updates["hyperbolic_embedding"] = hyperbolic_embedding
            # Track node as having hyperbolic embedding
            self.hyperbolic_embedding["initialized"] = True
        
        # Update node
        if updates:
            return await core_graph.update_node(node_id, updates)
        
        return True
    
    def to_hyperbolic(self, euclidean_vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Project a Euclidean vector into the Poincaré ball model of hyperbolic space.
        
        Args:
            euclidean_vector: Vector in Euclidean space
            
        Returns:
            Vector in hyperbolic space (Poincaré ball model)
        """
        if not self.hyperbolic_embedding["enabled"]:
            return euclidean_vector
            
        # Validate input
        if euclidean_vector is None:
            # Return zero vector of appropriate dimension
            return np.zeros(self.default_embedding_dimension)
            
        # Convert to numpy array if needed
        if not isinstance(euclidean_vector, np.ndarray):
            try:
                euclidean_vector = np.array(euclidean_vector, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Error converting to numpy array: {e}")
                return np.zeros(self.default_embedding_dimension)
        
        # Check for NaN or Inf values
        if np.isnan(euclidean_vector).any() or np.isinf(euclidean_vector).any():
            self.logger.warning("NaN or Inf values detected in embedding, replaced with zeros")
            # Return zero vector of appropriate dimension
            return np.zeros(self.default_embedding_dimension)
            
        # Normalize the Euclidean vector
        norm = np.linalg.norm(euclidean_vector)
        if norm == 0:
            return euclidean_vector  # Return zero vector as is
            
        # Scale vector to fit within the Poincaré ball (norm < 1)
        # Apply a tanh-based scaling to ensure the norm is < 1
        scale_factor = np.tanh(norm / (self.hyperbolic_embedding["curvature"] * 4))
        return (euclidean_vector / norm) * scale_factor
    
    def euclidean_to_hyperbolic(self, euclidean_vector):
        """Convert a Euclidean vector to hyperbolic space.
        
        This is an alias for to_hyperbolic to maintain backward compatibility.
        
        Args:
            euclidean_vector: Vector in Euclidean space
            
        Returns:
            Vector in hyperbolic space
        """
        return self.to_hyperbolic(euclidean_vector)
    
    def from_hyperbolic(self, hyperbolic_vector: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Project a vector from the Poincaré ball model back to Euclidean space.
        
        Args:
            hyperbolic_vector: Vector in hyperbolic space (Poincaré ball)
            
        Returns:
            Vector in Euclidean space
        """
        if not self.hyperbolic_embedding["enabled"]:
            return hyperbolic_vector
            
        # Validate input
        if hyperbolic_vector is None:
            # Return zero vector of appropriate dimension
            return np.zeros(self.default_embedding_dimension)
            
        # Convert to numpy array if needed
        if not isinstance(hyperbolic_vector, np.ndarray):
            try:
                hyperbolic_vector = np.array(hyperbolic_vector, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Error converting to numpy array: {e}")
                return np.zeros(self.default_embedding_dimension)
                
        # Check for NaN or Inf values
        if np.isnan(hyperbolic_vector).any() or np.isinf(hyperbolic_vector).any():
            self.logger.warning("NaN or Inf values detected in hyperbolic embedding, replaced with zeros")
            # Return zero vector of appropriate dimension
            return np.zeros(self.default_embedding_dimension)
            
        # Get the norm of the hyperbolic vector
        norm = np.linalg.norm(hyperbolic_vector)
        
        # If the vector is at or outside the ball boundary, adjust it
        if norm >= 1.0:
            return hyperbolic_vector * 0.99 / norm
            
        if norm == 0:
            return hyperbolic_vector  # Return zero vector as is
        
        # Apply inverse of the tanh-based scaling used in to_hyperbolic
        scale_factor = np.arctanh(norm) * (self.hyperbolic_embedding["curvature"] * 4)
        return (hyperbolic_vector / norm) * scale_factor
    
    def hyperbolic_similarity(self, v1, v2):
        """Calculate the similarity between two vectors in hyperbolic space.
        
        Args:
            v1: First hyperbolic vector
            v2: Second hyperbolic vector
            
        Returns:
            float: Similarity score between 0 and 1, where 1 indicates identical vectors
        """
        try:
            # Calculate hyperbolic distance
            dist = self.hyperbolic_distance(v1, v2)
            
            # Convert distance to similarity (1 - normalized distance)
            # Use a scaling factor based on the dimensionality and curvature
            max_distance = np.sqrt(self.hyperbolic_embedding["dimensions"]) * (1 / np.abs(self.hyperbolic_embedding["curvature"]))            
            similarity = 1.0 - min(dist / max_distance, 1.0)
            
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating hyperbolic similarity: {e}")
            return 0.0

    def hyperbolic_distance(self, v1, v2):
        """Calculate the distance between two vectors in hyperbolic space.
        
        Args:
            v1: First hyperbolic vector
            v2: Second hyperbolic vector
            
        Returns:
            float: Distance in hyperbolic space
        """
        try:
            # Ensure both vectors are numpy arrays
            v1 = np.array(v1)
            v2 = np.array(v2)
            
            # Calculate hyperbolic distance using the Poincaré distance formula
            # d(x,y) = arcosh(1 + 2||x-y||^2/((1-||x||^2)(1-||y||^2)))
            
            # Get the squared norms
            norm_x_sq = np.sum(v1**2)
            norm_y_sq = np.sum(v2**2)
            
            # Hyperbolic space constraint checks
            if norm_x_sq >= 1.0 or norm_y_sq >= 1.0:
                self.logger.warning("Vectors must be within the unit ball for hyperbolic calculations")
                # Project back to unit ball if needed
                if norm_x_sq >= 1.0:
                    v1 = v1 * (0.999 / np.sqrt(norm_x_sq))
                    norm_x_sq = np.sum(v1**2)
                    
                if norm_y_sq >= 1.0:
                    v2 = v2 * (0.999 / np.sqrt(norm_y_sq))
                    norm_y_sq = np.sum(v2**2)
            
            # Calculate squared distance between the points
            delta_sq = np.sum((v1 - v2)**2)
            
            # Calculate the denominator term
            denom = (1 - norm_x_sq) * (1 - norm_y_sq)
            
            # Avoid division by zero
            if denom < 1e-10:
                denom = 1e-10
                
            # Calculate the argument for arcosh
            arg = 1 + 2 * delta_sq / denom
            
            # Ensure the argument is valid for arcosh (must be >= 1)
            if arg < 1.0:
                arg = 1.0
                
            # Apply the curvature adjustment
            c = abs(self.hyperbolic_embedding["curvature"])
            distance = np.arccosh(arg) / np.sqrt(c)
            
            return distance
        except Exception as e:
            self.logger.error(f"Error calculating hyperbolic distance: {e}")
            return float('inf')  # Return infinite distance on error

    def calculate_similarity(self, vector_a: Union[List[float], np.ndarray], 
                        vector_b: Union[List[float], np.ndarray], 
                        use_hyperbolic: bool = True) -> float:
        """
        Calculate similarity between two embedding vectors.
        
        Args:
            vector_a: First vector
            vector_b: Second vector
            use_hyperbolic: Whether to use hyperbolic similarity if available
            
        Returns:
            Similarity score (0-1)
        """
        # Convert to numpy arrays if needed
        if not isinstance(vector_a, np.ndarray):
            vector_a = np.array(vector_a, dtype=np.float32)
        if not isinstance(vector_b, np.ndarray):
            vector_b = np.array(vector_b, dtype=np.float32)
        
        # Check for NaN or Inf values
        if (np.isnan(vector_a).any() or np.isinf(vector_a).any() or
            np.isnan(vector_b).any() or np.isinf(vector_b).any()):
            self.logger.warning("NaN or Inf values detected in vectors for similarity calculation")
            return 0.0
        
        # Use hyperbolic or standard similarity based on configuration
        if use_hyperbolic and self.hyperbolic_embedding["enabled"]:
            # Calculate hyperbolic distance
            distance = self.hyperbolic_distance(vector_a, vector_b)
            
            # Convert to similarity score
            similarity = 1.0 / (1.0 + distance)
            return similarity
        else:
            # Standard cosine similarity
            norm_a = np.linalg.norm(vector_a)
            norm_b = np.linalg.norm(vector_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            dot_product = np.dot(vector_a, vector_b)
            similarity = dot_product / (norm_a * norm_b)
            
            # Ensure similarity is in [0, 1] range
            similarity = (similarity + 1) / 2
            
            return similarity
    
    async def calculate_node_similarity(self, node_a: str, node_b: str, use_hyperbolic: bool = True) -> float:
        """
        Calculate similarity between two nodes or text strings.
        
        Args:
            node_a: First node ID or text
            node_b: Second node ID or text
            use_hyperbolic: Whether to use hyperbolic similarity if available
            
        Returns:
            Similarity score (0-1)
        """
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return 0.0
        
        # Check if inputs are node IDs
        node_a_data = None
        node_b_data = None
        
        if await core_graph.has_node(node_a):
            node_a_data = await core_graph.get_node(node_a)
        
        if await core_graph.has_node(node_b):
            node_b_data = await core_graph.get_node(node_b)
        
        # Get or generate embeddings
        embedding_a = None
        embedding_b = None
        
        # For node A
        if node_a_data:
            # Use existing embedding if available
            if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_a_data:
                embedding_a = node_a_data["hyperbolic_embedding"]
            elif "embedding" in node_a_data:
                embedding_a = node_a_data["embedding"]
            else:
                # Generate embedding
                await self.generate_embedding(node_a, node_a_data, use_hyperbolic)
                # Refresh node data
                node_a_data = await core_graph.get_node(node_a)
                if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_a_data:
                    embedding_a = node_a_data["hyperbolic_embedding"]
                elif "embedding" in node_a_data:
                    embedding_a = node_a_data["embedding"]
        else:
            # Generate embedding from text
            embedding_a = self._generate_embedding_from_text(node_a).tolist()
            if use_hyperbolic and self.hyperbolic_embedding["enabled"]:
                embedding_a = self.to_hyperbolic(embedding_a).tolist()
        
        # For node B
        if node_b_data:
            # Use existing embedding if available
            if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_b_data:
                embedding_b = node_b_data["hyperbolic_embedding"]
            elif "embedding" in node_b_data:
                embedding_b = node_b_data["embedding"]
            else:
                # Generate embedding
                await self.generate_embedding(node_b, node_b_data, use_hyperbolic)
                # Refresh node data
                node_b_data = await core_graph.get_node(node_b)
                if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_b_data:
                    embedding_b = node_b_data["hyperbolic_embedding"]
                elif "embedding" in node_b_data:
                    embedding_b = node_b_data["embedding"]
        else:
            # Generate embedding from text
            embedding_b = self._generate_embedding_from_text(node_b).tolist()
            if use_hyperbolic and self.hyperbolic_embedding["enabled"]:
                embedding_b = self.to_hyperbolic(embedding_b).tolist()
        
        # Calculate similarity
        if embedding_a is not None and embedding_b is not None:
            return self.calculate_similarity(embedding_a, embedding_b, use_hyperbolic)
        else:
            self.logger.warning("Could not calculate similarity due to missing embeddings")
            return 0.0
    
    def align_embeddings(self, vec_a: Union[List[float], np.ndarray], 
                     vec_b: Union[List[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two embeddings to have the same dimension.
        
        Args:
            vec_a: First embedding vector
            vec_b: Second embedding vector
            
        Returns:
            Tuple of aligned vectors
        """
        if vec_a is None or vec_b is None:
            # Return zero vectors of matching dimension
            dim = self.default_embedding_dimension
            return np.zeros(dim), np.zeros(dim)
            
        # Convert to numpy arrays if needed
        if not isinstance(vec_a, np.ndarray):
            vec_a = np.array(vec_a, dtype=np.float32)
        if not isinstance(vec_b, np.ndarray):
            vec_b = np.array(vec_b, dtype=np.float32)
            
        # Check if dimensions match
        if len(vec_a) == len(vec_b):
            return vec_a, vec_b
            
        # Align dimensions
        len_a = len(vec_a)
        len_b = len(vec_b)
        
        if len_a < len_b:
            # Pad vec_a with zeros
            padded_a = np.zeros(len_b, dtype=np.float32)
            padded_a[:len_a] = vec_a
            return padded_a, vec_b
        else:
            # Pad vec_b with zeros
            padded_b = np.zeros(len_a, dtype=np.float32)
            padded_b[:len_b] = vec_b
            return vec_a, padded_b
    
    async def find_nearest_nodes(self, reference: Union[str, List[float], np.ndarray], 
                           limit: int = 10, threshold: float = 0.5, 
                           use_hyperbolic: bool = True,
                           node_types: Optional[List[str]] = None,
                           domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Find nodes most similar to a reference node or embedding.
        
        Args:
            reference: Reference node ID, text, or embedding vector
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            use_hyperbolic: Whether to use hyperbolic similarity if available
            node_types: Optional list of node types to search within
            domains: Optional list of domains to search within
            
        Returns:
            List of nearest nodes with similarity scores
        """
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph module not found")
            return []
        
        # Get reference embedding
        reference_embedding = None
        
        if isinstance(reference, (list, np.ndarray)):
            # Reference is already an embedding
            reference_embedding = reference
        elif isinstance(reference, str):
            # Check if reference is a node ID
            if await core_graph.has_node(reference):
                # Get node data
                node_data = await core_graph.get_node(reference)
                
                # Use existing embedding if available
                if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_data:
                    reference_embedding = node_data["hyperbolic_embedding"]
                elif "embedding" in node_data:
                    reference_embedding = node_data["embedding"]
                else:
                    # Generate embedding
                    await self.generate_embedding(reference, node_data, use_hyperbolic)
                    # Refresh node data
                    node_data = await core_graph.get_node(reference)
                    if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_data:
                        reference_embedding = node_data["hyperbolic_embedding"]
                    elif "embedding" in node_data:
                        reference_embedding = node_data["embedding"]
            else:
                # Generate embedding from text
                reference_embedding = self._generate_embedding_from_text(reference).tolist()
                if use_hyperbolic and self.hyperbolic_embedding["enabled"]:
                    reference_embedding = self.to_hyperbolic(reference_embedding).tolist()
        
        if reference_embedding is None:
            self.logger.error("Could not get reference embedding")
            return []
        
        # Convert to numpy array if needed
        if not isinstance(reference_embedding, np.ndarray):
            reference_embedding = np.array(reference_embedding, dtype=np.float32)
        
        # Get nodes to search
        nodes_to_search = set()
        
        # Filter by node types if specified
        if node_types:
            for node_type in node_types:
                type_nodes = await core_graph.get_nodes_by_type(node_type)
                nodes_to_search.update(type_nodes)
        # Filter by domains if specified
        elif domains:
            for domain in domains:
                domain_nodes = await core_graph.get_nodes_by_domain(domain)
                nodes_to_search.update(domain_nodes)
        else:
            # Get all nodes
            nodes_to_search = set(core_graph.graph.nodes())
        
        # If reference is a node ID, exclude it from search
        if isinstance(reference, str) and reference in nodes_to_search:
            nodes_to_search.remove(reference)
        
        # Calculate similarity for each node
        similarity_results = []
        
        for node_id in nodes_to_search:
            node_data = await core_graph.get_node(node_id)
            
            # Get node embedding
            node_embedding = None
            
            if use_hyperbolic and self.hyperbolic_embedding["enabled"] and "hyperbolic_embedding" in node_data:
                node_embedding = node_data["hyperbolic_embedding"]
            elif "embedding" in node_data:
                node_embedding = node_data["embedding"]
            else:
                # Skip nodes without embeddings
                continue
            
            # Calculate similarity
            similarity = self.calculate_similarity(reference_embedding, node_embedding, use_hyperbolic)
            
            # Add to results if above threshold
            if similarity >= threshold:
                similarity_results.append({
                    "node_id": node_id,
                    "similarity": similarity,
                    "node_type": node_data.get("type", "unknown"),
                    "domain": node_data.get("domain", "general_knowledge")
                })
        
        # Sort by similarity (descending)
        similarity_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Limit results
        return similarity_results[:limit]
    
    async def batch_convert_to_hyperbolic(self, node_types: Optional[List[str]] = None,
                                  domains: Optional[List[str]] = None,
                                  batch_size: int = 100) -> Dict[str, Any]:
        """
        Batch convert existing node embeddings to hyperbolic space.
        
        Args:
            node_types: Optional list of node types to convert
            domains: Optional list of domains to convert
            batch_size: Number of nodes to process in each batch
            
        Returns:
            Dictionary with conversion statistics
        """
        if not self.hyperbolic_embedding["enabled"]:
            return {"success": False, "error": "Hyperbolic embeddings not enabled"}
        
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"success": False, "error": "Core graph module not found"}
        
        # Get nodes to convert
        nodes_to_convert = set()
        
        # Filter by node types if specified
        if node_types:
            for node_type in node_types:
                type_nodes = await core_graph.get_nodes_by_type(node_type)
                nodes_to_convert.update(type_nodes)
        # Filter by domains if specified
        elif domains:
            for domain in domains:
                domain_nodes = await core_graph.get_nodes_by_domain(domain)
                nodes_to_convert.update(domain_nodes)
        else:
            # Get all nodes
            nodes_to_convert = set(core_graph.graph.nodes())
        
        # Process nodes in batches
        processed = 0
        skipped = 0
        converted = 0
        errors = 0
        
        # Divide nodes into batches
        batches = [list(nodes_to_convert)[i:i+batch_size] for i in range(0, len(nodes_to_convert), batch_size)]
        
        for batch in batches:
            for node_id in batch:
                processed += 1
                
                try:
                    # Get node data
                    node_data = await core_graph.get_node(node_id)
                    if not node_data:
                        skipped += 1
                        continue
                    
                    # Skip if node already has hyperbolic embedding
                    if "hyperbolic_embedding" in node_data:
                        skipped += 1
                        continue
                    
                    # Skip if node doesn't have standard embedding
                    if "embedding" not in node_data:
                        skipped += 1
                        continue
                    
                    # Convert to hyperbolic
                    embedding = node_data["embedding"]
                    if not isinstance(embedding, np.ndarray):
                        embedding = np.array(embedding, dtype=np.float32)
                    
                    hyperbolic_emb = self.to_hyperbolic(embedding)
                    
                    # Update node
                    await self._update_node_embedding(node_id, None, hyperbolic_emb.tolist())
                    converted += 1
                    
                except Exception as e:
                    self.logger.error(f"Error converting node {node_id} to hyperbolic: {e}")
                    errors += 1
            
            # Emit progress event
            await self.event_bus.emit("hyperbolic_conversion_progress", {
                "processed": processed,
                "skipped": skipped,
                "converted": converted,
                "errors": errors,
                "total": len(nodes_to_convert)
            })
        
        # Mark as initialized if any nodes were converted
        if converted > 0:
            self.hyperbolic_embedding["initialized"] = True
        
        return {
            "success": True,
            "processed": processed,
            "skipped": skipped,
            "converted": converted,
            "errors": errors,
            "total": len(nodes_to_convert),
            "hyperbolic_initialized": self.hyperbolic_embedding["initialized"]
        }
    
    async def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get statistics about embeddings in the knowledge graph.
        
        Returns:
            Dictionary with embedding statistics
        """
        # Get core graph for node access
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            return {"error": "Core graph module not found"}
        
        # Count nodes with embeddings
        nodes_with_embeddings = 0
        nodes_with_hyperbolic = 0
        total_nodes = 0
        
        # Count by node type
        embedding_by_type = defaultdict(int)
        hyperbolic_by_type = defaultdict(int)
        
        # Count by domain
        embedding_by_domain = defaultdict(int)
        hyperbolic_by_domain = defaultdict(int)
        
        # Check all nodes
        for node_id, node_data in core_graph.graph.nodes(data=True):
            total_nodes += 1
            node_type = node_data.get("type", "unknown")
            domain = node_data.get("domain", "general_knowledge")
            
            if "embedding" in node_data:
                nodes_with_embeddings += 1
                embedding_by_type[node_type] += 1
                embedding_by_domain[domain] += 1
            
            if "hyperbolic_embedding" in node_data:
                nodes_with_hyperbolic += 1
                hyperbolic_by_type[node_type] += 1
                hyperbolic_by_domain[domain] += 1
        
        return {
            "total_nodes": total_nodes,
            "nodes_with_embeddings": nodes_with_embeddings,
            "nodes_with_hyperbolic": nodes_with_hyperbolic,
            "embedding_coverage": nodes_with_embeddings / total_nodes if total_nodes > 0 else 0,
            "hyperbolic_coverage": nodes_with_hyperbolic / total_nodes if total_nodes > 0 else 0,
            "embedding_by_type": dict(embedding_by_type),
            "hyperbolic_by_type": dict(hyperbolic_by_type),
            "embedding_by_domain": dict(embedding_by_domain),
            "hyperbolic_by_domain": dict(hyperbolic_by_domain),
            "hyperbolic_enabled": self.hyperbolic_embedding["enabled"],
            "hyperbolic_initialized": self.hyperbolic_embedding["initialized"],
            "embedding_model": self.embedding_model,
            "cache_stats": {
                "size": len(self.embedding_cache),
                "limit": self.cache_limit,
                "hits": self.cache_hit_count,
                "misses": self.cache_miss_count,
                "hit_rate": self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) if (self.cache_hit_count + self.cache_miss_count) > 0 else 0
            }
        }

    async def convert_nodes_to_hyperbolic(self):
        """Convert all node embeddings to hyperbolic space.
        
        This method processes all nodes in the graph and converts their
        Euclidean embeddings to hyperbolic embeddings based on the
        configured curvature parameter.
        
        Returns:
            dict: Statistics about the conversion process
        """
        self.logger.info("Converting nodes to hyperbolic space")
        
        # Get all nodes from the core graph manager
        core_graph = self.module_registry.get_module("core_graph")
        if not core_graph:
            self.logger.error("Core graph manager not found, cannot convert nodes")
            return {"success": False, "error": "Core graph manager not found"}
        
        all_nodes = await core_graph.get_all_nodes()
        processed_count = 0
        error_count = 0
        
        for node_id, node_data in all_nodes.items():
            # Skip nodes without embeddings
            if "embedding" not in node_data:
                continue
                
            try:
                # Get the current embedding
                euclidean_embedding = node_data["embedding"]
                
                # Validate the embedding
                if not self._validate_embedding(euclidean_embedding):
                    continue
                    
                # Convert to hyperbolic space
                hyperbolic_embedding = self.to_hyperbolic(euclidean_embedding)
                
                # Store the hyperbolic embedding in the node attributes
                await core_graph.update_node(node_id, {"hyperbolic_embedding": hyperbolic_embedding})
                
                processed_count += 1
            except Exception as e:
                self.logger.error(f"Error converting node {node_id} to hyperbolic space: {e}")
                error_count += 1
        
        conversion_stats = {
            "total_nodes": len(all_nodes),
            "processed": processed_count,
            "errors": error_count,
            "success_rate": processed_count / len(all_nodes) if len(all_nodes) > 0 else 0
        }
        
        self.logger.info(f"Hyperbolic conversion complete. Stats: {conversion_stats}")
        return conversion_stats

    async def _handle_embedding_transform(self, data):
        """Handle embedding transformation events."""
        # Implementation would go here
        pass
        
    async def _handle_embedding_validation(self, data):
        """Handle embedding validation events."""
        # Implementation would go here
        pass
        
    async def _handle_embedding_comparison(self, data):
        """Handle embedding comparison events."""
        # Implementation would go here
        pass
        
    async def _handle_hyperbolic_conversion(self, data):
        """Handle hyperbolic conversion events."""
        # Implementation would go here
        pass
        
    async def _handle_batch_processing(self, data):
        """Handle batch processing of embeddings."""
        # Implementation would go here
        pass