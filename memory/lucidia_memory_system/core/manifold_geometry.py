"""
Manifold geometry handling for hypersphere embeddings in the Lucidia memory system.

Ensures embedding operations maintain consistent hypersphere geometry across
different model versions and embedding operations.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)

class ManifoldGeometryRegistry:
    """Ensures embedding operations maintain consistent hypersphere geometry."""
    
    def __init__(self):
        self.geometries = {}
        self.compatibility_cache = {}
        self.embeddings = {}  # Add this line to initialize the embeddings dictionary
        self.lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        
    async def register_geometry(self, model_version: str, dimensions: int, curvature: float, parameters: Dict[str, Any] = None) -> None:
        """Register a new model version with its hypersphere geometry parameters.
        
        Args:
            model_version: Version identifier for the model
            dimensions: Number of dimensions in the embedding space
            curvature: Hypersphere curvature parameter
            parameters: Additional parameters specific to the geometry
        """
        if parameters is None:
            parameters = {}
            
        async with self.lock:
            self.geometries[model_version] = {
                "dimensions": dimensions,
                "curvature": curvature,
                "parameters": parameters,
                "registered_at": time.time()
            }
            # Invalidate cached compatibility results
            self.compatibility_cache.clear()
            self.logger.info(f"Registered geometry for model {model_version}: {dimensions} dimensions, curvature {curvature}")
            
    async def get_geometry(self, model_version: str) -> Optional[Dict[str, Any]]:
        """Get the geometry parameters for a model version.
        
        Args:
            model_version: The model version to retrieve geometry for
            
        Returns:
            Dictionary of geometry parameters or None if not found
        """
        async with self.lock:
            return self.geometries.get(model_version)
            
    async def check_compatibility(self, version_a: str, version_b: str) -> bool:
        """Determine if two model versions can share a hypersphere space.
        
        Args:
            version_a: First model version
            version_b: Second model version
            
        Returns:
            True if compatible, False otherwise
        """
        if version_a == version_b:
            return True  # Same version is always compatible
            
        cache_key = f"{version_a}:{version_b}"
        reverse_cache_key = f"{version_b}:{version_a}"
        
        async with self.lock:
            # Check cache first
            if cache_key in self.compatibility_cache:
                return self.compatibility_cache[cache_key]
            if reverse_cache_key in self.compatibility_cache:
                return self.compatibility_cache[reverse_cache_key]
                
            geom_a = self.geometries.get(version_a)
            geom_b = self.geometries.get(version_b)
            
            if not geom_a or not geom_b:
                compatible = False
                self.logger.warning(f"Compatibility check failed - missing geometry: "
                                   f"version_a={version_a}, version_b={version_b}")
            else:
                # Check dimension equality and curvature/radius similarity
                dimension_match = geom_a["dimensions"] == geom_b["dimensions"]
                curvature_match = abs(geom_a["curvature"] - geom_b["curvature"]) < 1e-6
                
                compatible = dimension_match and curvature_match
                
            # Cache the result for future lookups
            self.compatibility_cache[cache_key] = compatible
            self.logger.debug(f"Compatibility between {version_a} and {version_b}: {compatible}")
            return compatible
            
    async def get_compatible_versions(self, target_version: str) -> list:
        """Get all model versions compatible with the target version.
        
        Args:
            target_version: The model version to find compatibility for
            
        Returns:
            List of compatible model versions
        """
        compatible_versions = []
        
        async with self.lock:
            for version in self.geometries.keys():
                if await self.check_compatibility(target_version, version):
                    compatible_versions.append(version)
                    
        return compatible_versions
    
    async def register_embedding(self, memory_id: str, embedding: List[float], model_version: str) -> None:
        """Register an embedding with its model version.
        
        Args:
            memory_id: Unique identifier for the memory
            embedding: The embedding vector
            model_version: Model version used to generate the embedding
        
        Raises:
            ValueError: If the embedding is not compatible with the registered geometry
        """
        if await self.has_geometry(model_version):
            # Verify compatibility with the registered geometry
            if not await self.check_embedding_compatibility(model_version, embedding):
                raise ValueError(f"Embedding for memory {memory_id} is not compatible with {model_version} geometry")
        
        async with self.lock:
            self.embeddings[memory_id] = (embedding, model_version)
        
    async def has_geometry(self, model_version: str) -> bool:
        """Check if a geometry is registered for a model version.
        
        Args:
            model_version: The model version to check
            
        Returns:
            True if the geometry exists, False otherwise
        """
        return model_version in self.geometries
    
    async def check_embedding_compatibility(self, model_version: str, embedding: List[float]) -> bool:
        """Check if an embedding is compatible with a model's geometry.
        
        Args:
            model_version: The model version to check against
            embedding: The embedding to check
            
        Returns:
            True if compatible, False otherwise
        """
        if not await self.has_geometry(model_version):
            logger.warning(f"Cannot check compatibility: No geometry registered for model {model_version}")
            # If we don't have the geometry yet, we can't verify - assume compatible
            return True
        
        geometry = self.geometries[model_version]
        expected_dim = geometry["dimensions"]
        
        # Check dimensions
        if len(embedding) != expected_dim:
            logger.warning(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
            return False
        
        # If curvature is defined, verify embedding lies on the hypersphere
        if "curvature" in geometry and geometry["curvature"] != 0:
            # Calculate the norm of the embedding
            norm = np.linalg.norm(embedding)
            expected_norm = 1.0  # For unit hypersphere
            
            # Allow for small numerical errors (0.1% tolerance)
            tolerance = 0.001
            if abs(norm - expected_norm) > tolerance:
                logger.warning(f"Embedding norm mismatch: expected {expected_norm}, got {norm}")
                return False
        
        return True
    
    async def verify_batch_compatibility(self, embeddings: List[List[float]], model_versions: List[str]) -> bool:
        """Verify that a batch of embeddings is compatible for processing together.
        
        Args:
            embeddings: List of embedding vectors
            model_versions: Corresponding model versions for each embedding
            
        Returns:
            True if all embeddings are compatible for batch processing
        """
        if len(embeddings) != len(model_versions):
            logger.error("Number of embeddings does not match number of model versions")
            return False
        
        if not embeddings:
            return True  # Empty batch is trivially compatible
        
        # Get the reference geometry from the first embedding
        ref_model = model_versions[0]
        if not await self.has_geometry(ref_model):
            logger.warning(f"Reference model {ref_model} has no registered geometry")
            return False
        
        ref_geometry = self.geometries[ref_model]
        ref_dim = ref_geometry["dimensions"]
        ref_curvature = ref_geometry["curvature"]
        
        # Check that all embeddings are compatible
        for i, (embedding, model) in enumerate(zip(embeddings, model_versions)):
            # First, check that this embedding is compatible with its own model
            if not await self.check_embedding_compatibility(model, embedding):
                logger.warning(f"Embedding {i} is not compatible with its model {model}")
                return False
            
            # Then, check that this model's geometry is compatible with the reference
            if await self.has_geometry(model):
                model_geometry = self.geometries[model]
                
                # Check dimensions match
                if model_geometry["dimensions"] != ref_dim:
                    logger.warning(f"Model {model} has different dimensions from reference {ref_model}")
                    return False
                
                # Check curvature is compatible
                if abs(model_geometry["curvature"] - ref_curvature) > 1e-6:
                    logger.warning(f"Model {model} has different curvature from reference {ref_model}")
                    return False
        
        return True
    
    async def get_compatible_model_versions(self, reference_model: str) -> List[str]:
        """Get a list of model versions compatible with a reference model.
        
        Args:
            reference_model: The reference model version
            
        Returns:
            List of compatible model versions
        """
        if not await self.has_geometry(reference_model):
            return []  # No geometry information for reference model
        
        ref_geometry = self.geometries[reference_model]
        ref_dim = ref_geometry["dimensions"]
        ref_curvature = ref_geometry["curvature"]
        
        compatible_versions = []
        
        for model, geometry in self.geometries.items():
            # Check geometry compatibility
            if (geometry["dimensions"] == ref_dim and 
                abs(geometry["curvature"] - ref_curvature) < 1e-6):
                compatible_versions.append(model)
        
        return compatible_versions
    
    async def transform_embedding(self, embedding: List[float], source_model: str, target_model: str) -> List[float]:
        """Transform an embedding from one model's geometry to another, if possible.
        
        Args:
            embedding: The embedding to transform
            source_model: The source model version
            target_model: The target model version
            
        Returns:
            Transformed embedding
            
        Raises:
            ValueError: If the models are not compatible or transformation is not possible
        """
        if source_model == target_model:
            return embedding  # No transformation needed
        
        if not (await self.has_geometry(source_model) and await self.has_geometry(target_model)):
            raise ValueError(f"Missing geometry information for {source_model} or {target_model}")
        
        source_geo = self.geometries[source_model]
        target_geo = self.geometries[target_model]
        
        # Check basic compatibility
        if source_geo["dimensions"] != target_geo["dimensions"]:
            raise ValueError(f"Cannot transform between different dimensions: {source_model} ({source_geo['dimensions']}) -> {target_model} ({target_geo['dimensions']})")
        
        # For now, we only support trivial transformations (same dimensionality)
        # In a real system, you might need more complex transformations based on the
        # specific geometric properties and relationships between models
        
        # If we need to adjust for curvature differences, we would do that here
        if abs(source_geo["curvature"] - target_geo["curvature"]) > 1e-6:
            # Simple rescaling for curvature adjustment (this is just an example)
            # In a real system, proper geometric transformations would be required
            scale_factor = (target_geo["curvature"] / source_geo["curvature"]) if source_geo["curvature"] != 0 else 1.0
            return [x * scale_factor for x in embedding]
        
        return embedding  # No transformation needed if curvatures are the same

    def register_model_geometry(self, model_version: str, model_profile: Dict[str, Any]) -> None:
        """
        Register a model's geometry with the registry.
        
        This is a synchronous wrapper around the async register_geometry method,
        used for compatibility with non-async code.
        
        Args:
            model_version: Version identifier for the model
            model_profile: Dictionary containing geometry parameters including dimensions
        """
        dimensions = model_profile.get("dimensions", 768)
        curvature = model_profile.get("curvature", 1.0)
        
        # Create a task to register the geometry
        async def _register():
            await self.register_geometry(
                model_version=model_version,
                dimensions=dimensions,
                curvature=curvature,
                parameters=model_profile
            )
        
        # Create and run the task in the event loop if one is running
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(_register())
            else:
                loop.run_until_complete(_register())
        except RuntimeError:
            # If no event loop is available in this thread, log a warning
            logger.warning(f"No event loop available to register model geometry for {model_version}")
