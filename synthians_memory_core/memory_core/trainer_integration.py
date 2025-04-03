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
        
        # Get the embedding dimension from the main memory core config for consistency
        embedding_dim = self.memory_core.config.get('embedding_dim', 768)  # Default to 768 if not found
        
        # Correctly initialize GeometryManager with a config dictionary
        self.geometry_manager = GeometryManager(config={
            'embedding_dim': embedding_dim,
            'normalization_enabled': True,
            'alignment_strategy': 'truncate'
        })
    
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
        memory = await self.memory_core.get_memory_by_id_async(memory_id)
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
            # Raise exception instead of returning error dict to ensure
            # proper propagation to API error handlers
            error_msg = f"Failed to update quickrecal score for memory {memory_id} in core."
            logger.error(f"[TrainerIntegration] {error_msg}")
            raise RuntimeError(error_msg)
