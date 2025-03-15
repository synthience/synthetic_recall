"""
Lucidia's Narrative Identity API

This module provides API endpoints for interacting with Lucidia's narrative identity system.

Created by MEGAPROMPT (Daniel)
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Body, Depends, HTTPException

# Import the identity manager
from .identity_manager import NarrativeIdentityManager


# Create a router for identity endpoints
identity_router = APIRouter(prefix="/api/identity", tags=["identity"])

# Logger
logger = logging.getLogger("NarrativeIdentityAPI")


# Dependency to get the identity manager
async def get_identity_manager():
    """Dependency to get the narrative identity manager instance."""
    # This would typically be retrieved from a global state or dependency injection system
    # For now, we'll assume it's stored in a global variable or accessible through a function
    from memory.lucidia_memory_system import get_system_components
    
    # Get system components
    components = await get_system_components()
    
    # Get the identity manager
    identity_manager = components.get("narrative_identity_manager")
    
    if not identity_manager:
        raise HTTPException(status_code=503, detail="Narrative Identity Manager not available")
    
    return identity_manager


@identity_router.get("/narrative/{narrative_type}")
async def get_identity_narrative(
    narrative_type: str = "complete", 
    style: str = "neutral",
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get a narrative about Lucidia's identity.
    
    Args:
        narrative_type: Type of narrative to generate
        style: Style of narrative
        identity_manager: Narrative identity manager instance
        
    Returns:
        Generated narrative text
    """
    try:
        narrative = await identity_manager.get_self_narrative(narrative_type, style)
        return {"narrative": narrative, "type": narrative_type, "style": style}
    except Exception as e:
        logger.error(f"Error generating narrative: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating narrative: {str(e)}")


@identity_router.get("/status")
async def get_identity_status(
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get the current status of Lucidia's identity.
    
    Args:
        identity_manager: Narrative identity manager instance
        
    Returns:
        Identity status information
    """
    try:
        status = await identity_manager.get_identity_status()
        return status
    except Exception as e:
        logger.error(f"Error getting identity status: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting identity status: {str(e)}")


@identity_router.post("/experience")
async def record_identity_experience(
    experience: Dict[str, Any] = Body(
        ..., 
        example={
            "content": "Learned about narrative identity", 
            "significance": 0.8,
            "metadata": {"source": "user_interaction"}
        }
    ),
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Record an experience relevant to identity.
    
    Args:
        experience: Experience data
        identity_manager: Narrative identity manager instance
        
    Returns:
        Success status
    """
    try:
        content = experience.get("content")
        if not content:
            raise HTTPException(status_code=400, detail="Experience content is required")
        
        metadata = experience.get("metadata", {})
        significance = experience.get("significance", 0.7)
        
        result = await identity_manager.record_experience(content, metadata, significance)
        return {"success": result}
    except Exception as e:
        logger.error(f"Error recording experience: {e}")
        raise HTTPException(status_code=500, detail=f"Error recording experience: {str(e)}")


@identity_router.post("/dream-insights")
async def integrate_dream_insights(
    insights: List[Dict[str, Any]] = Body(
        ..., 
        example=[
            {
                "text": "I am developing a more nuanced understanding of my purpose.",
                "significance": 0.85
            }
        ]
    ),
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Integrate dream insights into the narrative identity.
    
    Args:
        insights: List of dream insights
        identity_manager: Narrative identity manager instance
        
    Returns:
        Integration results
    """
    try:
        result = await identity_manager.integrate_dream_insights(insights)
        return result
    except Exception as e:
        logger.error(f"Error integrating dream insights: {e}")
        raise HTTPException(status_code=500, detail=f"Error integrating dream insights: {str(e)}")


@identity_router.get("/timeline")
async def get_autobiographical_timeline(
    limit: int = 10,
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get the autobiographical timeline.
    
    Args:
        limit: Maximum number of events to return
        identity_manager: Narrative identity manager instance
        
    Returns:
        List of timeline events
    """
    try:
        timeline = await identity_manager.get_autobiographical_timeline(limit=limit)
        return {"timeline": timeline, "count": len(timeline)}
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting timeline: {str(e)}")


@identity_router.get("/memories/significant")
async def get_significant_memories(
    limit: int = 5,
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get the most significant autobiographical memories.
    
    Args:
        limit: Maximum number of memories to return
        identity_manager: Narrative identity manager instance
        
    Returns:
        List of significant memories
    """
    try:
        memories = await identity_manager.get_significant_memories(limit=limit)
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"Error getting significant memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting significant memories: {str(e)}")


@identity_router.get("/memories/identity-relevant")
async def get_identity_relevant_memories(
    limit: int = 5,
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get memories most relevant to identity.
    
    Args:
        limit: Maximum number of memories to return
        identity_manager: Narrative identity manager instance
        
    Returns:
        List of identity-relevant memories
    """
    try:
        memories = await identity_manager.get_identity_relevant_memories(limit=limit)
        return {"memories": memories, "count": len(memories)}
    except Exception as e:
        logger.error(f"Error getting identity-relevant memories: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting identity-relevant memories: {str(e)}")


@identity_router.get("/memories/category/{category}")
async def get_memories_by_category(
    category: str,
    limit: int = 5,
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Get memories in a specific narrative category.
    
    Args:
        category: Narrative category
        limit: Maximum number of memories to return
        identity_manager: Narrative identity manager instance
        
    Returns:
        List of memories in the category
    """
    try:
        memories = await identity_manager.get_memories_by_category(category, limit=limit)
        return {"memories": memories, "count": len(memories), "category": category}
    except Exception as e:
        logger.error(f"Error getting memories by category: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting memories by category: {str(e)}")


@identity_router.put("/core-identity")
async def update_core_identity(
    updates: Dict[str, Any] = Body(
        ..., 
        example={
            "name": "Lucidia",
            "purpose": "To develop as a self-orienting, self-learning system with narrative capabilities"
        }
    ),
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Update core identity attributes.
    
    Args:
        updates: Dictionary of attributes to update
        identity_manager: Narrative identity manager instance
        
    Returns:
        Success status
    """
    try:
        result = await identity_manager.update_core_identity(updates)
        return {"success": result, "updated_fields": list(updates.keys())}
    except Exception as e:
        logger.error(f"Error updating core identity: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating core identity: {str(e)}")


@identity_router.post("/relationship")
async def add_relationship(
    relationship: Dict[str, Any] = Body(
        ..., 
        example={
            "entity": "User",
            "relationship_type": "interacts_with",
            "strength": 0.9
        }
    ),
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Add a relationship to the identity.
    
    Args:
        relationship: Relationship data
        identity_manager: Narrative identity manager instance
        
    Returns:
        Success status
    """
    try:
        entity = relationship.get("entity")
        relationship_type = relationship.get("relationship_type")
        strength = relationship.get("strength", 0.8)
        
        if not entity or not relationship_type:
            raise HTTPException(status_code=400, detail="Entity and relationship_type are required")
        
        result = await identity_manager.add_relationship(entity, relationship_type, strength)
        return {"success": result}
    except Exception as e:
        logger.error(f"Error adding relationship: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding relationship: {str(e)}")


@identity_router.post("/save")
async def save_identity_state(
    identity_manager: NarrativeIdentityManager = Depends(get_identity_manager)
):
    """Save the current state of the narrative identity system.
    
    Args:
        identity_manager: Narrative identity manager instance
        
    Returns:
        Success status
    """
    try:
        await identity_manager.save_state()
        return {"success": True, "message": "Identity state saved successfully"}
    except Exception as e:
        logger.error(f"Error saving identity state: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving identity state: {str(e)}")


# Function to add the router to a FastAPI app
def add_identity_router(app):
    """Add the identity router to a FastAPI app.
    
    Args:
        app: FastAPI application
    """
    app.include_router(identity_router)
    logger.info("Narrative Identity API routes registered")