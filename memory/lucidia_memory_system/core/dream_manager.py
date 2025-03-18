#!/usr/bin/env python3
"""
Dream Manager for Lucidia Memory System

This module provides functionality for managing Lucidia's dreams and insights,
including storage, retrieval, analysis, and integration with the conversation system.
"""

import json
import logging
import os
import random
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import the DreamAPIClient
from .dream_api_client import DreamAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DreamAnalyzer:
    """
    Analyzes dreams and their impact on Lucidia's cognitive processes.
    """
    
    def __init__(self, log_path: str = "logs/dream_analysis"):
        """
        Initialize the dream analyzer.
        
        Args:
            log_path: Path to store dream analysis logs
        """
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        
        # Configure specialized dream analysis logger
        self.dream_logger = logging.getLogger('lucidia.dream_analysis')
        self.dream_logger.setLevel(logging.INFO)
        
        # Add file handler for dream analysis
        analysis_log = self.log_path / "dream_analysis.log"
        try:
            file_handler = logging.FileHandler(analysis_log)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.dream_logger.addHandler(file_handler)
            self.dream_logger.info("Dream analysis system initialized")
        except Exception as e:
            logger.error(f"Failed to set up dream analysis logger: {e}")
        
        # Configuration for dream analysis
        self.analysis_config = {
            'insight_decay_rate': 0.1,  # Rate at which insights lose influence per interaction
            'max_active_insights': 5,   # Maximum number of insights active at once
            'relevance_threshold': 0.4,  # Minimum relevance to consider an insight applicable
            'influence_metrics': {},    # Track influence of dreams on responses
            'dream_session_file': self.log_path / "dream_session.json"  # Session persistence
        }
        
        # Attempt to load existing session data
        self._load_session_data()
    
    def _load_session_data(self) -> None:
        """
        Load session data from disk if available.
        """
        try:
            if self.analysis_config['dream_session_file'].exists():
                with open(self.analysis_config['dream_session_file'], 'r') as f:
                    session_data = json.load(f)
                    
                # Update metrics from saved session
                if 'influence_metrics' in session_data:
                    self.analysis_config['influence_metrics'] = session_data['influence_metrics']
                    logger.info(f"Loaded dream influence metrics from previous session")
        except Exception as e:
            logger.error(f"Error loading dream session data: {e}")
    
    def _save_session_data(self) -> None:
        """
        Save session data to disk for persistence.
        """
        try:
            session_data = {
                'influence_metrics': self.analysis_config['influence_metrics'],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.analysis_config['dream_session_file'], 'w') as f:
                json.dump(session_data, f, indent=2)
                
            logger.debug("Saved dream session data to disk")
        except Exception as e:
            logger.error(f"Error saving dream session data: {e}")
    
    def record_dream_influence(self, insight_id: str, influence_type: str, 
                             strength: float, context: Dict[str, Any]) -> None:
        """
        Record the influence of a dream insight on Lucidia's cognitive processes.
        
        Args:
            insight_id: ID of the dream insight
            influence_type: Type of influence (e.g., 'response', 'emotion', 'reflection')
            strength: Strength of influence (0.0-1.0)
            context: Additional context about the influence
        """
        # Ensure metrics dictionary exists for this insight
        if insight_id not in self.analysis_config['influence_metrics']:
            self.analysis_config['influence_metrics'][insight_id] = {
                'total_influences': 0,
                'influence_types': {},
                'influence_history': [],
                'last_influence': None,
                'cumulative_strength': 0.0
            }
        
        # Update metrics
        metrics = self.analysis_config['influence_metrics'][insight_id]
        metrics['total_influences'] += 1
        
        # Update influence type counters
        if influence_type not in metrics['influence_types']:
            metrics['influence_types'][influence_type] = 0
        metrics['influence_types'][influence_type] += 1
        
        # Add to history
        influence_record = {
            'timestamp': datetime.now().isoformat(),
            'type': influence_type,
            'strength': strength,
            'context': context
        }
        metrics['influence_history'].append(influence_record)
        
        # Update last influence and cumulative strength
        metrics['last_influence'] = datetime.now().isoformat()
        metrics['cumulative_strength'] += strength
        
        # Log the influence
        self.dream_logger.info(
            f"Dream influence: {insight_id} affected {influence_type} with strength {strength:.2f}"
        )
        
        # Save session data for persistence
        self._save_session_data()
    
    def get_dream_influence_report(self, insight_id: Optional[str] = None, 
                                 time_period: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a report on dream influence metrics.
        
        Args:
            insight_id: Optional ID to filter for a specific insight
            time_period: Optional time period in hours to filter by
            
        Returns:
            Report dictionary with influence metrics
        """
        metrics = self.analysis_config['influence_metrics']
        
        # Filter by insight ID if provided
        if insight_id and insight_id in metrics:
            insights_to_report = {insight_id: metrics[insight_id]}
        else:
            insights_to_report = metrics
        
        # Filter by time period if provided
        if time_period is not None:
            cutoff_time = datetime.now() - timedelta(hours=time_period)
            cutoff_str = cutoff_time.isoformat()
            
            # Filter each insight's history
            for insight_id, insight_metrics in insights_to_report.items():
                filtered_history = []
                for record in insight_metrics['influence_history']:
                    if record['timestamp'] >= cutoff_str:
                        filtered_history.append(record)
                
                # Update history with filtered version
                insight_metrics['influence_history'] = filtered_history
        
        # Compile report
        report = {
            'total_insights': len(insights_to_report),
            'insights': insights_to_report,
            'generated_at': datetime.now().isoformat(),
            'time_period_hours': time_period
        }
        
        return report

class DreamManager:
    """
    Manages Lucidia's dreams, including storage, retrieval, and integration with memory.
    """
    
    def __init__(self, memory_integration=None, dream_analyzer: Optional[DreamAnalyzer] = None,
                 dream_api_client: Optional[DreamAPIClient] = None, dream_api_url: Optional[str] = None,
                 use_dream_api: bool = None):
        """
        Initialize the dream manager.
        
        Args:
            memory_integration: Memory integration instance for storing dreams
            dream_analyzer: Optional dream analyzer for tracking dream influence
            dream_api_client: Optional client to connect to the Dream API
            dream_api_url: Optional URL for the Dream API if client not provided
            use_dream_api: Optional flag to force enable/disable Dream API usage
        """
        self.memory_integration = memory_integration
        self.dream_analyzer = dream_analyzer or DreamAnalyzer()
        
        # Initialize Dream API client if provided or create new one with URL
        self.dream_api_client = dream_api_client
        if self.dream_api_client is None and dream_api_url is not None:
            self.dream_api_client = DreamAPIClient(api_base_url=dream_api_url)
        
        # Active dream insights and their influence on the system
        self.active_insights = []
        
        # Configuration for dream management
        self.config = {
            'dream_probability': 0.25,  # Probability of generating a spontaneous dream
            'dream_types': ['reflection', 'integration', 'creative', 'predictive'],
            'max_dream_age_days': 30,  # Maximum age for retrieving dreams
            'dream_log_file': 'logs/dreams/dream_log.json',
            'use_dream_api': use_dream_api if use_dream_api is not None else (self.dream_api_client is not None)  # Use explicitly provided flag or infer from client
        }
        
        # Ensure dream log directory exists
        dream_log_path = Path(self.config['dream_log_file']).parent
        dream_log_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dream manager initialized with {'Dream API support' if self.dream_api_client else 'memory integration only'}")
    
    async def store_dream(self, content: str, dream_type: str = 'reflection', 
                       significance: float = 0.7, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Store a dream in memory and log it.
        
        Args:
            content: The dream content/insight
            dream_type: Type of dream (reflection, integration, creative, predictive)
            significance: How significant this dream is (0.0-1.0)
            metadata: Additional metadata for the dream
            
        Returns:
            Dictionary with dream storage result
        """
        try:
            # Validate dream type
            if dream_type not in self.config['dream_types']:
                dream_type = random.choice(self.config['dream_types'])
            
            # Create base metadata
            if metadata is None:
                metadata = {}
                
            # Add required fields
            timestamp = datetime.now().isoformat()
            dream_id = f"dream_{dream_type}_{timestamp.replace(':', '-')}_{uuid.uuid4().hex[:8]}"
            
            # Merge with provided metadata
            dream_metadata = {
                "memory_type": "DREAM",
                "dream_type": dream_type,
                "timestamp": time.time(),
                "creation_date": timestamp,
                "significance": significance,
                **metadata
            }
            
            # Store in memory if integration available
            result = {
                "dream_id": dream_id,
                "stored": False,
                "logged": False,
                "enhanced": False
            }

            # If Dream API client is available, try to enhance the dream content
            if self.config['use_dream_api'] and self.dream_api_client:
                try:
                    # Attempt to enhance the dream with additional context
                    enhanced_result = await self.dream_api_client.enhance_dream_seed(
                        seed_content=content,
                        seed_type="dream",
                        depth=0.7
                    )
                    
                    if enhanced_result.get("status") == "success" and "enhanced_seed" in enhanced_result:
                        content = enhanced_result["enhanced_seed"]
                        dream_metadata["enhanced"] = True
                        dream_metadata["related_fragments"] = enhanced_result.get("related_fragments", [])
                        result["enhanced"] = True
                        logger.info(f"Enhanced dream content via Dream API: {dream_id}")
                except Exception as e:
                    logger.warning(f"Failed to enhance dream via Dream API: {e}")
            
            if self.memory_integration and hasattr(self.memory_integration, 'store_memory'):
                stored = await self.memory_integration.store_memory(
                    memory_id=dream_id,
                    content=content,
                    metadata=dream_metadata,
                    memory_type="DREAM"
                )
                result["stored"] = stored
                logger.info(f"Dream stored in memory: {dream_id}")
            
            # Log the dream regardless of memory storage
            try:
                dream_log = {
                    "dream_id": dream_id,
                    "content": content,
                    "metadata": dream_metadata,
                    "logged_at": datetime.now().isoformat()
                }
                
                # Read existing logs if file exists
                log_path = Path(self.config['dream_log_file'])
                dreams = []
                if log_path.exists():
                    with open(log_path, 'r') as f:
                        try:
                            dreams = json.load(f)
                        except json.JSONDecodeError:
                            dreams = []
                
                # Append new dream
                dreams.append(dream_log)
                
                # Write back all dreams
                with open(log_path, 'w') as f:
                    json.dump(dreams, f, indent=2)
                
                result["logged"] = True
                logger.debug(f"Dream logged to file: {dream_id}")
                
            except Exception as e:
                logger.error(f"Error logging dream: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error storing dream: {e}")
            return {"status": "error", "message": str(e)}
    
    async def retrieve_dreams(self, query: str, limit: int = 3, 
                           dream_type: Optional[str] = None, 
                           min_significance: float = 0.4) -> List[Dict[str, Any]]:
        """
        Retrieve dreams relevant to a query.
        
        Args:
            query: Search query to find relevant dreams
            limit: Maximum number of dreams to retrieve
            dream_type: Optional type of dreams to filter for
            min_significance: Minimum significance threshold
            
        Returns:
            List of matching dreams
        """
        try:
            # Determine which memory types to include
            memory_types = ["DREAM"]
            
            # Try to retrieve from memory integration first
            memories = []
            
            if self.memory_integration and hasattr(self.memory_integration, 'retrieve_memories'):
                memories = await self.memory_integration.retrieve_memories(
                    query=query,
                    limit=limit,
                    min_significance=min_significance,
                    memory_types=memory_types
                )
                
                # Filter by dream type if specified
                if dream_type and memories:
                    memories = [m for m in memories if m.get("metadata", {}).get("dream_type") == dream_type]
                
                logger.info(f"Retrieved {len(memories)} dreams from memory")
            
            # If we got no results and Dream API is available, try that as a fallback
            if not memories and self.config['use_dream_api'] and self.dream_api_client:
                try:
                    api_result = await self.dream_api_client.generate_dream_insight(
                        dream_content=query,
                        theme=dream_type,
                        depth=0.8,
                        creativity=0.7
                    )
                    
                    if api_result.get("status") == "success" and "insights" in api_result:
                        # Create a synthetic memory from the generated insight
                        insight = api_result["insights"][0] if isinstance(api_result["insights"], list) else api_result["insights"]
                        
                        timestamp = datetime.now().isoformat()
                        dream_id = f"dream_api_{timestamp.replace(':', '-')}_{uuid.uuid4().hex[:8]}"
                        
                        # Create synthetic memory
                        synthetic_memory = {
                            "id": dream_id,
                            "content": insight,
                            "metadata": {
                                "memory_type": "DREAM",
                                "dream_type": dream_type or "creative",
                                "timestamp": time.time(),
                                "creation_date": timestamp,
                                "significance": 0.8,
                                "synthetic": True,
                                "generated_from": query
                            },
                            "embedding": None,
                            "match_score": 0.9  # High match score since it was generated for this query
                        }
                        
                        memories = [synthetic_memory]
                        logger.info(f"Generated synthetic dream insight via API for query: {query[:30]}...")
                except Exception as e:
                    logger.warning(f"Error generating dream insight via API: {e}")
            
            return memories
        
        except Exception as e:
            logger.error(f"Error retrieving dreams: {e}")
            return []
    
    def record_dream_influence(self, dream_id: str, influence_context: Dict[str, Any]) -> None:
        """
        Record a dream's influence on the system.
        
        Args:
            dream_id: ID of the dream that influenced the system
            influence_context: Context of the influence
        """
        if self.dream_analyzer:
            influence_type = influence_context.get('type', 'response')
            strength = influence_context.get('strength', 0.6)
            
            self.dream_analyzer.record_dream_influence(
                insight_id=dream_id,
                influence_type=influence_type,
                strength=strength,
                context=influence_context
            )
            
            # Add to active insights list if not already present
            already_active = any(insight['dream_id'] == dream_id for insight in self.active_insights)
            if not already_active:
                self.active_insights.append({
                    'dream_id': dream_id,
                    'first_used': datetime.now().isoformat(),
                    'last_used': datetime.now().isoformat(),
                    'use_count': 1,
                    'influence_strength': strength
                })
            else:
                # Update existing insight
                for insight in self.active_insights:
                    if insight['dream_id'] == dream_id:
                        insight['last_used'] = datetime.now().isoformat()
                        insight['use_count'] += 1
                        insight['influence_strength'] = strength
            
            logger.info(f"Recorded dream influence: {dream_id} -> {influence_type} with strength {strength:.2f}")
    
    def decay_dream_influences(self) -> None:
        """
        Decay the influence of active dreams over time and remove expired ones.
        """
        # Decay influence for all active insights
        active_insights = []
        for insight in self.active_insights:
            # Parse dates
            last_used = datetime.fromisoformat(insight['last_used'])
            days_since_use = (datetime.now() - last_used).total_seconds() / (24 * 60 * 60)
            
            # Apply decay based on time since last use
            if days_since_use < 1:  # Less than a day
                decay_factor = 0.95  # Slight decay
            elif days_since_use < 3:  # 1-3 days
                decay_factor = 0.7  # Moderate decay
            else:  # More than 3 days
                decay_factor = 0.3  # Heavy decay
            
            # Update strength
            insight['influence_strength'] *= decay_factor
            
            # Keep if still relevant
            if insight['influence_strength'] > 0.2:
                active_insights.append(insight)
        
        # Update active insights list
        self.active_insights = active_insights
        logger.debug(f"Active dream insights after decay: {len(self.active_insights)}")
    
    def get_active_dream_influences(self) -> List[Dict[str, Any]]:
        """
        Get all currently active dream influences.
        
        Returns:
            List of active dream influences
        """
        return self.active_insights
    
    def get_dream_influence_report(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Get a report on dream influences over a time period.
        
        Args:
            time_period_hours: Number of hours to look back
            
        Returns:
            Report dictionary
        """
        if self.dream_analyzer:
            return self.dream_analyzer.get_dream_influence_report(time_period=time_period_hours)
        else:
            return {"error": "Dream analyzer not available"}

    async def start_dream_session(self, mode: str = "full", duration_minutes: int = 10, 
                               settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a new dream processing session using the Dream API.
        
        Args:
            mode: Processing mode ('full', 'consolidate', 'insights', 'reflection', 'optimize')
            duration_minutes: Duration of the dream session in minutes
            settings: Additional settings for the dream session
            
        Returns:
            Response with session ID and status
        """
        if not self.config['use_dream_api'] or not self.dream_api_client:
            return {
                "status": "error",
                "message": "Dream API client not available"
            }
        
        try:
            result = await self.dream_api_client.start_dream_session(
                mode=mode,
                duration_minutes=duration_minutes,
                settings=settings
            )
            
            if result.get("status") == "success":
                logger.info(f"Started dream session: {result.get('session_id')}")
            
            return result
        except Exception as e:
            logger.error(f"Error starting dream session: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def get_dream_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of current dream session or all active sessions.
        
        Args:
            session_id: Optional ID of specific session to check
            
        Returns:
            Status information
        """
        if not self.config['use_dream_api'] or not self.dream_api_client:
            return {
                "status": "error",
                "message": "Dream API client not available"
            }
        
        try:
            return await self.dream_api_client.get_dream_status(session_id=session_id)
        except Exception as e:
            logger.error(f"Error getting dream status: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def generate_dream_report(self, memory_ids: Optional[List[str]] = None, 
                                 timeframe: str = "recent", limit: int = 20) -> Dict[str, Any]:
        """
        Generate a dream report from specified memories or recent dreams.
        
        Args:
            memory_ids: Optional list of memory IDs to include in the report
            timeframe: Timeframe for selecting memories ('recent', 'all', 'today')
            limit: Maximum number of memories to include
            
        Returns:
            Generated dream report
        """
        if not self.config['use_dream_api'] or not self.dream_api_client:
            return {
                "status": "error",
                "message": "Dream API client not available"
            }
        
        try:
            return await self.dream_api_client.generate_dream_report(
                memory_ids=memory_ids,
                timeframe=timeframe,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Error generating dream report: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def perform_self_reflection(self, focus_areas: Optional[List[str]] = None, 
                                   depth: str = "standard") -> Dict[str, Any]:
        """
        Perform self-reflection using the dream processor.
        
        Args:
            focus_areas: Areas to focus reflection on
            depth: Depth of reflection ('light', 'standard', 'deep')
            
        Returns:
            Reflection results
        """
        if not self.config['use_dream_api'] or not self.dream_api_client:
            return {
                "status": "error",
                "message": "Dream API client not available"
            }
        
        try:
            return await self.dream_api_client.perform_self_reflection(
                focus_areas=focus_areas,
                depth=depth
            )
        except Exception as e:
            logger.error(f"Error performing self-reflection: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def consolidate_memories(self, target: str = "all", limit: int = 100, 
                                min_significance: float = 0.3,
                                min_quickrecal_score: float = None) -> Dict[str, Any]:
        """
        Consolidate memories in the memory system using the Dream API.
        
        Args:
            target: Memory target to consolidate ('all', 'recent', 'dreams')
            limit: Maximum number of memories to consolidate
            min_significance: Minimum significance threshold (deprecated, use min_quickrecal_score)
            min_quickrecal_score: Minimum quick recall score threshold
            
        Returns:
            Consolidation results
        """
        if not self.config['use_dream_api'] or not self.dream_api_client:
            return {
                "status": "error",
                "message": "Dream API client not available"
            }
        
        try:
            return await self.dream_api_client.consolidate_memories(
                target=target,
                limit=limit,
                min_significance=min_significance,
                min_quickrecal_score=min_quickrecal_score
            )
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
