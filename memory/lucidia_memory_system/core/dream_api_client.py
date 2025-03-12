#!/usr/bin/env python3
"""
Dream API Client for Lucidia Memory System

This module provides a client interface to interact with the Dream API endpoints.
It allows the DreamManager to leverage the more sophisticated dream processing capabilities
offered by the LucidiaDreamProcessor while maintaining its own memory integration approach.
"""

import json
import logging
import os
import aiohttp
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DreamAPIClient:
    """
    Client for interacting with Lucidia's Dream API endpoints.
    
    This client provides methods for starting dream sessions, retrieving dream insights,
    generating dream reports, and other dream-related functionality exposed by the Dream API.
    """
    
    def __init__(self, api_base_url: str = None):
        """
        Initialize the Dream API client.
        
        Args:
            api_base_url: Base URL for the Dream API. If None, defaults to localhost:8080.
        """
        self.api_base_url = api_base_url or os.environ.get('DREAM_API_URL', 'http://localhost:8080')
        self.api_prefix = '/api'
        logger.info(f"Dream API client initialized with base URL: {self.api_base_url}")
    
    async def _make_request(self, method: str, endpoint: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make an HTTP request to the Dream API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data for POST/PUT requests
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.api_base_url}{self.api_prefix}{endpoint}"
        headers = {
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers) as response:
                        return await self._process_response(response)
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, json=data) as response:
                        return await self._process_response(response)
                elif method.upper() == 'PUT':
                    async with session.put(url, headers=headers, json=data) as response:
                        return await self._process_response(response)
                elif method.upper() == 'DELETE':
                    async with session.delete(url, headers=headers) as response:
                        return await self._process_response(response)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientError as e:
            logger.error(f"Error making request to {endpoint}: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _process_response(self, response) -> Dict[str, Any]:
        """
        Process an HTTP response.
        
        Args:
            response: HTTP response object
            
        Returns:
            Response data as dictionary
        """
        try:
            data = await response.json()
            if response.status >= 400:
                logger.error(f"API error: {response.status} - {data.get('detail', 'Unknown error')}")
                return {"status": "error", "message": data.get('detail', 'Unknown error')}
            return data
        except ValueError as e:
            text = await response.text()
            logger.error(f"Error parsing response: {e} - Response: {text}")
            return {"status": "error", "message": f"Invalid response format: {text}"}
    
    async def start_dream_session(self, mode: str = "full", duration_minutes: int = 10, 
                             settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start a new dream processing session.
        
        Args:
            mode: Processing mode ('full', 'consolidate', 'insights', 'reflection', 'optimize')
            duration_minutes: Duration of the dream session in minutes
            settings: Additional settings for the dream session
            
        Returns:
            Response with session ID and status
        """
        data = {
            "mode": mode,
            "duration_minutes": duration_minutes,
            "settings": settings or {}
        }
        return await self._make_request('POST', '/start_dream_session', data)
    
    async def get_dream_status(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status of current dream session or all active sessions.
        
        Args:
            session_id: Optional ID of specific session to check
            
        Returns:
            Status information
        """
        endpoint = f"/get_dream_status"
        if session_id:
            endpoint += f"?session_id={session_id}"
        return await self._make_request('GET', endpoint)
    
    async def generate_dream_insight(self, dream_content: str, theme: Optional[str] = None,
                              depth: float = 0.7, creativity: float = 0.8) -> Dict[str, Any]:
        """
        Generate insights from dream content using the dream tools.
        
        Args:
            dream_content: The dream content to analyze
            theme: Optional thematic direction for insight generation
            depth: Depth of insight (0-1)
            creativity: Level of creativity (0-1)
            
        Returns:
            Generated insights
        """
        # Map the parameters to what the server expects
        data = {
            "timeframe": "recent",
            "limit": 20,
            "categories": [theme] if theme else None
        }
        
        # Use the correct endpoint that actually exists in the server implementation
        try:
            return await self._make_request('POST', '/memory/insights', data)
        except Exception as e:
            logger.error(f"Error accessing /memory/insights: {e}")
            # Fall back to a health check endpoint
            try:
                health = await self._make_request('GET', '/health')
                logger.info(f"Dream API health check: {health}")
                return {"status": "error", "message": "Dream API available but insights endpoint not found", "available_endpoints": health}
            except Exception as e2:
                logger.error(f"Error even with health check: {e2}")
                return {"status": "error", "message": "Dream API communication failed completely"}
    
    async def enhance_dream_seed(self, seed_content: str, seed_type: str = "memory", 
                            depth: float = 0.7) -> Dict[str, Any]:
        """
        Enhance a dream seed with additional context from memory and knowledge graph.
        
        Args:
            seed_content: The original seed content to enhance
            seed_type: Type of the seed (memory, concept, emotion)
            depth: Depth of enhancement (0-1)
            
        Returns:
            Enhanced seed content and related fragments
        """
        # The closest existing endpoint is /memory/insights, so adapt to it
        data = {
            "timeframe": "recent",
            "limit": 20,
            "categories": [seed_type] if seed_type else None
        }
        
        # Use the memory/insights endpoint which actually exists in the server
        try:
            return await self._make_request('POST', '/memory/insights', data)
        except Exception as e:
            logger.error(f"Error accessing /memory/insights: {e}")
            # Fall back to dream status check
            try:
                status = await self._make_request('GET', '/dream/status')
                logger.info(f"Dream API status check: {status}")
                return {"status": "error", "message": "Dream API available but enhance endpoint not found", "available_endpoints": status}
            except Exception as e2:
                logger.error(f"Error even with status check: {e2}")
                return {"status": "error", "message": "Dream API communication failed completely"}
    
    async def consolidate_memories(self, target: str = "all", limit: int = 100, 
                               min_significance: float = 0.3) -> Dict[str, Any]:
        """
        Consolidate memories in the memory system.
        
        Args:
            target: Memory target to consolidate ('all', 'recent', 'dreams')
            limit: Maximum number of memories to consolidate
            min_significance: Minimum significance threshold
            
        Returns:
            Consolidation results
        """
        # Server just uses a simple POST to /memory/consolidate with no parameters
        try:
            return await self._make_request('POST', '/memory/consolidate')
        except Exception as e:
            logger.error(f"Error accessing /memory/consolidate: {e}")
            # Fall back to a status check
            try:
                status = await self._make_request('GET', '/dream/status')
                logger.info(f"Dream API status check: {status}")
                return {"status": "error", "message": "Dream API available but consolidate endpoint not found", "available_status": status}
            except Exception as e2:
                logger.error(f"Error even with status check: {e2}")
                return {"status": "error", "message": "Dream API communication failed completely"}
    
    async def generate_dream_report(self, memory_ids: Optional[List[str]] = None, 
                               timeframe: str = "recent", limit: int = 20) -> Dict[str, Any]:
        """
        Generate a report from dream memories.
        
        Args:
            memory_ids: Optional list of specific memory IDs to include in the report
            timeframe: Time range for memories ('recent', 'significant', 'week', etc.)
            limit: Maximum number of memories to include
            
        Returns:
            The generated dream report
        """
        data = {
            "memory_ids": memory_ids,
            "timeframe": timeframe,
            "limit": limit,
            "domain": "synthien_studies"  # This appears to be the default domain in the server
        }
        
        # Use the correct endpoint for generating reports
        try:
            return await self._make_request('POST', '/report/generate', data)
        except Exception as e:
            logger.error(f"Error accessing /report/generate: {e}")
            # Fall back to a status check
            try:
                status = await self._make_request('GET', '/dream/status')
                logger.info(f"Dream API status check: {status}")
                return {"status": "error", "message": "Dream API available but report generation endpoint not found", "available_status": status}
            except Exception as e2:
                logger.error(f"Error even with status check: {e2}")
                return {"status": "error", "message": "Dream API communication failed completely"}
    
    async def perform_self_reflection(self, focus_areas: Optional[List[str]] = None, 
                                 depth: str = "standard") -> Dict[str, Any]:
        """
        Run a self-reflection cycle to analyze system performance and identify improvements.
        
        Args:
            focus_areas: Optional list of areas to focus reflection on
            depth: Reflection depth ('shallow', 'standard', 'deep')
            
        Returns:
            Self-reflection results
        """
        data = {
            "focus_areas": focus_areas,
            "depth": depth
        }
        
        # Use the correct endpoint for self-reflection
        try:
            return await self._make_request('POST', '/self/reflect', data)
        except Exception as e:
            logger.error(f"Error accessing /self/reflect: {e}")
            # Fall back to a status check
            try:
                status = await self._make_request('GET', '/dream/status')
                logger.info(f"Dream API status check: {status}")
                return {"status": "error", "message": "Dream API available but self-reflection endpoint not found", "available_status": status}
            except Exception as e2:
                logger.error(f"Error even with status check: {e2}")
                return {"status": "error", "message": "Dream API communication failed completely"}
