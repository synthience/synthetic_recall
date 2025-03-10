"""User Activity Tracker for Lucidia

This module tracks user activity and provides AFK (Away From Keyboard) detection
to help the system determine when to enter idle states for background processing.
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger("UserActivityTracker")

class UserActivityTracker:
    """Tracks user activity and provides AFK detection capabilities"""
    
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance of UserActivityTracker"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = UserActivityTracker()
            return cls._instance
    
    def __init__(self):
        """Initialize the activity tracker"""
        self.last_activity_time = time.time()
        self.activity_history = []
        self.max_history_size = 100
        
        # Configuration with defaults
        self.config = {
            'afk_threshold_seconds': int(os.getenv('AFK_THRESHOLD_SECONDS', 300)),  # 5 minutes
            'extended_afk_threshold_seconds': int(os.getenv('EXTENDED_AFK_THRESHOLD_SECONDS', 1800)),  # 30 minutes
            'activity_types': ["api_call", "ui_interaction", "voice_input", "text_input"]
        }
        
        # Track different types of activity separately
        self.activity_by_type = {activity_type: [] for activity_type in self.config['activity_types']}
        
        # Session tracking
        self.active_sessions = {}
        self.session_activity = {}
        
        logger.info(f"UserActivityTracker initialized with AFK threshold: {self.config['afk_threshold_seconds']} seconds")
    
    def record_activity(self, activity_type: str = "api_call", session_id: Optional[str] = None,
                       details: Optional[Dict[str, Any]] = None) -> None:
        """Record user activity of a specific type
        
        Args:
            activity_type: Type of activity (api_call, ui_interaction, etc.)
            session_id: Optional ID of the user session
            details: Optional details about the activity
        """
        current_time = time.time()
        activity_record = {
            'timestamp': current_time,
            'type': activity_type,
            'session_id': session_id,
            'details': details or {}
        }
        
        # Update last activity time
        self.last_activity_time = current_time
        
        # Add to general history
        self.activity_history.append(activity_record)
        if len(self.activity_history) > self.max_history_size:
            self.activity_history.pop(0)
        
        # Add to type-specific history
        if activity_type in self.activity_by_type:
            self.activity_by_type[activity_type].append(activity_record)
            if len(self.activity_by_type[activity_type]) > self.max_history_size:
                self.activity_by_type[activity_type].pop(0)
        
        # Update session tracking
        if session_id:
            self.active_sessions[session_id] = current_time
            if session_id not in self.session_activity:
                self.session_activity[session_id] = []
            self.session_activity[session_id].append(activity_record)
            
            # Trim session history if needed
            if len(self.session_activity[session_id]) > self.max_history_size:
                self.session_activity[session_id].pop(0)
        
        logger.debug(f"Recorded {activity_type} activity" + 
                     (f" for session {session_id}" if session_id else ""))
    
    def end_session(self, session_id: str) -> None:
        """End a user session
        
        Args:
            session_id: ID of the session to end
        """
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Ended session {session_id}")
    
    def is_afk(self) -> bool:
        """Check if user is away from keyboard based on activity threshold
        
        Returns:
            True if user is considered AFK, False otherwise
        """
        time_since_activity = time.time() - self.last_activity_time
        return time_since_activity > self.config['afk_threshold_seconds']
    
    def is_extended_afk(self) -> bool:
        """Check if user has been away for an extended period
        
        Returns:
            True if user is considered away for an extended period, False otherwise
        """
        time_since_activity = time.time() - self.last_activity_time
        return time_since_activity > self.config['extended_afk_threshold_seconds']
    
    def get_active_sessions_count(self) -> int:
        """Get count of currently active sessions
        
        Returns:
            Number of active sessions
        """
        # Clean up any sessions that should be considered inactive
        self._cleanup_inactive_sessions()
        return len(self.active_sessions)
    
    def _cleanup_inactive_sessions(self, max_inactive_time: Optional[int] = None) -> None:
        """Clean up inactive sessions
        
        Args:
            max_inactive_time: Maximum inactive time in seconds before session is removed
        """
        if max_inactive_time is None:
            max_inactive_time = self.config['afk_threshold_seconds'] * 2
        
        current_time = time.time()
        sessions_to_remove = []
        
        for session_id, last_time in self.active_sessions.items():
            if current_time - last_time > max_inactive_time:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.active_sessions[session_id]
            logger.info(f"Removed inactive session {session_id}")
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """Get activity statistics
        
        Returns:
            Dictionary with activity statistics
        """
        current_time = time.time()
        return {
            'last_activity_time': self.last_activity_time,
            'time_since_activity': current_time - self.last_activity_time,
            'is_afk': self.is_afk(),
            'is_extended_afk': self.is_extended_afk(),
            'active_sessions_count': self.get_active_sessions_count(),
            'activity_counts_by_type': {k: len(v) for k, v in self.activity_by_type.items()},
            'total_activity_count': len(self.activity_history)
        }
    
    def has_recent_activity(self, seconds: int = 60) -> bool:
        """Check if there was any activity in the last specified seconds
        
        Args:
            seconds: Number of seconds to check for recent activity
            
        Returns:
            True if there was activity within the specified timeframe
        """
        if not self.activity_history:
            return False
            
        current_time = time.time()
        threshold_time = current_time - seconds
        
        # Check the most recent activity first (optimization)
        if self.activity_history[-1]['timestamp'] >= threshold_time:
            return True
            
        # If we have a lot of history, use binary search
        if len(self.activity_history) > 20:
            return self._binary_search_recent_activity(threshold_time)
            
        # Otherwise just iterate from newest to oldest
        for activity in reversed(self.activity_history):
            if activity['timestamp'] >= threshold_time:
                return True
                
        return False
    
    def _binary_search_recent_activity(self, threshold_time: float) -> bool:
        """Use binary search to efficiently find recent activity
        
        Args:
            threshold_time: The threshold timestamp to search for
            
        Returns:
            True if there is activity after the threshold time
        """
        # Assuming activity_history is sorted by timestamp (which it should be)
        left, right = 0, len(self.activity_history) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if self.activity_history[mid]['timestamp'] < threshold_time:
                left = mid + 1
            else:
                right = mid - 1
                
        # At this point, left is the index of the first activity >= threshold_time
        # or len(self.activity_history) if no such activity exists
        return left < len(self.activity_history)
