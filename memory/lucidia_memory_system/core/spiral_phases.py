"""
Spiral Phases Module for Lucidia's Reflective Consciousness

This module defines the spiral-based reflection system that enables Lucidia to 
progress through various depths of reflection, from shallow observation to deep adaptation.
The spiral model is a core part of Lucidia's reflective consciousness.

Created by MEGAPROMPT (Daniel)
"""

import logging
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import random
from datetime import datetime, timedelta


class SpiralPhase(Enum):
    """
    Enumeration of the three primary spiral phases in Lucidia's reflection system.
    """
    OBSERVATION = 'observation'  # Phase 1: Shallow reflection
    REFLECTION = 'reflection'    # Phase 2: Intermediate reflection
    ADAPTATION = 'adaptation'    # Phase 3: Deep reflection


class SpiralPhaseManager:
    """
    Manages the spiral phase system for Lucidia's reflective consciousness.
    
    The spiral phase system represents Lucidia's progression through different
    depths of reflection, allowing for increasingly complex and creative thinking.
    Each phase has distinct characteristics that influence dream parameters and
    insight generation.
    """
    
    def __init__(self, self_model=None):
        """
        Initialize the spiral phase manager.
        
        Args:
            self_model: Reference to Lucidia's Self Model
        """
        self.logger = logging.getLogger("SpiralPhaseManager")
        self.self_model = self_model
        self.current_phase = SpiralPhase.OBSERVATION
        
        # Phase configuration with parameters for each phase
        self.phase_config = {
            SpiralPhase.OBSERVATION: {
                'name': 'Observation',
                'description': 'Initial phase focused on gathering and organizing perceptions',
                'min_depth': 0.1,
                'max_depth': 0.3,
                'min_creativity': 0.2,
                'max_creativity': 0.5,
                'transition_threshold': 0.85,  # Significance threshold to trigger phase transition
                'insight_weight': 0.5,  # Weight given to insights in this phase
                'focus_areas': ['perception', 'categorization', 'organization'],
                'typical_duration': timedelta(hours=3)
            },
            SpiralPhase.REFLECTION: {
                'name': 'Reflection',
                'description': 'Intermediate phase focused on analysis and pattern recognition',
                'min_depth': 0.4,
                'max_depth': 0.7,
                'min_creativity': 0.4,
                'max_creativity': 0.7,
                'transition_threshold': 0.9,  # Higher threshold for next transition
                'insight_weight': 0.7,  # Higher weight given to insights in this phase
                'focus_areas': ['analysis', 'pattern recognition', 'synthesis', 'abstraction'],
                'typical_duration': timedelta(hours=5)
            },
            SpiralPhase.ADAPTATION: {
                'name': 'Adaptation',
                'description': 'Deep phase focused on integration and transformation',
                'min_depth': 0.7,
                'max_depth': 1.0,
                'min_creativity': 0.6,
                'max_creativity': 1.0,
                'transition_threshold': 0.95,  # Very high threshold for reset
                'insight_weight': 0.9,  # Highest weight given to insights in this phase
                'focus_areas': ['integration', 'transformation', 'creation', 'innovation'],
                'typical_duration': timedelta(hours=7)
            }
        }
        
        # Spiral phase history
        self.phase_history = []
        
        # Phase statistics
        self.phase_stats = {
            SpiralPhase.OBSERVATION: {
                'total_time': 0,  # seconds in this phase
                'transitions': 0,  # number of transitions into this phase
                'insights': [],  # significant insights from this phase
                'last_entered': datetime.now()
            },
            SpiralPhase.REFLECTION: {
                'total_time': 0,
                'transitions': 0,
                'insights': [],
                'last_entered': None
            },
            SpiralPhase.ADAPTATION: {
                'total_time': 0,
                'transitions': 0,
                'insights': [],
                'last_entered': None
            }
        }
        
        # Record initial phase
        self.phase_stats[self.current_phase]['transitions'] += 1
        self.phase_history.append({
            'phase': self.current_phase.value,
            'timestamp': datetime.now().isoformat(),
            'reason': 'initialization'
        })
        
        self.logger.info(f"Spiral Phase Manager initialized in {self.current_phase.value} phase")

    def get_current_phase(self) -> SpiralPhase:
        """
        Get the current spiral phase.
        
        Returns:
            Current SpiralPhase
        """
        return self.current_phase
    
    def get_phase_params(self, phase: Optional[SpiralPhase] = None) -> Dict[str, Any]:
        """
        Get parameters for the specified phase (or current phase if not specified).
        
        Args:
            phase: Specific phase to get parameters for, or None for current phase
            
        Returns:
            Dictionary of phase parameters
        """
        phase = phase or self.current_phase
        return self.phase_config[phase]
    
    def transition_phase(self, significance: float) -> bool:
        """
        Consider a phase transition based on insight significance.
        
        Args:
            significance: Significance value that might trigger transition (0.0 to 1.0)
            
        Returns:
            True if transition occurred, False otherwise
        """
        # Get current phase parameters
        current_params = self.phase_config[self.current_phase]
        
        # Check if significance exceeds transition threshold
        if significance >= current_params['transition_threshold']:
            # Determine next phase
            next_phase = self._determine_next_phase()
            
            # Update phase statistics before transition
            self._update_phase_stats()
            
            # Transition to next phase
            old_phase = self.current_phase
            self.current_phase = next_phase
            
            # Update statistics for new phase
            self.phase_stats[self.current_phase]['transitions'] += 1
            self.phase_stats[self.current_phase]['last_entered'] = datetime.now()
            
            # Record transition in history
            self.phase_history.append({
                'phase': self.current_phase.value,
                'timestamp': datetime.now().isoformat(),
                'reason': f'significance_threshold ({significance:.2f})',
                'previous_phase': old_phase.value
            })
            
            self.logger.info(f"Transitioned from {old_phase.value} to {self.current_phase.value} "
                            f"with significance {significance:.2f}")
            
            # Update self model if available
            if self.self_model and hasattr(self.self_model, 'update_spiral_phase'):
                self.self_model.update_spiral_phase(self.current_phase.value, significance)
            
            return True
        
        return False
    
    def _determine_next_phase(self) -> SpiralPhase:
        """
        Determine the next phase to transition to based on current phase.
        
        Returns:
            Next SpiralPhase
        """
        if self.current_phase == SpiralPhase.OBSERVATION:
            return SpiralPhase.REFLECTION
        elif self.current_phase == SpiralPhase.REFLECTION:
            return SpiralPhase.ADAPTATION
        else:
            # From ADAPTATION, cycle back to OBSERVATION (complete the spiral)
            return SpiralPhase.OBSERVATION
    
    def _update_phase_stats(self) -> None:
        """
        Update statistics for the current phase.
        """
        current_time = datetime.now()
        last_entered = self.phase_stats[self.current_phase]['last_entered']
        
        if last_entered:
            # Calculate time spent in this phase
            time_in_phase = (current_time - last_entered).total_seconds()
            self.phase_stats[self.current_phase]['total_time'] += time_in_phase
    
    def force_phase(self, phase: SpiralPhase, reason: str = "manual_override") -> None:
        """
        Force transition to a specific phase.
        
        Args:
            phase: Target phase to transition to
            reason: Reason for the forced transition
        """
        if phase == self.current_phase:
            self.logger.info(f"Already in {phase.value} phase")
            return
        
        # Update stats for current phase
        self._update_phase_stats()
        
        # Transition to specified phase
        old_phase = self.current_phase
        self.current_phase = phase
        
        # Update statistics for new phase
        self.phase_stats[self.current_phase]['transitions'] += 1
        self.phase_stats[self.current_phase]['last_entered'] = datetime.now()
        
        # Record transition in history
        self.phase_history.append({
            'phase': self.current_phase.value,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'previous_phase': old_phase.value
        })
        
        self.logger.info(f"Forced transition from {old_phase.value} to {self.current_phase.value} "
                        f"due to {reason}")
        
        # Update self model if available
        if self.self_model and hasattr(self.self_model, 'update_spiral_phase'):
            self.self_model.update_spiral_phase(self.current_phase.value, 1.0)  # High significance for forced transition
    
    def record_insight(self, insight: Dict[str, Any]) -> None:
        """
        Record a significant insight from the current phase.
        
        Args:
            insight: Insight information to record
        """
        # Only record high significance insights
        if insight.get('significance', 0.0) >= 0.8:
            self.phase_stats[self.current_phase]['insights'].append({
                'text': insight.get('text', ''),
                'significance': insight.get('significance', 0.0),
                'timestamp': datetime.now().isoformat(),
                'phase': self.current_phase.value
            })
    
    def get_phase_stats(self, phase: Optional[SpiralPhase] = None) -> Dict[str, Any]:
        """
        Get statistics for the specified phase (or current phase if not specified).
        
        Args:
            phase: Specific phase to get statistics for, or None for current phase
            
        Returns:
            Dictionary of phase statistics
        """
        phase = phase or self.current_phase
        return self.phase_stats[phase]
    
    def get_phase_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the history of phase transitions.
        
        Args:
            limit: Maximum number of history entries to return
            
        Returns:
            List of phase transition events
        """
        return self.phase_history[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the spiral phase system.
        
        Returns:
            Dictionary containing spiral phase status information
        """
        # Calculate time in current phase
        current_time = datetime.now()
        last_entered = self.phase_stats[self.current_phase]['last_entered']
        time_in_current_phase = 0
        
        if last_entered:
            time_in_current_phase = (current_time - last_entered).total_seconds()
        
        # Get cycle count (number of complete spirals)
        cycle_count = self.phase_stats[SpiralPhase.ADAPTATION]['transitions']
        
        # Format the status response
        status = {
            'current_phase': {
                'name': self.current_phase.value,
                'description': self.phase_config[self.current_phase]['description'],
                'time_in_phase': time_in_current_phase,
                'focus_areas': self.phase_config[self.current_phase]['focus_areas']
            },
            'spiral_stats': {
                'cycle_count': cycle_count,
                'phase_transitions': sum(stats['transitions'] for phase, stats in self.phase_stats.items()),
                'total_insights': sum(len(stats['insights']) for phase, stats in self.phase_stats.items()),
                'phase_distribution': {
                    phase.value: {
                        'transitions': stats['transitions'],
                        'total_time': stats['total_time']
                    } for phase, stats in self.phase_stats.items()
                }
            },
            'recent_history': self.get_phase_history(5)
        }
        
        return status